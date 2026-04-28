import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import lightning.pytorch as pl
from transformers import AutoModel, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from Qformermoudel.qformermoudel import SimpleQFormerWrapper
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ========== 自定义视觉编码器和 Q-Former（已注释，暂时使用 LLaVA 自带）==========
        # # 使用自定义的视觉编码器和 Q-Former（而不是 LLaVA 自带的）
        # # 这样可以：
        # # 1. 使用医学影像专用的视觉编码器（如 Rad-DINO）
        # # 2. 支持两个视图（正面+侧面）的特殊处理
        # # 3. 通过 Q-Former 的两个独立 query token 分别查询两个视图
        # self.qformer=SimpleQFormerWrapper(num_hidden_layers=6,use_separate_queries=True)
        # print(f'Loading vision encoder:{args.vision_model}')
        #
        # self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        # self.visual_encoder.embeddings.mask_token.requires_grad = False
        # if args.freeze_vm:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #
        # # 获取 LLaVA 的实际 hidden_size（从 text_config 中获取，因为 LLaVA 的 LLM 部分使用这个维度）
        # # LLaVA 的配置结构：vision_config 和 text_config
        # if hasattr(self.llava_model.config, 'text_config'):
        #     llava_hidden_size = self.llava_model.config.text_config.hidden_size
        # elif hasattr(self.llava_model.config, 'hidden_size'):
        #     llava_hidden_size = self.llava_model.config.hidden_size
        # else:
        #     # 默认值，但应该从 config 获取
        #     llava_hidden_size = 4096
        #     print(f"Warning: Could not find hidden_size in LLaVA config, using default {llava_hidden_size}")
        #
        # print(f"LLaVA hidden_size: {llava_hidden_size}")
        #
        # # 投影层：从 Q-Former 输出维度 (768) 到 LLaVA 的 hidden_size
        # # 这个投影层将自定义视觉编码器+Q-Former 的输出映射到 LLaVA LLM 的输入空间
        # self.llava_proj = nn.Linear(768, llava_hidden_size)
        # self.layer_norm = nn.LayerNorm(llava_hidden_size)
        # # 添加位置嵌入：正面（front）和侧面（lateral），帮助Q-Former学习语义
        # self.front_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        # self.lateral_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        # ===================================================================================

        # 使用 LLaVA 自带的图像处理器和视觉编码器
        print('Loading LLAVA')
        self.llava_tokenizer = AutoTokenizer.from_pretrained(args.llava_model, use_fast=True)

        if self.llava_tokenizer.pad_token is None:
            self.llava_tokenizer.pad_token_id = self.llava_tokenizer.eos_token_id
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            args.llava_model,
            torch_dtype=torch.bfloat16,
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.llm_r,
            lora_alpha=args.llm_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.llava_model = get_peft_model(self.llava_model, peft_config)

        self.llava_model.print_trainable_parameters()
        print('Loading LLAMA LoRA Done')

        # # 冻结 LLaVA 的所有参数，只解冻多模态投影层用于任务适配
        # for name, param in self.llava_model.named_parameters():
        #     # 解冻多模态投影层（mm_projector 或 multi_modal_projector）
        #     if 'mm_projector' in name or 'multi_modal_projector' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            # (Cider(), "CIDEr")
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    # ========== 自定义图像编码方法（已注释，暂时使用 LLaVA 自带）==========
    # def encode_img(self, images, count):
    #     """
    #     编码图像，添加位置嵌入并根据count创建mask来忽略复制的图像
    #     Args:
    #         images: 图像列表，[front_images, lateral_images]，每个是 [B, C, H, W]
    #         count: tensor [B]，每个样本的图像数量，1或2
    #     """
    #     batch_size = images[0].shape[0]
    #     device = images[0].device
    #     num_patches = 1370  # 每个图像编码后的patch数量
    #
    #     pooled_list = []
    #     # 编码第一张图像（正面）
    #     outputs = self.visual_encoder(pixel_values=images[0], output_hidden_states=True)
    #     front_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
    #     # 添加正面位置嵌入
    #     front_pos_embed = self.front_position_embedding.expand(batch_size, num_patches, -1)
    #     front_features = front_features + front_pos_embed
    #     pooled_list.append(front_features)
    #
    #     # 编码第二张图像（可能是侧面或复制的正面）
    #     outputs = self.visual_encoder(pixel_values=images[1], output_hidden_states=True)
    #     second_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
    #     # 添加侧面位置嵌入
    #     lateral_pos_embed = self.lateral_position_embedding.expand(batch_size, num_patches, -1)
    #     second_features = second_features + lateral_pos_embed
    #     pooled_list.append(second_features)
    #
    #     # 拼接两个图像的特征
    #     pooled = torch.cat(pooled_list, dim=1)  # [B, 2740, 768]
    #
    #     # 根据count创建attention_mask
    #     # count=1: 第二个图像（索引1370:2740）应该被mask掉，虽然加了侧面位置嵌入，但会被mask忽略
    #     # count=2: 两个图像都有效，第一个是正面位置嵌入，第二个是侧面位置嵌入
    #     encoder_attention_mask = torch.ones(batch_size, num_patches * 2, dtype=torch.long, device=device)
    #     # 对于count=1的样本，mask掉第二个图像的位置（设为0）
    #     mask_indices = (count == 1)  # [B]
    #     encoder_attention_mask[mask_indices, num_patches:] = 0
    #
    #     q_tokens = self.qformer(encoder_hidden_states=pooled, encoder_attention_mask=encoder_attention_mask)
    #
    #     inputs_llava = self.llava_proj(q_tokens)
    #     atts_llava = torch.ones(inputs_llava.size()[:-1], dtype=torch.long).to(device)
    #     return inputs_llava, atts_llava
    #
    # def prompt_wrap(self, img_embeds, atts_img, count):
    #     prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
    #     batch_size = img_embeds.shape[0]
    #     p_before, p_after = prompt.split('<ImageHere>')
    #     p_before_tokens = self.llava_tokenizer(
    #         p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #     p_after_tokens = self.llava_tokenizer(
    #         p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #     p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
    #     p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
    #     wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
    #     wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
    #     return wrapped_img_embeds, wrapped_atts_img
    # ===================================================================================

    def forward(self, samples):
        """
        使用 LLaVA 的标准调用方式：pixel_values + input_ids
        注意：图像和文本已经在 data_helper.py 中用 processor 处理过了
        """
        # 从 samples 中获取已经处理好的数据
        pixel_values = samples["pixel_values"]  # [B, C, H, W]
        input_ids = samples["input_ids"]  # [B, seq_len]
        attention_mask = samples["attention_mask"]  # [B, seq_len]
        prompt_lengths = samples["prompt_length"]  # 可能是标量或 tensor

        # 处理 prompt_lengths：确保是 tensor 或标量（用于推理阶段截断生成）
        if isinstance(prompt_lengths, (int, float)):
            prompt_length = int(prompt_lengths)
        elif isinstance(prompt_lengths, torch.Tensor):
            prompt_length = int(prompt_lengths[0].item() if prompt_lengths.dim() > 0 else prompt_lengths.item())
        else:
            prompt_length = int(prompt_lengths[0] if isinstance(prompt_lengths, (list, tuple)) else prompt_lengths)

        # 创建 labels：只对目标文本部分计算 loss，prompt 和 padding 部分设为 -100
        labels = input_ids.clone()

        # ---- 简化版：只考虑左填充，直接使用 data_helper 传入的 GT 起始位置 ----
        # 结构假设为：[PAD...PAD][PROMPT+<image>][GT TOKENS][PAD...PAD]
        gt_start = samples.get("gt_start", None)
        if gt_start is not None:
            if isinstance(gt_start, torch.Tensor):
                gt_start = gt_start.to(input_ids.device)
                seq_len = input_ids.size(1)
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, L]
                labels[positions < gt_start.unsqueeze(1)] = -100
            else:
                labels[:, :int(gt_start)] = -100
        else:
            # 回退：如果没有 gt_start，就简单按 prompt_length 处理（旧逻辑）
            labels[:, :prompt_length] = -100

        # mask padding tokens（右侧 PAD 不参与 loss）
        pad_token_id = self.llava_tokenizer.pad_token_id
        if pad_token_id is not None:
            labels = labels.masked_fill(input_ids == pad_token_id, -100)

        # # 可选调试：在进入 LLM 之前，打印「参与 loss 的 GT 文本」和「对应的 prompt」
        # if self.training and getattr(self.hparams, "debug_gt", True):
        #     try:
        #         with torch.no_grad():
        #             idx = 0  # 只看第一个样本
        #             ids = input_ids[idx]
        #             labs = labels[idx]
        #             # 参与 loss 的 token：labels != -100
        #             gt_token_ids = ids[labs != -100]
        #             # 不参与 loss 的部分（prompt + pad），方便对比
        #             prompt_token_ids = ids[labs == -100]
        #
        #             gt_text = self.llava_tokenizer.decode(
        #                 gt_token_ids,
        #                 skip_special_tokens=True,
        #                 clean_up_tokenization_spaces=True,
        #             )
        #             prompt_text = self.llava_tokenizer.decode(
        #                 prompt_token_ids,
        #                 skip_special_tokens=True,
        #                 clean_up_tokenization_spaces=True,
        #             )
        #             print("\n========== [DEBUG GT ALIGNMENT] ==========")
        #             print(f"Prompt (ignored in loss):\n{prompt_text}\n")
        #             print(f"GT used for loss:\n{gt_text}\n")
        #             print("=========================================\n")
        #     except Exception as e:
        #         print(f"[DEBUG_GT] decode failed: {e}")

        # 调用 LLaVA 模型
        outputs = self.llava_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'],
                                                                        eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        """
        使用 LLaVA 的标准调用方式生成文本
        注意：生成时不依赖GT，只使用图像和prompt
        """
        # 从 samples 中获取已经处理好的数据
        pixel_values = samples["pixel_values"]  # [B, C, H, W]
        input_ids = samples["input_ids"]  # [B, seq_len]
        attention_mask = samples["attention_mask"]  # [B, seq_len]
        prompt_length = samples.get("prompt_length", 31)  # 可能是标量 / tensor / list

        device = pixel_values.device

        # 只使用 prompt 部分进行生成，不包含GT文本
        # input_ids 已经包含了 prompt，直接使用即可
        # 使用 LLaVA 生成（只基于图像和prompt，不依赖GT）
        outputs = self.llava_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
        )

        # 计算每个样本的真实 prompt 长度（等于输入的有效 token 数）
        # attention_mask: [B, seq_len]，其中 prompt + prompt 后的输入 token 为1，padding为0
        input_lengths = attention_mask.sum(dim=1).tolist()

        # 解码生成结果：只解码新生成的部分（从输入长度之后开始）
        hypo = []
        for seq, in_len in zip(outputs, input_lengths):
            in_len = int(in_len)
            new_tokens = seq[in_len:]
            hypo.append(self.decode(new_tokens))

        # 如果 samples 中有 GT（用于最终评估），直接使用原始文本，不再 tokenizer+decode，避免出现 <pad>
        ref = None
        if 'input_text' in samples:
            # 此时 samples['input_text'] 应该是长度为 B 的字符串列表
            ref = samples['input_text']

        # 保存结果（如果 ref 为 None，则只保存预测）
        if ref is not None:
            self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        else:
            self.val_step_outputs.append({"hypo": hypo, "id": samples["id"]})

        return hypo, ref if ref is not None else hypo

    def decode(self, output_token):
        """
        将模型输出的 token 序列解码为文本。

        注意：
        - 只传入 prompt 之后的 new tokens（在 validation_step/test_step 中已经截断）
        - 使用 skip_special_tokens=True 去掉 <pad>、<s> 等特殊符号
        - 额外去掉 LLaVA 里常见的 <image> 标记
        """
        # 防御性判断：空序列直接返回空字符串
        if output_token is None or output_token.numel() == 0:
            return ""

        # 使用 tokenizer 内部的 special_tokens_mask 去掉 <pad> / <bos> / <eos> 等
        output_text = self.llava_tokenizer.decode(
            output_token,
            skip_special_tokens=True,  # 关键：不再把 <pad> 等解码出来
            clean_up_tokenization_spaces=True,
        )

        # 去掉可能残留的图像占位符标记
        output_text = output_text.replace("<image>", "").replace("<ImageHere>", "")

        # 一些模型会在结尾附带 '</s>' 或多余空白，这里再做一次 strip
        output_text = output_text.split('</s>')[0].strip()

        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    # def on_validation_epoch_end(self):
    #     """
    #     验证轮结束：聚合结果并保存
    #
    #     注意：如果使用多 GPU，会自动聚合所有 GPU 的结果
    #     """
    #     ref, hypo, ids = [], [], []
    #     has_ref = False
    #     for i in self.val_step_outputs:
    #         hypo.extend(i['hypo'])
    #         ids.extend(i['id'])
    #         if 'ref' in i and i['ref'] is not None:
    #             ref.extend(i['ref'])
    #             has_ref = True
    #
    #     # 在 DDP 模式下，跨所有 GPU 聚合结果
    #     if self.trainer.num_devices > 1:
    #         # 使用 torch.distributed.all_gather_object 来收集 Python 对象
    #         if dist.is_available() and dist.is_initialized():
    #             gathered_hypo = [None] * dist.get_world_size()
    #             gathered_ids = [None] * dist.get_world_size()
    #             gathered_ref = [None] * dist.get_world_size() if has_ref else None
    #
    #             dist.all_gather_object(gathered_hypo, hypo)
    #             dist.all_gather_object(gathered_ids, ids)
    #             if has_ref:
    #                 dist.all_gather_object(gathered_ref, ref)
    #
    #             # 展平所有 GPU 的结果
    #             hypo = [item for sublist in gathered_hypo for item in sublist]
    #             ids = [item for sublist in gathered_ids for item in sublist]
    #             if has_ref:
    #                 ref = [item for sublist in gathered_ref for item in sublist]
    #
    #     # 只在 rank 0 上保存文件和评估（避免重复）
    #     if self.trainer.global_rank == 0:
    #         result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
    #         os.makedirs(result_folder, exist_ok=True)
    #         current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
    #
    #         # 保存生成的预测结果
    #         hypo_dict = {k: [v] for k, v in zip(ids, hypo)}
    #         json.dump(hypo_dict, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
    #
    #         # 如果有GT，则计算评估指标并保存GT（但不影响生成过程）
    #         if has_ref and len(ref) > 0:
    #             ref_dict = {k: [v] for k, v in zip(ids, ref)}
    #             eval_res = self.score(ref=ref_dict, hypo=hypo_dict)
    #             self.log_dict(eval_res, sync_dist=True, logger=True)
    #             json.dump(ref_dict, open(os.path.join(result_folder, 'refs.json'), 'w'))
    #             self.print(eval_res)
    #
    #             val_score = 0
    #             for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
    #                 val_score += eval_res[score_type] * weight
    #
    #             if val_score > self.val_score:
    #                 self.save_checkpoint(eval_res)
    #                 self.val_score = val_score
    #         else:
    #             # 没有GT，只保存预测结果
    #             self.print(f"✅ Validation completed. Generated {len(hypo)} predictions (no GT for evaluation)")
    #
    #     self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        """
        使用 LLaVA 的标准调用方式生成文本
        注意：生成时不依赖GT，只使用图像和prompt
        """
        # 从 samples 中获取已经处理好的数据
        pixel_values = torch.stack(samples["pixel_values"])  # [B, C, H, W]
        input_ids = torch.stack(samples["input_ids"])  # [B, seq_len]
        attention_mask = torch.stack(samples["attention_mask"])  # [B, seq_len]
        prompt_length = samples.get("prompt_length", 31)  # 可能是标量 / tensor / list

        device = pixel_values.device

        # 只使用 prompt 部分进行生成，不包含GT文本
        # input_ids 已经包含了 prompt，直接使用即可
        # 使用 LLaVA 生成（只基于图像和prompt，不依赖GT）
        outputs = self.llava_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
        )

        # 计算每个样本的真实 prompt 长度（等于输入的有效 token 数）
        input_lengths = attention_mask.sum(dim=1).tolist()

        # 解码生成结果：只解码新生成的部分
        hypo = []
        for seq, in_len in zip(outputs, input_lengths):
            in_len = int(in_len)
            new_tokens = seq[in_len:]
            hypo.append(self.decode(new_tokens))

        # 如果 samples 中有 GT（用于最终评估），直接使用原始文本，不再 tokenizer+decode，避免出现 <pad>
        ref = None
        if 'input_text' in samples:
            # 此时 samples['input_text'] 应该是长度为 B 的字符串列表
            ref = samples['input_text']

        # 保存结果（如果 ref 为 None，则只保存预测）
        if ref is not None:
            self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        else:
            self.test_step_outputs.append({"hypo": hypo, "id": samples["id"]})

        return hypo, ref if ref is not None else hypo

    def on_test_epoch_end(self):
        """
        测试轮结束：聚合结果并保存

        注意：如果使用多 GPU，会自动聚合所有 GPU 的结果
        """
        ref, hypo, ids = [], [], []
        has_ref = False
        for i in self.test_step_outputs:
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
            if 'ref' in i and i['ref'] is not None:
                ref.extend(i['ref'])
                has_ref = True

        # 在 DDP 模式下，跨所有 GPU 聚合结果
        if self.trainer.num_devices > 1:
            # 使用 torch.distributed.all_gather_object 来收集 Python 对象
            if dist.is_available() and dist.is_initialized():
                gathered_hypo = [None] * dist.get_world_size()
                gathered_ids = [None] * dist.get_world_size()
                gathered_ref = [None] * dist.get_world_size() if has_ref else None

                dist.all_gather_object(gathered_hypo, hypo)
                dist.all_gather_object(gathered_ids, ids)
                if has_ref:
                    dist.all_gather_object(gathered_ref, ref)

                # 展平所有 GPU 的结果
                hypo = [item for sublist in gathered_hypo for item in sublist]
                ids = [item for sublist in gathered_ids for item in sublist]
                if has_ref:
                    ref = [item for sublist in gathered_ref for item in sublist]

        # 只在 rank 0 上执行评估和保存（避免重复）
        if self.trainer.global_rank == 0:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)

            # 保存生成的预测结果
            hypo_dict = {k: [v] for k, v in zip(ids, hypo)}
            json.dump(hypo_dict, open(os.path.join(result_folder, "test_result.json"), 'w'))

            # 如果有GT，则计算评估指标并保存GT（但不影响生成过程）
            if has_ref and len(ref) > 0:
                ref_dict = {k: [v] for k, v in zip(ids, ref)}
                eval_res = self.score(ref=ref_dict, hypo=hypo_dict)
                json.dump(ref_dict, open(os.path.join(result_folder, "test_refs.json"), 'w'))
                self.print(f"🧪 Test result of {self.hparams.delta_file}: {eval_res}")
            else:
                # 没有GT，只保存预测结果
                self.print(f"🧪 Test completed. Generated {len(hypo)} predictions (no GT for evaluation)")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # 分离视觉编码器参数、cross-attention参数（包括gate）、mm_projector参数和其他参数
        vision_params = []
        cross_attn_params = []
        cross_attn_gate_params = []  # 单独分离 gate 参数
        mm_projector_params = []  # LLaVA 的 mm_projector 参数
        other_params = []

        # 获取cross-attention参数的ID集合（用于排除）
        cross_attn_param_ids = set()
        if hasattr(self, 'text_cross_attn_adapter'):
            cross_attn_params_list = self.text_cross_attn_adapter.get_trainable_parameters()
            cross_attn_param_ids = {id(p) for p in cross_attn_params_list}

            # 分离 gate 参数和其他 cross-attention 参数
            # Gate 参数名称包含 'attn_gate' 或 'ff_gate'
            for name, param in self.named_parameters():
                if id(param) in cross_attn_param_ids:
                    if 'attn_gate' in name or 'ff_gate' in name:
                        cross_attn_gate_params.append(param)
                    else:
                        cross_attn_params.append(param)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # 跳过cross-attention参数（已在上面单独处理）
            if id(param) in cross_attn_param_ids:
                continue
            # 分离 mm_projector 参数（LLaVA 的多模态投影层）
            if 'mm_projector' in name or 'multi_modal_projector' in name:
                mm_projector_params.append(param)
            elif 'visual_encoder' in name or 'token_pooler' in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        # 获取各参数组的学习率
        vision_lr = getattr(self.hparams, 'vision_lr', self.hparams.learning_rate)
        cross_attn_lr = getattr(self.hparams, 'cross_attn_lr', self.hparams.learning_rate)
        mm_projector_lr = getattr(self.hparams, 'mm_projector_lr', self.hparams.learning_rate)
        base_lr = self.hparams.learning_rate

        # 获取 weight_decay 配置
        weight_decay = getattr(self.hparams, 'weight_decay', 0.01)
        cross_attn_weight_decay = getattr(self.hparams, 'cross_attn_weight_decay', weight_decay)
        gate_weight_decay = getattr(self.hparams, 'cross_attn_gate_weight_decay', 0.0)

        # 创建参数组
        param_groups = []
        if vision_params:
            param_groups.append({'params': vision_params, 'lr': vision_lr, 'weight_decay': weight_decay})
            print(f'Vision encoder learning rate: {vision_lr}, weight_decay: {weight_decay}')
        if cross_attn_params:
            param_groups.append(
                {'params': cross_attn_params, 'lr': cross_attn_lr, 'weight_decay': cross_attn_weight_decay})
            print(f'Cross-attention learning rate: {cross_attn_lr}, weight_decay: {cross_attn_weight_decay}')
        if cross_attn_gate_params:
            param_groups.append(
                {'params': cross_attn_gate_params, 'lr': cross_attn_lr, 'weight_decay': gate_weight_decay})
            print(
                f'Cross-attention GATE learning rate: {cross_attn_lr}, weight_decay: {gate_weight_decay} (set to 0 to avoid over-regularization)')
        if mm_projector_params:
            param_groups.append({'params': mm_projector_params, 'lr': mm_projector_lr, 'weight_decay': weight_decay})
            print(f'MM Projector learning rate: {mm_projector_lr}, weight_decay: {weight_decay}')
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay})
            print(f'Other parameters learning rate: {base_lr}, weight_decay: {weight_decay}')

        # 检查是否有可训练的参数
        if len(param_groups) == 0:
            print("⚠️  Warning: No trainable parameters found! All parameters are frozen.")
            print("   Creating a dummy parameter to avoid optimizer initialization error.")
            # 创建一个虚拟参数，避免优化器初始化错误
            dummy_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            param_groups.append({'params': [dummy_param], 'lr': base_lr, 'weight_decay': weight_decay})
            print("   Note: This is a placeholder. Consider unfreezing some parameters for training.")

        optimizer = torch.optim.AdamW(param_groups)

        # 打印优化器配置
        print(f'\n=== Optimizer Configuration ===')
        for i, group in enumerate(optimizer.param_groups):
            param_count = sum(p.numel() for p in group['params'])
            if group['params'] == cross_attn_gate_params:
                print(f'Group {i}: Cross-Attention GATE (special handling)')
            elif group['params'] == cross_attn_params:
                print(f'Group {i}: Cross-Attention (non-gate)')
            elif group['params'] == vision_params:
                print(f'Group {i}: Vision Encoder')
            else:
                print(f'Group {i}: Other Parameters')
            print(f'  - Learning rate: {group["lr"]}')
            print(f'  - Weight decay: {group["weight_decay"]}')
            print(f'  - Parameters: {param_count:,}')
        print('=' * 35 + '\n')

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()



