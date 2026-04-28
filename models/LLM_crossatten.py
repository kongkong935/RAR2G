import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer, MllamaForConditionalGeneration

from Qformermoudel.qformermoudel import SimpleQFormerWrapper
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
# 导入我们的cross-attention组件
from Flamingo.OpenFlamingo import TextCrossAttentionAdapter


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.qformer = SimpleQFormerWrapper(num_hidden_layers=6, hidden_size=768, num_attention_heads=12,
                                            num_query_tokens=49)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        self.visual_encoder.embeddings.mask_token.requires_grad = False
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert = AutoModel.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert.eval()
        # self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.freeze_tm:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen textal encoder:{args.bert} -- Done')
        else:
            print(f'Loading Trainable textal encoder:{args.textal_encoder} -- Done')
        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        # self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            # self.llama_model = MllamaForConditionalGeneration.from_pretrained(
            #     args.llama_model,
            #     torch_dtype=torch.float16,
            # )
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            # peft_config = LoraConfig(
            #     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha,
            #     lora_dropout=args.lora_dropout
            # )
            # self.llama_model = get_peft_model(self.llama_model, peft_config)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)

            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        # self.token_pooler = PatchTokenPooler(hidden_size=768, side_out=self.args.pool_patch)
        self.llama_proj = nn.Linear(768, 4096)
        self.layer_norm = nn.LayerNorm(4096)
        # 添加位置嵌入：正面（front）和侧面（lateral），帮助Q-Former学习语义
        self.front_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        self.lateral_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        # 添加文本cross-attention适配器
        if args.use_text_cross_attention:
            print("Adding text cross-attention to LLaMA...")
            # 默认从 0 开始，与 Flamingo 原版一致，避免初始阶段影响 LLM
            attn_gate_init = getattr(args, 'cross_attn_attn_gate_init_value', 0.3)
            ff_gate_init = getattr(args, 'cross_attn_ff_gate_init_value', 0.1)
            self.text_cross_attn_adapter = TextCrossAttentionAdapter(
                self.llama_model,
                text_dim=getattr(args, 'text_feature_dim', 768),
                cross_attn_every_n_layers=getattr(args, 'cross_attn_every_n_layers', 8),
                attn_gate_init_value=attn_gate_init,
                ff_gate_init_value=ff_gate_init
            )
            print(f"Text cross-attention added! Attn gate init: {attn_gate_init}, FF gate init: {ff_gate_init}")

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

    # def encode_img(self, images):
    #     image_embeds = []
    #     for image in images:
    #         device = image.device
    #         if self.hparams.global_only:
    #             image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
    #         else:
    #             image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
    #         image_embeds.append(image_embed)
    #
    #     image_embeds = torch.stack(image_embeds).mean(0)
    #     inputs_llama = self.llama_proj(image_embeds)
    #     atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
    #     return inputs_llama, atts_llama

    def encode_img(self, images, count):
        """
        编码图像，添加位置嵌入并根据count创建mask来忽略复制的图像
        Args:
            images: 图像列表，[front_images, lateral_images]，每个是 [B, C, H, W]
            count: tensor [B]，每个样本的图像数量，1或2
        """
        batch_size = images[0].shape[0]
        device = images[0].device
        num_patches = 1370  # 每个图像编码后的patch数量

        pooled_list = []
        # 编码第一张图像（正面）
        outputs = self.visual_encoder(pixel_values=images[0], output_hidden_states=True)
        front_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
        # 添加正面位置嵌入
        front_pos_embed = self.front_position_embedding.expand(batch_size, num_patches, -1)
        front_features = front_features + front_pos_embed
        pooled_list.append(front_features)

        # 编码第二张图像（可能是侧面或复制的正面）
        outputs = self.visual_encoder(pixel_values=images[1], output_hidden_states=True)
        second_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
        # 添加侧面位置嵌入
        lateral_pos_embed = self.lateral_position_embedding.expand(batch_size, num_patches, -1)
        second_features = second_features + lateral_pos_embed
        pooled_list.append(second_features)

        # 拼接两个图像的特征
        pooled = torch.cat(pooled_list, dim=1)  # [B, 2740, 768]

        # 根据count创建attention_mask
        # count=1: 第二个图像（索引1370:2740）应该被mask掉，虽然加了侧面位置嵌入，但会被mask忽略
        # count=2: 两个图像都有效，第一个是正面位置嵌入，第二个是侧面位置嵌入
        encoder_attention_mask = torch.ones(batch_size, num_patches * 2, dtype=torch.long, device=device)
        # 对于count=1的样本，mask掉第二个图像的位置（设为0）
        mask_indices = (count == 1)  # [B]
        encoder_attention_mask[mask_indices, num_patches:] = 0

        q_tokens = self.qformer(encoder_hidden_states=pooled, encoder_attention_mask=encoder_attention_mask)

        inputs_llama = self.llama_proj(q_tokens)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama, q_tokens

    def prompt_wrap(self, img_embeds, atts_img, count):
        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    # def prompt_wrap(self, img_embeds, atts_img, count=None):
    #     """
    #     包装图像嵌入和prompt
    #     Args:
    #         img_embeds: 图像嵌入 [B, N, H]
    #         atts_img: 图像attention mask [B, N]
    #         count: tensor [B]，每个样本的图像数量，1或2
    #     """
    #     batch_size = img_embeds.shape[0]
    #     device = img_embeds.device
    #
    #     # 为每个样本生成包含count信息的prompt
    #     if count is not None:
    #         # 确保count是list或tensor
    #         if isinstance(count, torch.Tensor):
    #             count_list = count.cpu().tolist()
    #         else:
    #             count_list = count if isinstance(count, list) else [count] * batch_size
    #
    #         prompts = []
    #         for b_idx in range(batch_size):
    #             sample_count = count_list[b_idx] if b_idx < len(count_list) else 1
    #             if sample_count == 1:
    #                 image_info = "1 image (front view)"
    #             else:
    #                 image_info = "2 images (front and lateral view)"
    #             prompt_text = f'Human:You are an expert chest radiologist <Img><ImageHere></Img> Generate a comprehensive and detailed radiology report for this chest xray image ({image_info}). \nAssistant:'
    #             prompts.append(prompt_text)
    #     else:
    #         # 如果没有count信息，使用默认prompt
    #         prompt_text = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
    #         prompts = [prompt_text] * batch_size
    #
    #     # 为每个样本分别处理prompt（因为可能不同）
    #     wrapped_img_embeds_list = []
    #     wrapped_atts_img_list = []
    #     max_length = 0
    #
    #     for b_idx in range(batch_size):
    #         prompt = prompts[b_idx]
    #         p_before, p_after = prompt.split('<ImageHere>')
    #         p_before_tokens = self.llama_tokenizer(
    #             p_before, return_tensors="pt", add_special_tokens=False).to(device)
    #         p_after_tokens = self.llama_tokenizer(
    #             p_after, return_tensors="pt", add_special_tokens=False).to(device)
    #         p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)  # [1, L1, H]
    #         p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)  # [1, L2, H]
    #
    #         # 拼接当前样本的prompt和图像嵌入
    #         sample_img_embeds = img_embeds[b_idx:b_idx+1]  # [1, N, H]
    #         wrapped_embeds = torch.cat([p_before_embeds, sample_img_embeds, p_after_embeds], dim=1)  # [1, L1+N+L2, H]
    #         wrapped_img_embeds_list.append(wrapped_embeds)
    #         max_length = max(max_length, wrapped_embeds.shape[1])
    #
    #     # Padding到相同长度并堆叠
    #     hidden_size = wrapped_img_embeds_list[0].shape[2]
    #     padded_embeds_list = []
    #     padded_atts_list = []
    #
    #     for b_idx in range(batch_size):
    #         wrapped_embeds = wrapped_img_embeds_list[b_idx]  # [1, L, H]
    #         current_length = wrapped_embeds.shape[1]
    #         if current_length < max_length:
    #             # Padding
    #             pad_length = max_length - current_length
    #             pad_embeds = torch.zeros(1, pad_length, hidden_size, device=device)
    #             wrapped_embeds = torch.cat([wrapped_embeds, pad_embeds], dim=1)
    #
    #         padded_embeds_list.append(wrapped_embeds)
    #
    #         # 创建attention mask（padding部分为0）
    #         wrapped_atts = torch.ones(max_length, dtype=torch.long, device=device).unsqueeze(0)
    #         if current_length < max_length:
    #             wrapped_atts[:, current_length:] = 0
    #         padded_atts_list.append(wrapped_atts)
    #
    #     # 堆叠所有样本
    #     wrapped_img_embeds = torch.cat(padded_embeds_list, dim=0)  # [B, max_length, H]
    #     wrapped_atts_img = torch.cat(padded_atts_list, dim=0)  # [B, max_length]
    #
    #     return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        count = samples["count"]
        img_embeds, atts_img, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, count=count)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        # 在LLM调用之前设置文本条件
        if self.args.use_text_cross_attention:
            # text_tokenizer = self.bert_tokenizer(list(samples["prereport"]),
            #                                          padding=True,
            #                                          truncation=True,
            #                                          max_length=100,
            #                                          return_tensors='pt')
            # text_tokenizer = text_tokenizer.to(self.device)
            # text_emb = self.bert(**text_tokenizer).last_hidden_state
            self.text_cross_attn_adapter.condition_text_features(q_tokens)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # for name, p in self.named_parameters():
        #     if p.requires_grad and p.grad is None:
        #         print("unused:", name)
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
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        count = samples["count"]
        img_embeds, atts_img, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, count=count)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        # 在LLM调用之前设置文本条件
        # if self.args.use_text_cross_attention:
        #     text_tokenizer = self.bert_tokenizer(list(samples["prereport"]),
        #                                              padding=True,
        #                                              truncation=True,
        #                                              max_length=100,
        #                                              return_tensors='pt')
        #     text_tokenizer = text_tokenizer.to(self.device)
        #     text_emb = self.bert(**text_tokenizer).last_hidden_state
        #     self.text_cross_attn_adapter.condition_text_features(text_emb)
        if self.args.use_text_cross_attention:
            # text_tokenizer = self.bert_tokenizer(list(samples["prereport"]),
            #                                          padding=True,
            #                                          truncation=True,
            #                                          max_length=100,
            #                                          return_tensors='pt')
            # text_tokenizer = text_tokenizer.to(self.device)
            # text_emb = self.bert(**text_tokenizer).last_hidden_state
            self.text_cross_attn_adapter.condition_text_features(q_tokens)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        """
        验证轮结束：聚合结果并保存

        注意：如果使用多 GPU，会自动聚合所有 GPU 的结果
        """
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # 在 DDP 模式下，跨所有 GPU 聚合结果
        if self.trainer.num_devices > 1:
            # 使用 torch.distributed.all_gather_object 来收集 Python 对象
            if dist.is_available() and dist.is_initialized():
                gathered_ref = [None] * dist.get_world_size()
                gathered_hypo = [None] * dist.get_world_size()
                gathered_ids = [None] * dist.get_world_size()

                dist.all_gather_object(gathered_ref, ref)
                dist.all_gather_object(gathered_hypo, hypo)
                dist.all_gather_object(gathered_ids, ids)

                # 展平所有 GPU 的结果
                ref = [item for sublist in gathered_ref for item in sublist]
                hypo = [item for sublist in gathered_hypo for item in sublist]
                ids = [item for sublist in gathered_ids for item in sublist]

        # 所有节点都计算评估结果（用于同步）
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        # 只在 rank 0 上保存文件和 checkpoint（避免重复）
        if self.trainer.global_rank == 0:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
            self.print(eval_res)

            val_score = 0
            for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
                val_score += eval_res[score_type] * weight

            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score

        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        count = samples["count"]
        img_embeds, atts_img, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, count=count)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        # 在LLM调用之前设置文本条件
        # if self.args.use_text_cross_attention:
        #     text_tokenizer = self.bert_tokenizer(list(samples["prereport"]),
        #                                              padding=True,
        #                                              truncation=True,
        #                                              max_length=100,
        #                                              return_tensors='pt')
        #     text_tokenizer = text_tokenizer.to(self.device)
        #     text_emb = self.bert(**text_tokenizer).last_hidden_state
        #     self.text_cross_attn_adapter.condition_text_features(text_emb)
        if self.args.use_text_cross_attention:
            # text_tokenizer = self.bert_tokenizer(list(samples["prereport"]),
            #                                          padding=True,
            #                                          truncation=True,
            #                                          max_length=100,
            #                                          return_tensors='pt')
            # text_tokenizer = text_tokenizer.to(self.device)
            # text_emb = self.bert(**text_tokenizer).last_hidden_state
            self.text_cross_attn_adapter.condition_text_features(q_tokens)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        测试轮结束：聚合结果并保存

        注意：如果使用多 GPU，会自动聚合所有 GPU 的结果
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # 在 DDP 模式下，跨所有 GPU 聚合结果
        if self.trainer.num_devices > 1:
            # 使用 torch.distributed.all_gather_object 来收集 Python 对象
            if dist.is_available() and dist.is_initialized():
                gathered_ref = [None] * dist.get_world_size()
                gathered_hypo = [None] * dist.get_world_size()
                gathered_ids = [None] * dist.get_world_size()

                dist.all_gather_object(gathered_ref, ref)
                dist.all_gather_object(gathered_hypo, hypo)
                dist.all_gather_object(gathered_ids, ids)

                # 展平所有 GPU 的结果
                ref = [item for sublist in gathered_ref for item in sublist]
                hypo = [item for sublist in gathered_hypo for item in sublist]
                ids = [item for sublist in gathered_ids for item in sublist]

        # 只在 rank 0 上执行评估和保存（避免重复）
        if self.trainer.global_rank == 0:
            ref = {k: [v] for k, v in zip(ids, ref)}
            hypo = {k: [v] for k, v in zip(ids, hypo)}
            eval_res = self.score(ref=ref, hypo=hypo)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), 'w'))
            json.dump(ref, open(os.path.join(result_folder, "test_refs.json"), 'w'))
            self.print(f"🧪 Test result of {self.hparams.delta_file}: {eval_res}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # 分离视觉编码器参数、cross-attention参数（包括gate）和其他参数
        vision_params = []
        cross_attn_params = []
        cross_attn_gate_params = []  # 单独分离 gate 参数
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
            if 'visual_encoder' in name or 'token_pooler' in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        # 获取各参数组的学习率
        vision_lr = getattr(self.hparams, 'vision_lr', self.hparams.learning_rate)
        cross_attn_lr = getattr(self.hparams, 'cross_attn_lr', self.hparams.learning_rate)
        base_lr = self.hparams.learning_rate

        # 获取 weight_decay 配置
        weight_decay = getattr(self.hparams, 'weight_decay', 0.01)
        cross_attn_weight_decay = getattr(self.hparams, 'cross_attn_weight_decay', weight_decay)
        # Gate 参数使用 weight_decay = 0，避免过度抑制
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
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay})
            print(f'Other parameters learning rate: {base_lr}, weight_decay: {weight_decay}')

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


# class PatchTokenPooler(nn.Module):
#     def __init__(self, hidden_size, target_tokens=49):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.target_tokens = target_tokens
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, hidden_size),
#         )

# def forward(self, hidden_states):
#     cls_token = hidden_states[:, :1]  # (B,1,H)
#     patch_tokens = hidden_states[:, 1:]  # (B,N,H)
#     b, n, h = patch_tokens.size()
#     side = int(math.sqrt(n))
#     if side * side != n:
#         side += 1
#         needed = side * side - n
#         pad = patch_tokens.new_zeros(b, needed, h)
#         patch_tokens = torch.cat([patch_tokens, pad], dim=1)
#     patch_tokens = patch_tokens.transpose(1, 2).reshape(b, h, side, side)
#     pooled = F.adaptive_avg_pool2d(patch_tokens, (7, 7))
#     pooled = pooled.reshape(b, h, -1).transpose(1, 2)  # (B,49,H)
#     if self.target_tokens > 1:
#         pooled = torch.cat([cls_token, pooled[:, : self.target_tokens - 1]], dim=1)
#     else:
#         pooled = cls_token
#     return self.mlp(pooled)


class PatchTokenPooler(nn.Module):
    def __init__(self, hidden_size, side_out):
        super().__init__()
        self.hidden_size = hidden_size
        self.side_out = side_out
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states):
        cls_token = hidden_states[:, :1]
        patch_tokens = hidden_states[:, 1:]  # [B, N, H]
        b, n, h = patch_tokens.size()
        side = int(math.sqrt(n))
        x = patch_tokens.transpose(1, 2).reshape(b, h, side, side)  # [B,H,37,37]
        x = self.conv(x)  # [B,H,37,37]
        x = F.adaptive_avg_pool2d(x, (self.side_out, self.side_out))
        x = x.reshape(b, h, -1).transpose(1, 2)
        tokens = torch.cat([cls_token, x], dim=1)
        return self.mlp(tokens)
