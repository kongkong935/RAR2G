import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from lightning.fabric.utilities.data import AttributeDict
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
from mertic.mertic import compute_all_scores
torch.serialization.add_safe_globals([AttributeDict])
torch.set_float32_matmul_precision('medium')

DEFAULT_ARGS = {
    'chexbert_path': "/root/autodl-tmp/checkpoints/chexbert.pth",
    'bert_path': "/root/autodl-tmp/checkpoints/bert-base-uncased",
    'radgraph_path': "/root/autodl-tmp/checkpoints/radgraph.tar.gz",
}

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.compute_all_scores = compute_all_scores
        self.qformer = SimpleQFormerWrapper(num_hidden_layers=6,use_separate_queries=args.use_separate_queries)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        self.visual_encoder.embeddings.mask_token.requires_grad = False

        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
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
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                dtype=torch.float16,
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            # peft_config = LoraConfig(
            #     task_type=TaskType.CAUSAL_LM,
            #     inference_mode=False,
            #     r=args.llm_r,
            #     lora_alpha=args.llm_alpha,
            #     lora_dropout=args.lora_dropout,
            #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            #     bias="none",
            # )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                bias='none',
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


        if args.delta_file is not None :
            is_resume = getattr(args, "resume", False)
            # 加载增量权重
            state_obj = torch.load(
                args.delta_file,
                map_location=torch.device(f'cuda:{torch.cuda.current_device()}')
            )
            state_dict = state_obj['model']

            # 加载权重（允许部分 key 对不上）
            self.load_state_dict(state_dict=state_dict, strict=False)


            # 根据是否为 resume 模式，决定是否冻结加载的参数
            loaded_param_names = set(state_dict.keys())
            for name, param in self.named_parameters():
                if name in loaded_param_names:
                    # resume=True：只加载，不额外改 requires_grad
                    # resume=False：加载并冻结（用于两阶段训练第二阶段）
                    if not is_resume and param.requires_grad:
                        param.requires_grad = False


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


        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
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
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # compute_all_scores 需要字典格式 {id: [text]}，但 compute_ce_scores 需要列表
        # 所以传入字典，compute_all_scores 内部会处理
        ref_dict = {k:[v] for k, v in zip(ids, ref)}
        hypo_dict = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.compute_all_scores(gts=ref_dict, res=hypo_dict, args=DEFAULT_ARGS)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        # 只在 rank 0 上保存文件（避免多 GPU 重复保存）
        if self.trainer.global_rank == 0:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step

            # 保存生成结果和参考文本
            json.dump(hypo_dict, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            json.dump(ref_dict, open(os.path.join(result_folder, 'refs.json'), 'w'))

            # 保存验证指标到 JSON 文件
            metrics_folder = os.path.join(self.hparams.savedmodel_path, 'metrics')
            os.makedirs(metrics_folder, exist_ok=True)

            # 构建指标记录（包含 epoch、step 和所有评估指标）
            metrics_record = {
                'epoch': current_epoch,
                'step': global_step,
                **eval_res  # 展开所有评估指标
            }

            # 保存单次验证指标
            metrics_file = os.path.join(metrics_folder, f"metrics_epoch{current_epoch}_step{global_step}.json")
            json.dump(metrics_record, open(metrics_file, 'w'), indent=2)

            # 更新汇总指标文件（追加模式）
            summary_file = os.path.join(metrics_folder, 'metrics_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            else:
                summary = []
            summary.append(metrics_record)
            json.dump(summary, open(summary_file, 'w'), indent=2)

        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
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

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
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
            # compute_all_scores 需要字典格式 {id: [text]}
            ref_dict = {k: [v] for k, v in zip(ids, ref)}
            hypo_dict = {k: [v] for k, v in zip(ids, hypo)}
            eval_res = self.compute_all_scores(gts=ref_dict, res=hypo_dict, args=DEFAULT_ARGS)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            json.dump(hypo_dict, open(os.path.join(result_folder, "test_result.json"), 'w'))
            json.dump(ref_dict, open(os.path.join(result_folder, "test_refs.json"), 'w'))

            # 保存测试指标到 JSON 文件
            metrics_folder = os.path.join(self.hparams.savedmodel_path, 'metrics')
            os.makedirs(metrics_folder, exist_ok=True)

            # 构建测试指标记录
            test_metrics_record = {
                'test': True,
                'delta_file': getattr(self.hparams, 'delta_file', None),
                **eval_res
            }

            # 保存测试指标
            test_metrics_file = os.path.join(metrics_folder, 'test_metrics.json')
            json.dump(test_metrics_record, open(test_metrics_file, 'w'), indent=2)

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


