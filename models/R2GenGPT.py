import os
import json
# 在导入 transformers 之前设置环境变量，避免 torch.load 安全检查问题
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
import torch
import torch.nn as nn
import lightning.pytorch as pl


from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer

from Qformermoudel.qformermoudel import SimpleQFormerWrapper
from peft import get_peft_model, LoraConfig, TaskType
from mertic.mertic import compute_all_scores
from retrieval.external_memory import ExternalMemoryRetriever
torch.set_float32_matmul_precision('medium')

DEFAULT_ARGS = {
    'chexbert_path': "/root/autodl-tmp/checkpoint/chexbert.pth",
    'bert_path': "/root/autodl-tmp/checkpoint/bert-base-uncased",
    'radgraph_path': "/root/autodl-tmp/checkpoint/radgraph.tar.gz",
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
        qformer_pretrained_path = getattr(args, 'stage1_checkpoint', None)
        
        # 第一个 Q-Former：用于训练（随机初始化）
        self.qformer_train = SimpleQFormerWrapper(num_hidden_layers=3, use_separate_queries=args.use_separate_queries)
        # 第二个 Q-Former：加载预训练权重（冻结，不训练）
        self.qformer_pretrained = SimpleQFormerWrapper(num_hidden_layers=3, pretrained_path=qformer_pretrained_path)
        # 冻结预训练的 Q-Former
        for param in self.qformer_pretrained.parameters():
            param.requires_grad = False
        self.qformer_pretrained.eval()

        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert = AutoModel.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert.eval()

        if args.freeze_tm:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen textal encoder:{args.bert} -- Done')
        else:
            print(f'Loading Trainable textal encoder:{args.textal_encoder} -- Done')

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
        self.llama_model = LlamaForCausalLM.from_pretrained(
            args.llama_model,
            dtype=torch.bfloat16,
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0


        if args.llm_use_lora:
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
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        self.embed_tokens = self.llama_model.get_input_embeddings()



        self.llama_proj = nn.Linear(768, 4096)  # 用于投影视觉特征和检索向量
        self.layer_norm = nn.LayerNorm(4096)
        self.bert_proj = nn.Linear(768, 4096)  # 用于投影视觉特征和检索向量
        self.bert_norm = nn.LayerNorm(4096)
        # 添加位置嵌入：正面（front）和侧面（lateral），帮助Q-Former学习语义
        self.front_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        self.lateral_position_embedding = nn.Parameter(torch.randn(1, 1, 768))
        self.end_sym = args.end_sym
        # 尽量短：图像为主，检索为辅；不要照抄；冲突以图像为准。
        self.prompt = (
            "Generate a comprehensive and detailed diagnosis report for this chest X-ray image by incorporating the retrieved case evidence."
        )
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
        # 加载外部记忆库（如果提供了路径）
        ext_memory_path = getattr(args, 'ext_memory_path', None)
        ext_memory_topn = getattr(args, 'ext_memory_topn', 9)
        self.memory_retriever = ExternalMemoryRetriever(ext_memory_path, topn=ext_memory_topn)
        # 融合数量：如果 ext_memory_fusion 为 None，则使用 topn（融合所有检索到的报告）
        ext_memory_fusion = getattr(args, 'ext_memory_fusion', None)
        self.ext_memory_fusion = ext_memory_fusion if ext_memory_fusion is not None else ext_memory_topn
        # 检索融合权重：统一 train/val/test
        self.ext_memory_alpha = getattr(args, 'ext_memory_alpha', 0.5)

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
        front_features_raw = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
        front_features = front_features_raw
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

        # 使用训练的 Q-Former
        q_tokens_train = self.qformer_train(encoder_hidden_states=pooled, encoder_attention_mask=encoder_attention_mask)
        
        # 使用预训练的 Q-Former（冻结，不需要梯度）
        with torch.no_grad():
            q_tokens_pretrained = self.qformer_pretrained(encoder_hidden_states=pooled, encoder_attention_mask=encoder_attention_mask)
            # 检索专用：只使用正面图像，不加位置嵌入
            front_attention_mask = torch.ones(batch_size, num_patches, dtype=torch.long, device=device)
            q_tokens_retrieval = self.qformer_pretrained(
                encoder_hidden_states=front_features_raw,
                encoder_attention_mask=front_attention_mask
            )
        
        # 拼接两个 Q-Former 的输出：[B, num_query_tokens_train + num_query_tokens_pretrained, 768]
        q_tokens = torch.cat([q_tokens_train, q_tokens_pretrained], dim=1)

        inputs_llama = self.llama_proj(q_tokens)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        query_features = q_tokens_retrieval.mean(dim=1)  # [B, 768]
        return inputs_llama, atts_llama, query_features, q_tokens_retrieval


    def _encode_retrieved_text(self, retrieved_ids, batch_size, device, similarities=None, num_fusion=3):
        """
        根据检索到的 reports_pure 文本进行 BERT 编码，支持融合 top-N 报告
        
        Args:
            retrieved_ids: List[List[str]]，每个样本检索到的 reports_pure 文本列表
            batch_size: batch 大小
            device: 设备
            similarities: torch.Tensor [B, topn]，相似度分数，用于加权融合（可选）
            num_fusion: int，融合多少个 top 报告（默认 3）
            
        Returns:
            retrieved_text_embeds: [B, 40, 4096]，编码后的文本特征（融合后的）
            retrieved_text_atts: [B, 40]，文本 attention mask（融合后的）
        """
        flat_texts = []
        sample_to_flat_idx = [[] for _ in range(batch_size)]
        flat_weights = []

        for b in range(batch_size):
            top_reports = retrieved_ids[b][:num_fusion] if retrieved_ids[b] else []
            for i, reports_pure in enumerate(top_reports):
                if not reports_pure:
                    continue
                sample_to_flat_idx[b].append(len(flat_texts))
                flat_texts.append(reports_pure)

                if similarities is not None and i < similarities.shape[1]:
                    flat_weights.append(similarities[b, i])
                else:
                    flat_weights.append(None)

        retrieved_text_embeds = torch.zeros(batch_size, 40, 768, device=device)
        retrieved_text_atts = torch.zeros(batch_size, 40, dtype=torch.long, device=device)
        has_text = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if len(flat_texts) > 0:
            with torch.no_grad():
                text_inputs = self.bert_tokenizer(
                    flat_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=40,
                    add_special_tokens=True
                ).to(device)
                text_outputs = self.bert(**text_inputs)
                flat_embeds = text_outputs.last_hidden_state
                flat_atts = text_inputs.attention_mask

            for b in range(batch_size):
                idxs = sample_to_flat_idx[b]
                if not idxs:
                    continue

                has_text[b] = True
                encoded_reports = flat_embeds[idxs]
                encoded_atts = flat_atts[idxs]

                if similarities is not None:
                    weights = torch.stack([
                        flat_weights[j] if flat_weights[j] is not None else similarities.new_tensor(1.0)
                        for j in idxs
                    ]).to(device=device, dtype=torch.float32)
                else:
                    weights = torch.ones(len(idxs), device=device, dtype=torch.float32)
                weights = torch.softmax(weights, dim=0)

                fused_embeds = (encoded_reports * weights.view(-1, 1, 1)).sum(dim=0)
                fused_atts = encoded_atts.sum(dim=0).clamp(0, 1).long()

                retrieved_text_embeds[b] = fused_embeds
                retrieved_text_atts[b] = fused_atts
        
        # 投影和归一化检索到的文本特征
        retrieved_text_embeds = self.bert_proj(retrieved_text_embeds)  # [B, 40, 4096]
        retrieved_text_embeds = self.bert_norm(retrieved_text_embeds)  # [B, 40, 4096]

        # 空样本保持全零，避免线性层 bias 引入伪检索信号
        retrieved_text_embeds[~has_text] = 0
        
        return retrieved_text_embeds, retrieved_text_atts

    def prompt_wrap(self, img_embeds, atts_img, retrieved_text_embeds=None, retrieved_text_atts=None):
        """
        将(视觉特征 + 可选检索特征)拼接，并包裹到 prompt 中形成 LLaMA 输入 embedding。
        - 在视觉与检索特征之间插入分隔标记，让模型感知边界。
        """
        batch_size = img_embeds.shape[0]
        device = img_embeds.device

        if retrieved_text_embeds is not None:
            if retrieved_text_atts is None:
                raise ValueError("retrieved_text_atts must be provided when retrieved_text_embeds is not None")
            # sep token embeddings
            sep_tokens = self.llama_tokenizer(
                " <RET> ", return_tensors="pt", add_special_tokens=False).to(device)
            sep_embeds = self.embed_tokens(sep_tokens.input_ids).expand(batch_size, -1, -1)
            sep_atts = torch.ones(batch_size, sep_embeds.shape[1], dtype=torch.long, device=device)

            img_embeds = torch.cat([img_embeds, sep_embeds, retrieved_text_embeds], dim=1)
            atts_img = torch.cat([atts_img, sep_atts, retrieved_text_atts], dim=1)

        prompt = f'Human: <Img><ImageHere></Img>\n{self.prompt}\nAssistant:'
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        p_before_atts = torch.ones(batch_size, p_before_embeds.shape[1], dtype=torch.long, device=img_embeds.device)
        p_after_atts = torch.ones(batch_size, p_after_embeds.shape[1], dtype=torch.long, device=img_embeds.device)
        wrapped_atts_img = torch.cat([p_before_atts, atts_img, p_after_atts], dim=1)
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        count = samples["count"]
        img_embeds, atts_img, query_features, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)

        # 使用正面图像检索特征查询外部记忆库
        retrieved_ids, similarities = self.memory_retriever.retrieve(
            query_features, q_tokens=q_tokens, alpha=self.ext_memory_alpha
        )  # List[List[str]], [B, topn]
        
        # 融合 top-3 的 reports_pure 并使用 BERT 编码（根据相似度分数加权）
        batch_size = img_embeds.shape[0]
        retrieved_text_embeds, retrieved_text_atts = self._encode_retrieved_text(
            retrieved_ids, batch_size, img_embeds.device, 
            similarities=similarities, num_fusion=self.ext_memory_fusion  # 融合 top-N 报告（默认等于 topn）
        )
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img,
            retrieved_text_embeds=retrieved_text_embeds,
            retrieved_text_atts=retrieved_text_atts
        )

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
            output_hidden_states=True,  # 获取所有层的隐藏状态
        )
        return {"loss": outputs.loss}

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
        img_embeds, atts_img, query_features, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)

        # 使用正面图像检索特征查询外部记忆库
        retrieved_ids, similarities = self.memory_retriever.retrieve(
            query_features, q_tokens=q_tokens, alpha=self.ext_memory_alpha
        )  # List[List[str]], [B, topn]
        
        # 融合 top-3 的 reports_pure 并使用 BERT 编码（根据相似度分数加权）
        batch_size = img_embeds.shape[0]
        retrieved_text_embeds, retrieved_text_atts = self._encode_retrieved_text(
            retrieved_ids, batch_size, img_embeds.device, 
            similarities=similarities, num_fusion=self.ext_memory_fusion  # 融合 top-N 报告（默认等于 topn）
        )
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img,
            retrieved_text_embeds=retrieved_text_embeds,
            retrieved_text_atts=retrieved_text_atts
        )

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
        ref_dict = {k: [v] for k, v in zip(ids, ref)}
        hypo_dict = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.compute_all_scores(gts=ref_dict, res=hypo_dict, args=DEFAULT_ARGS)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        # 只在 rank 0 上保存文件（避免多 GPU 重复保存）
        if self.trainer.global_rank == 0:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step

            # 保存生成结果和参考文本
            json.dump(hypo_dict,
                      open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
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
        img_embeds, atts_img, query_features, q_tokens = self.encode_img(image, count)
        img_embeds = self.layer_norm(img_embeds)

        # 使用正面图像检索特征查询外部记忆库
        retrieved_ids, similarities = self.memory_retriever.retrieve(
            query_features, q_tokens=q_tokens, alpha=self.ext_memory_alpha
        )  # List[List[str]], [B, topn]
        
        # 融合 top-3 的 reports_pure 并使用 BERT 编码（根据相似度分数加权）
        batch_size = img_embeds.shape[0]
        retrieved_text_embeds, retrieved_text_atts = self._encode_retrieved_text(
            retrieved_ids, batch_size, img_embeds.device, 
            similarities=similarities, num_fusion=self.ext_memory_fusion  # 融合 top-N 报告（默认等于 topn）
        )
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img,
            retrieved_text_embeds=retrieved_text_embeds,
            retrieved_text_atts=retrieved_text_atts
        )

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
        self.test_step_outputs.append({
            "hypo": hypo,
            "ref": ref,
            "id": samples["id"],
            "retrieved_topk": retrieved_ids,
        })
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids, retrieved_topk = [], [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
            retrieved_topk.extend(i['retrieved_topk'])

        ref_dict = {k: [v] for k, v in zip(ids, ref)}
        hypo_dict = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.compute_all_scores(gts=ref_dict, res=hypo_dict, args=DEFAULT_ARGS)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        detailed_results = []
        for sample_id, reference_report, generated_report, topk_reports in zip(ids, ref, hypo, retrieved_topk):
            detailed_results.append({
                "id": sample_id,
                "reference_report": reference_report,
                "generated_report": generated_report,
                "retrieved_topk_reports": topk_reports,
            })
        json.dump(
            detailed_results,
            open(os.path.join(result_folder, "test_detailed_with_retrieval.json"), 'w'),
            ensure_ascii=False,
            indent=2
        )
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        
        # 清空test_step_outputs，为下一轮测试做准备
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # 简化：所有可训练参数使用同一个学习率
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            print("⚠️  Warning: No trainable parameters found! All parameters are frozen.")
            print("   Creating a dummy parameter to avoid optimizer initialization error.")
            # 创建一个虚拟参数，避免优化器初始化错误
            dummy_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            trainable_params = [dummy_param]
            print("   Note: This is a placeholder. Consider unfreezing some parameters for training.")
        
        # 获取学习率和weight_decay
        base_lr = self.hparams.learning_rate
        weight_decay = getattr(self.hparams, 'weight_decay', 0.01)
        
        # 创建优化器，所有参数使用同一个学习率
        optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
        
        # 打印优化器配置
        param_count = sum(p.numel() for p in trainable_params)
        print(f'\n=== Optimizer Configuration ===')
        print(f'Learning rate: {base_lr}')
        print(f'Weight decay: {weight_decay}')
        print(f'Total trainable parameters: {param_count:,}')
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




