import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from lightning.fabric.utilities.data import AttributeDict
import lightning.pytorch as pl
from transformers import AutoModel, AutoTokenizer

from Qformermoudel.qformermoudel import SimpleQFormerWrapper
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from lightning_tools.contrastive_loss import ContrastiveProjection, compute_contrastive_loss
from peft import get_peft_model, LoraConfig, TaskType

from mertic.mertic import compute_all_scores

torch.set_float32_matmul_precision('medium')
import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_SAFE"] = "True"
DEFAULT_ARGS = {
    'chexbert_path': "/root/autodl-tmp/checkpoints/chexbert.pth",
    'bert_path': "/root/autodl-tmp/checkpoints/bert-base-uncased",
    'radgraph_path': "/root/autodl-tmp/checkpoints/radgraph.tar.gz",
}



class Stage1Model(pl.LightningModule):
    """
    Stage 1 Model - 内容与 R2GenGPT 相同
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.qformer = SimpleQFormerWrapper(num_hidden_layers=3, use_separate_queries=args.use_separate_queries)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert = AutoModel.from_pretrained(args.bert, trust_remote_code=True, local_files_only=True)
        self.bert.eval()
        if args.freeze_tm:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen textal encoder:{args.bert} -- Done')
        else:
            print(f'Loading Trainable textal encoder:{args.textal_encoder} -- Done')


        self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        self.visual_encoder.embeddings.mask_token.requires_grad = False
        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')
        
        # 对比学习投影模块
        self.contrastive_projection = ContrastiveProjection(hidden_size=768)


    def encode_img(self, images, count):
        """
        编码图像，添加位置嵌入并根据count创建mask来忽略复制的图像
        Args:
            images: 图像列表，[front_images, lateral_images]，每个是 [B, C, H, W]
            count: tensor [B]，每个样本的图像数量，1或2
        Returns:
            img_embed: [B, 64, 768] Q-Former输出的图像特征
        """
        batch_size = images[0].shape[0]
        device = images[0].device
        num_patches = 1370  # 每个图像编码后的patch数量

        outputs = self.visual_encoder(pixel_values=images[0], output_hidden_states=True)
        front_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0
        encoder_attention_mask = torch.ones(batch_size, num_patches, dtype=torch.long, device=device)
        q_tokens = self.qformer(encoder_hidden_states=front_features, encoder_attention_mask=encoder_attention_mask)
        return q_tokens  # [B, 64, 768]




    def forward(self, samples):
        """
        阶段一：对比学习任务
        Args:
            samples: 包含 "image", "count", "reports_pure" 的字典
        Returns:
            {"loss": contrastive_loss}
        """
        image = samples["image"]
        count = samples["count"]
        
        # 编码图像: [B, 64, 768]
        img_embed = self.encode_img(image, count)  # [B, 64, 768]
        
        # 编码文本
        # 处理文本列表，确保是字符串列表
        reports = samples["reports_pure"]
        # if isinstance(reports, torch.Tensor):
        #     # 如果是tensor，需要转换
        #     reports = [str(r) for r in reports]
        # elif not isinstance(reports, list):
        #     reports = [reports]
        
        text_tokenizer = self.bert_tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=40,
            return_tensors='pt'
        )
        text_tokenizer = {k: v.to(img_embed.device) for k, v in text_tokenizer.items()}
        
        # BERT编码: [B, 40, 768]
        text_emb = self.bert(**text_tokenizer).last_hidden_state  # [B, 40, 768]
        
        # 计算对比学习loss（使用lightning_tools中的模块）
        contrastive_loss = compute_contrastive_loss(img_embed, text_emb, self.contrastive_projection)
        
        return {"loss": contrastive_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self):
        """
        保存checkpoint（阶段一：保存对齐后的权重）
        """
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
            "stage1_checkpoint_epoch{}_step{}.pth".format(current_epoch, global_step),
        )
        self.print("Saving stage1 checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def on_train_epoch_end(self):
        """
        训练epoch结束：保存checkpoint
        """
        # 每个epoch结束时保存一次checkpoint
        if self.trainer.global_rank == 0:
            self.save_checkpoint()

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

