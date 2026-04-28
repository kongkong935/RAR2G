import torch
import torch.nn as nn
from transformers import AutoModel

from Qformermoudel.qformermoudel import SimpleQFormerWrapper
torch.set_float32_matmul_precision('medium')


class Stage2Model(nn.Module):
    """
    Stage 2 Model - 仅用于构建外部记忆库
    只需要 Vision Encoder + Q-Former，不需要 LLM，不需要训练
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.qformer = SimpleQFormerWrapper(num_hidden_layers=3, use_separate_queries=args.use_separate_queries)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = AutoModel.from_pretrained(args.vision_model)
        self.visual_encoder.embeddings.mask_token.requires_grad = False

        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')






