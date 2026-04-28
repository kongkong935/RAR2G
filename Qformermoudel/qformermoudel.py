from transformers import Blip2QFormerConfig, Blip2QFormerModel
import torch
import torch.nn as nn
import os

class SimpleQFormerWrapper(nn.Module):
    def __init__(self, num_hidden_layers=6, hidden_size=768, num_attention_heads=12, num_query_tokens=32,
                 use_separate_queries=False, pretrained_path=None):
        """
        Args:
            num_hidden_layers: Q-Former 的隐藏层数
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
            num_query_tokens: query token 数量
            use_separate_queries: 是否使用分离的 query tokens（正面和侧面）
            pretrained_path: 预训练权重路径（checkpoint 文件），如果提供，会从中加载 qformer 和 query_tokens 的权重
        """
        super().__init__()
        config = Blip2QFormerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            vocab_size=1,              # 你其实不做LM，这里随便
            cross_attention_frequency=1,  # 每层都带 cross-attn
            encoder_hidden_size=hidden_size,  # KV 的 dim
        )
        self.qformer = Blip2QFormerModel(config)
        self.use_separate_queries = use_separate_queries
        
        if use_separate_queries:
            # 两个独立的 query token，每个有 num_query_tokens 个 token
            self.query_tokens_front = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
            self.query_tokens_lateral = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        else:
            # 单个 query token
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        
        # 如果提供了预训练权重路径，加载权重
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """
        从 checkpoint 文件中加载 qformer 和 query_tokens 的权重
        
        Args:
            pretrained_path: checkpoint 文件路径，可以是：
                - 包含 'model' 键的字典（如 Lightning checkpoint）
                - 直接的 state_dict
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_path}")
        
        print(f"📥 从 {pretrained_path} 加载 Q-Former 权重...")
        
        # 加载 checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # 处理不同的 checkpoint 格式
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 提取 qformer 相关的权重
        qformer_state_dict = {}
        query_tokens_state_dict = {}
        
        for key, value in state_dict.items():
            # 提取 qformer 的权重
            # Stage1Model 中保存的键名格式：'qformer.qformer.xxx' (Stage1Model.qformer.qformer.xxx)
            # 需要去掉 'qformer.qformer.' 前缀，得到 'xxx'，然后加载到 self.qformer (Blip2QFormerModel)
            if key.startswith('qformer.qformer.'):
                new_key = key[len('qformer.qformer.'):]  # 去掉 'qformer.qformer.' 前缀，得到 'xxx'
                qformer_state_dict[new_key] = value
            # 提取 query_tokens 的权重（Stage1Model 中格式：'qformer.query_tokens'）
            elif key == 'qformer.query_tokens':
                query_tokens_state_dict['query_tokens'] = value
            elif key == 'qformer.query_tokens_front':
                query_tokens_state_dict['query_tokens_front'] = value
            elif key == 'qformer.query_tokens_lateral':
                query_tokens_state_dict['query_tokens_lateral'] = value
            # 如果键名直接是 query_tokens（没有 qformer 前缀），也支持
            elif key == 'query_tokens':
                query_tokens_state_dict['query_tokens'] = value
            elif key == 'query_tokens_front':
                query_tokens_state_dict['query_tokens_front'] = value
            elif key == 'query_tokens_lateral':
                query_tokens_state_dict['query_tokens_lateral'] = value
        
        # 加载 qformer 权重
        if qformer_state_dict:
            missing_keys, unexpected_keys = self.qformer.load_state_dict(qformer_state_dict, strict=False)
            if missing_keys:
                print(f"  ⚠️  Q-Former 缺失的权重: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"  ⚠️  Q-Former 意外的权重: {len(unexpected_keys)} 个")
            print(f"  ✅ Q-Former 权重加载完成")
        else:
            print(f"  ⚠️  未找到 Q-Former 权重")
        
        # 加载 query_tokens 权重
        if query_tokens_state_dict:
            if self.use_separate_queries:
                # 分离的 query tokens
                if 'query_tokens_front' in query_tokens_state_dict:
                    self.query_tokens_front.data = query_tokens_state_dict['query_tokens_front']
                    print(f"  ✅ query_tokens_front 权重加载完成")
                if 'query_tokens_lateral' in query_tokens_state_dict:
                    self.query_tokens_lateral.data = query_tokens_state_dict['query_tokens_lateral']
                    print(f"  ✅ query_tokens_lateral 权重加载完成")
            else:
                # 单个 query token
                if 'query_tokens' in query_tokens_state_dict:
                    self.query_tokens.data = query_tokens_state_dict['query_tokens']
                    print(f"  ✅ query_tokens 权重加载完成")
        else:
            print(f"  ⚠️  未找到 query_tokens 权重，使用随机初始化")

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        """
        encoder_hidden_states: 视觉编码器输出，形状 [B, N, H]
        encoder_attention_mask: attention mask，形状 [B, N]
        
        如果 use_separate_queries=True，期望 encoder_hidden_states 包含两个图像的特征拼接
        """
        B = encoder_hidden_states.size(0)
        
        if self.use_separate_queries:
            # 分别查询正面和侧面图像
            num_patches_per_image = encoder_hidden_states.size(1) // 2  # 假设两个图像拼接
            
            # 第一个 query token 查询前 num_patches_per_image 个 patch（正面）
            front_features = encoder_hidden_states[:, :num_patches_per_image, :]  # [B, N/2, H]
            front_mask = encoder_attention_mask[:, :num_patches_per_image] if encoder_attention_mask is not None else None
            query_tokens_front = self.query_tokens_front.expand(B, -1, -1)  # [B, num_query, H]
            outputs_front = self.qformer(
                query_embeds=query_tokens_front,
                encoder_hidden_states=front_features,
                encoder_attention_mask=front_mask,
                return_dict=True,
            )
            front_query_output = outputs_front.last_hidden_state  # [B, num_query, H]
            
            # 第二个 query token 查询后 num_patches_per_image 个 patch（侧面）
            lateral_features = encoder_hidden_states[:, num_patches_per_image:, :]  # [B, N/2, H]
            lateral_mask = encoder_attention_mask[:, num_patches_per_image:] if encoder_attention_mask is not None else None
            query_tokens_lateral = self.query_tokens_lateral.expand(B, -1, -1)  # [B, num_query, H]
            outputs_lateral = self.qformer(
                query_embeds=query_tokens_lateral,
                encoder_hidden_states=lateral_features,
                encoder_attention_mask=lateral_mask,
                return_dict=True,
            )
            lateral_query_output = outputs_lateral.last_hidden_state  # [B, num_query, H]
            
            # 拼接两个 query token 的输出
            combined_output = torch.cat([front_query_output, lateral_query_output], dim=1)  # [B, 2*num_query, H]
            return combined_output
        else:
            # 原始方式：单个 query token
            query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, num_query, H]

            outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            # 输出的 last_hidden_state 就是更新后的 query token: [B, num_query, H]
            return outputs.last_hidden_state
