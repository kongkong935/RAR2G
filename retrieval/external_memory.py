"""
外部记忆库检索模块
提供外部记忆库的加载和检索功能
使用 PyTorch GPU 检索，避免 CPU-GPU 数据传输开销
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemoryRetriever(nn.Module):
    """
    外部记忆库检索器
    负责加载外部记忆库并执行相似度检索
    使用 PyTorch GPU 检索，适合中等规模（几百到几千）的记忆库
    """

    def __init__(self, ext_memory_path=None, topn=9):
        """
        Args:
            ext_memory_path: 外部记忆库 pkl 文件路径
            topn: 默认检索 Top-N 个相似样本
        """
        super().__init__()
        self.topn = topn
        self.ext_base_img = None  # torch.Tensor [N, 768]，常驻 GPU
        self.ext_base_img_id = None  # List[str]（实际语义：reports_pure 文本）
        self.ext_base_qtokens = None  # torch.Tensor [N, num_query_tokens, 768]（Q-Former 原始输出，默认保留在 CPU）

        if ext_memory_path:
            self.load_memory(ext_memory_path)

    def load_memory(self, ext_memory_path):
        """加载外部记忆库（三项）：(pooled[N,768], q_tokens[N,Q,768], texts[N])."""
        print(f"📚 加载外部记忆库: {ext_memory_path}")
        with open(ext_memory_path, 'rb') as f:
            features_pooled, all_features, all_texts = pickle.load(f)

        self.ext_base_img = torch.as_tensor(features_pooled, dtype=torch.float32)  # [N, 768]
        self.ext_base_qtokens = torch.as_tensor(all_features, dtype=torch.float32)  # [N, Q, 768]
        self.ext_base_img_id = all_texts  # texts[N]

        # 把“点积检索”变成“余弦相似度检索”：预先把库向量做 L2 归一化
        # 检索时只需再对 query 做一次归一化即可
        self.ext_base_img = F.normalize(self.ext_base_img, dim=-1)
        # 未来做局部/patch（token 级）匹配时也建议归一化：
        # token 点积 == token 余弦相似度（数值更稳定、阈值更可解释）
        self.ext_base_qtokens = F.normalize(self.ext_base_qtokens, dim=-1)

        print("✅ 外部记忆库加载完成")
        print(f"   pooled: {tuple(self.ext_base_img.shape)}  # [N, 768] (L2 norm)")
        print(f"   q_tokens: {tuple(self.ext_base_qtokens.shape)}  # [N, Q, 768]")
        print(f"   texts: {len(self.ext_base_img_id)}  # N")

        return self.ext_base_img, self.ext_base_qtokens, self.ext_base_img_id

    def retrieve(self, query_features, top_k=None, q_tokens=None, alpha=0.8):
        """
        外部记忆库检索（GPU 上做相似度）。

        Args:
            query_features: torch.Tensor, shape [B, 768]
                query 的 pooled 向量（通常是 q_tokens.mean(dim=1)）。
            top_k: int | None
                返回的 Top-K；None 表示使用 self.topn。
            q_tokens: torch.Tensor | None, shape [B, Qq, 768]
                query 的 token 级特征（例如 Qq=32）。提供时会启用“局部匹配”分数。
            alpha: float
                融合权重：score = alpha * global + (1 - alpha) * local。

        Returns:
            retrieved_texts: List[List[str]]
                每个样本的 Top-K 文本（这里存的是 reports_pure）。
            similarities: torch.Tensor, shape [B, top_k]
                Top-K 的相似度分数（融合后的）。
        """
        top_k = self.topn if top_k is None else top_k
        batch_size = query_features.shape[0]
        device = query_features.device

        if self.ext_base_img is None:
            return [[] for _ in range(batch_size)], torch.zeros(batch_size, top_k, device=device)

        with torch.no_grad():
            if self.ext_base_img.device != device:
                self.ext_base_img = self.ext_base_img.to(device)
            if q_tokens is not None and self.ext_base_qtokens.device != device:
                self.ext_base_qtokens = self.ext_base_qtokens.to(device)

            # 全局分数：pooled(query) vs pooled(memory)，点积(单位向量) == 余弦相似度
            scores = F.normalize(query_features, dim=-1) @ self.ext_base_img.T  # [B, N]

            if q_tokens is not None:
                # 局部分数：token(query) vs token(memory) 的 token-to-token 最大相似度
                q = F.normalize(q_tokens, dim=-1)  # [B, Qq, 768]
                m = self.ext_base_qtokens  # [N, Qm, 768]
                sim = q.reshape(batch_size * q.shape[1], -1) @ m.reshape(m.shape[0] * m.shape[1], -1).T
                sim = sim.reshape(batch_size, q.shape[1], m.shape[0], m.shape[1])  # [B, Qq, N, Qm]
                local = sim.max(dim=3).values.max(dim=1).values  # [B, N]
                # 融合：global 为主，local 为辅（或相反），由 alpha 控制
                scores = alpha * scores + (1 - alpha) * local

            # Top-K：返回对应 texts（reports_pure）和融合后的分数
            topk = scores.topk(k=top_k, dim=-1)
            retrieved = [[self.ext_base_img_id[i.item()] for i in topk.indices[b]] for b in range(batch_size)]
            return retrieved, topk.values


class FrontImageEncoder(nn.Module):
    """
    正面图像编码器
    只处理第一张正面图像，使用冻结的 qformer_pretrained 进行编码
    """

    def __init__(self, visual_encoder, qformer_pretrained):
        """
        Args:
            visual_encoder: 视觉编码器
            qformer_pretrained: 冻结的预训练 Q-Former
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.qformer_pretrained = qformer_pretrained
        self.num_patches = 1370  # 每个图像编码后的patch数量

    def encode(self, front_image):
        """
        编码第一张正面图像

        Args:
            front_image: 第一张正面图像 [B, C, H, W]

        Returns:
            query_features: 池化后的查询特征 [B, 768]
        """
        batch_size = front_image.shape[0]
        device = front_image.device

        with torch.no_grad():
            # 编码第一张正面图像
            outputs = self.visual_encoder(pixel_values=front_image, output_hidden_states=True)
            front_features = (outputs.hidden_states[2] + outputs.hidden_states[8] + outputs.last_hidden_state) / 3.0

            # 创建 attention mask（只有一张图像，全部为1）
            encoder_attention_mask = torch.ones(batch_size, self.num_patches, dtype=torch.long, device=device)

            # 使用冻结的预训练 Q-Former 编码
            q_tokens = self.qformer_pretrained(encoder_hidden_states=front_features, encoder_attention_mask=encoder_attention_mask)

            # 平均池化：[B, num_query_tokens, 768] -> [B, 768]
            query_features = q_tokens.mean(dim=1)  # [B, 768]

        return query_features, q_tokens

