"""
构建外部记忆库工具函数
提供构建外部记忆库的工具函数，供 train.py 调用
"""
import os
import json
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Stage2Model import Stage2Model
from dataset.data_helper import ParseDataset


@torch.no_grad()
def build_memory_database(model, dataloader, output_path):
    """
    构建外部记忆库的工具函数
    直接进行图像编码，不调用encode_img函数
    Args:
        model: Stage2Model 实例，已经加载了阶段一的权重
        dataloader: DataLoader，用于加载数据
        output_path: str，输出目录路径，函数会自动在该目录下生成 ext_memory.pth 文件
    Returns:
        tuple: (features_pooled, all_features, all_texts)
            - features_pooled: torch.Tensor [N, 768] 池化后的向量（检索用）
            - all_features: torch.Tensor [N, num_query_tokens, 768] 完整的Q-Former输出
            - all_texts: List[str] reports_pure 文本列表（与特征一一对应）
    """
    model.eval()
    device = next(model.parameters()).device
    
    print("\n开始编码图像特征...")
    all_features = []  # 保存完整的Q-Former输出，每个元素是 [num_query_tokens, 768]
    all_texts = []  # 保存 reports_pure（句子/报告文本），与 all_features 一一对应
    
    num_patches = 1370  # 每个图像编码后的patch数量
    
    for batch in tqdm(dataloader, desc="编码图像"):
        images = batch["image"]
        batch_texts = batch.get("reports_pure", [])
        
        # 处理图像：现在只有第一张图像（正面图像）
        if isinstance(images, list):
            image1 = images[0].to(device)  # [B, C, H, W]
        else:
            image1 = images.to(device)
        
        batch_size = image1.shape[0]
        
        # 直接进行图像编码（只编码第一张图像）
        outputs1 = model.visual_encoder(pixel_values=image1, output_hidden_states=True)
        front_features = (outputs1.hidden_states[2] + outputs1.hidden_states[8] + outputs1.last_hidden_state) / 3.0
        
        # 只使用第一张图像的特征
        pooled = front_features  # [B, 1370, 768]
        
        # attention_mask：只有一张图像，全部为1
        encoder_attention_mask = torch.ones(batch_size, num_patches, dtype=torch.long, device=device)
        
        # Q-Former编码，输出完整的query tokens [B, num_query_tokens, 768]
        q_tokens = model.qformer(encoder_hidden_states=pooled, encoder_attention_mask=encoder_attention_mask)  # [B, 32, 768] 或 [B, 64, 768]
        
        # 将 batch 的特征和 ID 一一对应地保存
        q_tokens_cpu = q_tokens.cpu()  # [B, num_query_tokens, 768]
        for i in range(batch_size):
            # 保存单个样本的特征 [num_query_tokens, 768]
            all_features.append(q_tokens_cpu[i])  # 取第 i 个样本的特征
            
            # 保存文本（reports_pure）
            if i < len(batch_texts):
                text = batch_texts[i]
            else:
                text = ""
            all_texts.append(text)
    
    # 拼接所有特征：将列表中的每个 [num_query_tokens, 768] 拼接成 [N, num_query_tokens, 768]
    if len(all_features) > 0:
        all_features = torch.stack(all_features, dim=0)  # [N, num_query_tokens, 768]
    else:
        all_features = torch.empty(0, 0, 768)  # 空张量
    
    # 确保特征和文本数量一致
    assert len(all_features) == len(all_texts), f"特征数量 ({len(all_features)}) 与 文本数量 ({len(all_texts)}) 不匹配！"
    
    print(f"\n编码完成！")
    print(f"特征形状: {all_features.shape}")  # [N, num_query_tokens, 768]
    print(f"样本数量: {len(all_texts)}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 对特征进行平均池化，得到 [N, 768] 格式
    features_pooled = all_features.mean(dim=1)  # [N, num_query_tokens, 768] -> [N, 768]
    
    # 保存 pkl 文件（三项）：(features_pooled, all_features, all_texts)
    pkl_file = os.path.join(output_path, 'ext_memory.pkl')
    evcap_data = (features_pooled, all_features, all_texts)
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(evcap_data, f)
    
    print(f"\n✅ 外部记忆库已保存到: {pkl_file}")
    print(f"   特征张量形状: {features_pooled.shape}  # [N, 768]")
    print(f"   原始Q-Former特征形状: {all_features.shape}  # [N, num_query_tokens, 768]")
    print(f"   文本数量: {len(all_texts)}")

    
    return features_pooled, all_features, all_texts


def prepare_memory_dataloader(args):
    """
    准备用于构建记忆库的DataLoader
    Args:
        args: 配置参数
    Returns:
        DataLoader: 数据加载器
    """
    memory_split = getattr(args, 'memory_split', 'train')
    
    print(f"\n准备数据加载器...")
    print(f"  使用split: {memory_split}")
    
    # 检查是否有memory split
    with open(args.annotation, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'memory' in data and memory_split == 'memory':
        print(f"✅ 发现 'memory' split，包含 {len(data['memory'])} 个样本")
    elif memory_split == 'memory' and 'memory' not in data:
        print(f"⚠️  未找到 'memory' split，将使用 'train' split")
        memory_split = 'train'
    
    # 创建数据集（使用现有的ParseDataset）
    memory_dataset = ParseDataset(args, split=memory_split)
    print(f"  数据集大小: {len(memory_dataset)} 个样本")
    
    # 创建DataLoader
    memory_loader = DataLoader(
        memory_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    return memory_loader

