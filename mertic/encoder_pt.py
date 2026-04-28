# """
# 将JSON文件编码成pt文件
# - id: label_content去掉逗号后的字符串
# - embedding: 使用BiomedVLP-CXR-BERT编码的CLS向量（1*768）
# """

# import json
# import os

# # 检查依赖
# try:
#     import torch
# except ImportError:
#     print("错误: 未安装PyTorch")
#     print("请运行: pip install torch")
#     exit(1)

# try:
#     from transformers import AutoTokenizer, AutoModel
# except ImportError:
#     print("错误: 未安装transformers")
#     print("请运行: pip install transformers")
#     exit(1)

# try:
#     from tqdm import tqdm
# except ImportError:
#     print("警告: 未安装tqdm，将不使用进度条")
#     def tqdm(iterable, desc=""):
#         return iterable

# def encode_sentences_to_pt(json_file, output_file, model_name='microsoft/BiomedVLP-CXR-BERT-general', 
#                            batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     """
#     将句子编码成pt文件
    
#     Args:
#         json_file: 输入的JSON文件路径
#         output_file: 输出的pt文件路径
#         model_name: 模型名称
#         batch_size: 批处理大小
#         device: 设备（cuda或cpu）
#     """
#     print(f"正在读取文件: {json_file}")
    
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     print(f"总样本数: {len(data):,}")
    
#     # 检查设备
#     print(f"使用设备: {device}")
#     if device == 'cuda' and not torch.cuda.is_available():
#         print("警告: CUDA不可用，将使用CPU")
#         device = 'cpu'
    
#     # 加载模型和tokenizer
#     print(f"\n正在加载模型: {model_name}")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
#         model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
#         model.to(device)
#         model.eval()
#         print("模型加载成功!")
#     except Exception as e:
#         print(f"加载模型失败: {e}")
#         print("请确保已安装transformers库: pip install transformers")
#         print("如果模型不存在，可能需要从HuggingFace下载")
#         return
    
#     # 准备数据
#     print(f"\n正在处理数据...")
#     ids = []
#     sentences = []
    
#     for sample in data:
#         # 将label_content转换为id（去掉逗号）
#         label_content = sample['label_content']
#         id_str = label_content.replace(',', '')
#         ids.append(id_str)
#         sentences.append(sample['sentence'])
    
#     # 批量编码
#     embeddings = []
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(sentences), batch_size), desc="编码进度"):
#             batch_sentences = sentences[i:i+batch_size]
            
#             # Tokenize
#             inputs = tokenizer(
#                 batch_sentences,
#                 padding=True,
#                 truncation=True,
#                 max_length=512,
#                 return_tensors='pt'
#             )
            
#             # 移动到设备
#             inputs = {k: v.to(device) for k, v in inputs.items()}
            
#             # 编码
#             outputs = model(**inputs)
            
#             # 获取CLS向量（第一个token的embedding）
#             cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
            
#             # 移动到CPU
#             cls_embeddings = cls_embeddings.cpu()
            
#             embeddings.append(cls_embeddings)
    
#     # 合并所有embeddings
#     all_embeddings = torch.cat(embeddings, dim=0)  # [total_samples, 768]
    
#     print(f"\n编码完成!")
#     print(f"  IDs数量: {len(ids)}")
#     print(f"  Embeddings形状: {all_embeddings.shape}")
    
#     # 创建字典
#     result = {
#         'id': ids,
#         'embedding': all_embeddings
#     }
    
#     # 保存为pt文件
#     print(f"\n正在保存到: {output_file}")
#     torch.save(result, output_file)
    
#     print(f"完成! 文件已保存")
#     print(f"  文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
#     # 验证文件
#     print(f"\n验证文件...")
#     loaded = torch.load(output_file)
#     print(f"  加载的键: {list(loaded.keys())}")
#     print(f"  ID数量: {len(loaded['id'])}")
#     print(f"  Embedding形状: {loaded['embedding'].shape}")
#     print(f"  Embedding类型: {loaded['embedding'].dtype}")
    
#     return result

# if __name__ == '__main__':
#     json_file = 'filtered_sentences_proportional_final.json'
#     output_file = 'sentences_encoded.pt'
    
#     # 如果模型名称不对，可以尝试其他可能的名称
#     # 常见的医学BERT模型：
#     # - microsoft/BiomedVLP-CXR-BERT-general
#     # - emilyalsentzer/Bio_ClinicalBERT
#     # - dmis-lab/biobert-base-cased-v1.1
    
#     model_name = '/root/autodl-tmp/hf_cache/BiomedVLP-CXR-BERT-general'
    
#     try:
#         result = encode_sentences_to_pt(
#             json_file=json_file,
#             output_file=output_file,
#             model_name=model_name,
#             batch_size=32,
#             device='cuda' if torch.cuda.is_available() else 'cpu'
#         )
#         print("\n编码完成!")
#     except Exception as e:
#         print(f"错误: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\n如果模型加载失败，请检查：")
#         print("1. 是否安装了transformers: pip install transformers")
#         print("2. 模型名称是否正确")
#         print("3. 网络连接是否正常（首次下载需要）")

"""
将label_content_sentences.json文件编码成pt文件
- id: label_content去掉逗号后的字符串（840个）
- embedding: 每种类型所有句子的平均CLS向量（840 * 768）
"""

import json
import os

# 检查依赖
try:
    import torch
except ImportError:
    print("错误: 未安装PyTorch")
    print("请运行: pip install torch")
    exit(1)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("错误: 未安装transformers")
    print("请运行: pip install transformers")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("警告: 未安装tqdm，将不使用进度条")
    def tqdm(iterable, desc=""):
        return iterable

def encode_label_content_to_pt(json_file, output_file, model_name='microsoft/BiomedVLP-CXR-BERT-general', 
                               batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    将每种label_content类型的所有句子编码，取平均CLS向量
    
    Args:
        json_file: 输入的JSON文件路径（label_content_sentences.json）
        output_file: 输出的pt文件路径
        model_name: 模型名称
        batch_size: 批处理大小
        device: 设备（cuda或cpu）
    """
    print(f"正在读取文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总类型数: {len(data):,}")
    
    # 统计信息
    total_sentences = sum(len(item['sentences']) for item in data)
    print(f"总句子数: {total_sentences:,}")
    print(f"平均每种类型句子数: {total_sentences / len(data):.2f}")
    
    # 检查设备
    print(f"使用设备: {device}")
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device = 'cpu'
    
    # 加载模型和tokenizer
    print(f"\n正在加载模型: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model.to(device)
        model.eval()
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试不使用local_files_only...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.to(device)
            model.eval()
            print("模型加载成功!")
        except Exception as e2:
            print(f"加载模型失败: {e2}")
            print("请确保模型路径正确或网络连接正常")
            return
    
    # 准备数据
    print(f"\n正在处理数据...")
    ids = []
    type_embeddings = []
    
    # 对每种类型进行处理
    for type_idx, item in enumerate(tqdm(data, desc="处理类型")):
        label_content = item['label_content']
        sentences = item['sentences']
        
        # 将label_content转换为id（去掉逗号）
        id_str = label_content.replace(',', '')
        ids.append(id_str)
        
        # 如果该类型没有句子，使用零向量
        if len(sentences) == 0:
            type_embeddings.append(torch.zeros(768))
            continue
        
        # 对该类型的所有句子进行编码
        sentence_embeddings = []
        
        with torch.no_grad():
            # 批量处理该类型的所有句子
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 移动到设备
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 编码
                outputs = model(**inputs)
                
                # 获取CLS向量（第一个token的embedding）
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
                
                # 移动到CPU
                cls_embeddings = cls_embeddings.cpu()
                
                sentence_embeddings.append(cls_embeddings)
        
        # 合并该类型所有句子的embeddings
        if sentence_embeddings:
            all_sentence_embeddings = torch.cat(sentence_embeddings, dim=0)  # [num_sentences, 768]
            # 计算平均向量
            avg_embedding = torch.mean(all_sentence_embeddings, dim=0)  # [768]
            type_embeddings.append(avg_embedding)
        else:
            # 如果没有句子，使用零向量
            type_embeddings.append(torch.zeros(768))
    
    # 合并所有类型的embeddings
    all_embeddings = torch.stack(type_embeddings, dim=0)  # [840, 768]
    
    print(f"\n编码完成!")
    print(f"  类型数: {len(ids)}")
    print(f"  Embeddings形状: {all_embeddings.shape}")
    
    # 创建字典
    result = {
        'id': ids,
        'embedding': all_embeddings
    }
    
    # 保存为pt文件
    print(f"\n正在保存到: {output_file}")
    torch.save(result, output_file)
    
    print(f"完成! 文件已保存")
    print(f"  文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 验证文件
    print(f"\n验证文件...")
    loaded = torch.load(output_file)
    print(f"  加载的键: {list(loaded.keys())}")
    print(f"  ID数量: {len(loaded['id'])}")
    print(f"  Embedding形状: {loaded['embedding'].shape}")
    print(f"  Embedding类型: {loaded['embedding'].dtype}")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    print(f"  总类型数: {len(ids)}")
    print(f"  每种类型的平均向量维度: 768")
    
    # 显示前几个样本的ID
    print(f"\n前5个样本的ID:")
    for i, id_str in enumerate(ids[:5], 1):
        print(f"  {i}. {id_str}")
    
    return result

if __name__ == '__main__':
    json_file = 'label_content_sentences.json'
    output_file = 'label_content_embeddings.pt'
    
    # 模型路径（根据实际情况修改）
    model_name = '/root/autodl-tmp/hf_cache/BiomedVLP-CXR-BERT-general'
    # 如果本地没有，可以尝试从HuggingFace下载：
    # model_name = 'microsoft/BiomedVLP-CXR-BERT-general'
    
    try:
        result = encode_label_content_to_pt(
            json_file=json_file,
            output_file=output_file,
            model_name=model_name,
            batch_size=32,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("\n编码完成!")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n如果模型加载失败，请检查：")
        print("1. 是否安装了transformers: pip install transformers")
        print("2. 模型路径是否正确")
        print("3. 网络连接是否正常（如果从HuggingFace下载）")

