# import torch
# from pathlib import Path
# from lightning.fabric.utilities.data import AttributeDict
#
# # 允许 AttributeDict 被反序列化（仅在信任来源的前提下使用）
# torch.serialization.add_safe_globals([AttributeDict])
#
# STOP_KEYS = {
#     "model.llama_model",
#     "model.llama_proj",
#     "model.front_position_embedding",
#     "model.qformer",
# }
#
#
# def _summarize_obj(obj, prefix="", depth=0, max_depth=None):
#     """
#     递归打印/总结对象。
#     - tensor: shape/dtype
#     - dict/Mapping: 继续深入（受 max_depth 限制）
#     - 其他: type 信息
#     """
#     summary = {}
#
#     # 如果命中停止键，或达到深度限制：只汇报类型，不再展开
#     if prefix in STOP_KEYS or (max_depth is not None and depth >= max_depth):
#         summary[prefix or "<root>"] = {"type": f"{type(obj).__name__} (depth limit)"}
#         return summary
#
#     if torch.is_tensor(obj):
#         summary[prefix or "<root>"] = {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
#     elif isinstance(obj, dict):
#         sub = {}
#         for k, v in obj.items():
#             sub.update(_summarize_obj(v, prefix=f"{prefix}.{k}" if prefix else k, depth=depth + 1, max_depth=max_depth))
#         summary.update(sub)
#     else:
#         summary[prefix or "<root>"] = {"type": type(obj).__name__}
#     return summary
#
#
# def inspect_pth(path, max_depth=None):
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"{path} not found")
#
#     # PyTorch 2.6 默认 weights_only=True，会导致含自定义对象的 checkpoint 失败
#     # 这里显式设为 False，并已将 AttributeDict 加入安全白名单（需信任来源）
#     obj = torch.load(path, map_location="cpu", weights_only=False)
#     # 如果是常见的 checkpoint 结构，优先取 state_dict
#     state = obj.get("state_dict", obj)
#     return _summarize_obj(state, max_depth=max_depth)
#
# if __name__ == "__main__":
#     pth_path = "/root/autodl-tmp/sava/多视角模块+lora/checkpoints/checkpoint_epoch4_step32922_bleu0.127054_cider0.262630.pth"  # 换成实际 .pth 路径
#     result = inspect_pth(pth_path, max_depth=2)
#     for k, v in result.items():
#         print(k, "->", v)
"""
从指定的 JSON 文件中随机采样数据并生成新文件
- train: 随机选取 5000 条
- val: 随机选取 100 条
- test: 随机选取 100 条
- sam: 随机选取 100 条（如果不足则全部选取）
"""

import json
import random
import os
from pathlib import Path

# ==================== 配置区域 ====================
# 在这里填写你的配置，直接运行脚本即可

# 输入 JSON 文件路径
INPUT_FILE = "/root/autodl-tmp/LW2L_12_11.json"

# 输出 JSON 文件路径（如果为 None，则自动生成：输入文件名_sampled.json）
OUTPUT_FILE = None  # 例如: "output.json" 或 None

# 采样数量配置
TRAIN_SIZE = 5000  # train 数据集采样数量
VAL_SIZE = 100  # val 数据集采样数量
TEST_SIZE = 100  # test 数据集采样数量
SAM_SIZE = 100  # sam 数据集采样数量

# 随机种子（用于可重复性，设置为 None 则每次随机）
RANDOM_SEED = None  # 例如: 42


# ==================== 配置区域结束 ====================


def sample_json_data(input_file, output_file, train_size=5000, val_size=100, test_size=100, sam_size=100, seed=None):
    """
    从输入 JSON 文件中随机采样数据并保存到新文件

    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径
        train_size: train 数据集采样数量，默认 5000
        val_size: val 数据集采样数量，默认 100
        test_size: test 数据集采样数量，默认 100
        sam_size: sam 数据集采样数量，默认 100
        seed: 随机种子，用于可重复性
    """
    # 设置随机种子
    if seed is not None:
        random.seed(seed)

    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查文件结构
    required_keys = ['train', 'val', 'test', 'sam']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"输入文件缺少必需的键: {key}")

    result = {}

    # 采样 train 数据
    train_data = data['train']
    if len(train_data) < train_size:
        print(f"警告: train 数据只有 {len(train_data)} 条，少于请求的 {train_size} 条，将全部选取")
        result['train'] = train_data
    else:
        result['train'] = random.sample(train_data, train_size)
    print(f"train: 从 {len(train_data)} 条中选取了 {len(result['train'])} 条")

    # 采样 val 数据
    val_data = data['val']
    if len(val_data) < val_size:
        print(f"警告: val 数据只有 {len(val_data)} 条，少于请求的 {val_size} 条，将全部选取")
        result['val'] = val_data
    else:
        result['val'] = random.sample(val_data, val_size)
    print(f"val: 从 {len(val_data)} 条中选取了 {len(result['val'])} 条")

    # 采样 test 数据
    test_data = data['test']
    if len(test_data) < test_size:
        print(f"警告: test 数据只有 {len(test_data)} 条，少于请求的 {test_size} 条，将全部选取")
        result['test'] = test_data
    else:
        result['test'] = random.sample(test_data, test_size)
    print(f"test: 从 {len(test_data)} 条中选取了 {len(result['test'])} 条")

    # 采样 sam 数据
    sam_data = data['sam']
    if len(sam_data) < sam_size:
        print(f"警告: sam 数据只有 {len(sam_data)} 条，少于请求的 {sam_size} 条，将全部选取")
        result['sam'] = sam_data
    else:
        result['sam'] = random.sample(sam_data, sam_size)
    print(f"sam: 从 {len(sam_data)} 条中选取了 {len(result['sam'])} 条")

    # 保存结果
    print(f"正在保存到: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"完成! 新文件已保存: {output_file}")
    print(
        f"总计: train={len(result['train'])}, val={len(result['val'])}, test={len(result['test'])}, sam={len(result['sam'])}")


def main():
    """主函数：使用配置区域的参数执行采样"""
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件不存在: {INPUT_FILE}")
        return

    # 如果没有指定输出文件，自动生成
    if OUTPUT_FILE is None:
        input_path = Path(INPUT_FILE)
        output_file = str(input_path.parent / f"{input_path.stem}_sampled.json")
    else:
        output_file = OUTPUT_FILE

    # 执行采样
    sample_json_data(
        INPUT_FILE,
        output_file,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        sam_size=SAM_SIZE,
        seed=RANDOM_SEED
    )


if __name__ == '__main__':
    main()


