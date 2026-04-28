retrieval-augmented approach for automated radiology report generation.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

## Overview

This project implements a retrieval-augmented radiology report generation system based on R2GenGPT. Given chest X-ray images as input, the model generates diagnostic reports automatically. The key innovation is the **case-level external memory** mechanism that retrieves similar cases to enhance report quality.

### Key Features

- **Three-Stage Training Pipeline**: Shallow alignment → Memory construction → Deep alignment
- **Case-Level Memory**: External memory database stores training cases for retrieval
- **Multi-Query Retrieval**: Retrieves top-K similar cases using both image and text queries
- **Frozen LLM**: Leverages pre-trained Llama-2-7B while keeping it frozen
- **Flexible Fusion**: Configurable fusion weight α between global and local representations

## Method Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Image     │ →   │  Vision CNN  │ →   │  Q-Former       │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐              ↓
│   Memory    │ ←   │  Retrieval   │←── Multi-Query
│   Database  │     │    (Top-K)   │
└─────────────┘     └──────────────┘              ↓
                                          ┌────────────────┐
                                          │  Llama-2-7B    │
                                          │  (Frozen LLM)  │
                                          └────────┬───────┘
                                                   ↓
                                          ┌────────────────┐
                                          │ Radiology      │
                                          │ Report         │
                                          └────────────────┘
```

### Training Stages

| Stage | Description |
|-------|-------------|
| **Stage 1** | Shallow alignment between vision encoder and LLM |
| **Stage 2** | Build external memory database from training set |
| **Stage 3** | Deep alignment with retrieval-augmented generation |

## Installation

```bash
git clone https://github.com/your-repo/EVAP.git
cd EVAP
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)
- 16GB+ GPU memory recommended

### Pre-trained Models

Download the following models and specify paths via command-line arguments:

| Model | Description |
|-------|-------------|
| `rad_dino` | Vision encoder for chest X-rays |
| `BiomedVLP-CXR-BERT` | Text encoder for medical reports |
| `Llama-2-7b-chat-hf` | Frozen language model |

## Datasets

### IU-Xray

Download from [Google Drive](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view).

### MIMIC-CXR

- Annotations: [Google Drive](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing)
- Images: [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Place downloaded data in `./dataset/`.

## Quick Start

### Training

**Stage 1 — Shallow Alignment:**
```bash
bash scripts/4-1.shallow_run.sh
```

**Stage 2 — Build Memory Database:**
```bash
python train.py \
    --stage 2 \
    --stage1_checkpoint /path/to/stage1_checkpoint.pth \
    --output_path ./ext_memory.pkl \
    --memory_split train
```

**Stage 3 — Deep Alignment with Memory:**
```bash
python train.py \
    --stage 3 \
    --ext_memory_path ./ext_memory.pkl \
    --ext_memory_topn 4 \
    --ext_memory_alpha 0.5
```

### Testing

```bash
bash scripts/4-2.shallow_test.sh   # Shallow alignment
bash scripts/6-2.deep_test.sh      # Deep alignment with memory
```

## Key Arguments

### Memory Retrieval (Stage 3)

| Argument | Default | Description |
|----------|---------|-------------|
| `--ext_memory_path` | - | Path to memory database pickle file |
| `--ext_memory_topn` | 4 | Number of similar cases to retrieve |
| `--ext_memory_alpha` | 0.5 | Fusion weight: score = α×global + (1-α)×local |

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm_use_lora` | False | Use LoRA for LLM |
| `--vis_use_lora` | False | Use LoRA for vision encoder |
| `--freeze_vm` | True | Freeze vision model |
| `--freeze_tm` | True | Freeze text model |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 50 | Training batch size |
| `--max_epochs` | 10 | Maximum epochs |
| `--learning_rate` | 3e-4 | Learning rate |
| `--precision` | bf16-mixed | Mixed precision training |

## Project Structure

```
EVAP/
├── configs/              # Configuration and hyperparameters
├── dataset/             # Data loading and preprocessing
├── models/              # Model definitions
│   ├── Stage1Model.py   # Stage 1: Shallow alignment
│   ├── Stage2Model.py   # Stage 2: Memory construction
│   └── R2GenGPT.py      # Stage 3: Deep alignment + generation
├── lightning_tools/     # PyTorch Lightning callbacks
├── mertic/              # Evaluation metrics
├── retrieval/           # Retrieval mechanisms
├── scripts/             # Training/testing scripts
├── evalcap/             # Evaluation pipeline
├── train.py             # Main training entry point
└── fenxi.py             # Analysis utilities
```

## Citation

If you find this work helpful in your research, please cite:

```bibtex
@article{evap2024,
  title={Retrieval-augmented radiology report generation with case-level memory and multi-query retrieval},
  author={},
  year={2024}
}
```

## Acknowledgements

This project is built upon [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and uses [Llama-2](https://github.com/facebookresearch/llama) as the frozen language model. We acknowledge their excellent work.

## License

This repository is under [BSD 3-Clause License](LICENSE).
