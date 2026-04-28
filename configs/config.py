import argparse
parser = argparse.ArgumentParser(description="hyper-parameter for R2GenGPT")
# ========================= Dataset Configs ==========================
parser.add_argument('--test', action='store_true', help="only run test set")
parser.add_argument('--validate', action='store_true', help="only run validation set")
parser.add_argument('--dataset', type=str, default='mimic_cxr', help="iu-xray or mimic-cxr")
parser.add_argument('--annotation', type=str, default='/root/autodl-tmp/dataset/EVAP_cleaned_fixed_reports_filled.json')
# parser.add_argument('--annotation', type=str, default='/root/autodl-tmp/dataset/dataset_730.json')
parser.add_argument('--base_dir', type=str, default=r'/root/autodl-tmp/dataset/mimic_cxr/images')
parser.add_argument('--batch_size', default=50, type=int, help="use for training duration per worker")
parser.add_argument('--val_batch_size', default=5, type=int, help="use for validation duration per worker")
parser.add_argument('--test_batch_size', default=5, type=int, help="use for testing duration per worker")
parser.add_argument('--prefetch_factor', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--num_workers', default=16, type=int, help="Cpu num for dataloaders")

# ========================= Model Settings ============================
parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3], help="训练阶段: 1=Stage1Model, 2=Stage2Model, 3=R2GenGPT")
parser.add_argument('--vision_model',default='/root/autodl-tmp/hf_cache/rad_dino',type=str, help="使用的视觉模型")
parser.add_argument('--bert',default='/root/autodl-tmp/hf_cache/BiomedVLP-CXR-BERT',type=str, help="使用的文本模型")
parser.add_argument('--use_separate_queries', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--llama_model', default='/root/autodl-tmp/hf_cache/Llama-2-7b-chat-hf', type=str, help="使用的 LLM 模型")
parser.add_argument('--freeze_vm', default=True, type=lambda x: (str(x).lower() == 'true'), help='freeze vision model')
parser.add_argument('--freeze_tm', default=True, type=lambda x: (str(x).lower() == 'true'), help='freeze textal model')
parser.add_argument('--llm_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'), help="whether use lora for LLM model")
parser.add_argument('--llm_r', default=16, type=int, help='The dimension used by the LoRA update matrices')
parser.add_argument('--llm_alpha', default=16, type=int, help='Scaling factor.')
parser.add_argument('--vis_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'), help="whether use lora for vision model")
parser.add_argument('--vis_r', default=16, type=int, help='The dimension used by the LoRA update matrices')
parser.add_argument('--vis_alpha', default=16, type=int, help='Scaling factor.')
parser.add_argument('--lora_dropout', default=0.1, type=float, help='lora dropout')
parser.add_argument('--global_only', default=False, type=lambda x: (str(x).lower() == 'true'), help='use global embedding only')
parser.add_argument('--low_resource', default=False, type=bool)
parser.add_argument('--end_sym', default='</s>', type=str)

# ======================== SavedModel Configs ===========================
parser.add_argument('--savedmodel_path', type=str, default='.evcap')
parser.add_argument('--ckpt_file', type=str, default=None, help='the checkpoint file to load')
#parser.add_argument('--delta_file', type=str,default='/root/autodl-tmp/sava/SEG_test1/checkpoints/checkpoint_epoch0_step6509_bleu0.102376_cider0.141687.pth')
parser.add_argument('--delta_file', type=str)
parser.add_argument('--resume', action='store_true', help='resume training from delta_file without freezing loaded params')
parser.add_argument('--weights', type=list, default=[0.9, 0.1])
parser.add_argument('--scorer_types', type=list, default=['Bleu_4', 'micro_p'])

# ========================= Learning Configs ==========================
parser.add_argument('--learning_rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--gradient_clip_val', default=None, type=int, help='gradient clip value')

# ========================= External Memory Settings (Stage 2) ==========================
parser.add_argument('--ext_memory_path', type=str, default="/root/autodl-tmp/EVAP-autodl/checkpoint/ext_memory.pkl",
                    help="外部记忆库pickle文件路径（阶段三使用）")
parser.add_argument('--ext_memory_topn', type=int, default=4, help="外部记忆库检索Top-N个相似案例")
parser.add_argument('--ext_memory_alpha', type=float, default=0.5,
                    help="外部记忆库检索融合权重 alpha：score = alpha*global + (1-alpha)*local（阶段三使用）")
parser.add_argument('--ext_memory_fusion', type=int, default=None,
                    help="融合多少个检索到的报告（默认等于ext_memory_topn，即融合所有检索到的报告）")
parser.add_argument('--stage1_checkpoint', type=str,
                    default="/root/autodl-tmp/EVAP-autodl/checkpoint/stage1_checkpoint_epoch9_step6500.pth",
                    help="阶段一的checkpoint路径（阶段二构建记忆库时需要）")
parser.add_argument('--memory_split', type=str, default='train', choices=['train', 'val', 'test', 'memory'],
                    help="使用哪个split的数据构建记忆库（阶段二）")
parser.add_argument('--memory_limit', type=int, default=None, help="限制使用的样本数量（阶段二，None表示使用全部）")
parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/EVAP-autodl/checkpoint', help="输出pickle文件路径（阶段二）")

# ========================= Decoding Settings ==========================
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--num_beam_groups', type=int, default=1)
parser.add_argument('--min_new_tokens', type=int, default=80)
parser.add_argument('--max_new_tokens', type=int, default=120)
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--repetition_penalty', type=float, default=2.0)
parser.add_argument('--length_penalty', type=float, default=2.0)
parser.add_argument('--diversity_penalty', type=float, default=0)

# ====================== Pytorch Lightning ===========================
parser.add_argument('--devices', type=int, default=1, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
parser.add_argument('--strategy', type=str, default="auto", help='default ddp for multi-gpus')
parser.add_argument('--precision', type=str, default='bf16-mixed', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_test_batches', type=float, default=1.0, help='How much of test dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
parser.add_argument('--max_epochs', type=int, default=10, help='Stop training once this number of epochs is reached')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=0.5, help='How often to check the validation set')
parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates gradients over k batches before stepping the optimizer')
parser.add_argument("--num_sanity_val_steps", type=int, default=0, help='Sanity check runs n validation batches before starting the training routine')
