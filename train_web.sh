export HF_HOME=/workspace/jyh/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=3 /workspace/jyh/miniconda3/envs/llama_factory/bin/python src/train_web.py