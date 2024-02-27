TASK=CSC_cscd-ns

export HF_HOME=/workspace/jyh/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

/workspace/jyh/miniconda3/envs/llama_factory/bin/deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path THUDM/chatglm3-6b-base \
    --finetuning_type full \
    --template default \
    --dataset_dir data \
    --dataset NaCGEC \
    --cutoff_len 1000 \
    --learning_rate 0.0001 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --neftune_noise_alpha 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target query_key_value \
    --output_dir saves/ChatGLM3-6B-Base/full/train_$(date +"%Y-%m-%d-%H-%M")_GEC_full_ft_1e-4 \
    --plot_loss True 