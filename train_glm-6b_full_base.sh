export HF_HOME=/workspace/jyh/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

date=$(date +"%Y-%m-%d-%H-%M-%S")

lr=1e-5

save_dir=saves/ChatGLM3-6B-Base/full/train_${lr}_${date}

/workspace/jyh/miniconda3/envs/llama_factory/bin/deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path THUDM/chatglm3-6b-base \
    --finetuning_type full \
    --template chatglm3 \
    --dataset_dir data \
    --dataset haruhi_zero_0,haruhi_zero_1,haruhi_zero_2,haruhi_zero_3,haruhi_zero_4,haruhi_zero_5,alpaca_gpt4_zh \
    --cutoff_len 1024 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --learning_rate $lr \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target query_key_value \
    --output_dir $save_dir \
    --fp16 True \
    --plot_loss True > train_full_${lr}_${date}_base.log 2>&1