export HF_HOME=/workspace/jyh/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

date=$(date +"%Y-%m-%d-%H-%M-%S")

lr=1e-4

save_dir=saves/Gemma-7B-Chat/full/train_${lr}_${date}
echo $save_dir

/workspace/jyh/miniconda3/envs/llama_factory/bin/deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path google/gemma-7b-it \
    --finetuning_type full \
    --template gemma \
    --dataset_dir data \
    --dataset haruhi_zero_0,haruhi_zero_1,haruhi_zero_2,haruhi_zero_3,haruhi_zero_4,haruhi_zero_5,haruhi_zero_6 \
    --cutoff_len 1024 \
    --num_train_epochs 3.0 \
    --learning_rate $lr \
    --max_samples 100000 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target query_key_value \
    --output_dir  $save_dir\
    --bf16 True \
    --plot_loss True > train_full_${lr}_${date}_Gemma-7B-Chat.log 2>&1