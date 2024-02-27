TASK=CSC_cscd-ns

accelerate launch src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path THUDM/chatglm3-6b-base \
    --finetuning_type full \
    --template default \
    --dataset_dir data \
    --dataset cscd-ns \
    --cutoff_len 1024 \
    --learning_rate 0.0001 \
    --num_train_epochs 10.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --neftune_noise_alpha 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target query_key_value \
    --output_dir saves/ChatGLM3-6B-Base/full/train_2024-01-07-08-43_CSC_Full_ft \
    --fp16 True \
    --plot_loss True 