TASK=CSC_cdcd-ns

deepspeed --num_gpus 8 --master_port=9901 launch src/train_bash.py \
    --stage sft \
    --model_name_or_path THUDM/chatglm3-6b \
    --finetuning_type lora \
    --template chatglm3 \
    --dataset_dir data \
    --dataset cscd-ns \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/ChatGLM3-6B-Chat/lora/eval_$(date +"%Y-%m-%d-%H-%M")_CSC_cscd-ns_noFT \
    --do_predict True 