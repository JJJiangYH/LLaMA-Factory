TASK=CSC_cdcd-ns

deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --stage sft \
    --model_name_or_path THUDM/chatglm3-6b \
    --adapter_name_or_path saves/ChatGLM3-6B-Base/full/train_2024-01-08-03-46_CSC_full_ft_1e-4 \
    --finetuning_type full \
    --template chatglm3 \
    --dataset_dir data \
    --dataset cscd-ns \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/ChatGLM3-6B-Base/full/eval_$(date +"%Y-%m-%d-%H-%M")_CSC_full_ft_1e-4 \
    --do_predict True 