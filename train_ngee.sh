export CUDA_VISIBLE_DEVICES="1"
 
python3 uniform_finetune.py --model_type llama --model_name_or_path decapoda-research/llama-7b-hf \
    --data negg_train --val_set_size 0 --lora_target_modules q_proj v_proj \
    --per_gpu_train_batch_size 8 --learning_rate 3e-4 --epochs 1