export CUDA_VISIBLE_DEVICES="0,1"


python3 -m torch.distributed.launch --nproc_per_node 2 \
    uniform_finetune.py --local_rank 0 \
    --model_type chatglm --model_name_or_path THUDM/chatglm-6b \
    --data negg_mini --lora_target_modules query_key_value --val_set_size 0 \
    --lora_r 16 --lora_alpha 16 --lora_dropout 0.1 --per_gpu_train_batch_size 1 \
    --learning_rate 2e-5 --epochs 1

#python3 -m torch.distributed.launch --nproc_per_node 2  \
#    uniform_finetune.py --local_rank 0 \
#    --model_type chatglm --model_name_or_path THUDM/chatglm-6b \
#    --data mini --val_set_size 0 --cutoff_len 256 --lora_target_modules query_key_value \
#    --per_gpu_train_batch_size 4 --learning_rate 3e-4 --epochs 1