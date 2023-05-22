export CUDA_VISIBLE_DEVICES="0"


python finetune.py  --data negg --size 7

# python3 -m torch.distributed.launch --nproc_per_node 2  \
#    --nnodes=1 --node_rank=0 finetune.py --data alpaca --size 7

# python3 finetune.py --model_type llama --model_name_or_path decapoda-research/llama-7b-hf \
#     --data alpaca-belle-cot --lora_target_modules q_proj v_proj \
#    --per_gpu_train_batch_size 4 --learning_rate 3e-4 --epochs 1 