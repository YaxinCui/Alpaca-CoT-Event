import argparse
from transformers import pipeline
from utils.tools import *
import json

from datasets import load_dataset
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

# add the custom dataset
DATA_PATH = {
             "alpaca": "./data/alpaca_data_cleaned.json",
             "belle": "./data/belle_data_cn.json",
             "alpaca-belle": "./data/alpaca_plus_belle_data.json",
             "cot": "./data/CoT_data.json",
             "alpaca-cot": "./data/alcapa_plus_cot.json",
             "alpaca-belle-cot": "./data/alcapa_plus_belle_plus_cot.json",
             "belle1.5m": "./data/belle_data1.5M_cn.json",
             "finance": "./data/finance_en.json",
             "multiturn_chat": "./data/multiturn_chat_0.8M.json",
             "negg_train": "./data/negg_train_data.json",
             "negg_dev": "./data/negg_dev_data.json",
             "negg_test": "./data/negg_test_data.json",
            }

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


CUTOFF_LEN = 250
def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

def predict(args):
    model, tokenizer = get_fine_tuned_model(args)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    input_data = get_predict_data(args)
    
    input_data = load_dataset("json", data_files=DATA_PATH[args.data])
    
    def predict_and_write_to_file(input_data, batch_size):
        with open(args.result_dir, 'w') as f:
            for i in range(0, len(input_data['input']), batch_size):
                batch = input_data['input'][i:i + batch_size]
                instructions = input_data['instruction'][i:i + batch_size]
                
                generated_text = generator(batch, max_length=args.cutoff_len, num_return_sequences=1)
                
                train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
                
                for instruction, prompt, result in zip(instructions, batch, generated_text):
                    res = result[0]['generated_text']
                    filter_res = generate_service_output(res, prompt, args.model_type, args.lora_dir)
                    instruction['generate'] = filter_res
                    str_info = json.dumps(instruction, ensure_ascii=False)
                    f.write(str_info + "\n")
                    f.flush()
    predict_and_write_to_file(input_data, args.predict_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some llm info.')
    parser.add_argument('--model_type', type=str, default="belle_bloom", choices=AVAILABLE_MODEL,
                        help='the base structure (not the model) used for model or fine-tuned model')
    parser.add_argument('--size', type=str, default="7b",
                        help='the type for base model or the absolute path for fine-tuned model')
    parser.add_argument('--data', type=str, default="negg_test", help='the data used for predicting')
    parser.add_argument('--lora_dir', type=str, default="none",
                        help='the path for fine-tuned lora params, none when not in use')
    parser.add_argument('--result_dir', default="saved_result.txt", type=str)
    parser.add_argument('--predict_batch_size', default=4, type=int)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--cutoff_len', default=512, type=int)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed serving')
    args = parser.parse_args()
    print(args)
    predict(args)
