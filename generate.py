import sys
import torch
from peft import PeftModel
import transformers
import argparse
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, 
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, help='the data used for instructing tuning')
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
parser.add_argument('--size', type=str, help='the size of llama model')
parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
parser.add_argument('--lora_path', default="decapoda-research/llama-7b-hf", type=str)

args = parser.parse_args()
CUTOFF_LEN = 500
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


if args.model_type == "llama":
    BASE_MODEL = f"decapoda-research/llama-{args.size}b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    LORA_WEIGHTS = "./saved-"+args.data+args.size+"b"
elif args.model_type == "bloom":
    BASE_MODEL = "bigscience/bloomz-7b1-mt"
    tokenizer = BloomTokenizerFast.from_pretrained(BASE_MODEL)
    LORA_WEIGHTS = "./saved_bloominstinwild-belle1.5m/middle"
elif args.model_type == "chatglm":
    BASE_MODEL = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
    LORA_WEIGHTS = "./saved_chatglm" + args.data 


#LORA_WEIGHTS = "./saved_models/llama-7b-hf_egg_trai/lora_old2/"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

LOAD_8BIT = True

if device == "cuda":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("model load from huggingface success")
        """
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
        """
        # print("lora load from local success")
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        """
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
        )
        """
        # print("lora load from local success")
elif device == "mps":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
else:
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

    elif args.model_type == "bloom":
        model = BloomForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )   
    elif args.model_type == "chatglm":
        model = AutoModel.from_pretrained(
            BASE_MODEL,trust_remote_code=True,
            device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )   
        
tokenizer.save_pretrained(BASE_MODEL)
model.save_pretrained(BASE_MODEL)
def generate_prompt(instruction, input=None):
    return f"""### EXAM
Single-choice Question: Based on the presented event processes sequence, please select the most likely subsequent event from the provided choices.
Processes: {instruction}

### Event Options:
{input}

### Response:
The format for your answer should be "?. (xxx, xxx, xxx…)". So the next predected correct option is: 
"""

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=1.0,
    top_p=0.9,
    top_k=40,
    num_beams=4,
    max_new_tokens=20,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.8,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.split("So the next predected correct option is: ")[-1].strip()
    return output

from datasets import load_dataset

from tqdm import tqdm
if __name__ == "__main__":
    # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    TEST_DATA_PATH = "data/negg_test_data.json"
    data = load_dataset("json", data_files=TEST_DATA_PATH)
    
    all_num = 0
    accurate_num = 0
    for index, example in tqdm(enumerate(data['train'])):
        all_num += 1
        instruction = example['instruction']
        input_ = example['input']
        output = example['output']
        response = evaluate(instruction, input=input_)
        print(str(index)+"-"*100)
        print("Model answer:", response.strip())
        print("-"*50)
        print("True answer:", output)
        print("-"*100)
        if response[0]==output[0]:
            accurate_num += 1
        if index > 300:
            break
    print(f"accurate = {accurate_num}/{all_num} = ", accurate_num/all_num)