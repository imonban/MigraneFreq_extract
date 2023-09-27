import torch
import argparse
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os
import pandas as pd

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def process(model_name, dataset, batch_size,device, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side='left'
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # loading dataset
    
    
    model = model.to(device)
    

    steps = math.ceil(len(dataset)/batch_size)

    write_to_dict = []

    for step in tqdm(range(steps), total=steps):
    # for td in tqdm(test_data, total= len(test_data)): 
        inputs = []
        start = step*batch_size
        end = min(step*batch_size+batch_size, len(dataset))
        td = dataset.iloc[start:end]
        for ind, t in td.iterrows():
            # print(t)
            t['input'] = t['SENT']
            t['instruction'] = "given the input, generate the headache frequency per month."
            prompt_input = PROMPT_DICT['prompt_input'].format_map(t)
            inputs.append(prompt_input)
        
        inputs_token = tokenizer(inputs, max_length=2020, padding = True, truncation=True, return_tensors='pt').to(device)
        # inputs_token = tokenizer(inputs, return_tensors='pt').to(device)
        # print(tokenizer.decode(inputs_token['input_ids'][0]))
        with torch.no_grad():
            beam_output = model.generate(**inputs_token,
                                        min_length=2,
                                        # max_new_tokens = 10,
                                        max_length=2048,
                                        # num_beams=5,
                                        early_stopping=True
                                        )
            # print(beam_output)
            prediction = tokenizer.batch_decode(beam_output, skip_special_tokens=True)
            write_to_dict.extend(prediction)
        
    print(len(write_to_dict))
    dataset['prediction'] = write_to_dict
    dataset.to_csv(output_file, index=False)

    # save_file = os.path.join( model_name,'predictions')
    # save_file = os.path.join(save_file, "pred_train.jsonl")

    # # Get the directory name from the file path.
    # dir_name = os.path.dirname(save_file)

    # # Check if the directory exists.
    # if not os.path.exists(dir_name):
    #     # If the directory does not exist, create it.
    #     try:
    #         os.makedirs(dir_name)
    #         print(f"Directory {dir_name} created.")
    #     except Exception as e:
    #         print(f"Failed to create directory {dir_name}. Error: {e}")
    # else:
    #     print(f"Directory {dir_name} already exists.")
    
    # write_jsonlines(save_file, write_to_dict)
    




        