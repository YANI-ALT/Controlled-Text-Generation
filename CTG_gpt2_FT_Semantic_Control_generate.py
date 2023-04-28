
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json

parser = argparse.ArgumentParser(description='generation args.')
parser.add_argument('--num_gen', type=int, default=10, help='')
parser.add_argument('--model_path',type=str,default='gpt2',help='')
parser.add_argument('--output_file',type=str,default='',help='')
parser.add_argument('--input_file',type=str,default='dataset/control_target/target_attribute.json',help='')
parser.add_argument('--sample_strat',type=str,default='Topk',help='') #topk beam greedy
parser.add_argument('--sample_strat_n',type=int,default=3,help='')
# COMMAND : python text_generation_gpt2_generate.py --num_gen 100 --model_path trained_models_text_generation/
args = parser.parse_args()

NUM_GEN=args.num_gen
MODEL_PATH=args.model_path
SAMPLE_STRAT=args.sample_strat
SAMPLE_N=args.sample_strat_n

def get_parsed_input_data_json_pos(file_path):
    POS_tags=[]
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)
        for line in json_list:
            POS_tags.append(json.loads(line)['pos'])
            # print(json.loads(line)['pos'])

    return POS_tags

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def get_parsed_input_data_json(input_path):
    data = []
    temp_data={}
    with open(input_path) as f:
        for line in f:
            temp_data={}
            temp=json.loads(line)
            value=""
            start=2
            if temp[0]=='family' or temp[0]=='customer':
                start=3

            for val in temp[start:]:
                value=value+" "+val
            if temp[0]=='family':
                temp_data[temp[0]+" friendly"]=value.strip()
            elif temp[0]=='customer':
                temp_data[temp[0]+" rating"]=value.strip()
            else:
                temp_data[temp[0]]=value.strip()
            data.append(temp_data)
    return data

def get_attribute_string(semantic_control):
    prefix_control=""
    key_val=""
    for key in semantic_control:
        key_val=key+" : "+semantic_control[key]
        prefix_control=prefix_control+key_val+"|"
    return prefix_control


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)

model.load_state_dict(torch.load(MODEL_PATH,map_location=device))

output_file_path = args.output_file

model.eval()
if os.path.exists(output_file_path):
    os.remove(output_file_path)
    
sent_num = 0
input_data=get_parsed_input_data_json(args.input_file)
print(input_data)
if len(output_file_path)==0:    
    if SAMPLE_STRAT=='greedy':
        output_file_path='generated_output/CTG_FT_semantic_control_generated_greedy.txt'
    elif SAMPLE_STRAT=='beam':
        output_file_path='generated_output/CTG_FT_semantic_control_generated_beam_{}.txt'.format(SAMPLE_N)
    elif SAMPLE_STRAT=='Topk':
        output_file_path='generated_output/CTG_FT_semantic_control_generated_Top{}.txt'.format(SAMPLE_N)
    else:
        output_file_path='generated_output/CTG_FT_semantic_control_generated.txt'

with torch.no_grad():
   
        for semantic_control in input_data:
            condition_prefix=get_attribute_string(semantic_control)
            # print(condition_prefix)
            sent_finished = False

            cur_ids = torch.tensor(tokenizer.encode(condition_prefix)).unsqueeze(0).to(device)
            if SAMPLE_STRAT=='Topk':
                # perform TOpk sampling decoding
                for i in range(100):
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                    if i < 3:
                        n = 20
                    else:
                        n = SAMPLE_N
                    next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                    if next_token_id in tokenizer.encode('<|endoftext|>'):
                        sent_finished = True
                        break

                
                if sent_finished:
                    
                    sent_num = sent_num + 1
                    
                    output_list = list(cur_ids.squeeze().to('cpu').numpy())
                    output_text = tokenizer.decode(output_list)
                    print(output_text)
                    with open(output_file_path, 'a') as f:
                        f.write(f"{output_text} \n\n")

            elif SAMPLE_STRAT == 'beam':
                # perform decoding by beam search with beam length as input argument
                beam_output = model.generate(
                    cur_ids, 
                    max_length=100, 
                    num_beams=SAMPLE_N, 
                    early_stopping=True
                )
                output_text=tokenizer.decode(beam_output[0], skip_special_tokens=True)+"<|endoftext|>"
                print(output_text)
                with open(output_file_path, 'a') as f:
                        f.write(f"{output_text} \n\n")
            
            elif SAMPLE_STRAT == 'greedy':
                # perform decoding by greedy
                greedy_output = model.generate(cur_ids, max_length=50)
                output_text=tokenizer.decode(greedy_output[0], skip_special_tokens=True)+"<|endoftext|>"
                print(output_text)
                with open(output_file_path, 'a') as f:
                        f.write(f"{output_text} \n\n")