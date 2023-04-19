
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os

parser = argparse.ArgumentParser(description='generation args.')
parser.add_argument('--num_gen', type=int, default=10, help='')
parser.add_argument('--model_path',type=str,default='gpt2',help='')
parser.add_argument('--output_file',type=str,default='generated_output/generated.txt',help='')

# COMMAND : python text_generation_gpt2_generate.py --num_gen 100 --model_path trained_models_text_generation/
args = parser.parse_args()

NUM_GEN=args.num_gen
MODEL_PATH=args.model_path



def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

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
with torch.no_grad():
   
        for idx in range(NUM_GEN):
        
            sent_finished = False

            cur_ids = torch.tensor(tokenizer.encode("REVIEW ")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
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