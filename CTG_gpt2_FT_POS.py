import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os

from transformers import AdamW,get_linear_schedule_with_warmup
import spacy


# python -m spacy download en_core_web_sm
# import logging
# logging.getLogger().setLevel(logging.CRITICAL)

# import warnings
# warnings.filterwarnings('ignore')
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class e2eDataset(Dataset):
    def __init__(self, dataset_path = 'dataset/e2e_data/',data_portion='train'):
        super().__init__()
        e2e_path=None
        if data_portion=='train':
          e2e_path = os.path.join(dataset_path, 'src1_train.txt')
        elif data_portion=='test':
          e2e_path = os.path.join(dataset_path, 'src1_test.txt')
        elif data_portion=='valid':
          e2e_path = os.path.join(dataset_path, 'src1_valid.txt')
        self.nlp = spacy.load("en_core_web_sm")
        # doc = nlp('Alimentum at high prices for a family atmosphere at the Riverside Inn that can suit the entire Family,')
        # output_sent_pos=[]
        # for token in doc:
        #     output_sent_pos.append(token.pos_)
        self.data_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(e2e_path, 'r') as ff:
            for row in ff:
                # word_lst = row.split('||')[1]
                parsed_sent=self.nl(row)
                output_sent_pos=[]
                for token in parsed_sent:
                    output_sent_pos.append(token.pos_)
                ctrl=" ".join(output_sent_pos)
                self.data_list.append(ctrl+"||"+row+self.end_of_text_token)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--dataset_path', type=str, default='dataset/e2e_data', help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--model_name_or_path',type=str,default='gpt2',help='')
    parser.add_argument('--epochs',type=int,default=2,help='')

    args = parser.parse_args()

    MODEL_STRING = args.model_name_or_path
    MODE=args.mode # train 
    EPOCHS=args.epochs

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    print("Loading tokenzier...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_STRING)
    print("Loading Model...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_STRING)
    
    model = model.to(device)
    print('Model on device=',device)

    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    WARMUP_STEPS = 5000
    MAX_SEQ_LEN = 400
    
    models_folder='CTG_POS_control_models'

    dataset = e2eDataset(data_portion='train')
    e2e_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # setup for training
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    tmp_tens = None
    
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(EPOCHS):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx,data_pt in enumerate(e2e_loader):
            
            #################### "Fit as many sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            data_tens = torch.tensor(tokenizer.encode(data_pt[0])).unsqueeze(0).to(device)
            #Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if data_tens.size()[1] > MAX_SEQ_LEN:
                continue
            
            #The first joke sequence in the sequence
            if not torch.is_tensor(tmp_tens):
                tmp_tens = data_tens
                continue
            else:
                #The next joke does not fit in so we process the sequence and leave the last joke 
                #as the start for next sequence 
                if tmp_tens.size()[1] + data_tens.size()[1] > MAX_SEQ_LEN:
                    work_tens = tmp_tens
                    tmp_tens = data_tens
                else:
                    #Add the joke to sequence, continue and try to add more
                    tmp_tens = torch.cat([tmp_tens, data_tens[:,1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################
                
            outputs = model(work_tens, labels=work_tens)
            loss, logits = outputs[:2]                        
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
                        
            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0    
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0
        
        # Store the model after each epoch to compare the performance of them
        if epoch%5==0 or epoch==EPOCHS-1:
            print("Saving model ")
            torch.save(model.state_dict(), os.path.join(models_folder, f"{MODEL_STRING}_e2e_{epoch}.pt"))
    
    print("DONE TRAINING all results saved in {} ".format(models_folder))
    print("x-----x------x-------x-------x-----------x---------x")