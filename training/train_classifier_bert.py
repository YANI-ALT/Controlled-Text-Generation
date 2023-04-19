from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification,AutoTokenizer, TrainingArguments, Trainer
from transformers import GPT2Config,GPT2Tokenizer,GPT2ForSequenceClassification
import os
import json
import csv
import numpy as np
import evaluate
import argparse
import torch
import pickle
from sklearn.metrics import accuracy_score

def read_data_file(path='datasets/e2e_data/src1_train.txt'):
  data_list=[]
  end_of_text_token="<|endoftext|>"
  with open(path, 'r') as ff:
    for row in ff:
        text=word_lst = row.split('||')[1]
        labels=word_lst = row.split('||')[0]
        data_list.append((labels,"REVIEW:"+text+end_of_text_token))

  return data_list

def split_label(label_string):
  pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in label_string.split('|')}
  return pair_lst

def obtain_class_list(data_list):
  types={'name':[],'Type':[],'price':[],'customer rating':[],'near':[],'food':[],'family friendly':[],'area':[]}
  for data_pt in data_list:
    pair_lst=split_label(data_pt[0])
    for t in types:
      if t in pair_lst and pair_lst[t] not in types[t]:
        types[t].append(pair_lst[t])
  
  return types

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
    

def get_label_map(classif_label):
    with open('label_maps/label2id_{}.pickle'.format(classif_label), 'rb') as handle:
        label2id_maps = pickle.load(handle)
    with open('label_maps/id2label_{}.pickle'.format(classif_label), 'rb') as handle:
        id2label_maps = pickle.load(handle)
    
    return label2id_maps,id2label_maps

class e2eDataset(Dataset):
    def __init__(self,id2label,label2id,tokenizer,classification_task='name',dataset_path = 'dataset/e2e_data/',data_portion='train'):
        super().__init__()
        self.label2id=label2id
        
        e2e_path=None
        if data_portion=='train':
            print("CLASSIFICATION TASK : ",classification_task)
            print("label2id Mapping: " ,self.label2id)
            e2e_path = os.path.join(dataset_path, 'src1_train.txt')
        elif data_portion=='test':
            e2e_path = os.path.join(dataset_path, 'src1_test.txt')
        elif data_portion=='valid':
            e2e_path = os.path.join(dataset_path, 'src1_valid.txt')

        self.data_list = []
        self.end_of_text_token = "<|endoftext|>"
        print("Loading dataset from path={}".format(e2e_path))
        with open(e2e_path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                label_mapping=split_label(row.split('||')[0])
                if classification_task=='customer_rating' and 'customer rating' in label_mapping:
                    label_value=self.label2id[label_mapping['customer rating']]
                elif classification_task=='family_friendly' and 'family friendly' in label_mapping:
                    label_value=self.label2id[label_mapping['family friendly']]
                elif classification_task in label_mapping:
                    label_value=self.label2id[label_mapping[classification_task]]
                else:
                    continue
                word_lst=word_lst.split(' ')
                prefix=""
                for i in range(1,len(word_lst)):
                  prefix+=" "+word_lst[i]
                  prefix=prefix.strip()
                  # decide to take this prefix or not
                  prob=np.random.uniform(0,1,1)[0]
                  if prob>0.6 and i!=len(word_lst)-1:
                    continue
                  token_val=tokenizer(prefix,truncation=True)
                  token_val['label']=label_value
                #   token_val['text']=prefix
                  self.data_list.append(token_val)
                #   print(token_val)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--dataset_path', type=str, default='dataset/e2e_data', help='')
    parser.add_argument('--mode', type=str, default='evaluate', help='')
    parser.add_argument('--model_name_or_path',type=str,default='gpt2-small',help='')
    parser.add_argument('--class_label',type=str,default='name',help='')
    parser.add_argument('--epochs',type=int,default=2,help='')
    parser.add_argument('--batch_size',type=int,default=16,help='')
    parser.add_argument('--lr',type=float,default=2e-5,help='')
    
    # COMMAND : python train_classifier_bert.py --mode train --mode_name_or_path gpt2 --class_label name --epochs 2
    np.random.seed(28)
    args = parser.parse_args()

    MODEL_STRING = 'distilbert-base-cased'
    DATASET_PATH=args.dataset_path # 
    CLASS_LABEL=args.class_label # ['name', 'Type', 'price', 'customer_rating', 'near']
    EPOCHS=args.epochs
    OUTPUT_DIR="bert_classifier_models/models_for_{}".format(CLASS_LABEL)
    MAX_LENGTH=60
    BACTH_SIZE=args.batch_size
    lr=args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("------------------------------")
    print("MODEL_STRING : ",MODEL_STRING)
    print("MODE : ",args.mode)
    print("CLASS_LABEL : ",CLASS_LABEL)
    print("------------------------------")

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    label2id,id2label=get_label_map(classif_label=CLASS_LABEL)
    n_labels=len(label2id)
    print("N_LABEL:",n_labels)
    e2e_dataset_test=e2eDataset(id2label,label2id,tokenizer,classification_task=CLASS_LABEL,data_portion='test')
    e2e_dataset_train=e2eDataset(id2label,label2id,tokenizer,classification_task=CLASS_LABEL,data_portion='train')
    e2e_dataset_valid=e2eDataset(id2label,label2id,tokenizer,classification_task=CLASS_LABEL,data_portion='valid')

    print("TRAIN : ",len(e2e_dataset_train))
    print("TEST : ",len(e2e_dataset_test))
    print("VALID : ",len(e2e_dataset_valid))


    # Get the actual model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=n_labels, id2label=id2label, label2id=label2id)

    
    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=BACTH_SIZE,
        per_device_eval_batch_size=BACTH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=e2e_dataset_train,
        eval_dataset=e2e_dataset_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.mode=='train':
        trainer.train()
        trainer.evaluate()
    elif args.mode=='test':
        trainer.evaluate()