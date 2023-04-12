import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
from transformers import GPT2Config,GPT2Tokenizer,GPT2ForSequenceClassification


def get_label_map(type_):
  with open('label_maps/label2id_{}.pickle'.format(type_), 'rb') as handle:
    label2id = pickle.load(handle)
  with open('label_maps/id2label_{}.pickle'.format(type_), 'rb') as handle:
    id2label = pickle.load(handle)
  return label2id,id2label


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    # print(ind)
    top_prob = probs[ind]
    # print(top_prob)
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def get_classifier_logits(classifer_model,classifier_tokenizer,text,id2label,label2id):
  encoding = classifier_tokenizer(text, return_tensors="pt")
  encoding = {k: v.to(classifer_model.device) for k,v in encoding.items()}

  outputs = classifer_model(**encoding) 
  
  logits = outputs.logits.detach()[0]
#   print("classifier logits: ",logits)
  # apply sigmoid + threshold
#   sigmoid = torch.nn.Sigmoid()
#   probs = sigmoid(logits.squeeze().cpu())
#   predictions = np.zeros(probs.shape)
#   predictions[np.where(probs >= 0.5)] = 1
  # turn predicted id's into actual label names
  #predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
  # print(predicted_labels)
  preds=torch.softmax(logits,dim=0)
#   print("classifier preds: ",preds)
  logits=torch.log(preds)
#   print(logits)
  return logits

def get_sentences(curr_ids,ind,tokenizer,device):
  output_sent=[]
  for next_token_id in ind:
    new_cur_ids = torch.cat([curr_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
    output_list = list(new_cur_ids.squeeze().to('cpu').numpy())
    output_text = tokenizer.decode(output_list)
    output_sent.append(output_text)
#   print(output_sent)
  return output_sent

def get_all_classifier_logits(classifer_data,sentence,values,id2label_dict,label2id_dict):
    all_classif_logit=0
    for context in values:
        value=values[context] # (context,value) pair
        model_data=classifer_data[context]
        classifier_model=model_data['model']
        classifier_tokenizer=model_data['tokenizer']
        id2label=id2label_dict[context]
        label2id=label2id_dict[context]
        classif_logits=get_classifier_logits(classifier_model,classifier_tokenizer,sentence,id2label,label2id)
        classif_logits_for_value=classif_logits.numpy()[label2id[value]]

        all_classif_logit=all_classif_logit+classif_logits_for_value

    # print("ALL CLASSIF combined: ",all_classif_logit)
    return all_classif_logit

def choose_from_top_controlled(classifer_data,gen_tokenizer,cur_ids,logits,label2id_dict,id2label_dict,values,sample='max',n=10,lambda_condition=1,debug=False):
    softmax_logits = torch.softmax(logits, dim=0)
    ind = np.argpartition(softmax_logits, -n)[-n:] # get the top n indices\

    text_preds=get_sentences(cur_ids,ind,tokenizer=gen_tokenizer,device=device) # get the entire senteces for these words form the generative model
    if debug:
        print("-----"*10)
        print("Sentences after top n={} Predicted nextword ".format(n))
    
    classif_logits=[]
    for i,txt in enumerate(text_preds):
    #   classif_logit=get_all_classifier_logits(txt.split('REVIEW:')[1],id2label,label2id).detach().numpy()[0][label2id[value]]
        if len(txt.split('REVIEW '))==1:
            classif_logits.append(0)
            continue
        classif_logit=get_all_classifier_logits(classifer_data,txt.split('REVIEW ')[1],values,id2label_dict,label2id_dict)
        classif_logits.append(classif_logit)

    classif_logits_tensor=torch.tensor(classif_logits)

    classif_preds=torch.softmax(classif_logits_tensor,dim=0)
    # logits=torch.log(softmax_logits)
    conditioned_logits=logits[ind]+lambda_condition*classif_logits_tensor
    # print(torch.softmax(logits[ind]+classif_logits_tensor,dim=0))
    # print(torch.softmax(logits[ind]+lambda_condition*classif_logits_tensor,dim=0))
    conditioned_probs=torch.softmax(conditioned_logits, dim=0)
    if debug :
        for i,txt in enumerate(text_preds):
            print("{} Prob_word :{} Prob_classif={} Prod={} Prod={} (These should be same)".format(txt,softmax_logits.numpy()[ind[i]],classif_preds.numpy()[i],conditioned_probs.numpy()[i],classif_preds.numpy()[i]*softmax_logits.numpy()[ind[i]]))
        print(conditioned_probs)
        print()
    if sample=='max':
        argmax_ind=np.argmax(conditioned_probs)
    else :
        argmax_ind=choose_from_top(conditioned_probs.detach().numpy(),n=5)

    if debug:
        print("SAMPLING RESULT : ",text_preds[argmax_ind])

    return ind[argmax_ind]

def read_all_label_maps(control_list):
    label2id_dict={}
    id2label_dict={}
    for control_name in control_list:
        label2id_dict[control_name],id2label_dict[control_name]=get_label_map(control_name)

    return label2id_dict,id2label_dict

def split_label(label_string):
  pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in label_string.split('|')}
  return pair_lst

def get_parsed_input_data_json(input_path):
    data = []
    temp_data={}
    with open(input_path) as f:
        for line in f:
            temp_data={}
            temp=json.loads(line)
            value=""
            start=2
            if temp[0]=='family':
                start=3
            for val in temp[start:]:
                value=value+" "+val
            if temp[0]=='family':
                temp_data[temp[0]+" friendly"]=value.strip()
            else:
                temp_data[temp[0]]=value.strip()
            data.append(temp_data)
    return data
    

def get_parsed_input_data(input_path):
    print("Reading input file from {}...".format(input_path))
    input_data=[]
    with open(input_path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                label_mapping=split_label(row.split('||')[0])
                input_data.append(label_mapping)

    return input_data


def get_classifier(model_path,n_labels,device):
    print("-------------------------------")
    print('Accessing model from ',model_path)
    print('Loading configuraiton..')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token


    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)
    print("-------------------------------")
    return model,tokenizer

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def collect_classifiers(control_list,n_labels_dict,device):
    classifier_data={}
    model_paths=read_json('classifier_models/model_paths.json')
    for control_name in control_list:
        if control_name not in model_paths:
            print("{} - no model assigned for this".format(control_name))
            continue

        model,tokenizer=get_classifier(model_paths[control_name],n_labels_dict[control_name],device)
        classifier_data[control_name]={'model':model,'tokenizer':tokenizer}
        
    
    return classifier_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    parser.add_argument('--input_file', type=str, default='dataset/control_task/test.txt', help='')
    parser.add_argument('--gen_model_path', type=str, default='', help='')
    parser.add_argument('--per_control',type=int,default=3,help='')
    parser.add_argument('--lambda_condition',type=int,default=1,help='')
    parser.add_argument('--sample_stratergy',type=str,default='max',help='')

    # COMMAND : python3 controlled_text_generation.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1

    args = parser.parse_args()
    
    TG_MODEL_PATH=args.gen_model_path
    PER_CONTROL=args.per_control
    INPUT_PATH=args.input_file
    lambda_condition=args.lambda_condition
    sample_strat=args.sample_stratergy

    # get the generator model tokenizer and model
    gen_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gen_model = GPT2LMHeadModel.from_pretrained('gpt2')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # load the model weights
    gen_model.load_state_dict(torch.load(TG_MODEL_PATH,map_location=device))

    output_file_path = f'generated_output/controlled_generation.txt'

    gen_model.eval()

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    control_list=['area', 'name', 'price', 'food', 'customer_rating', 'family_friendly', 'near', 'Type']
    # control_list=['area', 'customer_rating', 'name', 'near', 'Type']
    # control_list=['area']

    label2id_dict,id2label_dict=read_all_label_maps(control_list)

    input_data=[]

    if INPUT_PATH.endswith('.txt'):
        input_data=get_parsed_input_data(INPUT_PATH)
    elif INPUT_PATH.endswith('.json'):
        input_data=get_parsed_input_data_json(INPUT_PATH)
    
    n_labels_dict={}
    for control_name in control_list:
        n_labels_dict[control_name]=len(label2id_dict[control_name])

    classifier_data=collect_classifiers(control_list,n_labels_dict,device=device)


    sent_num = 0
    control_val_flag=False
    with torch.no_grad():
            for control_val_pairs in input_data:
                print(control_val_pairs)
                if 'family friendly' in control_val_pairs:
                    control_val_pairs['family_friendly'] = control_val_pairs.pop('family friendly')
                
                if 'customer rating' in control_val_pairs:
                    control_val_pairs['customer_rating'] = control_val_pairs.pop('customer rating')

                for control_val in control_val_pairs:
                    # print(control_val)
                    if control_val=='family friendly':
                        control_val='family_friendly'
                    elif control_val=='customer rating':
                        control_val='customer_rating'

                    if control_val not in control_list :
                        control_val_flag=True
                        break
                if control_val_flag:
                    control_val_flag=False
                    continue
                
                for idx in range(PER_CONTROL):
                
                    sent_finished = False

                    cur_ids = torch.tensor(gen_tokenizer.encode("REVIEW ")).unsqueeze(0).to(device)

                    for i in range(100):
                        outputs = gen_model(cur_ids, labels=cur_ids)
                        loss, logits = outputs[:2]
                        softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                        # next_token_id = choose_from_top_controlled(logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                        next_token_id=choose_from_top_controlled(classifier_data,gen_tokenizer,cur_ids,logits[0,-1],label2id_dict,id2label_dict,control_val_pairs,debug=False,sample=sample_strat,n=10,lambda_condition=lambda_condition)
                        
                        cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                        if next_token_id in gen_tokenizer.encode('<|endoftext|>'):
                            sent_finished = True
                            break

                    # print("here")
                    sent_finished=True
                    if sent_finished:
                        
                        sent_num = sent_num + 1
                        
                        output_list = list(cur_ids.squeeze().to('cpu').numpy())
                        output_text = gen_tokenizer.decode(output_list)
                        prefix=""
                        for vals in control_val_pairs:
                            prefix+=vals+":"+control_val_pairs[vals]
                        prefix=prefix+"|| "
                        output_text=output_text.split("REVIEW")[1]
                        output_text=output_text.strip()
                        print(prefix+output_text)
                        output_text=prefix+output_text
                        with open(output_file_path, 'a') as f:
                            f.write(f"{output_text} \n\n")

