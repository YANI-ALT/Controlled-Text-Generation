from flair.nn import Classifier
from flair.models import SequenceTagger
from flair.data import Sentence
import argparse
from transformers import GPT2Config,GPT2Tokenizer,GPT2ForSequenceClassification,AutoModelForSequenceClassification,AutoTokenizer,AutoConfig
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import os
import json

def get_parsed_input_data_json(file_path):
    POS_tags=[]
    with open('dataset/control_target/target_pos.json', 'r') as json_file:
        json_list = list(json_file)
        for line in json_list:
            POS_tags.append(json.loads(line)['pos'])
            # print(json.loads(line)['pos'])

    return POS_tags

def get_sentences(curr_ids,ind,tokenizer,device):
  output_sent=[]
  for next_token_id in ind:
    new_cur_ids = torch.cat([curr_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
    output_list = list(new_cur_ids.squeeze().to('cpu').numpy())
    output_text = tokenizer.decode(output_list)
    output_sent.append(output_text)
#   print(output_sent)
  return output_sent

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    # print(ind)
    top_prob = probs[ind]
    # print(top_prob)
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def get_classifier_logits(classif_model,classif_token,text,isSentEnd,debug):
    # make a sentence
    encoding = classif_token(text, return_tensors="pt")
    encoding = {k: v.to(classif_model.device) for k,v in encoding.items()}

    outputs = classif_model(**encoding) 
    
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
    label_index=0
    if isSentEnd:
        label_index=1
    
    return logits[label_index]
    

def choose_from_top_controlled(classif_model,classif_token,gen_tokenizer,cur_ids,logits,isSentEnd,debug,sample,n,lambda_condition):
    softmax_logits = torch.softmax(logits, dim=0)
    ind = np.argpartition(softmax_logits, -n)[-n:] # get the top n indices\

    text_preds=get_sentences(cur_ids,ind,tokenizer=gen_tokenizer,device=device) # get the entire senteces for these words form the generative model
    if debug:
        print("-----"*10)
        print("Sentences after top n={} Predicted nextword ".format(n))
    
    classif_logits=[]
    
    for i,txt in enumerate(text_preds):
        txt=txt.strip()
    #   classif_logit=get_all_classifier_logits(txt.split('REVIEW:')[1],id2label,label2id).detach().numpy()[0][label2id[value]]
        if len(txt.split('REVIEW '))==1:
            classif_logits.append(torch.inf)
            continue
        classif_logit=get_classifier_logits(classif_model,classif_token,txt.split('REVIEW ')[1],isSentEnd,debug)
        classif_logits.append(classif_logit)

    classif_logits_tensor=torch.tensor(classif_logits)

    classif_preds=torch.softmax(classif_logits_tensor,dim=0)
    # logits=torch.log(softmax_logits)
    conditioned_logits=logits[ind]+lambda_condition*classif_logits_tensor
    # print(torch.softmax(logits[ind]+classif_logits_tensor,dim=0))
    # print(torch.softmax(logits[ind]+lambda_condition*classif_logits_tensor,dim=0))
    conditioned_probs=torch.softmax(conditioned_logits, dim=0)
    if debug==True:
        for i,txt in enumerate(text_preds):
            print("{} Prob_word :{} Prob_classif={} Prod={} Prod={} (These should be same)".format(txt,softmax_logits.numpy()[ind[i]],classif_preds.numpy()[i],conditioned_probs.numpy()[i],classif_preds.numpy()[i]*softmax_logits.numpy()[ind[i]]))
        print(conditioned_probs)
        print()
    if sample=='max':
        argmax_ind=np.argmax(conditioned_probs)
    else :
        argmax_ind=choose_from_top(conditioned_probs.detach().numpy(),n=5)

    if debug==True:
        print("CONTROL : ",control_len)
        print("SAMPLING RESULT : ",text_preds[argmax_ind])

    return ind[argmax_ind]

def get_classifier_bert(model_path,n_labels,device):
    print("-------------------------------")
    print('Accessing model from ',model_path)
    print('Loading configuraiton..')
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    # # default to left padding
    # tokenizer.padding_side = "left"
    # # Define PAD Token = EOS Token = 50256
    # tokenizer.pad_token = tokenizer.eos_token


    # Get the actual model.
    print('Loading model...')
    # model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)
    # resize model embedding to match new tokenizer
    # model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    # model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)
    print("-------------------------------")
    return model,tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    # parser.add_argument('--input_file', type=str, default='dataset/control_task/test.txt', help='')
    parser.add_argument('--gen_model_path', type=str, default='', help='')
    parser.add_argument('--classif_model', type=str, default='', help='')
    parser.add_argument('--per_control',type=int,default=3,help='')
    parser.add_argument('--lambda_condition',type=int,default=1,help='')
    parser.add_argument('--sample_stratergy',type=str,default='max',help='')
    # parser.add_argument('--use_bert',type=bool,default=False,help='')
    parser.add_argument('--debug',type=bool,default=False,help='')
    parser.add_argument('--n_lm',type=int,default=20,help='the top-n samples from the LM')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1

    args = parser.parse_args()
    
    TG_MODEL_PATH=args.gen_model_path
    PER_CONTROL=args.per_control
    # INPUT_PATH=args.input_file
    lambda_condition=args.lambda_condition
    sample_strat=args.sample_stratergy
    # use_bert=args.use_bert
    debug_flag=args.debug
    n_lm=args.n_lm

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # get the generator model tokenizer and model
    gen_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gen_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    len_classifier,len_tokenizer=get_classifier_bert(args.classif_model,2,device)

    

    gen_model.load_state_dict(torch.load(TG_MODEL_PATH,map_location=device))
    
    output_file_path = f'generated_output/controlled_len_generation_n_lm_{n_lm}_sample_strat_{sample_strat}.txt'

    gen_model.eval()
    
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    

    # if INPUT_PATH.endswith('.txt'):
    #     print("Not handling .txt yet")
    # elif INPUT_PATH.endswith('.json'):
    #     input_data=get_parsed_input_data_json(INPUT_PATH)

    output_data=[]
    with torch.no_grad():
        for control_len in [10,15,20,25,30,35,40]:
            if debug_flag :
                print("CONTROL Len :",control_len)
            
            
            sent_finished = False

            cur_ids = torch.tensor(gen_tokenizer.encode("REVIEW ")).unsqueeze(0).to(device)
            gen_len=0
            for i in range(100):
                outputs = gen_model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                # next_token_id = choose_from_top_controlled(logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                isSentEnd=i==control_len
                next_token_id=choose_from_top_controlled(len_classifier,len_tokenizer,gen_tokenizer,cur_ids,logits[0,-1],isSentEnd,debug=debug_flag,sample=sample_strat,n=n_lm,lambda_condition=lambda_condition)
                
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence
                gen_len+=1
                if next_token_id in gen_tokenizer.encode('<|endoftext|>'):
                    sent_finished = True
                    break

            sent_finished=True
            if sent_finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = gen_tokenizer.decode(output_list)
                if len(output_text.split("REVIEW "))>1:
                    output_text=output_text.split("REVIEW ")[1]
                output_text=output_text.strip()
                output_dict={"len":control_len,"sentence":output_text}
                output_data.append(output_dict)
                
                print(output_text)
                print("EXPECTED Len : ",control_len)
                sent=Sentence(output_text)
                print("OBTAINED : ",gen_len)
                    

    with open(output_file_path, 'a') as f:
        for entry in output_data:
            json.dump(entry, f)
            f.write('\n')
    print("Succesfully written to ",output_file_path)
            