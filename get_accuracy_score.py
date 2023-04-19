import json
import os
import argparse
import spacy

def read_file_lbyl(path):
    lines=[]
    with open(path) as f:
        for line in f:
            lines.append(line)
    
    return lines

def read_jsonl(path):
    json_list=None
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    return json_list

def get_pos_score(json_list):
    nlp=spacy.load("en_core_web_sm")
    count_correct=0
    count_total=0
    for json_str in json_list:
        entry = json.loads(json_str)
        pos_tags = nlp(entry["sentence"])
        # print("SENTENCE : ",entry["sentence"])
        # print("TRUTH : ",entry["pos"])
        sample_list = []
        for token in pos_tags:
            sample_list.append(token.pos_)
        n = min(len(entry["pos"])-1, len(sample_list))
        pos_seq = entry["pos"][1:n+1]
        sample_list = sample_list[0:n]
        # print("POS : ",pos_seq)
        # print("TAG : ",sample_list)
        # print(list(zip(pos_seq,sample_list)))
        for i in range(n):
            count_total += 1
            if pos_seq[i]==sample_list[i] or pos_seq[i]=='START' or pos_seq[i]=='END':
                count_correct += 1

    print("Accuracy Score : ",100*count_correct/count_total)

def get_ft_pos_score(input_data):
    nlp=spacy.load("en_core_web_sm")
    count_correct=0
    count_total=0
    for line in input_data:
        pos_tags_control=line.split("||")[0].split()
        if len(line.strip())==0:
            continue
        txt=line.split("||")[1].strip().split('<|endoftext|>')[0]
        # print(txt)
        pos_tags = nlp(txt)
        sample_list = []
        for token in pos_tags:
            sample_list.append(token.pos_)
        n = min(len(pos_tags_control)-1, len(sample_list))
        pos_seq = pos_tags_control[1:n+1]
        sample_list = sample_list[0:n]
        # print(pos_seq)
        # print(sample_list)
        for i in range(n):
            count_total += 1
            if pos_seq[i]==sample_list[i] or pos_seq[i]=='START' or pos_seq[i]=='END':
                count_correct += 1

    print("Accuracy Score : ",100*count_correct/count_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    parser.add_argument('--input_file', type=str, default='generated_output/controlled_generation.txt', help='')
    # parser.add_argument('--only_pos',type=bool,default=False,help='')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1
    eos_tag="<|endoftext|>" # this is for the teacher model
    args = parser.parse_args()
    count_correct = 0
    count_total = 0
    if args.input_file.endswith('.txt'):
        incorrect=[]
        data=read_file_lbyl(args.input_file)
        print("Succesfully read file : ",args.input_file)
        if 'FT_POS' in args.input_file:
            get_ft_pos_score(data)
        elif 'POS' in args.input_file:
            # POS tag control
            get_pos_score(data)
        else:
            # semantic control
            for line in data:
                if '||' not in line :
                    continue

                contrl_pair=line.strip().split("||")[0]
                # print(contrl_pair)
                contrl_pair=contrl_pair.split("|")[0]
                txt=line.strip().split("||")[1]
                count_total+=1
                contrl=contrl_pair.split(":")[0].strip()
                value=contrl_pair.split(":")[1].strip()
                # print(value)
                # print(contrl)
                # print(txt)
                if value in txt:
                    count_correct+=1

            print("Accuracy Score : ",100*count_correct/count_total)

    elif args.input_file.endswith('.jsonl'):
        json_list=read_jsonl(args.input_file)
        print("Succesfully read file : ",args.input_file)
        for json_str in json_list:
            entry = json.loads(json_str)
            for key in entry.keys():
                control = key[key.rfind(':')+4:-1].split(', ')
                for i in range(len(control)):
                    control[i] = control[i][1:-1]
                control_string = ''
                for word in control:
                    control_string += word
                    control_string += ' '
                control_string = control_string[:-1]
                for i in range(len(entry[key])):
                    count_total += 1
                    if control_string in entry[key][i]:
                        count_correct += 1
        
        print("Accuracy Score : ",100*count_correct/count_total)
    else :
        print("Error in reading file : unknown format for ",args.input_file)
    