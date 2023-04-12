import json
import os
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    parser.add_argument('--input_file', type=str, default='generated_output/controlled_generation.txt', help='')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1
    eos_tag="<|endoftext|>" # this is for the teacher model
    args = parser.parse_args()

    count_correct = 0
    count_total = 0
    if args.input_file.endswith('.txt'):
        incorrect=[]
        data=read_file_lbyl(args.input_file)
        print("Succesfully read file : ",args.input_file)
        for line in data:
            if '||' not in line :
                continue

            contrl_pair=line.strip().split("||")[0]
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
    