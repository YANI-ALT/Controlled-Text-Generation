import argparse
import os
import json

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
    parser.add_argument('--output_dir', type=str, default='perplex_score_input/', help='')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1
    eos_tag="<|endoftext|>" # this is for the teacher model
    args = parser.parse_args()
    if args.input_file.endswith('.txt'):
        # fudge output is .txt
        # each line is a sentence with :
        # a line : <control> : <value> || <sentence> \n eos_tag
        print("Reading file ",args.input_file)
        file_data=read_file_lbyl(args.input_file)
        

        output_data=[]
        for line in file_data:
            if "||" not in line :
                continue
            output_data.append(line.strip().split("||")[1].strip()+" "+eos_tag)
        # print(output_data)
        output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1])
        
    elif args.input_file.endswith('.jsonl'):
        print("Reading file ",args.input_file)
        file_data=read_jsonl(args.input_file)

        output_data=[]
        for json_str in file_data:
            entry = json.loads(json_str)
            for key in entry.keys():
                for sent in entry[key]:
                    print(sent.split())
                    new_sent=""
                    for i in sent.split():
                        if i=='END':
                            output_data.append(sent.strip()+" "+eos_tag)
                            sent=""
                            continue
                        elif i=='START':
                            sent=""
                            continue
                        sent=sent+" "+i
        # print(output_data)
        output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1].split(".jsonl")[0]+".txt")

    else:
        print("Error reading file")
    
    with open(output_file, 'w') as f:
        for line in output_data:
            f.write(line)
            f.write('\n')
    print("File written {}".format(output_file))
