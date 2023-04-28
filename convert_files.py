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

def get_tokenizer(model_name_or_path):
    path_save_tokenizer = '{}/vocab.json'.format(model_name_or_path)
    with open(path_save_tokenizer, 'r') as f:
        vocab = json.load(f)
    tokenizer = {v: k for k, v in vocab.items()}
    rev_tokenizer={k:v  for k,v in vocab.items()}
    return tokenizer,rev_tokenizer

def handle_unks(rev_tokenizer,file_data):
    transformed_sent=[]
    eos_tag='<|endoftext|>'
    for line in file_data:
        line=line.split("<|endoftext|>")[0]
        line=line.strip()
        line=line.replace(","," , ")
        line=line.replace("."," . ")
        word_lst=line.split()
        # print(word_lst[-1])
        new_sent_wrd_list=[]
        for wrd in word_lst:
            if wrd=='<|endoftext|>':
                new_sent_wrd_list.append(wrd)
                continue
            if wrd not in rev_tokenizer:
                new_sent_wrd_list.append("UNK")
            else:
                new_sent_wrd_list.append(wrd)
        
        new_sentence=" ".join(new_sent_wrd_list)
        # print("OLD : ",line)
        # print("NEW : ",new_sentence)
        transformed_sent.append(new_sentence+eos_tag)
        print(transformed_sent[-1])
    
    # file_name=file_path.split('/')[-1]
    # file_loc="/".join(file_path.split('/')[:-1])
    # output_file=os.path.join(file_loc,"handleUNK_"+file_name)
    # write_file_lbyl(output_file,transformed_sent)
    return transformed_sent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    parser.add_argument('--input_file', type=str, default='generated_output/controlled_generation.txt', help='')
    parser.add_argument('--output_dir', type=str, default='perplex_score_input/', help='')
    parser.add_argument('--handleUNK',type=bool,default=False,help='')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1
    eos_tag="<|endoftext|>" # this is for the teacher model
    args = parser.parse_args()
    print(args.input_file)
    if "controlled_POS" in args.input_file:
        file_data=read_file_lbyl(args.input_file)
        output_data=[]
        for line in file_data:
            json_parse=json.loads(line)
            # print(json_parse['sentence'])
            output_data.append(json_parse['sentence'])
        # print(output_data)
        # output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1])
        outfile_name="perplexity_input_"+os.path.split(args.input_file)[-1]

    elif args.input_file.endswith('.txt'):
        # fudge output is .txt
        # each line is a sentence with :
        # a line : <control> : <value> || <sentence> \n eos_tag
        print("Reading file ",args.input_file)
        file_data=read_file_lbyl(args.input_file)
        # print("__------------HERE--------------")
        # print(file_data)
        output_data=[]
        for line in file_data:
            if "||" not in line :
                continue
            sent=line.strip().split("||")[1].strip()
            if eos_tag in sent:
                output_data.append(sent)
            else:
                output_data.append(sent+" "+eos_tag)
            
        # print(output_data)
        # output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1])
        outfile_name="perplexity_input_"+os.path.split(args.input_file)[-1]
    elif args.input_file.endswith('.jsonl') and 'pos' in args.input_file:
        print("Reading file ",args.input_file)
        file_data=read_jsonl(args.input_file)

        output_data=[]
        for json_str in file_data:
            entry = json.loads(json_str)
            for key in entry.keys():
                for sent in entry[key]:
                    # print(sent.split())
                    sent=sent.strip()
                    new_sent=""
                    for i in sent.split():
                        if i=='END':
                            print(new_sent)
                            output_data.append(new_sent.strip()+" "+eos_tag)
                            new_sent=""
                            continue
                        elif i=='START':
                            new_sent=""
                            continue
                        new_sent=new_sent+" "+i
        # print(output_data)
        # output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1].split(".jsonl")[0]+".txt")
        outfile_name="perplexity_input_"+os.path.split(args.input_file)[-1].split(".jsonl")[0]+".txt"
    elif args.input_file.endswith('.jsonl'):
        print("Reading file ",args.input_file)
        file_data=read_jsonl(args.input_file)

        output_data=[]
        for json_str in file_data:
            entry = json.loads(json_str)
            for key in entry.keys():
                for sent in entry[key]:
                    # print(sent.split())
                    sent=sent.strip()
                    for i in sent.split():
                        if i=='END':
                            print(new_sent)
                            output_data.append(new_sent.strip()+" "+eos_tag)
                            new_sent=""
                            continue
                        elif i=='START':
                            new_sent=""
                            continue
                        new_sent=new_sent+" "+i
        # print(output_data)
        # output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1].split(".jsonl")[0]+".txt")
        outfile_name="perplexity_input_"+os.path.split(args.input_file)[-1].split(".jsonl")[0]+".txt"
    else:
        print("Error reading file")

    if args.handleUNK==True:
        DIFF_LM_PATH='diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/'
        tokenizer,rev_tokenizer=get_tokenizer(DIFF_LM_PATH)
        output_data=handle_unks(rev_tokenizer,output_data)
        output_file=os.path.join(args.output_dir,"handleUNK_"+outfile_name)
    else :
        output_file=os.path.join(args.output_dir,outfile_name)

    with open(output_file, 'w') as f:
        for line in output_data:
            f.write(line)
            f.write('\n')
    print("File written {}".format(output_file))
