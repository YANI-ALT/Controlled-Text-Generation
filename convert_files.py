import argparse
import os


def read_file_lbyl(path):
    lines=[]
    with open(path) as f:
        for line in f:
            lines.append(line)
    
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='controlled text generation args.')
    parser.add_argument('--input_file', type=str, default='generated_output/controlled_generation.txt', help='')
    parser.add_argument('--output_dir', type=str, default='perplex_score_input/', help='')
    # COMMAND : python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1

    args = parser.parse_args()
    
    file_data=read_file_lbyl(args.input_file)
    eos_tag="<|endoftext|>"

    output_data=[]
    for line in file_data:
        if "||" not in line :
            continue
        output_data.append(line.strip().split("||")[1].strip()+" "+eos_tag)
    # print(output_data)
    output_file=os.path.join(args.output_dir,"perplexity_input_"+os.path.split(args.input_file)[-1])
    with open(output_file, 'w') as f:
        for line in output_data:
            f.write(line)
            f.write('\n')
    print("File written {}".format(output_file))
