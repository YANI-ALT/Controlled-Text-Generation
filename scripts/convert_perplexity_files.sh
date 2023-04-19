for file in generated_output/bert_controlled_gen*
do
    python convert_files.py --input_file $file --handleUNK True
    python convert_files.py --input_file $file 
done