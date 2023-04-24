for file in generated_output/controlled_POS_*
do  
    echo $file
    python convert_files.py --input_file $file --handleUNK True
    python convert_files.py --input_file $file 
done