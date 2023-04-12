for file in generated_output/*
do
    python3 get_accuracy_score.py --input_file $file
done