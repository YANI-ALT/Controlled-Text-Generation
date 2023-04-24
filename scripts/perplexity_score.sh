for file in perplex_score_input_POS/*
do
    python3 improved-diffusion/scripts/ppl_under_ar.py --input_text $file --model_name_or_path classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetune_None --mode eval
done


# for file in perplex_score_input/*
# do
#     python3 improved-diffusion/scripts/ppl_under_ar.py --input_text $file --model_name_or_path classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetuneUNK_None --mode eval
# done