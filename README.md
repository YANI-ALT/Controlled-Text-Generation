# Controlled-Text-Generation
With the recent explosion in the popularity of
large language models, controlling their behav-
ior has become an important problem. The
task of controlled generation, which entails
performing control on the output of a lan-
guage model without heavy retraining, has re-
ceived increased attention from the research
community in recent years. For this replication
study, we look at Diffusion-LM and FUDGE,
two popular methods for handling controlled
generation

## Installation
The environment.yml file installations are required to run the scripts in this folder.

Install using conda
```python
conda env create -f environment.yml
```
For the diffusion-lm folder setup refer to the README in the folder.

## Usage

For Controlled Generation Task, controls are made availble in dataset/control_target. The following naming scheme is followed :

- ```controlled_{task}_generation.py``` : Files for running FUDGE for len, POS and semantic (text) control.

    - Command for Semantic Control Task :
    ```python
    python3 controlled_text_generation.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1 --use_bert True --lambda_condition 7
    ```

    - Command for POS Sequence Control Task :
    ```python
    python3 controlled_POS_generation.py --input_file dataset/e2e_data/src1_valid.txt --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1
    ```

- ```CTG_gpt2_FT_{task}_generate.py``` : Files for running Fine tuned model for POS and semantic control.

    - Command for Semantic Control Task :
    ```python
    python3 CTG_gpt2_FT_Semantic_Control_generate.py --model_path CTG_semantic_control_models/gpt2_e2e_5.pt --num_gen 1 --sample_strat beam --sample_strat_n 5
    ```

    - Command for POS Sequence Control Task :
    ```python
    CTG_gpt2_FT_POS_generate.py --model_path CTG_pos_control_models/gpt2_e2e_5.pt --num_gen 1 --sample_strat beam --sample_strat_n 5
    ```



- ```text_generate_gpt2_generate.py``` : File for generating sentences from base LM of FUDGE.

    - Command : 
    
    ```python
    python text_generation_gpt2_generate.py --num_gen 100 --model_path trained_models_text_generation/
    ```

- ```generated_output``` : all files above will write output to this folder.


## References

## Contributors
This repo contains code done as part of the NLP course project at UT-Austin. The project team consists of : [Anubhav Goel](https://github.com/anubhavgoel26) and [Devyani Maladkar](https://github.com/YANI-ALT).


<!-- !python3 improved-diffusion/scripts/ppl_under_ar.py --input_text perplex_score_input/perplexity_input_BEST_revised_special_bert_controlled_generation_n_lm_10_lambda_8_sample_strat_max.txt --model_name_or_path classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetune_None --mode eval -->