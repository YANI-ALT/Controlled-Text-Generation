Command for running the SLURM files

```sbatch -J fudge_classif -o fudge_classif.o%j -e fudge_classif.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/FUDGE_train_classifiers_GPT2.slurm```

```sbatch -J fudge_classif_2 -o fudge_classif_2.o%j -e fudge_classif_2.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/FUDGE_train_classifiers_GPT2_2.slurm```

```sbatch -J fudge_classif_3 -o fudge_classif_3.o%j -e fudge_classif_3.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/FUDGE_train_classifiers_GPT2_3.slurm```

```sbatch -J bert_classif -o bert_classif.o%j -e bert_classif.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/bert_classifier.slurm```

```sbatch -J fudge_tg -o fudge_tg.o%j -e fudge_tg.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/FUDGE_text_generation_GPT2.slurm```

```sbatch -J control_gen -o control_gen.o%j -e control_gen.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/controlled_text_generation.slurm```

```sbatch -J control_gen_pos -o control_gen_pos.o%j -e control_gen_pos.e%j -p development -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/controlled_text_generation.slurm```

```sbatch -J control_gen -o perplex.o%j -e perplex.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/perplexity_training.slurm```

```sbatch -J perplex_gen -o perplex_gen.o%j -e perplex_gen.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=devyani.maladkar@utexas.edu --mail-type=all -A CCR23005 ../SLURM_FILES/generate_perplexity_scores.slurm```

Command for obtaining the idev for testing

```idev -p gpu-a100 -N 1 -n 10 -t 0:10:00```
```idev -p development -N 1 -n 10 -t 0:10:00```

Command for checking queue on tacc

```squeue -u yani28```

Command to copy files 

```scp train_classifier.py yani28@ls6.tacc.utexas.edu:/work/09396/yani28/ls6/NLP_PROJECT/```
### Command for python venv

Create venv

```python python -m venv ./name_of_venv ```

Activate the venv

```source name_of_venv/bin/activate```

Now one can do pip install for the required packages

Command to deactivate (Only this nothing else)
``` deactivate ```

Controlled Generation Command
```python3 controlled_text_generation_v2.py --input_file dataset/e2e_data/target_attribute.json --gen_model_path trained_models_text_generation/gpt2_e2e_5.pt --per_control 1 --use_bert True --debug True --lambda_condition 5```

Generate Perplexity Scores
```python3 improved-diffusion/scripts/ppl_under_ar.py --input_text datasets/test_perplexity.txt --model_name_or_path classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetune_None --mode eval --experiment random --model_path /work/09396/yani28/ls6/Diffusion-LM/classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetune_None/pytorch_model.bin ``` 





