## Diffusion-LM

This repository consists of code from [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM). Clone the repo here and follow the setup instructions.

To obtain perplexity scores we train and run the perplexity model as follows.

Training :

```python
python train_run.py --experiment e2e-tgt --task finetune --dataset_name e2e --app "--e2e_train datasets/e2e_data/"

```

Perplexity Scores :

```python
python3 improved-diffusion/scripts/ppl_under_ar.py --input_text datasets/test_perplexity.txt --model_name_or_path classifier_models/e2e-tgt_e\=5_b\=10_m\=gpt2_wikitext-103-raw-v1_101_finetune_None --mode eval
```