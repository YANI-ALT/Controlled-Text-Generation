## Training

This folder contains all the code required for training the various models in the repo.

The following is the description of the files :
- ```training/CTG_gpt2_FT_POS.py``` : Fine-tunes a gpt2-small model for conditional text generation for the POS control task.

```python
python3 CTG_gpt2_FT_POS.py --dataset_path ../dataset/e2e_data/ --mode train --model_name_or_path gpt2 --class_label near --epochs 5
```

- ```training/CTG_gpt2_FT_Semantic_Control.py``` : Fine-tunes a gpt2-small model for conditional text generation for the Semantic Control control task.

```python
python3 CTG_gpt2_FT_Semantic_Control.py --dataset_path ../dataset/e2e_data/ --mode train --model_name_or_path gpt2 --class_label near --epochs 5
```

- ```training/text_generation_gpt2.py``` : Fine tune a gpt2-small on the E2E dataset for text generation required as base LM in FUDGE.

```python
python3 text_generation_gpt2.py --dataset_path ../dataset/e2e_data/ --mode train --model_name_or_path gpt2 --class_label near --epochs 5
```

- ```training/train_classifier_bert.py``` : Train distilled-bert classifiers as future discriminators for various attributes in FUDGE semantic control.

```python
python3 train_classifier_bert.py --mode train --model_name_or_path gpt2 --class_label near --epochs 5
```

- ```training/train_classifier.py``` : Train gpt2 classifiers as future discriminators for various attributes in FUDGE semantic control.
```python
python3 train_classifier.py --mode train --model_name_or_path gpt2 --class_label near --epochs 5
```