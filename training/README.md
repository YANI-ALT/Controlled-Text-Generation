## Training

This folder contains all the code required for training the various models in the repo.

The following is the description of the files :
- ```training/CTG_gpt2_FT_POS.py``` : Fine-tunes a gpt2-small model for conditional text generation for the POS control task.

- ```training/CTG_gpt2_FT_Semantic_Control.py``` : Fine-tunes a gpt2-small model for conditional text generation for the Semantic Control control task.

- ```training/text_generation_gpt2.py``` : Fine tune a gpt2-small on the E2E dataset for text generation required as base LM in FUDGE.

- ```training/train_classifier_bert.py``` : Train distilled-bert classifiers as future discriminators for various attributes in FUDGE semantic control.

- ```training/train_classifier.py``` : Train gpt2 classifiers as future discriminators for various attributes in FUDGE semantic control.