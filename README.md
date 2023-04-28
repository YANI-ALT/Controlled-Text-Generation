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
```
conda env create -f environment.yml
```
For the diffusion-lm folder setup refer to the README in the folder.

## Usage

For Controlled Generation Task, controls are made availble in dataset/control_target. The following naming scheme is followed :

- ```controlled_{task}_generation.py``` : Files for running FUDGE for len, POS and semantic (text) control.
- ```CTG_gpt2_FT_{}_generate.py``` : Files for running Fine tuned model for POS and semantic control.
- ```text_generate_gpt2_generate.py``` : File for generating sentences from base LM of FUDGE.



## References

## Contributors
This repo contains code done as part of the NLP course project at UT-Austin. The project team consists of : [Anubhav Goel](https://github.com/anubhavgoel26) and [Devyani Maladkar](https://github.com/YANI-ALT).


