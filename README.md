# CHIME: Cross-passage Hierarchical Memory Network for Generative Review Question Answering
This repository contains PyTorch implementation of the [corresponding COLING 2020 Paper](https://waitforadding.com).

## Breif Introduction
CHIME is a cross-passage hierarchical memory network for generative question answering (QA). It extends
[XLNet](https://github.com/zihangdai/xlnet) introducing an auxiliary memory module consisting of two components:
the **context memory** collecting cross-passage evidences, and the **answer memory** working as a buffer continually
refining the generated answers.

The following syntactically well-formed answers show the efficacy of CHIME.
- *Question1: can this chair be operated with battery only?*
- *yes, it can be operated by battery, but it is not recommended to use this chair with batteries only*

## Dependency
```
python 3.7.7
apex 0.1
bert-score 0.3.4
BLEURT 0.0.1
nltk 3.4.5
rouge 1.0.0
torch 1.4.0
torchtext 0.5.0
transformers 2.8.0
```

## Data
- AmazonQA

## Running
Change task type for model training, testing, predicting, analyzing and evaluating. In particular, analyzing refers to
reveal model's interoperability by listing intermediate answers
```
python3 run_chime.py --devices 0,1 --model xlnet-base-cased --root_path YourDataPath --data_size 1.0 --epochs 5
--batch_size_perGPU 1 --model_output_path YourModelPath --prediction_output_path YourPredictionPath
--evaluation_output_path YourEvaluationPath --rev_num 10 --ans_num 1 --task Train
```

## Licence
MIT

## Citation
TBD
