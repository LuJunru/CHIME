# CHIME: Cross-passage Hierarchical Memory Network for Generative Review Question Answering
This repository contains PyTorch implementation of the [corresponding COLING 2020 Paper](https://waitforadding.com).

## Breif Introduction
CHIME is a cross-passage hierarchical memory network for generative question answering (QA). It extends
[XLNet](https://github.com/zihangdai/xlnet) introducing an auxiliary memory module consisting of two components:
the **context memory** collecting cross-passage evidences, and the **answer memory** working as a buffer continually
refining the generated answers. A sample of syntactically well-formed [answers](https://github.com/LuJunru/CHIME/blob/main/evaluation_output/evaluation_samples.txt) show the efficacy of CHIME.

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
- AmazonQA: Raw data can be downloaded from [AmazonQA Project](https://github.com/amazonqa/amazonqa)
- Details of building experimental subsets are included in [build_data.py](https://github.com/LuJunru/CHIME/blob/main/build_data.py). The subsets we used are uploaded [here](https://drive.google.com/drive/folders/1DkS0afPh93cut2gpgSXOWRgyfFXIVLIp?usp=sharing)

## Running
Change task type for model training, testing, predicting, analyzing and evaluating. In particular, analyzing refers to
reveal model's interoperability by listing intermediate answers
```
python3 run_chime.py --devices 0,1 --model xlnet-base-cased --root_path YourDataPath --data_size 1.0 --epochs 3
--batch_size_perGPU 1 --model_output_path YourModelPath --prediction_output_path YourPredictionPath
--evaluation_output_path YourEvaluationPath --rev_num 10 --ans_num 1 --task Train
```

## Licence
MIT

## Citation
TBD
