

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilroberta-finetuned-bloomberg-classifier

## Model description

DistilRoberta base finetuned on Bloomberg financial articles, more on https://huggingface.co/oli1728/distilroberta-finetuned-bloomberg-classifier

## Intended uses & limitations

Intended for analysis of press releases and financial news relating to market sentiment. Not intended for everyday speech.

## Training and evaluation data

Finetuned on ~12,500 Bloomberg financial articles

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.3714        | 1.0   | 313  | 0.3035          |
| 0.4374        | 2.0   | 626  | 0.2983          |
| 0.2918        | 3.0   | 939  | 0.5378          |
| 0.1115        | 4.0   | 1252 | 0.6534          |
| 0.0002        | 5.0   | 1565 | 0.7559          |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.1+cpu
- Datasets 2.20.0
- Tokenizers 0.19.1
