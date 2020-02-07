# BiDAF
Re-implementation of [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) (ICLR 2017)

## Requirements
- python=3.7.0
- pytorch=1.1.0
- torchtext
- tensorboardX
- tqdm
- nltk

## Train model
- Squad1.1 where every question has its answer. 

```
MODEL_NAME="model"
python main.py --model_name ${MODEL_NAME}\
    --GPU 0\
    --train_batch_size 60\
    --train_file train-v1.1.json\
    --dev_batch_size 100\
    --dev_file dev-v1.1.json
```

## Train model with unanswerable questions (TBD)
- Squad 2.0 where some questions don't have their answers within the passages. 
```
MODEL_NAME="model"
python main.py --model_name ${MODEL_NAME}\
    --GPU 0\
    --train_batch_size 60\
    --train_file train-v2.0.json\
    --dev_batch_size 100\
    --dev_file dev-v2.0.json\
    --no_answer True
```

## Evaluatation
| checkout | dataset       | EM | F1 |
|----------|---------------|----|----|
|          | SQuAD1.1      | ?  | ?  |
|          | SQuAD2.0 (no_answer False)      | ?  | ?  |
|          | SQuAD2.0      | ?  | ?  |