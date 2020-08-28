# Example usage

Train bert with `bert_config.json`

```
python examples/run_konlpy_bert.py \
    --output_dir=output \
    --overwrite_output_dir \
    --model_type=bert \
    --config_name=examples/bert_config.json \
    --mlm \
    --vocab_file=tutorials/tokenizers/BertStyleMecab/notag-vocab.txt \
    --do_train \
    --train_data_file=data/2020-07-29_covid_news_sents.txt \
    --konlpy_name=mecab \
    --learning_rate=1e-4 \
    --num_train_epochs=3 \
    --save_total_limit=2 \
    --save_steps=2000 \
    --block_size=512 \
    --seed=42
```
