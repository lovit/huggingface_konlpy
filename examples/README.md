# Example usage

Train new tokenizer

```
python examples/prepare_tokenizer_and_train_data.py \
    --train_tokenizer \
    --train_data_file tests/data/sample_corpus.txt \
    --tokenizer_name komoran \
    --num_vocabs 2000 \
    --min_frequency 1 \
    --vocab_directory ./tmp/models/
```

Prepare TextDataset

```
python examples/prepare_tokenizer_and_train_data.py \
    --train_data_file tests/data/sample_corpus.txt \
    --tokenizer_name komoran \
    --vocab_directory ./tmp/models/ \
    --prepare_dataset \
    --line_by_line \
    --block_size 512
```

Train new language model with trained KoNLPy tokenizer

```
python examples/run_language_model.py \
    --output_dir ./tmp/models/ \
    --overwrite_output_dir \
    --overwrite_cache \
    --config_name ./tmp/test_bert.json \
    --vocab_file ./tmp/models/bert/komoran.vocab \
    --mlm \
    --do_train \
    --train_data_file ./tests/data/sample_corpus.txt \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --seed 42
```

Or replace `--config_name` with `--model_type` like below

```
python examples/run_language_model.py \
    --model_type gpt2
    ...
```

Then, trainer uses default configuration of GPT2 such as num layers and adapts only num vocabs to equal the number of lines in `vocab_file`

