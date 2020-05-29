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

