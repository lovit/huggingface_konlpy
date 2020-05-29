import os
import pickle
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from transformers_konlpy import get_tokenizer, train_konlpy_vocab


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which tokenizer we are train
    """

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "KoNLPy tokenizer name. Available ['komoran', 'kkma', 'hannanum', 'okt', 'mecab']"}
    )
    vocab_directory: Optional[str] = field(
        default=None, metadata={"help": "/path/to/tokenizer_name.vocab"}
    )
    num_vocabs: Optional[int] = field(
        default=-1, metadata={"help": "Maximal number of vocabulary"}
    )
    min_frequency: Optional[int] = field(
        default=-1, metadata={"help": "Minimum occurrence in `train_data_file`"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what vocabulary we are going to input our model for training.
    """ 

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    prepare_dataset: bool = field(
        default=False, metadata={"help": "Transform text file to DataSet consisting list of tensor"}
    )


def prepare_dataset(tokenizer, file_path, block_size):
    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
    )
    examples = []
    with open(file_path, encoding="utf-8") as f:
        text = f.readlines()

    tokenized_text = []
    for line in tqdm(text, ascii=True, desc='Preparing dataset'):
        tokenized_text += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        examples.append(
            tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
        )
    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
    # If your dataset is small, first you should loook for a bigger one :-) and second you
    # can change this behavior by adding (model specific) padding.

    with open(cached_features_file, "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(data_args.train_data_file):
        raise ValueError('Not found data file. Check `train_data_file`.')

    if model_args.num_vocabs < 0 and model_args.min_frequency < 0:
        raise ValueError('One of `num_vocabs` or `min_frequency` must be positive integer')

    if model_args.vocab_directory is None:
        raise ValueError('Set `vocab_directory`')
    if model_args.tokenizer_name is None:
        raise ValueError("Set `tokenizer_name`. Available ['komoran', 'kkma', 'hannanum', 'okt', 'mecab']")
    vocab_file = f'{model_args.vocab_directory}/{model_args.tokenizer_name}.vocab'

    train_konlpy_vocab(
        model_args.tokenizer_name,
        data_args.train_data_file,
        model_args.num_vocabs,
        model_args.min_frequency,
        None,  # TODO: available to customize special_tokens
        vocab_file
    )

    if data_args.prepare_dataset and data_args.line_by_line:
        tokenizer = get_tokenizer(vocab_file, model_args.tokenizer_name)
        prepare_dataset(tokenizer, data_args.train_data_file, data_args.block_size)

    # TODO: prepare dataset when `line_by_line=False`

if __name__ == '__main__':
    main()
