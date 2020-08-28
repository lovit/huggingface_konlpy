import os
from collections import Counter
from tokenizers import BertWordPieceTokenizer
from tokenizers import AddedToken, InputSequence, Encoding, EncodeInput
from tqdm import tqdm
from typing import Optional, List, Union

from .pretokenizers import KoNLPyWordPieceTokenizer


class KoNLPyPretokBertWordPieceTokenizer(BertWordPieceTokenizer):
    """
    Examples::
        >>> from konlpy.tag import Komoran
        >>> from transformers import BertTokenizer

        >>> konlpy_pretok = KoNLPyPreTokenizer(Komoran())
        >>> konlpy_bert_wordpiece_tokenizer = KoNLPyPretokBertWordPieceTokenizer(
        >>>     konlpy_pretok = konlpy_pretok,
        >>>     vocab_file = 'path/to/pretrained/vocab.txt')

        >>> sent_ko = '신종 코로나바이러스 감염증(코로나19) 사태가 심각합니다'
        >>> konlpy_bert_wordpiece_tokenizer.encode(sent_ko, add_special_tokens=False).tokens)
        $ ['신종', '코로나바이러스', '감염증', '(', '코로나', '19', ')', '사태', '가', '심각', '하', 'ᄇ니다']

        >>> bert_wordpiece_tokenizer = BertWordPieceTokenizer(
        >>>     vocab_file = 'path/to/pretrained/vocab.txt')
        >>> bert_wordpiece_tokenizer.encode(sent_ko).tokens
        $ ['신종', '코로나바이러스', '감염증', '(', '코로나', '##1', '##9', ')', '사태', '##가', '심각', '##합', '##니다']
    """
    def __init__(
        self,
        konlpy_pretok,
        vocab_file: Optional[str] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):
        super().__init__(
            vocab_file,
            unk_token,
            sep_token,
            cls_token,
            pad_token,
            mask_token,
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
            wordpieces_prefix
        )
        self.konlpy_pretok = konlpy_pretok

    def encode(
        self,
        sequence: InputSequence,
        pair: Optional[InputSequence] = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """ Encode the given sequence and pair. This method can process raw text sequences as well
        as already pre-tokenized sequences.
        Args:
            sequence: InputSequence:
                The sequence we want to encode. This sequence can be either raw text or
                pre-tokenized, according to the `is_pretokenized` argument:
                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`
            is_pretokenized: bool:
                Whether the input is already pre-tokenized.
            add_special_tokens: bool:
                Whether to add the special tokens while encoding.
        Returns:
            An Encoding
        """
        if sequence is None:
            raise ValueError("encode: `sequence` can't be `None`")

        sequence = self.konlpy_pretok(sequence)
        return self._tokenizer.encode(sequence, pair, is_pretokenized, add_special_tokens)

    def encode_batch(
        self,
        inputs: List[EncodeInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List[Encoding]:
        """ Encode the given inputs. This method accept both raw text sequences as well as already
        pre-tokenized sequences.
        Args:
            inputs: List[EncodeInput]:
                A list of single sequences or pair sequences to encode. Each `EncodeInput` is
                expected to be of the following form:
                    `Union[InputSequence, Tuple[InputSequence, InputSequence]]`
                Each `InputSequence` can either be raw text or pre-tokenized,
                according to the `is_pretokenized` argument:
                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`
            is_pretokenized: bool:
                Whether the input is already pre-tokenized.
            add_special_tokens: bool:
                Whether to add the special tokens while encoding.
        Returns:
            A list of Encoding
        """

        if inputs is None:
            raise ValueError("encode_batch: `inputs` can't be `None`")

        input_iterator = tqdm(inputs, desc='konlpy pretok', total=len(inputs))
        konlpy_pretok_inputs = [self.konlpy_pretok(sequence) for sequence in input_iterator]
        return self._tokenizer.encode_batch(konlpy_pretok_inputs, is_pretokenized, add_special_tokens)


class KoNLPyBertWordPieceTrainer:
    def __init__(self, konlpy_tagger, wordpieces_prefix="##", use_tag=False):
        konlpy_wordpiece = KoNLPyWordPieceTokenizer(
            konlpy_tagger,
            wordpieces_prefix,
            use_tag
        )
        self.tokenizer = konlpy_wordpiece

    def tokenize(self, sent):
        split_tokens = []
        for eojeol in sent.split():
            split_tokens += self.tokenizer.tokenize(eojeol)
        return split_tokens

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress: bool = True,
    ):
        if isinstance(files, str):
            files = [files]
        alphabets = initialize_alphabet(files, limit_alphabet, initial_alphabet, special_tokens, show_progress)
        self.vocab = train_vocab(files, vocab_size, min_frequency, show_progress, alphabets, self.tokenizer.tokenize)

    def save_model(self, directory: str, name: Optional[str] = None):
        if name is None:
            name = 'vocab.txt'
        else:
            name = f'{name}-vocab.txt'
        directory = os.path.abspath(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        vocab_file = f'{directory}/{name}'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for subword in self.vocab:
                f.write(f'{subword}\n')
        print(f'[{vocab_file}]')


def initialize_alphabet(files, limit_alphabet, initial_alphabet, special_tokens, show_progress):
    counter = Counter()
    n_files = len(files)
    for i_file, file in enumerate(files):
        with open(file, encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        if show_progress:
            iterator = tqdm(lines, desc=f'Initialize alphabet {i_file + 1}/{n_files}', total=len(lines))
        else:
            iterator = lines
        counter.update(Counter(char for line in iterator for char in line if char))
    del counter[' ']
    frequent_alphabets = sorted(counter, key=lambda x: -counter[x])
    frequent_alphabets = [alphabet for alphabet in frequent_alphabets if alphabet not in initial_alphabet]
    alphabets = special_tokens + frequent_alphabets
    alphabets = alphabets[:limit_alphabet]
    return alphabets


def train_vocab(files, vocab_size, min_frequency, show_progress, alphabets, tokenize):
    counter = Counter()
    n_files = len(files)
    for i_file, file in enumerate(files):
        with open(file, encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        if show_progress:
            iterator = tqdm(lines, desc=f'Train vocab {i_file + 1}/{n_files}', total=len(lines))
        else:
            iterator = lines
        counter.update(Counter(sub for line in iterator for sub in tokenize(line)))
    frequent_vocab = sorted(counter, key=lambda x: -counter[x])
    frequent_vocab = [vocab for vocab in frequent_vocab
                      if (vocab not in alphabets) and (counter[vocab] >= min_frequency)]
    vocab = alphabets + frequent_vocab
    vocab = vocab[:vocab_size]
    return vocab
