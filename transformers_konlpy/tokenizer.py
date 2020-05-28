import konlpy
import os
import transformers

from collections import defaultdict
from konlpy.tag import Hannanum, Kkma, Komoran, Mecab, Okt
from tokenizers import CharBPETokenizer, SentencePieceBPETokenizer
from transformers import BertTokenizer
from tqdm import tqdm


KONLPY_TAGGERS = {
    'hannanum': Hannanum,
    'kkma': Kkma,
    'komoran': Komoran,
    'okt': Okt,
    'mecab': Mecab
}

SPECIAL_TOKENS = [
    '[BOS]',
    '[EOS]',
    '[PAD]',
    '[MASK]',
    '[CLS]',
    '[SEP]',
    '[UNK]'
]


class KoNLPyTokenizer(BertTokenizer):
    def __init__(self, vocab_file, konlpy_name, unk_token="[UNK]", sep_token="[SEP]",
                 pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"):

        super().__init__(vocab_file, unk_token=unk_token, sep_token=sep_token,
                         pad_token=pad_token, cls_token=cls_token,
                         mask_toekn=mask_token, do_basic_tokenize=False)

        self.konlpy_name = konlpy_name
        self.konlpy_tagger = KONLPY_TAGGERS[konlpy_name]()

    def _tokenize(self, text):
        split_tokens = self.konlpy_tagger.pos(text, join=True)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError('KoNLPyTokenizer does not provide this function')

    def save_vocabulary(self, directory, suffix=''):
        directory = os.path.abspath(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if suffix:
            suffix = '_' + suffix
        vocab_file = f'{directory}/{self.kolnpy_name}{suffix}.vocab'
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


def get_tokenizer(vocab_file, tokenizer_name=None):

    raise NotImplementedError


def check_special_tokens(special_tokens):
    if isinstance(special_tokens, str):
        return [special_tokens]
    if special_tokens is None:
        return SPECIAL_TOKENS
    return special_tokens


def initialize_konlpy_tagger(tokenizer_name):
    if tokenizer_name in KONLPY_TAGGERS:
        return KONLPY_TAGGERS[tokenizer_name]()
    raise ValueError(f'Only available {KONLPY_TAGGERS.keys()}')


def save_vocabs(vocab_file, vocabs):
    vocab_file = os.path.abspath(vocab_file)
    dirpath = os.path.dirname(vocab_file)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for vocab in vocabs:
            f.write(f'{vocab}\n')


def load_text(path):
    with open(path, encoding='utf-8') as f:
        text = [line.strip() for line in f.readlines()]
    text = [line for line in text if line]
    return text


def train_konlpy_vocab(tokenizer_name, file_path, num_vocabs, min_frequency=1, special_tokens=None, vocab_file=None):
    special_tokens = check_special_tokens(special_tokens)
    tagger = initialize_konlpy_tagger(tokenizer_name)
    vocab_counter = defaultdict(int)
    text = load_text(file_path)
    for line in tqdm(text, ascii=True, desc='Scan KoNLPy vocabs'):
        for vocab in tagger.pos(line, join=True):
            vocab_counter[vocab] += 1
    if min_frequency > 1:
        vocab_counter = {vocab:count for vocab, count in vocab_counter.items() if count >= min_frequency}
    num_special_tokens = len(special_tokens)
    vocabs = special_tokens + sorted(vocab_counter, key=lambda v:-vocab_counter[v])[:num_vocabs - num_special_tokens]
    if isinstance(vocab_file, str):
        save_vocabs(vocab_file, vocabs)
    return vocabs


def initialize_tokenizers_tokenizer():
    return SentencePieceBPETokenizer(unk_token='[UNK]')


def train_tokenizers_vocab(file_path, num_vocabs, min_frequency=1, special_tokens=None):
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS
    tokenizer = initialize_tokenizers_tokenizer()
    tokenizer.train([file_path], vocab_size=num_vocabs, min_frequency=min_frequency, special_tokens=special_tokens)
    vocabs = tokenizer.get_vocab()
    vocabs = sorted(vocabs, key=lambda v:vocabs[v])

    def replace_prefix(vocab):
        if vocab in special_tokens:
            return vocab
        if vocab[0] == '▁':
            return vocab[1:]
        return f'##{vocab}'

    vocabs = [replace_prefix(v) for v in vocabs]
    if isinstance(vocab_file, str):
        save_vocabs(vocab_file, vocabs)
    return vocabs
