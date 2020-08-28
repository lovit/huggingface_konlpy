from konlpy.tag import Komoran, Mecab, Okt
from unicodedata import normalize
from .tokenizers_konlpy import KoNLPyWordPieceTokenizer
from .transformers_konlpy import KoNLPyBertTokenizer


KONLPY = {
    'komoran': Komoran,
    'mecab': Mecab,
    'Okt': Okt
}


def compose(tokens):
    return [normalize('NFKC', token) for token in tokens]


def get_tokenizer(vocab_file, konlpy_name, use_tag=False):
    if konlpy_name not in KONLPY:
        raise ValueError(f'Support only {set(KONLPY.keys())}')
    konlpy_bert_tokenizer = KoNLPyBertTokenizer(
        konlpy_wordpiece = KoNLPyWordPieceTokenizer(KONLPY[konlpy_name](), use_tag=use_tag),
        vocab_file = vocab_file
    )
    return konlpy_bert_tokenizer


def prepare_pretokenized_corpus(raw_path, pretokenized_path, pretok):
    """
    Examples::
        >>> from huggingface_konlpy.tokenizers import KoNLPyPreTokenizer
        >>> from konlpy.tag import Komoran

        >>> prepare_pretokenized_corpus(
        >>>     '../data/2020-07-29_covid_news_sents.txt',
        >>>     '../data/2020-07-29_covid_news_sents.komoran.txt',
        >>>     KoNLPyPreTokenizer(Komoran())
        >>> )
        """
    with open(raw_path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    with open(pretokenized_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{pretok(line)}\n')
