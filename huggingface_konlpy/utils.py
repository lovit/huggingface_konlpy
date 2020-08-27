from unicodedata import normalize


def compose(tokens):
    return [normalize('NFKC', token) for token in tokens]


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
