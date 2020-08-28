from transformers import BertTokenizer


class KoNLPyPretokBertTokenizer(BertTokenizer):
    """
    Examples::

        >>> from huggingface_konlpy.tokenizers import KoNLPyPreTokenizer
        >>> from huggingface_konlpy.transformers import KoNLPyPretokBertTokenizer
        >>> from konlpy.tag import Komoran, Mecab, Okt

        >>> konlpy_pretok = KoNLPyPreTokenizer(Komoran())
        >>> konlpy_bert_tokenizer = KoNLPyPretokBertTokenizer(
        >>>    konlpy_pretok = konlpy_pretok,
        >>>    vocab_file = 'path/to/pretrained/vocab.txt')

        >>> sent_ko = '신종 코로나바이러스 감염증(코로나19) 사태가 심각합니다'
        >>> konlpy_bert_tokenizer.tokenize(sent_ko)
        $ ['신종', '코로나바이러스', '감염증', '(', '코로나', '19', ')', '사태', '가', '심각', '하', 'ᄇ니다']
    """
    def __init__(
        self,
        konlpy_pretok,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents
        )
        self.konlpy_pretok = konlpy_pretok

    def _tokenize(self, text):
        text = self.konlpy_pretok(text)
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens


class KoNLPyBertTokenizer(BertTokenizer):
    """
    Examples::
        Train vocabulary using mecab as wordpiece tokenizer

            >>> from huggingface_konlpy.tokenizers_konlpy import KoNLPyBertWordPieceTokenizer
            >>> from konlpy.tag import Mecab

            >>> mecab_wordpiece_tokenizer = KoNLPyBertWordPieceTokenizer(Mecab(), use_tag=True)
            >>> mecab_wordpiece_tokenizer.train(
            >>>     files = ['../data/2020-07-29_covid_news_sents.txt']
            >>> )
            >>> mecab_wordpiece_tokenizer.save_model('./tokenizers/BertStyleMecab/', 'usetag')

        Load pretrained "mecab + bert tokenizer"

            >>> from huggingface_konlpy.transformers_konlpy import KoNLPyBertTokenizer

            >>> mecab_bert_tokenizer = KoNLPyBertTokenizer(
            >>>     konlpy_wordpiece = mecab_wordpiece_tokenizer,
            >>>     vocab_file = './tokenizers/BertStyleMecab/usetag-vocab.txt'
            >>> )
            >>> sent = '신종 코로나바이러스 감염증(코로나19) 사태가 심각합니다'
            >>> mecab_bert_tokenizer.tokenize(sent)
            $ ['신종/NNG', '코로나/NNP', '##바이러스/NNG', '감염증/NNG', '##(/SSO', '##코로나/NNP',
               '##19/SN', '##)/SSC', '사태/NNG', '##가/JKS', '심각/XR', '합', '니', '다']
    """
    def __init__(
        self,
        konlpy_wordpiece,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents
        )
        self.konlpy_wordpiece = konlpy_wordpiece

    def _tokenize(self, text):
        base_tokens = self.konlpy_wordpiece.tokenize(text)
        split_tokens = []
        for token in base_tokens:
            if token in self.vocab:
                split_tokens.append(token)
            else:
                split_tokens += self.konlpy_wordpiece.token_to_alphabets(token)
        return split_tokens
