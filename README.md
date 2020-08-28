# Hugging Face + KoNLPy

- `huggingface.tokenizers` + `KoNLPy` for training tokenizers
- `huggingface.transformers` + `KoNLPy` for training language models with pre-trained tokenizers using KoNLPy as base tokenizer

## Examples

BERT 학습 예시는 [`examples/`](https://github.com/lovit/huggingface_konlpy/tree/master/examples) 에 있습니다.

### From training tokenizer and load transformers.BertTokenizer (+ KoNLPy) with tag

어절 단위로 KoNLPy 모델의 형태소 분석을 적용하여 이를 Word Piece Tokenizer 로 이용합니다. 형태소 분석 결과의 품사 정보를 선택적으로 이용할 수 있습니다.

```python
from huggingface_konlpy.tokenizers_konlpy import KoNLPyBertWordPieceTrainer
from huggingface_konlpy.transformers_konlpy import KoNLPyBertTokenizer

mecab_wordpiece_usetag_trainer = KoNLPyBertWordPieceTrainer(
    Mecab(), use_tag=True)
mecab_wordpiece_usetag_trainer.train(
    files = ['../data/2020-07-29_covid_news_sents.txt'])
mecab_wordpiece_usetag_trainer.save_model('./tokenizers/BertStyleMecab/', 'usetag')

mecab_bert_usetag = KoNLPyBertTokenizer(
    konlpy_wordpiece = KoNLPyWordPieceTokenizer(Mecab(), use_tag=True),
    vocab_file = './tokenizers/BertStyleMecab/usetag-vocab.txt'
)
print(mecab_bert_usetag.tokenize(sent_ko))
```

```
Initialize alphabet 1/1: 100%|██████████| 70964/70964 [00:00<00:00, 87827.53it/s]
Train vocab 1/1: 100%|██████████| 70964/70964 [00:14<00:00, 4924.31it/s]

['신종/NNG', '코로나/NNP', '##바이러스/NNG', '감염증/NNG', '##(/SSO', '##코로나/NNP', '##19/SN', '##)/SSC', '사태/NNG', '##가/JKS', '심각/XR', '합', '니', '다']
```

### From training tokenizer and load transformers.BertTokenizer (+ KoNLPy) without tag

```python
mecab_wordpiece_notag_trainer = KoNLPyBertWordPieceTrainer(
    Mecab(), use_tag=False)
mecab_wordpiece_notag_trainer.train(
    files = ['../data/2020-07-29_covid_news_sents.txt'])
mecab_wordpiece_notag_trainer.save_model('./tokenizers/BertStyleMecab/', 'notag')

mecab_bert_notag = KoNLPyBertTokenizer(
    konlpy_wordpiece = KoNLPyWordPieceTokenizer(Mecab(), use_tag=False),
    vocab_file = './tokenizers/BertStyleMecab/notag-vocab.txt'
)
print(mecab_bert_notag.tokenize(sent_ko))
```

```
Initialize alphabet 1/1: 100%|██████████| 70964/70964 [00:00<00:00, 75675.21it/s]
Train vocab 1/1: 100%|██████████| 70964/70964 [00:14<00:00, 4888.77it/s]

['신종', '코로나', '##바이러스', '감염증', '##(', '##코로나', '##19', '##)', '사태', '##가', '심각', '##합니다']
```

### Use KoNLPy as pre-tokenizer

입력 문장을 KoNLPy 의 형태소열로 분해하는 전처리 모듈로 `KoNLPyPreTokenizer` 를 이용할 수 있습니다.

```python
from huggingface_konlpy.tokenizers_konlpy import KoNLPyPreTokenizer
from konlpy.tag import Komoran

sent_ko = '신종 코로나바이러스 감염증(코로나19) 사태가 심각합니다'
komoran_pretok = KoNLPyPreTokenizer(Komoran())
print(komoran_pretok(sent_ko))
```

```
신종 코로나바이러스 감염증 ( 코로나 19 ) 사태 가 심각 하 ㅂ니다
```

입력 문장을 KoNLPy 의 형태소열로 분해한 뒤 WordPieceTokenizer 를 적용할 수 있습니다. 이를 위해 vocabulary 을 합니다.

```python
from huggingface_konlpy.tokenizers_konlpy import KoNLPyPretokBertWordPieceTokenizer
from huggingface_konlpy.transformers_konlpy import KoNLPyPretokBertTokenizer


komoran_bertwordpiece_tokenizer = KoNLPyPretokBertWordPieceTokenizer(
    konlpy_pretok = komoran_pretok)

komoran_bertwordpiece_tokenizer.train(
    files = ['../data/2020-07-29_covid_news_sents.txt'],
    vocab_size = 3000)
komoran_bertwordpiece_tokenizer.save_model(
    directory='./tokenizers/KomoranBertWordPieceTokenizer/',
    name='covid')
```

기학습된 (pre-trained) KoNLPy (pretok) + transformers.BertTokenizer 를 사용합니다.

```python
from huggingface_konlpy import compose
from huggingface_konlpy.transformers_konlpy import KoNLPyPretokBertTokenizer

komoran_pretok_berttokenizer = KoNLPyPretokBertTokenizer(
    konlpy_pretok = komoran_pretok,
    vocab_file = './tokenizers/KomoranBertWordPieceTokenizer/covid-vocab.txt')

indices = komoran_pretok_berttokenizer.encode(sent_ko)
tokens = [komoran_pretok_berttokenizer.ids_to_tokens[ids] for ids in indices]
print(' '.join(compose(tokens)))
```

```
[CLS] 신종 코로나바이러스 감염증 ( 코로나 19 ) 사태 가 심 ##각 하 [UNK] [SEP]
```

## References
- [KoNLPy][konlpy]: Python package for natural language processing of the Korean language.
- [tokenizers][tokenizers]
- [transformers][transformers]

[konlpy]: https://konlpy.org/en/latest/
[tokenizers]: https://github.com/huggingface/tokenizers
[transformers]: https://github.com/huggingface/transformers
