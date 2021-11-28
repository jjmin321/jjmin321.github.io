---
title:  "Sentencepiece 사용하기"
excerpt: "서브워드 분절에서 사용하는 sentencepiece에 대해 알아보자."
toc: true
toc_sticky: true

categories:
  - NLP
tags:
  - BPE
  - sentencepiece
  - tutorial
last_modified_at: 2020-07-14
---

sentencepiece를 공부해보면 WPM, BPE, Unigram LM과 같은 것들이 튀어나와 헷갈린다.
특히, WPM, BPE 논문은 서로 바꿔 달아놓기도 하고 (어디서는 Senrich를 BPE라 하고, 어디서는 WPM이라 한다)
서로 차이점이 없어보이기도 한다.
한번 자세히 살펴보자.  

## Introduction

Subword model은 NLM에서 처음 시도된 것으로, [Luong et al., (2013)](https://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf)에서 OOV 문제를 해결하면서 시작되었다. OOV 문제를 해결하기 위해 복합어를 2-3개의 token으로 분리해야 한다고 주장하였다. 단어의 빈도수가 1-5개인 데이터는 junk word로 제외하고, 나머지는 (5, 10], (10, 100], ... 등으로 분리하여 tokenize하는 방식을 제안한 것이다.

![image](https://user-images.githubusercontent.com/47516855/87418367-887caa00-c60c-11ea-8440-085f5b05e7cc.png)

이렇게 빈도수로 나누었을 때 표와 같이 prefix, suffix가 공통적으로 나타나는 걸 확인할 수 있다. 이 중 빈도수가 낮은 단어들은 NLM이나 NMT에서 성능 저하의 원인이 되지만, *Obtainment*나 *acquirement*의 경우 일상생활에서 흔히 사용하기 때문에 함부로 지울 수 없다.

## Byte Pair Encoding (BPE)

데이터 압축 기법을 의미한다. 예를 들면 다음과 같은 데이터가 있을 때,

```
aaabdaaabac
```

byte pair 'aa'는 가장 자주 발생한다. 따라서 이를 이전에 사용하지 않던 byte "Z"로 바꿔준다.

```
ZabdZabac
Z=aa
```

그 후, "ab"가 그 다음 자주 발생하므로, 이를 바꿔준다.

```
ZYdZYac
Y=ab
Z=aa
```

이제는 pair의 횟수가 1이므로, 여기서 멈출 수도 있고, recursive하게 적용하여 "ZY"를 "X"로 표현할 수도 있다.

```
XdXac
X=ZY
Y=ab
Z=aa
```


## subword-nmt

[Sennrich et al. (2015)](https://arxiv.org/pdf/1508.07909.pdf)에선 Byte Pair Encoding (BPE) 알고리즘을 이용하여 subword dictionary를 만드는 방법을 제안했다.
[Radfor et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)에선 이를 이용하여 GPT-2의 subword vector를 구축하기도 하였다.

### Algorithm

1. Corpus를 준비
2. 적절한 vocab size를 지정
3. 단어를 character의 sequence로 분리하고, \</W>와 word frequency를 붙여준다. 따라서 이 단계에서의 기본 단위는 character이다. 예를 들어 "low"의 빈도수가 5라면, 이를 “l o w \</w>”: 5로 바꿔준다.
4. 높은 빈도수에 따라 새로운 subword를 생성한다.
5. step2에서 정한 vocab size에 도달하거나 다음으로 제일 높은 빈도수가 1이 될 때까지 step4를 반복한다.

### Python code

코드로 살펴보면 다음과 같다

<script src="https://gist.github.com/InhyeokYoo/ee6d03bb23a1bbdebabcc9a23c370d75.js"></script>

### Example

“low: 5”, “lower: 2”, “newest: 6” and “widest: 3”를 예시로 들자. 가장 높은 frequency는 `e`와 `s` 쌍이다. 
이는 `newest`로부터 6, `widest`로부터 3이 카운트 되어 총 9개를 갖는다. 이러면 새로운 subword `es`가 생성되고, 이는 다음 iteration의 후보가 된다.

다음은 위 코드의 결과물이다. subword는 `es -> est -> est\</w> -> lo -> low -> ne -> new -> newest -> low\</w> -> wi` 순이다.

```
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
{'lo w </w>': 5, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
{'low </w>': 5, 'low e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
{'low </w>': 5, 'low e r </w>': 2, 'ne w est</w>': 6, 'w i d est</w>': 3}
{'low </w>': 5, 'low e r </w>': 2, 'new est</w>': 6, 'w i d est</w>': 3}
{'low </w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
{'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
```

## WordPiece Model (WPM)

WordPiece는 또 다른 word segmentation algorithm으로, subword-nmt와 흡사하다. [Schuster and Nakajima, (2012)](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf)가 공개한 논문 'Japanese and Korean voice search'에서 처음 제안 되었다. subword-nmt와 비슷하지만 다른 점은 새로운 subword를 구성할 때 빈도수로 결정하는 것이 아니라 **likelihood**로 계산하는 것이다.

### Algorithm

1. Corpus를 준비한다.
2. Vocab size를 설정한다.
3. word를 character로 split한다.
4. step 3의 데이터를 토대로 language model을 준비한다.
5. step 4의 language model에 추가했을 때, corpus에 대해 가장 likelihood를 높게 만드는 새로운 word unit을 선택한다.
6. Step 5 vocab size에 도달하거나, likelihood가 특정 threshold를 만족할 때 까지 반복한다.

> 여기서 드는 의문은 step 4에서 준비한 LM에 따라 본 알고리즘의 성능이 크게 좌지우지 되지 않을까 하는 점이다. Implementation이 공개되지 않는 이상 알 방법은 없다.

## Unigram Language Model

[Kudo, (2018)](https://arxiv.org/pdf/1804.10959.pdf)에서 제안된 방법이다. 
BPE로 진행할 경우, 같은 vocabulary라도 한 문장은 여러 subword로 분리될 수 있다.

![image](https://user-images.githubusercontent.com/47516855/87444557-e7084f00-c631-11ea-867e-d852d49789e5.png)

따라서 이를 방지하기 위한 entropy encoder이다. 

이는 모든 sub-word의 출현이 독립적이고, subword sequence가 subword 등장 확률의 곱으로 계산된다고 가정한다.
WordPiece와 Unigram LM 모두 subword vocabulary를 만들기 위해 Language Model을 사용한다.

### Algorithm

1. Corpus를 준비한다.
2. Vocab size를 설정한다.
3. word sequence에 따른 word occurence 확률을 optimize한다.
4. 각 subword의 loss값을 계산한다.
5. symbole을 loss값으로 정렬하고, 상위 X%만 남긴다. OOV문제를 해결하기 위해 subword는 character-level로 진행하는게 좋다.
6. step2에서 정한 vocab size에 도달하거나, step 5에서 변화가 없을 때까지 3-5를 반복한다.


## SentencePiece

### Overview

[SentencePiece](https://github.com/google/sentencepiece)는 [arXiv](https://arxiv.org/pdf/1808.06226.pdf)에 공개된 알고리즘으로, 앞서 설명한 Sennrich의 BPE나 Unigram LM을 활용하여 sub-word unit를 만든다.

NMT는 일반적으로 고정된 vocab size를 다룬다. 대부분의 unsupervised word segmentation algorithm이
무한한 vocab size를 가정하는 것과는 다르게, SentencePiece는 마지막 vocab size가 고정된 segmentation 
model을 학습한다.

이전의 sub-word 알고리즘은 input sentence가 pre-tokenized하다고 가정하는데, 이는 효율적인 학습을
위해 필요하지만 미리 language dependent한 tokenizer를 실행해야 하므로 preprocessing을 더 복잡하게 만든다. SentencePiece는 raw text로부터 바로 학습할 수 있을 정도로 빠르고, 중국어와 일본어같이 
띄어쓰기가 없는 언어의 tokenizer와 detokenizer에 효과적이다.

Tokenization에서 원래의 input과 tokenized sequence는 **reversibly convertible**하지 않다.
예를 들어 "World"와 "." 사이에 공백이 없다는 정보가 tokenized sequence로부터 사라졌다고 해보자.
그러면  
```
Tokenize(“World.”) == Tokenize(“World .”)
```
가 된다.

SentencePiece는 input sequence를 Unicode character의 sequence로 취급하므로, 공백문자 또한 일반적인
symbol로 취급된다. 이를 basic token으로 명시적으로 취급하기 위해, SetencePiece는 공백문자를 meta symbol "▁" (U+2581)로 escape한다.  
```
Hello▁World.
```
그리고 이는 작은 조각으로 segmentation된다.
```
[Hello] [▁Wor] [ld] [.]
```

따라서 일반적인 tokenization과는 다르게, 별다른 언어 정보 없이도 detokenization을 수행할 수 있다.
```python
 detokenized = ''.join(pieces).replace('_', ' ')
```

즉, $Decode(Encode(Normalize(text))) = Normalize(text)$ 이며, lossless tokenization이라 부른다.

### 사용법

다음 방법을 통하여 SentencePiece를 설치한다.

```
pip install sentencepiece
```

그 후 다음을 통해 텍스트 파일을 준비하자 [https://github.com/google/sentencepiece/tree/master/data](https://github.com/google/sentencepiece/tree/master/data)

```
% wget https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt
```

다음은 학습에 필요한 parameter이다.

```
spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --character_coverage=1.0 --model_type=<type>
```
* `--input`: 한 라인 당 한 문장을 갖는 corpus로 **raw file**. Tokenizer, normalizer, preprocessor를 실행할 필요가 없음. default로 Unicode NFKC로 normalize 함. comma-separated로 파일 리스트를 넣을 수 있음.
* `--model_prefix`: 모델 이름에 붙는 prefix. `<model_name>.model`와 `<model_name>.vocab`가 생성.
* `--vocab_size`: vocabulary size, e.g., 8000, 16000, or 32000
* `--character_coverage`: 모델에 의해 cover할 character의 양. 일어나 중어 같은 rich character set은 `0.9995`가 좋은 결과를 내고, 그 외 character set이 작은 언어는 `1.0`이 좋음.
* `--model_type`: 모델 타입. `unigram` (default), `bpe`, `char`, or `word` 중 선택. `word`을 선택할 경우 반드시 pretokenized 되야 함.

이제 `templates`에다가 parameter를 집어넣어보자

```python
import sentencepiece as spm

templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={}'


train_input_file = "/root/KH-KIM/Ch04/botchan.txt"
pad_id = 0  # <pad>
vocab_size = 2000 # vocab size
prefix = 'botchan_spm'
bos_id = 1 # <start> token
eos_id = 2 # <end> token
unk_id = 3 # <unknown> token
character_coverage = 1.0 # 보통 1
model_type = 'unigram' # Choose from unigram (default), bpe, char, or word


cmd = templates.format(train_input_file,
                pad_id,
                bos_id,
                eos_id,
                unk_id,
                prefix,
                vocab_size,
                character_coverage,
                model_type)

cmd                
```

그리고 `SentencePieceTrainer`의 `Train` 클래스를 통해 학습할 수가 있다. 이를 통해 생성되는 것은
model과 vocab이며, 앞서 `model_prefix`를 통해 이름을 지정한 그대로 나오게된다.

```python
spm.SentencePieceTrainer.Train(cmd)

# botchan_spm.model
# botchan_spm.vocab
```
model은 tokenizer이며, vocab는 단어집합이다.

이제 모델을 불러서 tokenizing 해보자
```python
sp = spm.SentencePieceProcessor()
sp.Load('botchan_spm.model')

sp.SetEncodeExtraOptions('bos:eos') # <s>, </s>를 자동으로 넣어줌

text = 'This eBook is for the use of anyone anywhere at no cost'

tokens = sp.EncodeAsPieces(text)
print(tokens)
ids = sp.EncodeAsIds(text)
print(ids)
```
```
['▁This', '▁eBook', '▁is', '▁for', '▁the', '▁use', '▁of', '▁anyone', '▁any', 'w', 'here', '▁at', '▁no', '▁cost']
[210, 809, 33, 35, 6, 480, 12, 1467, 120, 84, 595, 42, 75, 968]
```

Detokenize는 `DecodePieces`와 `DecodeIds`를 통해 할 수 있다.
```python
sp.DecodePieces(tokens)
sp.DecodeIds(ids)
```
```
'This eBook is for the use of anyone anywhere at no cost'
'This eBook is for the use of anyone anywhere at no cost'
```

공식 github의 사용법을 번역한 문서를 colab으로 만들어서 배포 중에 있다. [다음](https://github.com/InhyeokYoo/NLP/blob/master/papers/2.Sub-word%20Model/Sentencepiece_python_module_example.ipynb)을 참고해서 따라해보자.




## Refence

[https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46)

[https://velog.io/@gwkoo/BPEByte-Pair-Encoding](https://velog.io/@gwkoo/BPEByte-Pair-Encoding)

[https://wikidocs.net/22592](https://wikidocs.net/22592)

[https://www.youtube.com/watch?v=1q67UzJWogE](https://www.youtube.com/watch?v=1q67UzJWogE)

[https://en.wikipedia.org/wiki/Byte_pair_encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)

[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

[https://donghwa-kim.github.io/SPM.html](https://donghwa-kim.github.io/SPM.html)