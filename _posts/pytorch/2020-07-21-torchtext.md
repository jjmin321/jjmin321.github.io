---
title:  "Torch text 살펴보기 (아직 작성 중)"
excerpt: ""
toc: true
toc_sticky: true
permalink: /pytorch/torchtext/
categories:
  - PyTorch
  - Torchtext
use_math: true
last_modified_at: 2020-09-21
---

# Introduction

PyTorch를 통해 NLP를 처리하다보면 torchtext를 사용하게 되는데, 마테리얼도 따로 없고... 공부할 방법이 마땅치가 않다.
이번 기회에 한번 정리해보자. Colab문서는 [다음](https://github.com/InhyeokYoo/NLP/blob/master/utils/1.%20torchtext/1_torchtext_tutorial.ipynb)을 참조하며 따라해보자.

torchtext로 할 수 있는 목록은 다음과 같다.
- File loading
- Tokenization
- Generate a vocabulary list(Vocab)
- Numericalize/Indexify
- Word Vector
- Batching
    
# Field

다음은 [공식문서](https://pytorch.org/text/data.html#torchtext.data.Field)에서 발췌한 내용이다.
> Field는 Tensor로 변환하기 위한 지시사항과 datatype을 함께 정의한다.  
Field 클래스는 tensor로 표현될 수 있는 공통의 text processing datatype을 모델링한다. 이에는 field의  요소에 대해 모든 가능한 값과 이에 대응되는 숫자표현을 정의하는 Vocab 객체를 갖고 있다. Field객체는 또한 datatype이 숫자로 변환되는 방법(Tokenization, Tensor의 생성 방법)에 관련된 parameter를 갖고 있다.  
만약 Field가 데이터 셋 내에서 두 개의 컬럼에 작용하는 경우(e.g. QA), 이 컬럼들은 같은 vocabulary를 갖는다.

뭔 말인진 알겠으면서 잘 모르겠다. Parameter를 제공한다 했으니 한번 살펴보자.
- `sequential=True`: dataset이 sequential한지 여부. 만일 False일 경우, tokenization은 적용되지 않는다. label이 대표적인 False 타입이다.
- `use_vocab=True`: Vocab 객체를 사용할지 여부. Vocab은 토큰을 숫자로 맵핑하므로, Flase라면 이 필드에 있는 데이터는 반드시 숫자여야 한다.
- `init_token=None`: 각 example의 앞에다가 붙일 token
- `eos_token=None`: EOS 토큰.
- `fix_length=None`: 최대 허용 길이. 나머지는 pad로 맞춰줌
- `dtype=torch.int64`: batch의 데이터 타입
- `preprocessing=None`: Pipeline. Tokenization - numericalizing 사이에 적용
- `postprocessing=None`: numericalizing과 실제 Tensor로 변하기 전에 적용될 Pipleline. Pipleline 함수는 batch를 list로 받고, filed의 Vocab을 받는다.
- `lower=False`: 소문자 전환 여부. 대문자가 들어있으면 sparse해지므로 소문자로 한다.
- `tokenize=None`: 토크나이징 함수. `konlpy`같은거를 사용하면 된다. `string.split()`이 default다.
- `tokenizer_language='en'`: 걍 언어. 오직 SpaCy에서만 다양한 언어를 제공함
- `include_lengths=False`: pad된 미니배치만 return할지, 각 example의 길이도 같이 반환할지 여부
- `batch_first=False`: 배치 우선 여부. `[Batch, Sequence_length]`의 차원이 됨.
- `pad_token='<pad>'` 패딩 토큰
- `unk_token='<unk>'`: 언노운 토큰
- `pad_first=False`: 패딩을 뒤에다 할지 앞에다 할지 여부
- `truncate_first=False`: 처음에 자를지 말지 여부?
- `stop_words=None`: 불용어
- ` is_target=False`: 레이블 데이터 여부

대충 text 데이터를 처리하는데 있어서 사용할 것이란 감이 온다.

다음은 실사용 예시이다. Many-to-one model을 가정하고, 텍스트에 관련된 `TEXT` Field와 이에 대한 label정보를 갖는 `LABEL` Field를 만들었다. 데이터 타입에 따라 파라미터가 어떻게 바뀌는지 직접 확인해보자.

<script src="https://gist.github.com/InhyeokYoo/48d680bb7f70cc773d4a702f428a4702.js"></script>

## Dataset

이제 데이터 셋을 만들어보자. 

## Reference

Allen Nie's article: ["A Tutorial on Torchtext"](http://anie.me/On-Torchtext/)

simonjisu's notebook: [TorchText Tutorials](https://github.com/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/01_TorchText.ipynb)
