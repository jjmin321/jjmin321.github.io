---
title:  "LANGUAGE TRANSLATION WITH TORCHTEXT"
excerpt: "LANGUAGE TRANSLATION WITH TORCHTEXT 번역 자료"
toc: true
toc_sticky: true

categories:
  - PyTorch
  - NLP
tags:
  - attention
  - NMT
  - torchtext
use_math: true
last_modified_at: 2020-07-12
---

# Introduction

이번 튜토리얼은 `torchtext`의 몇 몇 편리한 클래스를 이용하여 영어와 독일어 모두를 포함한 잘 알려진 데이터 셋에 대해 진행하고, 독일 문장을 영어로 번역할 수 있는 attention을 이용한 sequence-to-sequence 모델을 학습시켜보도록 하겠습니다.

이 튜토리얼이 끝나면, 여러분은 다음과 같은 것을 할 수 있습니다:

-   문장을 NLP 모델링에서 일반적으로 사용되는 포맷으로 전처리 할 수 있습니다. 이는 `torchtext`의 편리한 클래스를 이용합니다:
    -   [TranslationDataset](https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.TranslationDataset)
    -   [Field](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field)
    -   [BucketIterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator)

> 본 튜토리얼의 원본은 [다음](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)과 같습니다. 역자의 주석은 지금 이 문단과 같이 citation으로 남기겠습니다. 본 코드는 colab을 통해서 실행할 수 있습니다. 
> 
> [https://github.com/InhyeokYoo/PyTorch-tutorial-text/blob/master/LANGUAGE\_TRANSLATION\_WITH\_TORCHTEXT.ipynb](https://github.com/InhyeokYoo/PyTorch-tutorial-text/blob/master/LANGUAGE_TRANSLATION_WITH_TORCHTEXT.ipynb)

# Field and TranslationDataset

`torchtext`는 번역 모델을 만들기 위해 쉽게 반복할 수 있는 데이터 셋을 만드는 유용한 기능이 있습니다. [Field](https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L64)는 이러한 핵심 클래스 중 하나로, 각 문장이 전처리하는 방법을 구체화해주고, 다른 하나는 `TranslationDataset`로, 데이터셋이 담겨있습니다. 이번 튜토리얼서 사용할 데이터는 [Multi30k dataset](https://github.com/multi30k/dataset)으로, 약 3만 개의 영어, 독일어 문장을 포함합니다 (평균적으로 문장 당 13개의 단어).

**Note**  
이 튜토리얼에서의 tokenization은 [Spacy](https://spacy.io/)를 필요로 합니다. 이는 영어 이외의 언어에서 보다 강력한 tokenization기능을 지원하기 때문입니다. `torchtext`는 `basis_english` tokenizer를 제공하고 영어를 위한 다른 tokenizer 또한 제공합니다 (e.g. [Moses](https://bitbucket.org/luismsgomes/mosestokenizer/src/default/)). 그러나 여러 언어가 필요한 언어 번역에서는 Spacy가 제일 좋은 선택입니다.

이 튜토리얼을 실행시키기 위해 `pip`나 `conda`를 이용하여 `spacy`를 먼저 설치합니다. 그 다음, 영어와 독일어 Spacy tokenizer를 위한 raw data를 다운로드합니다.

```
!python -m spacy download en
!python -m spacy download de
```

다음 코드는 `TranslationDataset` 내의 각 문장을 `Field`에 정의된 토크나이저를 기반으로 tokenize합니다.

```python
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize='spacy', tokenizer_language='de', init_token='<SOS>', eos_token='<EOS>', lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en", init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts =('.de', '.en'), fields=(SRC, TRG))
```

> `Multi30k`는 `TranslationDataset`의 subclass입니다.

> `Field`에 대해 더 자세히 알아보겠습니다. 공식문서에 따르면 Field는 데이터타입과 이를 텐서로 변환할 지시사항과 함께 정의하는 것이라 되어있습니다. `Field`는 텐서로 표현 될 수 있는 텍스트 데이터 타입을 처리하고, 각 토큰을 숫자 인덱으로 맵핑시켜주는 단어장(Vocabulary) 객체가 있습니다. 또한 토큰화 하는 함수, 전처리 등을 지정할 수 있습니다.  
> Arguments:

-   sequential: text는 sequential 데이터이므로 인자를 True 로 두고, LABEL 데이터는 순서가 필요없기 때문에 False 로 둔다.
-   use\_vocab: Vocab 객체를 사용할지의 여부. text에만 True 로 인자를 전달한다.
-   tokenize: 이름 그대로 tokenize
-   lower: 소문자 전환 여부.
-   batch\_first: True이면 tensor는 \[B, 문장의 최대 길이\]가 된다.
-   preprocessing: 전처리는 토큰화 후, 수치화하기 전 사이에서 작동한다.

`train_data`를 정의했으므로, `torchtext`의 `Field`의 매우 유용한 특성을 볼 수 있습니다. `build_vocab` 메소드는 이제 우리로 하여금 각 언어에 관련된 사전을 만들 수 있게끔 허용합니다.

> `build_vocab`은 Positional, keyward argument 두 개를 받는데, positional argument의 경우, `Dataset` 오브젝트나 iterable한 데이터를 받아 `Vocab`객체를 생성합니다. keyward argument의 경우 `Vocab`의 생성자로 전달할 인자를 받습니다.

```python
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
```

한번 이러한 코드가 동작하고 나면, `SRC.vocab.stoi`는 각 토큰을 key로 하고 이에 상응하는 값을 value로 하는 dictionary가 될 것입니다. `SRC.vocab.itos`는 이를 거꾸로한 list입니다. 이번 튜토리얼에서 이에 대해 깊게 다루지 않을 것이지만, 앞으로 마주할 NLP task에서 도움될만한 정보입니다.

# `BucketIterator`

마지막으로 살펴볼 `torchtext`의 특성은 `BuckerIterator`로, 이는 `TranslationDataset`을 첫번째 인자로 받아 사용하기 쉽습니다. 구체적으로, API 문서에서 언급한 바와 같이, 비슷한 길이를 갖는 데이터를 함께 묶는(batch) Iterator를 정의합니다. 매 새로운 epoch에서 랜덤한 batch를 생성하는 과정에서 padding을 최소화합니다.

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)
```

이러한 iterator는 아래와 같이 `train`과 `evaluate` 함수 안에서 `DataLoader`와 같은 방법으로 호출할 수 있습니다:

```python
for i, batch in enumerate(iterator):
```

이러면 각 `batch`는 `src`와 `trg` 속성을 갖게됩니다.

```python
src = batch.src
trg = batch.trg
```

> `BucketIterator` 또한 살펴보도록 하겠습니다. 위에서 만든 `BucketIterator`를 한번 살펴보고, `SRC.vocab.itos`를 통해 다시 text로 변환시켜봅시다.

```python
for i, batch in enumerate(train_iterator):
    print(batch.trg.size())
    items = [" ".join([TRG.vocab.itos[item] for item in batch.trg[:, i]]) for i in range(128)]
    for item in items:
        print(item)

    print(batch.src.size())
    items = [" ".join([SRC.vocab.itos[item] for item in batch.src[:, i]]) for i in range(128)]
    for item in items:
        print(item)
    break
```

```
torch.Size([30, 128])
<sos> a group of young people lounging on the couch . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
<sos> three females and one male are walking at the edge of a road . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
<sos> a man on a city sidewalk in a coat playing a brass clarinet . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
<sos> a woman nibbles at a food item in her hands . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
......
```

> 최대 길이에 맞춰 padding된 것을 볼 수 있습니다.

# Defining our `nn.Module` and `Optimizer`

이는 `torchtext`관점에서 주로 이루어지는 것으로, dataset이 만들어지고 iterator가 정의되면 이 튜토리얼의 나머지 부분은 단순히 `nn.Module`로 우리의 모델을 만들고 `Optimizer`를 정한 후 학습시키면 끝납니다.

우리 모델은 구체적으로 특별히 [여기](https://arxiv.org/abs/1409.0473) 묘사된 구조를 따릅니다 (더 많은 설명은 [이곳](https://github.com/SethHWeidman/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb))

**Note**  
이 모델은 언어 번역에서 사용할 수 있는 단순한 예제로, 이가 standard한 모델이기 때문에 사용한 것이지 번역을 위해 추천되는 모델이기 때문이 아닙니다. 알다시피 SOTA(State\_Of\_The\_Art: 가장 좋은) 모델은 Transformer에 기반하고 있습니다. [이곳](https://pytorch.org/docs/stable/nn.html#transformer-layers)에서 PyTorch 구현을 볼 수 있습니다. 특히, 아래에서 사용된 "attention" 모델은 transformer에서 구현된 multi-headed self-attention과는 다릅니다.

```python
# Std. Lib.
import random
from typing import Tuple    # typing으로 Param.의 type을 강제함.

# Torch Lib.
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super(Encoder, self).__init__()
        # Constructor
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        # 단어의 idx가 들어오면 이를 embedding layer에 넣어 word vector를 얻어준다.
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # Seq2Seq는 GRU를 사용함. bidirectional한 이유는 언어에 따라 word order가 다르기 때문
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # dropout layer의 위치를 주목하자
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        # torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1): batch dimension으로 concat
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim) * 2 + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention)

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: int, attention: nn.Module):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)

class Seq2Seq(nn.Module):
    # 일종의 main 함수처럼
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)
optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

> ## 자세히 살펴보기
> 복잡하므로 하나씩 천천히 해석을 해보겠습니다. 코드 라인을 중심으로 해석할 것이니 위 아래로 왔다갔다해야합니다.  
> 우선 우리의 모델은 `Seq2Seq`에서 제어합니다. 이 모델은 encoder, decoder를 필요로 합니다.

>```python
>model = Seq2Seq(enc, dec, device).to(device)
>```

>`Seq2Seq`의 forward를 보겠습니다. 여기서는 `src`와 `trg` 텐서를 input으로 받고 있습니다. 이는 `BucketIterator`의 iterator로, 위에 코드로 확인했듯이 다음을 통해 접근 가능합니다.

> ```python
> for i, batch in enumerate(train_iterator):
>     src = batch.src  # [seq_len x B]
>     trg = batch.trg  # [seq_len x B]
> ```

>````
>번역을 하다보면 문장이 끊임없이 늘어날 수 있으므로, >`seq_len`만큼의 길이를 갖도록 제한을 해줍니다.
>```python
>max_len = trg.shape[0]
>````

>`outputs`는 decoder를 수행한 결과를 담을 >tensor입니다. 처음에 이를 초기화한 이후, 번역의 >결과를 담도록 합니다. 차원은 RNN의 input과 같이 **\>[Seq\_len, Batch, input\_dim\]**을 따를 것입니다.

>```python
>outputs = torch.zeros(max_len, batch_size, >trg_vocab_size).to(self.device)
>```

>이후 얻은 src는 `Encoder`에 넣도록 하겠습니다. 인코더로 얻어지는 결과는 `enocoder_outputs`와 `hidden`으로, `enocoder_outputs`은 input sequence의 back/forward 모든 hidden state이고, `hidden`은 마지막 hidden state로 linear layer에 쓰입니다.

>```python
>encoder_outputs, hidden = self.encoder(src)
>```

> ## `Encoder`
> 
> 앞서 본 src는 `nn.Embedding`으로 전달됩니다. 한 가지 특이사항으로 `nn.Embedding`은 **\[Seq\_len x B\]** 이나 **\[B x Seq\_len\]** 모두의 형태를 input을 받을 수 있습니다. 다음 예시를 봐볼까요?

>```python
>emb = nn.Embedding(len(SRC.vocab), 32)
>emb.to(device)
>data = batch.src.to(device)
>data_T = data.T
>print(f"Orignal: {data.size()}, Batch_first: {emb>(data_T).size()}, Batch_last: {emb(data).size()}")
>```

>```
># 결과
>Orignal: torch.Size([30, 128]), Batch_first: >torch.Size([128, 30, 32]), Batch_last: torch.Size(>[30, 128, 32])
>```

>결국 어느 경우든 input에 embedding\_dim이 추가되는 형태임을 알 수 있습니다. 따라서 어느 것을 사용할지는 RNN의 batch\_first에 달려있습니다.

> 다음은 GRU입니다. Seq2Seq (정확하게는 조경현 교수님의 Seq2Seq의 초기버전)에서는 GRU를 사용합니다. GRU가 처음 제안된 논문이기도 합니다. 번역 모델에서는 언어에 따라 word order가 달라질 수 있으므로 Bidirectional한 모델을 사용합니다. GRU의 input은 앞선 `nn.Embedding`의 output인 **\[Seq\_len x Batch x Emb\_dim\]**이 됩니다. GRU의 `hidden`은 마지막(t=src len) hidden state 값으로, **\[num\_layers \* num\_directions x Batch x Hid\_dim\]** 차원입니다. `output`은 GRU의 hidden state를 모아놓은 것으로, 세번째 차원 **\[hid dim \* num directions\]**에서 첫번째는 forward RNN, 두번째는 backward RNN을 의미합니다. 즉, $h\_1 = \[\\overrightarrow h\_1;\\overrightarrow h\_T \]$이고, $h\_2 = \[\\overrightarrow h\_2;\\overrightarrow h\_{T-1} \]$이 됩니다. 그리고, 이러한 stacked encoder hidden state를 $H = {h\_1, h\_2, ...h\_T} $로 나타낼 수 있습니다. 차원은 **\[src sent len, batch size, hid dim \* num directions\]**이 됩니다.

> 그 후 concat하게 되는데, `hidden [-2, :, : ]`은 forwards RNN을, `hidden[-1, :, : ]`은 backward RNN을 의미합니다. Batch는 변하면 안되므로 Batch차원은 유지(dim=1)합니다.  
> FC layer의 경우 GRU의 hidden state 두개를 concat하여 넣어줍니다. 이는 후에 decoder의 초기 hidden state가 될 것이므로, 디코더의 차원과 맞게끔 유지합니다. Embedding과는 다르게 matrix multiplication의 차원을 신경써줘야 합니다.

>```python
>embedded = self.dropout(self.embedding(src)) 
>outputs, hidden = self.rnn(embedded)
>hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
>```

> 최종적으로 차원을 정리하면 다음과 같습니다:  
> **GRU**
>-   src: \[Seq\_len, Batch, Emb\_dim\]
>-   outputs: \[src sent len, batch size, hid dim \* num directions\]
>-   hidden: \[n layers \* num directions, batch size, hid dim\]

> **torch.cat(, dim = 1)**
>-   `hidden[-2, :, :], hidden[-1, :, :]`: \[Batch, Hid\_dim\]
>-   output: \[batch\_size, enc\_hid\_dim \* 2\]

> **FC**
>-   input: \[batch\_size, enc\_hid\_dim \* 2\]
>-   output: \[batch size, dec hid dim\]

> ## 다시 `Seq2Seq`
> 
> 다시 Seq2Seq으로 돌아오겠습니다. Encoder가 끝나면, 이 결과에 대해 decoding을 할 차례입니다. 우선 문장의 시작을 알리는 토큰이 필요합니다. `trg`의 0번째 idx는 토큰이므로, 이를 이용하겠습니다. 그러면 `output`은 **\[Batch\]**의 vector가 됩니다.

>```python
>output = trg[0,:] # first input to the decoder is the <sos> token
>```

>이후에는 앞서 encoder의 output인 `hidden`와 'encoder\_outputs', 그리고 토큰인 `output`, 을 디코더에 넣겠습니다. 나머지 부분은 디코더를 확인하고 다시보겠습니다.

>```python
>for t in range(1, max_len):
>    output, hidden = self.decoder(output, hidden, encoder_outputs)
>    """
>    outputs[t] = output
>    teacher_force = random.random() < teacher_forcing_ratio
>    top1 = output.max(1)[1]
>    output = (trg[t] if teacher_force else top1)
>    """
>```

>## `Attention`

> Decoder를 보기에 앞서 `Attention`을 확인하겠습니다. 이는 디코더의 이전 hidden\_state인 $s\_{t-1}$과 encoder의 모든 forward와 backward를 쌓은 hidden state $H$를 필요로합니다. 이 레이어의 결과는 attnetion vector $a\_t$로, 길이가 source sentence의 길이와 같고 값이 0부터 1 사이이며, 모두 합치면 1이 됩니다.

>```python
>class Attention(nn.Module):
>    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
>        super(Attention, self).__init__()
>        self.enc_hid_dim = enc_hid_dim
>        self.dec_hid_dim = dec_hid_dim
>
>        self.attn_in = (enc_hid_dim) * 2 + dec_hid_dim
>        self.attn = nn.Linear(self.attn_in, attn_dim)
>
>    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
>        src_len = encoder_outputs.shape[0]
>        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
>        encoder_outputs = encoder_outputs.permute(1, 0, 2)
>        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden encoder_outputs), dim=2)))
>        attention = torch.sum(energy, dim=2)
>
>        return F.softmax(attention)
>```

> 먼저 이전 디코더 hidden state와 encoder hidden state사이의 energy를 계산해야합니다. Energy를 구하는 식은 다음과 같습니다.  
><p style="text-align: center;">
> $$  
> E_t = \tanh(\textrm{attn}(s_{t-1}, H))
> $$
> </p>

> 인코더의 히든 스테이트는 T (source len)개 tensor의 sequence이고, 디코더의 히든 스테이트는 **\[batch size, dec hid dim\]**의 single vector이므로, 길이를 맞춰주어야 합니다. 이를 위해 `unsqueeze(1)`을 하여 **\[batch size, 1, dec hid dim\]**로 바꾸고, T번 `repeat(1, T, 1)`합니다. 그러면 **\[batch size, seq\_len, dec hid dim\]**이 될 것입니다.

```python
repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
```

> `encoder_outputs`는 **\[src sent len, batch size, enc hid dim \* 2\]**의 차원을 갖고 있습니다. 이를 concat하고, FC에 feed하여 attn\_dim으로 나타내기 위해 `torch.Tensor.permute`를 통해 텐서 차원끼리 교환합니다. 이 결과 **\[batch size, src sent len, enc hid dim \* 2\]**차원이 됩니다.

>```python
>encoder_outputs = encoder_outputs.permute(1, 0, 2)
>```

>이후 이 둘을 concat합니다. **\[batch size, seq\_len, dec hid dim ; batch size, src sent len, enc hid dim \* 2\]** 이므로, **\[batch size, src sent len, enc hid dim \* 2 + dec hid dim\]**이 될 것입니다.

>```python
>torch.cat((repeated_decoder_hidden, encoder_outputs), dim = 2)
>```

>이제 energy를 계산합니다. 에너지는 FC인 `self.attn`을 통과하여 얻습니다. 차원은 **\[batch\_size, seq\_len, attn\_dim\]** 입니다. 그 후 tanh를 통과합니다.
>```python
>energy = torch.tanh(self.attn(torch.cat(>(repeated_decoder_hidden, encoder_outputs), dim=2)))
>```

>Addictive attention의 경우 $E\_t = v^T\\tanh (\\textrm{attn}(Ws\_{t-1} + Uh\_j))$ 가 되고, 사이즈는 **\[batch size, src len\]**입니다. $\\tanh (\\textrm{attn}(Ws\_{t-1} + Uh\_j))$ 부분은 앞서 concat하여 구했습니다. 여기서는 parameter $v^T$를 학습시키는 대신 이후 attention dim으로 sum하겠습니다. 사이즈는 마찬가지로 **\[batch size, src len\]**가 됩니다.

>```python
>attention = torch.sum(energy, dim=2)
>```

> ## `Decoder`
> 
> Encoder의 결과인 hidden vector와 output vector, attention의 attention score를 받아 번역할 언어의 단어를 차례대로 반환합니다. 따라서, trg 언어의 embedding이 필요할 것입니다. output\_dim은 trg언어의 look-up words의 개수, emb\_dim은 embedding vector의 차원입니다.

>```python
>self.embedding = nn.Embedding(output_dim, emb_dim)
>```

>이후엔 encoder와 마찬가지로 GRU를 이용해 번역합니다. 어텐션의 `attn_in`은 인코더의 context vector로부터 decoder의 attention score를 계산하는 layer의 input dimension입니다.

>```python
>self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
>self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
>self.dropout = nn.Dropout(dropout)
>```

>디코더는 어텐션을 이용해 인코더 히든 스테이트인 $H$와 어텐션 벡터 $a\_t$를 이용해 weighted source vector $w\_t$를 생성합니다.  

><p style="text-align: center;">
>$$  
>w_t = a_tH 
>$$  
></p>

> 이 과정은 함수 `_weighted_encoder_rep`에 나와 있습니다.
>```python
>def _weighted_encoder_rep(self, decoder_hidden: Tensor, >encoder_outputs: Tensor) -> Tensor:
>        a = self.attention(decoder_hidden, encoder_outputs)
>        a = a.unsqueeze(1)
>        encoder_outputs = encoder_outputs.permute(1, 0, 2)
>        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
>        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
>
>        return weighted_encoder_rep
>```

> 우선 attention의 결과로 얻어지는 attention vector `a`는 차원이 **\[Batch x seq len\]**이기 때문에, 이를 **\[Batch x 1 x seq len\]**로 바꾸어줍니다.

>```python
>a = self.attention(decoder_hidden, encoder_outputs)
>a = a.unsqueeze(1)
>```

>이후, Batch matrix multiplication을 하기 위해 `permute(1, 0, 2)`를 이용해 stacked hidden state인 `encoder_outputs`의 차원을 **\[src sent len, batch size, hid dim \* num directions\]** 에서 **\[batch size, src sent len, hid dim \* num directions\]**로 바꾸어줍니다.
>```python
>encoder_outputs = encoder_outputs.permute(1, 0, 2)
>```

>이제는 위에서 본 weighted source vector `w_t`를 구하면 됩니다.  
**\[Batch x 1 x seq len\]**와 **\[batch size, src sent len, hid dim \* num directions\]**의 배치곱이므로, 결과는 **\[batch size, 1, hid dim \* num directions\]**이 됩니다. 이를 다시 **\[1, batch size, enc hid dim \* 2\]**차원으로 바꿉니다.
>```python
>weighted_encoder_rep = torch.bmm(a, encoder_outputs)
>weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
>```

> 이제는 `forward`를 보겠습니다.  
> `input`은 글자의 idx tensor로, **\[batch size\]**차원입니다. 즉, 매 t번째 시점의 단어 (처음에는 토큰이 batch size만큼)가 들어오는 것입니다. seq\_len이 1이므로 이를 **\[1, batch size\]**로 `unsqueeze(0)`해줍니다. 그 후 각 단어의 idx는 임베딩되어 `embedded`가 되고, 이의 차원은 \*\*\[1, batch\_size, emb\_dim\]이 됩니다.
>```python
>input = input.unsqueeze(0)
>embedded = self.dropout(self.embedding(input))
>```

>임베딩된 input word $y\_t$(`embedded`)와 weighted source >vector $w\_t$(`weighted_encoder_rep`), 이전 시점의 >디코더의 히든 스테이트 $s\_{t-1}$(`decoder_hidden`)은 디코더 RNN으로 전달됩니다.
><p style="text-align: center;">
>$$  
>s_t = \textrm{DecoderGRU}(y_t, w_t, s_{t-1}) 
>$$
></p> 
>weighted source vector $w\_t$(`weighted_encoder_rep`)는 **\[1, batch size, enc hid dim \* 2\]**, $y\_t$와 $w\_t$는 concat되어 **\[1, batch size, (enc hid dim \* 2) + emb dim\]**이 됩니다.

>```python
>weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
>rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
>```

>Decoder의 hidden state $s\_{t-1}$은 encoder의 `hidden`으로 **\[batch size, dec hid dim\]**입니다. 이를 Decoder의 GRU에 넣기 위해 `unsqueeze(0)`하여 **\[1, batch size, dec hid dim\]**를 얻습니다. 마찬가지로, 1은 seq\_len입니다.
>```python
>output, decoder_hidden = self.rnn(rnn_input, >decoder_hidden.unsqueeze(0))
>```
>`output`은 마찬가지로 hidden state의 집함, `decoder_hidden`은 마지막 hidden state입니다.

> 그 후엔 linear layer $f$에 $y\_t, w\_t, s\_{t-1}$를 전달하여 target sentence $\\hat{y\_{t+1}}$을 예측합니다. 이는 이들 모두를 concat하여 수행할 수 있습니다. 
><p style="text-align: center;">
> $$  
> y_t = f(y_t, w_t, s_t)
> $$
> </p> 
> seq\_len은 전부 1이니까 이를 `squeeze(0)`하고 concat한 후 FC에 넣습니다.  
> `embedded`: **\[1, batch size\]** -> **\[batch size\]**  
> `output`: **\[1, batch size, dec hid dim \* n directions\]**\-> **\[batch size, dec hid dim \* n directions\]**,  
> `weighted_encoder_rep`: **\[1, batch size, (enc hid dim \* 2) + emb dim\]** -> **batch size, (enc hid dim \* 2) + emb dim\]**  
> 이루어직concat은 당연히 batch size를 중심으로 이루어집니다.  
> `output`은 **\[batch size, output dim\]**가 됩니다. Decoder의 결과는 이 `output`과, `decoder_hidden`을 `squeeze(0)`한 것입니다. `decoder_hidden`은 **\[ batch size, dec hid dim\]**이 됩니다.

```python
embedded = embedded.squeeze(0)
output = output.squeeze(0)
weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim = 1))
return output, decoder_hidden.squeeze(0)
```

> ## 다시 `Seq2seq`
> 
> 아까봤던 `Seq2seq`의 `forward`의 반복문을 보겠습니다. 앞서 저희는 `outputs`라는 텐서에 단어를 넣기로 하였습니다. Decoder의 결과물인 `output`은 softmax를 통하여 예측하는 다음단어가 됩니다.

```python
for t in range(1, max_len):
    '''
    output, hidden = self.decoder(output, hidden, encoder_outputs)
    '''
    outputs[t] = output
    '''
    teacher_force = random.random() < teacher_forcing_ratio
    top1 = output.max(1)[1]
    output = (trg[t] if teacher_force else top1)
    '''
```

> Decoder가 예측한 다음 단어는 top1이 되고 (softmax를 한 결과와 max를 한 결과가 같음), Teacher forcing을 사용하겠다면 `trg[t]`가 다음 `output`이 되어 decoder의 입력으로 들어가고, 그게 아니라면 `top1`을 넣어 teacher forcing을 사용하지 않을 것입니다.

```python
for t in range(1, max_len):
    """
    output, hidden = self.decoder(output, hidden, encoder_outputs)
    outputs[t] = output
    """
    teacher_force = random.random() < teacher_forcing_ratio
    top1 = output.max(1)[1]
    output = (trg[t] if teacher_force else top1)
```

**Note**  
언어 번역 모델의 성능을 평가할 때, `nn.CrossEntropyLoss`로 하여금 padding index를 알려주어야 합니다.

```python
PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

마지막으로 train과 eval을 해봅시다.

```python
import math
import time

def train(model: nn.Module, iterator: BucketIterator,optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model: nn.Module, iterator: BucketIterator, criterion: nn.Module):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```