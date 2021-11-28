---
title:  "pytorch로 BERT 구현하기"
excerpt: "BERT를 직접 구현하며 헷갈리는 것들을 살펴보자"
toc: true
toc_sticky: true
permalink: /project/nlp/bert-issue/
categories:
  - NLP
  - PyTorch
tags:
- Implementation
use_math: true
last_modified_at: 2020-12-06
---

# Intro

[이전 시간](/project/nlp/bert-review/)에는 BERT에 대해 공부해보았다. 이번에는 이를 구현해보도록 하자.

BERT는 크게 pre-train과 fine-tuning 두 가지의 task를 하게 된다. 이번 장에서는 데이터를 load하여 DataLoader를 만드는 것을 포함하여 각 task에서 필요로 하는 pre-processing을 다뤄보자.

아래는 본 포스트에서 다루지 못한 TODO 리스트이다.
- 80% [MASK], 10% RAND, 10% SAME for MLM
- 데이터 셋 shuffle
    - 이건 데이터 셋을 따로 만들 때 사용할 수 있다
- 두 개의 sentence 길이가 512가 넘을 때 대처방법
    - 구글에서는 둘의 길이가 512가 넘을 때 sentence A와 B를 임의로 잘랐음 (crop)
- Warm up optimizer
- pre-train에서 step의 90%는 128길이의 문서로, 나머지는 512길이의 문서로 학습 
- fine-tuning task

# Pre-train

Pre-train과정에서는 masked language model과 next sentence prediction을 수행한다. 구체적으로 필요한 요구사항은 다음과 같이 정리할 수 있을 것 같다.
- DataLoader
    - *`torchtext.data.Dataset` vs. `torch.utils.data.Dataset`*
- 학습 데이터에 대해 WordPiece 모델을 통해 tokenizing 하는 기능
    - *직접 만들기는 그렇고 어디선가 가져와야 함*
- 학습 데이터에 대해 `Vocab`으로 단어를 저장하는 기능
    - `torchtext.data.Field`쓰면 됨
- \<CLS>, \<SEP>, \<MASK> special token 추가
    - `torchtext.data.Field`쓰면 됨
- **각 task에 맞는 기능 추가하기**
    - NLM: \<MASK> 토큰 씌우는 기능
    - NSP: 문장 섞어주는 기능. 이러면 *BPTTIterator*를 사용할 필요가 없음

## Load data

가장 먼저 할 일은 데이터를 불러오는 것이다. BERT는 BooksCorpus와 wikipedia데이터를 통해 학습한다.

> For the pre-training corpus we use the BooksCorpus (800M  words)  (Zhu  et  al.,2015) and English  Wikipedia (2,500M  words).

BooksCorpus는 [허깅페이스](https://huggingface.co/datasets/bookcorpus)를 통해 다운받을 수 있다.

```shell
!wget https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2
!tar -xvf '/content/drive/MyDrive/Colab Notebooks/datasets/bookcorpus.tar.bz2'
```

혹은 huggingface의 `datasets`를 사용해서 다운받을 수도 있다.

```python
from datasets import load_dataset

dataset = load_dataset('bookcorpus', split='train')
```

데이터를 다운 받은 후엔 이를 불러온다. 내가 쓰는 colab pro 환경에서는 데이터를 다 부르면 너무 크니까 이를 적절하게 나누어 주었다. 구글에서 나온 코드는 문서 간 shuffle기능이 있지만, 이 데이터셋은 어느 문서가 어디에서 나오는지도 확인하기 어려우므로 패스한다.

```python
path = '/content/drive/MyDrive/Colab-Notebooks/datasets/'
files = glob.glob(f'{path}books_large*.txt')
files.sort()
print(files)
data = []

for file in files:
    with open(file) as f:
        for line in f.readlines():
            data.append(line.strip())

# (train+valid):test = 0.8:0.2
# train:valid: 0.8:0.2
length = len(data)
train = data[:int(length * 0.8 * 0.8)]
valid = data[int(length * 0.8 * 0.8):-int(length * 0.2)]
test = data[-int(length * 0.2):]

with open(path+'BookCorpus_train.txt', 'w') as f:
    for data in train:
        f.write("%s\n" % data)

with open(path+'BookCorpus_valid.txt', 'w') as f:
    for data in valid:
        f.write("%s\n" % data)

with open(path+'BookCorpus_test.txt', 'w') as f:
    for data in test:
        f.write("%s\n" % data)
```

## Tokenizer

BERT는 Wordpiece tokenizer를 사용한다. 따라서 `sentencepiece`의 tokenizer를 사용한다. 
Vocab 수는 30,000이다.
character_coverage의 경우 1.0으로 설정하지 않으면 특수문자를 학습하지 못하는 경향이 생긴다. 따라서 1.0으로 넣어주자.


```python
parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={} --model_type={} --character_coverage={}'

train_input_file = "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_train.txt"
vocab_size = 30000
prefix = 'bookcorpus_spm'
user_defined_symbols = '[PAD],[CLS],[SEP],[MASK]'
model_type = 'bpe'
character_coverage = 1.0 # default

cmd = parameter.format(train_input_file, prefix, vocab_size, user_defined_symbols, model_type, character_coverage)

spm.SentencePieceTrainer.Train(cmd)
```

## `torchtext.data.Dataset` vs. `torch.utils.data.Dataset`

데이터를 불러오는 `Dataset`선택지는 크게 두 개가 있다. 하나는 `torchtext.data.Dataset`를 쓰는 것이고, 나머지 하나는 `torch.utils.data.Dataset`를 쓰는 것이다.

`torchtext.data.Dataset`는 parameter로 *examples*와 *fields*를 받고, 자동적으로 vocab 등을 생성해준다는 장점이 있다. 그러나 `torchtext.data.Example`을 만들어줘야 한다. 이는 `torch.utils.data.Dataset`의 자식 클래스이다.

그러나 `torchtext.data.Field`는 곧 deprecation되어 없어질 예정이고, `torchtext.data.Dataset`는 `torch.utils.data.Dataset`와 호환되지 않으므로 `torch.utils.data.Dataset`을 사용하는게 더 좋아보인다. 이는 다음 [torchtext 레포에 남겨진 issue](https://github.com/pytorch/text/issues/936)와 [패치노트](https://github.com/pytorch/text/releases)에서도 확인할 수 있다.

> Several components and functionals were unclear and difficult to adopt. For example, the Field class coupled tokenization, vocabularies, splitting, batching and sampling, padding, and numericalization all together, and was opaque and confusing to users. We determined that these components should be divided into separate orthogonal building blocks. **For example, it was difficult to use HuggingFace's tokenizers with the Field class (issue #609)**. Modular pipeline components would allow a third party tokenizer to be swapped into the pipeline easily.
...
torchtext’s datasets were incompatible with DataLoader and Sampler in torch.utils.data, or even duplicated that code (e.g. torchtext.data.Iterator, torchtext.data.Batch). Basic inconsistencies confused users. For example, many struggled to fix the data order while using Iterator (issue #828), whereas with DataLoader, users can simply set shuffle=False to fix the data order.

또한, `torchtext.data.Field`, `torchtext.data.Example`도 같이 없어지기 때문에 이를 대체할 코드가 필요하다.

```python
# dataset 만들기
class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data: List, vocab: spm.SentencePieceProcessor):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
        """

        super(LanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data

    def __getitem__(self, i):
        return self.vocab.EncodeAsIds(self.data[i].strip())
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)
```

그러나 이렇게 할 경우 `torch.utils.data.Dataloader`를 통해 불러올 때 `collate_fn`을 적절하게 세팅해서 넣어줘야 한다.
MLM task라면 패딩만 신경쓰면 되지만 NSP task를 할 때는 index를 알 수 있는 방법이 전무하므로, 랜덤한 문장을 넣기가 어렵다.

따라서 아래와 같이 `__getitem__`부터 미리 pre-train에 맞는 데이터 셋을 반환하게끔 하였다.

```python
# dataset에서 미리 처리해주기
class BERTLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data: List, vocab: spm.SentencePieceProcessor, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data
        self.seq_len = seq_len
        self.sep_id = vocab.piece_to_id(sep_id)
        self.cls_id = vocab.piece_to_id(cls_id)
        self.mask_id = vocab.piece_to_id(mask_id)
        self.pad_id = vocab.piece_to_id(pad_id)
        self.p = p
        self.mask_frac = mask_frac

    def __getitem__(self, i):
        seq1 = self.vocab.EncodeAsIds(self.data[i].strip())
        seq2_idx = i+1
        # decide wheter use random next sentence or not for NSP task
        if random.random() > p:
            is_next = torch.tensor(1)
            while seq2_idx == i+1:
                seq2_idx = random.randint(0, len(data))
        else:
            is_next = torch.tensor(0)

        seq2 = self.vocab.EncodeAsIds(self.data[seq2_idx])

        if len(seq1) + len(seq2) >= self.seq_len - 3: # except 1 [CLS] and 2 [SEP]
            idx = self.seq_len - 3 - len(seq1)
            seq2 = seq2[:idx]

        # sentence embedding: 0 for A, 1 for B
        mlm_target = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        sent_emb = torch.ones((mlm_target.size(0)))
        _idx = len(seq1) + 2
        sent_emb[:_idx] = 0
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([torch.tensor([self.cls_id]), masking(seq1), torch.tensor([self.sep_id]), masking(seq1), torch.tensor([self.sep_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (512 - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target, sent_emb, is_next
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)
```

세부 기능별로 깔끔하게 다듬을 수 있을 것 같긴한데, 일단은 넘어가자.

이후 데이터로더를 통해 불러오면 깔끔하게 불러와지는 것을 확인할 수 있다.

```python
# 작동 테스트
dataloader = DataLoader(dataset, batch_size=48, shuffle=False)

for batch, (mlm_train, mlm_target, sent_emb, is_next) in enumerate(dataloader):
    print(mlm_train.size())
    print(mlm_target.size())
    print(sent_emb.size())
    print(is_next.size())
    break
    
# torch.Size([48, 512])
# torch.Size([48, 512])
# torch.Size([48, 512])
# torch.Size([48])
```

이 방법 외에 데이터를 불러와서 전처리하고 label 정보까지 미리 준 다음 dataloader에서 셔플하는 방법도 있다.

## Model architecture

이제는 모델을 짤 차례이다. 트랜스포머는 다른 곳에서도 많이 구현해놨고, torch에서도 제공하므로 생략하자.

나는 BERT의 몸체를 담당하는 `BertModle`과 MLM task/NSP task를 담당하는 head로 나눠놨다. 
이는 fine-tuning시에도 `nn.Module`을 붙일 수 있으므로 적절한 구조로 보인다.

```python
class BertModel(nn.Module):
    def __init__(self, voc_size:int=30000, seq_len: int=512, d_model: int=768, d_ff:int=3072, pad_idx: int=1,
                num_encoder: int=12, num_heads: int=12, dropout: float=0.1):
        super(BertModel, self).__init__()
        self.pad_idx = pad_idx
        self.emb = BERTEmbedding(seq_len, voc_size, d_model, dropout)
        self.encoders = Encoders(seq_len, d_model, d_ff, num_encoder, num_heads, dropout)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
        dim:
            input:
                input: [B, S]
            output:
                result: [B, S, V]
        '''
        pad_mask = get_attn_pad_mask(input, input, self.pad_idx)
        emb = self.emb(input, seg) # [B, S, D_model]
        output = self.encoders(emb, pad_mask) # [B, S, D_model]

        return output # [B, S, D_model]
```

논문에도 나와있듯 segment embedding, token embedding, positional embedding 세 개가 합쳐져서 input이 되고,
이를 transformer 인코더에 넣는 구조이다.

`BERTEmbedding`은 다음과 같은 구조를 갖고 있다.

```python
class BERTEmbedding(nn.Module):
    """
    Embeddings for BERT.
    It includes segmentation embedding, token embedding and positional embedding.
    I add dropout for every embedding layer just like the original transformer.
    """
    def __init__(self, seq_len: int=512, voc_size: int=30000, d_model: int=768, dropout: float=0.1) -> None:
        super(BERTEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model)
        self.tok_dropout = nn.Dropout(dropout)
        self.seg_emb = nn.Embedding(2, d_model)
        self.seg_dropout = nn.Dropout(dropout)
        self.pos_emb = PositionalEncoding(d_model, seq_len, dropout)

    def forward(self, tokens: torch.Tensor, seg: torch.Tensor):
        """
        tokens: [B, S]
        seg: [B, S]. seg is binary tensor. 0 indicates that the corresponding token for its index belongs sentence A
        """
        tok_emb = self.tok_emb(tokens) # [B, S, d_model]
        seg_emb = self.seg_emb(seg) # [B, S, d_model]
        pos_emb = self.pos_emb(tokens) # [B, S, d_model]

        return self.tok_dropout(tok_emb) + self.seg_dropout(seg_emb) + pos_emb  # [B, S, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: int=0.1):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(seq_len, d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [B, S]. x is tokens
        pos = torch.arange(self.seq_len, dtype=torch.long, device=x.device) # [S]
        pos = pos.unsqueeze(0).expand(x.size()) # [1, S] -> [B, S]
        pos_emb = self.emb(pos)
        return self.dropout(pos_emb) # [B, S, D_model]
```

다음은 pre-train의 task를 담당하는 `MaskedLanguageModeling`과 `NextSentencePrediction`이다.
각기 bert 몸체를 받아 동작하도록 만들어놨다.

```python
from typing import Optional
import torch
import torch.nn as nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert: nn.Module, voc_size:int=30000):
        super(MaskedLanguageModeling, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, voc_size)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''
        output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, voc_size]

        return output # [B, S, voc_size]

class NextSentencePrediction(nn.Module):
    def __init__(self, bert: nn.Module):
        super(NextSentencePrediction, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, 2)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''
        output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, 2]

        return output[:, 0, :] # [B, 2]
```

## Train

다음은 train 과정이다. 앞서 데이터를 불러올 때 `is_next`를 [batch, ] 형태로 불러왔다는 사실만 기억하자.

```python
import torch.nn as nn
import torch
import torch.optim as optim

def train(mlm_head: nn.Module, nsp_head: nn.Module, dataloader: torch.utils.data.DataLoader, mlm_optimizer: optim.Optimizer, nsp_optimizer: optim.Optimizer,
          criterion: nn.Module, clip: float):
    mlm_head.train()
    nsp_head.train()

    mlm_epoch_loss = 0
    nsp_epoch_loss = 0

    cnt = 0 # count length for avg loss
    for batch, (mlm_train, mlm_target, sent_emb, is_next) in enumerate(dataloader):
        # MLM task
        mlm_optimizer.zero_grad()
        mlm_output = mlm_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
        mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])
        mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE
        mlm_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlm_head.parameters(), 1)
        mlm_optimizer.step()

        # NSP task
        nsp_optimizer.zero_grad()
        nsp_output = nsp_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
        nsp_loss = criterion(nsp_output, is_next.to(DEVICE)) # no need for reshape target
        nsp_loss.backward()
        torch.nn.utils.clip_grad_norm_(nsp_head.parameters(), 1)
        nsp_optimizer.step()

        mlm_epoch_loss += mlm_loss.item()
        nsp_epoch_loss += nsp_loss.item()
        cnt += 1

    return mlm_epoch_loss / cnt, nsp_epoch_loss / cnt, 

def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module):
    model.eval()
    mlm_epoch_loss = 0
    nsp_epoch_loss = 0

    cnt = 0 # count length for avg loss
    with torch.no_grad():
        for batch, (mlm_train, mlm_target, sent_emb, is_next) in enumerate(dataloader):
            # MLM task
            mlm_output = mlm_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
            mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])
            mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE

            # NSP task
            nsp_optimizer.zero_grad()
            nsp_output = nsp_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
            nsp_loss = criterion(nsp_output.to(DEVICE), is_next.to(DEVICE)) # CE

            mlm_epoch_loss += mlm_loss.item()
            nsp_epoch_loss += nsp_loss.item()
            cnt += 1

    return epoch_loss / cnt

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

이후 다음을 통해 학습하면 된다.

```python
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
mlm_optimizer = optim.Adam(mlm_head.parameters(), lr=1e-4, betas=[0.9, 0.999], weight_decay=0.01)
nsp_optimizer = optim.Adam(nsp_head.parameters(), lr=1e-4, betas=[0.9, 0.999], weight_decay=0.01)
# you can also optimize the parameters like below:
# optim.Adam(list(mlm_head.parameters()) + list(nsp_head.parameters())
criterion = nn.CrossEntropyLoss()

import time
N_EPOCHS = 10

criterion = nn.CrossEntropyLoss()

for epoch in range(1, N_EPOCHS+1):
    start_time = time.time()
    mlm_loss, nsp_loss = train(mlm_head, nsp_head, dataloader, mlm_optimizer, nsp_optimizer, criterion, 1)
    
    end_time = time.time()
```