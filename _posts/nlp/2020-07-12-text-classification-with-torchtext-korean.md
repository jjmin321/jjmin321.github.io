---
title:  "TEXT CLASSIFICATION WITH TORCHTEXT"
excerpt: "TEXT CLASSIFICATION WITH TORCHTEXT 번역 자료"
toc: true
toc_sticky: true

categories:
  - PyTorch
  - NLP
tags:
  - torchtext
use_math: true
last_modified_at: 2020-07-12
---

# Introduction

이번 튜토리얼은 `torchtext` 내에 있는 text classification datasets을 어떻게 사용할 수 있는지 보여주고, 다음을 포함합니다.

```
- AG_NEWS,
- SogouNews,
- DBpedia,
- YelpReviewPolarity,
- YelpReviewFull,
- YahooAnswers,
- AmazonReviewPolarity,
- AmazonReviewFull
```

이 예제는 이러한 `TextClassification` datasets 중 하나를 이용하여 지도학습 분류 알고리즘을 어떻게 학습시키는지 보여줍니다.

> 본 튜토리얼은 파이토치 홈페이지 내의 튜토리얼을 번역한 자료입니다. 다수 의역이 되어있습니다. 역자주의 경우 지금과 같이 citation을 통해 남기도록 하겠습니다. 원문은 [다음](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)과 같습니다. 또한, 본 코드는 colab을 통해 실행할 수 있습니다. [https://github.com/InhyeokYoo/PyTorch-tutorial-text/blob/master/TEXT\_CLASSIFICATION\_WITH\_TORCHTEXT.ipynb](https://github.com/InhyeokYoo/PyTorch-tutorial-text/blob/master/TEXT_CLASSIFICATION_WITH_TORCHTEXT.ipynb)

# Load data with ngrams

A bag of ngrams features는 국소적인(local) 단어 순서에 대한 특정한 정보를 사로잡기 위해 이용됩니다. 실전에선 bi-gram이나 tri-gram은 단어의 그룹으로서 제공되어 단순히 하나의 단어만 이용하는 것보다 더 유용한 정보를 제공해줍니다.

```python
"load data with ngrams"
Bi-grams results: "load data", "data with", "with ngrams"
Tri-grams results: "load data with", "data with ngrams"
```

`TextClassification` Dataset은 ngrams 메소드를 지원합니다. ngrams를 2로 세팅함으로써 dataset에 있는 예제 텍스트는 single words에 bi-grams string을 더한 list가 될 것입니다.

```python
import torch
import torchtext
from torchtext.datasets import text_classification
# from torchtext.datasets import AG_NEWS # 다른 방법으로 importing하기
import os

NGRAMS = 2 # final로 선언합니다.

if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None
)
BATCH_SIZE = 16
device = torch.device('cpu' if torch.cuda.is_available() == False else 'cuda')
print(device)
```

# Define the model

이 모델은 [EmbeddingBag](https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag) 레이어와 linear layer(밑에 그림 참고)로 이루어져 있습니다. `nn.EmbeddingBag`은 embedding 주머니(bag)의 평균값을 계산합니다. 튜토리얼의 text entries는 서로 다른 길이를 갖고 있습니다. `nn.EmbeddingBag`은 여기서 padding이 필요 없는데, 이는 offsets안에 텍스트의 길이가 저장되어 있기 때문입니다.

추가적으로, `nn.EmbeddingBag`은 평균을 즉석에서 누적하므로, `nn.EmbeddingBag`은 tensor의 시퀀스를 다룰 때 메모리 효율과 성능을 향상시킬 수 있습니다.

> `nn.EmbeddingBag`은 `nn.Embedding` 후 `torch.mean(dim=0)`와 동일합니다. 그러나, `EmbeddingBag`은 시간과 메모리면에서 훨씬 효율적입니다.  
> Input은 `Input` (LongTensor), `offsets` (LongTensor, optional)으로,

-   만일 `input`이 (_B_, _N_)으로 2D이면, 각 길이가 고정으로 `N`인 `B`개의 주머니 (sequence)로 취급합니다. 이는 `mode`에 따라 합산된 `B`개의 값을 반환할 것이며, `offsets`는 이 경우에 무시됩니다.
-   만일 `input`이 (_N_)의 1D라면, 여러 주머니(sequences)의 concatenation으로 취급합니다. `offsets`는 1D tensor로, `input`안의 각 주머니가 시작하는 지점의 index를 포함합니다. 따라서, (_B_) 차원의 `offsets`에 대해서, `input`은 `B`개의 주머니를 갖는다고 할 수 있습니다. 빈 주머니(즉, 0짜리 길이)는 0으로 채워진 tensor를 반환합니다.

![](https://pytorch.org/tutorials/_images/text_sentiment_ngrams_model.png)

```python
import torch.nn as nn
import torch.nn.functional as F

class TextSentiment(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super(TextSentiment, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
```

# Initiate an instance

AG NEWS 데이터셋은 네 개의 레이블이있고, 따라서 클래스의 갯수는 4가 됩니다.

```python
1 : World
2 : Sports
3 : Business
4 : Sci/Tec
```

사전의 크기(vocab size)는 사전(vocab; 개별 단어와 ngrams를 포함)의 길이와 같습니다. 클래스의 개수는 레이블의 개수와 같고, AG\_NEWS의 경우 4가 됩니다.

```python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
```

```python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
```

# Functions used to generate batch

다음 요소가 서로 다른 길이를 갖기때문에, 사용자 함수인 generate\_batch()를 이용하여 data batches와 offsets를 생성합니다. 이 함수는 `torch.utils.data.DataLoader`안에 있는 `collate_fn`으로 전달됩니다. `collate_fn`의 인풋은 batch\_size만큼의 크기를 갖는 tensors로 이루어진 list이고, `collate_fn` 함수는 mini\_batch로 나눕니다. `collate_fn`은 최고 레벨의 함수로 선언되는 것에 주목합시다. 이는 이 함수가 각 worker에서 사용가능하게 합니다.

원본 data batch input의 텍스트는 list로 감싸져있고, 하나의 tensor로 concat되어 `nn.EmbeddingBag`의 input이 됩니다. Offset은 text tensor내 개별 sequence의 시작점의 인덱스를 나타내는 텐서입니다.

> `torchtext.datasets.TextClassificationDataset`의 data에 대한 설명은 다음과 같습니다: label/tokens의 튜플로 이루어진 리스트로, tokens은 string tokens를 numericalizing한 것이고, label은 integer입니다. `[(label1, tokens1), (label2, tokens2), (label2, tokens3)]`

```python
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.tensor.cumsum은 정해진 dimension dim
    # 안에 요소의 누적합을 반환합니다.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label
```

# Define functions to train the model and evaluate results.

`torch.utils.data.DataLoader`는 손쉽게 병렬 데이터 로딩을 가능케하고 PyTorch 사용자에게 권장됩니다. 여기서는 `DataLoader`를 사용하여 AG\_NEWS 데이터셋을 불러오고 모델로 보내 training과 validation을 해보도록 하겠습니다.

```python
from torch.utils.data import DataLoader

def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0

    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # 마지막엔 softmax layer를 추가해도 좋습니다.
        train_acc += (output.argmax(1) == cls).sum().item()

    # Learning rate 조정
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            # 마지막엔 softmax layer를 추가해도 좋습니다.
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
```

> 살짝 헷갈리므로 불러오는 데이터의 형식을 한번 살펴보도록 하겠습니다. Tractable 하기위해 `shuffle=False`로 설정하겠습니다.

```python
data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
for i, (text, offsets, cls) in enumerate(data):
    print(f"text: {text}, len: {len(text)}")
    print(f"offsets: {offsets}, length:{len(offsets)}")
    print(f"cls: {cls}, len: {len(cls)}")
    print("="*200)
    if i == 3:
        break
```

```
text: tensor([    572,     564,       2,  ..., 1194110,  300136,   10278]), len: 1432
offsets: tensor([   0,   57,  140,  219,  298,  383,  478,  571,  668,  843,  904,  991,
        1102, 1163, 1262, 1369]), length:16
cls: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), len: 16
========================================================================================================================================================================================================
text: tensor([   6312,    1934, ...,  5641,    4893,    1818]), len: 966
offsets: tensor([  0,  51, 154, 217, 278, 335, 390, 433, 500, 559, 624, 685, 740, 795,
        856, 907]), length:16
cls: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), len: 16
========================================================================================================================================================================================================
text: tensor([   2226,    4547,       5,  ..., 1083638,   24519,   11559]), len: 1552
offsets: tensor([   0,   71,  138,  189,  252,  353,  450,  587,  738,  907, 1010, 1077,
        1196, 1321, 1402, 1467]), length:16
cls: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), len: 16
========================================================================================================================================================================================================
text: tensor([   12,  1001,     3,  ..., 53174,    37,    37]), len: 1296
offsets: tensor([   0,   75,  110,  151,  208,  271,  354,  403,  440,  479,  522,  641,
         818,  911, 1078, 1187]), length:16
cls: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), len: 16
```

> 이로 미루어 알 수 있는 사실은 16개 (`len(offsets)` or `len(cls)`)의 sequences (bags)가 text 안에 담겨있고, offsets는 각 bag의 시작점을 알려주고 있습니다. 각 bag의 길이는 모두 서로 다릅니다. 본 데이터 셋은 `[Batch x length]`를 반환하는 대신, `[BATCH_SIZE]` concat된 bag과 offsets를 반환합니다.   참고로 원래의 dataset은 다음과 같이 생겼습니다.

```python
data = DataLoader(train_dataset, shuffle=False)
for i, (x) in enumerate(data):
    print(x, x[1].size())
    print("="*200)
    if i == 3:
        break
```

```
# Result
[tensor([2]), tensor([[    572,     564,       2,    2326,   49106,     150,      88,       3,
            1143,      14,      32,      15,      32,      16,  443749,       4,
             572,     499,      17,      10,  741769,       7,  468770,       4,
              52,    7019,    1050,     442,       2,   14341,     673,  141447,
          326092,   55044,    7887,     411,    9870,  628642,      43,      44,
             144,     145,  299709,  443750,   51274,     703,   14312,      23,
         1111134,  741770,  411508,  468771,    3779,   86384,  135944,  371666,
            4052]])] torch.Size([1, 57])
```

> 모든 데이터를 다 확인할 순 없지만 앞서 `collate_fn`을 이용한 결과와 위의 결과를 비교하면 `BATCH_SIZE`만큼 데이터를 concat하여 사용함을 확인할 수 있습니다.

# Split the dataset and run the model

원본 AG\_NEWS 데이터에는 validation set이 없으므로 training set을 0.95의 비율로 나누겠습니다. 이를 위해 우리는 [torch.utils.data.datasets.random\_split](https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split)을 사용하겠습니다.

[CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss)는 nn.LogSoftmax()와 nn.NLLLoss()를 하나로 합쳐놓은 것입니다. 이는 C개의 클래스를 분류하는 문제를 학습시킬 때 유용합니다. [SGD](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html)는 stochastic gradient descent를 optimizer로서 구현한 것입니다. 초기 learning rate값은 4.0으로 설정되어있습니다. [StepLR](https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR)은 여기서 epochs에 따라 learning rate를 조절하기 위해 사용되었습니다.

```python
import time
from torch.utils.data.dataset import random_split

N_EPOCHS = 5
min_valid_loss = float('inf')   # 왜 있는지 모르겠음

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
# StepLR은 step_size마다 lr을 gamma만큼 줄여서 작동한다: 이 경우, 4.0, 3.6, ..., 로 줄어든다.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
# random_split(dataset: Dataset, lengths: 'sequence')
# Arguments:
#        dataset (Dataset): Dataset to be split
#        lengths (sequence): lengths of splits to be produced
sub_train_, sub_valid_ = random_split(dataset=train_dataset, lengths=[train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    # f string
    print(f'Epoch: {epoch + 1}', f" | time in {mins} minutes, {secs} seconds")
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
```

```
Epoch: 1  | time in 0.13333333333333333 minutes, 8 seconds
    Loss: 0.0126(train)    |    Acc: 93.3%(train)
    Loss: 0.0001(valid)    |    Acc: 93.2%(valid)
Epoch: 2  | time in 0.13333333333333333 minutes, 8 seconds
    Loss: 0.0071(train)    |    Acc: 96.2%(train)
    Loss: 0.0001(valid)    |    Acc: 93.1%(valid)
Epoch: 3  | time in 0.13333333333333333 minutes, 8 seconds
    Loss: 0.0039(train)    |    Acc: 98.1%(train)
    Loss: 0.0000(valid)    |    Acc: 93.0%(valid)
Epoch: 4  | time in 0.13333333333333333 minutes, 8 seconds
    Loss: 0.0023(train)    |    Acc: 98.9%(train)
    Loss: 0.0001(valid)    |    Acc: 93.4%(valid)
Epoch: 5  | time in 0.13333333333333333 minutes, 8 seconds
    Loss: 0.0014(train)    |    Acc: 99.4%(train)
    Loss: 0.0000(valid)    |    Acc: 93.8%(valid) 
```

# Evaluate the model with test dataset

```python
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
```

# Test on a random news

제일 좋은 모델을 사용하여 골프 뉴스를 테스트해보자. label정보는 [이곳](https://pytorch.org/text/datasets.html?highlight=ag_news#torchtext.datasets.AG_NEWS)에서 확인할 수 있다.

```python
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
```