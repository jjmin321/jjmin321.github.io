---
title:  "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
toc: true
toc_sticky: true
permalink: /project/nlp/RoBERTa-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-09-26
---

## Introduction

본 논문은 Facebook AI Research (FAIR)에서 연구한 RoBERTa로, BERT가 underfitting되어있음을 확인하고 NSP objective와 minor design등의 변경을 통해 성능향상을 이끌어낸 논문이다. 원문과 깃헙은 다음에서 확인할 수 있다.

- [논문 보러가기](https://arxiv.org/pdf/1907.11692.pdf)
- [Github repo (FAIRSEQ)](https://github.com/pytorch/fairseq/tree/main/examples/roberta)

본 포스트의 순서는 논문의 순서와는 조금씩 다르며, 단순 번역보단 해설을 지향하였음을 미리 알린다. 따라서 디테일한 내용은 논문을 참고하길 권한다.

본 연구는 BERT에 대한 복제연구(replication study)로, **hyperparameter tuning과 학습 데이터의 양의 영향력을 평가**하는데 중점을 두었다. 본 연구를 통해 BERT가 **매우 undertraining** 되어있다는 사실을 알아냈으며, 이를 향상시키기 위한 방법으로 RoBERTa를 제안하도록 한다. 이는 BERT 이후의 모델들의 성능에 맞먹거나 능가하는 성능을 지녔다.

여기서 사용한 방법은 매우 간단하다.

1. 더 많은 배치사이즈와 데이터를 통해 더 오래 학습시킴
2. NSP objective 삭제
3. 더 긴 문장을 학습
4. 마스킹 패턴을 dynamic하게 변화

데이터를 통제하여 실험했을 때는 GLUE와 SQuAD에서 BERT보다 더 나은 성능을 보였다. 더 많은 데이터를 사용하였을 때는 XLNet과 비슷한 수준을 보였다. RoBERTa는 GLUE의 9개 실험 중 MNLI, QNLI, RTE, STS-B의 4개의 실험에서 SOTA를 달성하였다. SQuAD와 RACE에 대해서는 SOTA에 준하는 성능을 얻었다.

다음은 본 논문의 contribution이다.

1. BERT의 중요한 design choice와 학습 전략에 대해 소개하고, downstream task에서 더 나은 결과를 가져올 수 있는 방법에 대해 소개하였다.
2. 새로운 데이터셋인 CC-NEWS를 사용하여 더 많은 데이터를 사용하는 것이 성능을 향상시킨다는 것을 발견하였다.
3. 올바른 design choice하에서 MLM은 다른 방법론에 비해 성능이 전혀 밀리지 않는다.


## Background

2장에선 BERT에 대한 간략한 설명과 다음장에서 실험할 training choice에 대해 다루고 있다. BERT에 대해 잘 알지 못한다면 [다음](/project/nlp/bert-review/)을 참고하도록 하자.


## Experimental Setup

### Implementation

저자들이 FAIR소속이므로 FAIRSEQ를 사용하여 BERT를 재구현하였다. BERT에서 밝힌 hyperparameter 세팅 그대로를 따라했다고하며, peak learning rate와 warmup step의 수만 조정했다고 한다. 저자들은 추가적으로 Adam optimizer의 epsilon에 따라 학습이 민감해지는 것을 발견하였고, 이를 조절하여 더 나은 성능이나 안정성을 확보했다고 한다. 또한, 이와 비슷하게 배치사이즈가 큰 경우에는 $\beta _2=0.98$로 세팅하면 안정적인 학습이 가능한 것을 밝혀내었다고 한다.

길이의 경우 BERT와 같은 $T=512$로 설정하였다. 그러나 BERT에서 학습과정의 computation을 줄이기 위해 128의 짧은 문장 길이로 90%를 학습하는 테크닉은 사용하지 않고, full-length의 문장을 전부 학습시킨다.

학습의 경우 8개의 V100을 Infiniband로 연결한 환경에서 진행하였으며, DGX-1 machines에서 mixed precision floating point을 사용하였다고 한다.

### Data

BERT-style의 pretraining은 다량의 데이터에 크게 의존한다. [Baevski et al. (2019)](https://arxiv.org/abs/1903.07785)는 데이터의 사이즈를 키울수록 end-task의 성능이 올라감을 증명하였으며, 기존의 BERT보다 더욱 크고 다양한 데이터셋을 통해 학습시키는 노력들이 존재해왔다 ([Radford et al., 2019 (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); [Yang et al., 2019 (XLNet)](https://arxiv.org/abs/1906.08237), [리뷰 보기](/project/nlp/xlnet-review/); [Zellers et al., 2019](https://arxiv.org/abs/1905.12616)). 그러나 안타깝게도 여기에 사용된 모든 데이터가 공개되지는 않았기에 저자들은 ㅌ기존의 문헌들과 비교하기 적합한 양질의 데이터를 가능한 모으는데 집중했다고 한다.

여기서는 다양한 사이즈와 도메인을 갖는 5개의 영어 corpus를 수집하였으며, 총 160GB의 텍스트 파일을 사용했다고한다.

- BookCorpus + English WIKIPEDIA: BERT 학습에 사용됨 (16GB)
- CC-News: CommonCrawl News dataset에서 영어만 수집. 16년 9월부터 19년 2월 사이의 63M의 뉴스기사를 포함 (76GB).
- OpenWebText: WebText의 오픈소스 버전. GPT-2에서 사용. (38GB).
- STORIES: [Trinh and Le(2018)](https://arxiv.org/abs/1806.02847)에서 소개된 데이터. CommonCrawl data에서 이야기스러운 데이터만 필터링.


### Evaluation

이전의 연구들과 마찬가지로 본 연구의 pretrained model을 테스트한다. 테스트에 사용된 데이터는 GLUE, SQuAD, RACE의 세가지 벤치마크이다.

## Training Procedure Analysis

본 장을 통해 BERT의 어느 부분이 중요한지를 탐구한다. 모델의 구조는 BERT_BASE (L =
12, H = 768, A = 12, 110M params)와 같은 구조를 사용한다. 지금은 구조를 그대로 사용하였지만 구조의 변경 또한 future work의 중요한 부분이라고 한다.

### Static vs. Dynamic Masking

BERT의 성능은 임의의 단어를 마스킹하고 이를 복원하므로서 달성된다. 기존의 BERT 구현체는 전처리과정에서 마스킹을 딱 한번만 수행하기 때문에 single **static** mask을 하게된다. 매 에폭 때마다 똑같은 마스킹을 피하기 위해 데이터를 10번 복사하고 매번 다른 마스킹을 씌워 40 에폭동안 학습했다고 한다. 따라서 학습에 사용된 각 문장은 같은 마스킹으로 4번 (40 epochs/10 duplicate) 학습하게 된다.

이렇게 모델에 넣을 때마다 각기 다른 마스킹 패턴을 사용하는 것을 **dynamic masking**이라 한다. 이는 큰 데이터셋이나 더 많은 스텝을 사용하여 pretraining할 때 중요하다.

**Results**:  
아래 Table 1은 BERT와 여기서 사용한 static/dynamic masking을 비교한 결과이다. Static masking을 사용할 경우 기존의 BERT와 비슷했지만, dynamic masking을 사용할 경우 static masking보다 더 나은 성능을 보임을 확인하였다. 

![image](https://user-images.githubusercontent.com/47516855/136080135-b9527693-e13c-49ba-8d86-d178676aaa0e.png){: .align-center}{: width="400"}

따라서 앞으로의 실험 또한 dynamic masking으로 진행한다.

### Model Input Format and Next Sentence Prediction

BERT의 pretraining에선 두 개의 document segment를 concat을 인풋으로 받게된다. 그리고 MLM Objective에 더해 두 문장이 같은 문서에서 왔는지 아닌지를 판별하는 NSP loss를 보조로 사용하게 된다.

BERT의 경우 NSP loss를 학습하는데 중요한 요인 중 하나로 인식하였고, 실제로도 NSP를 제거할 경우 QNLI, MNLI, and SQuAD 1.1과 같은 태스크에서 심각한 성능하락이 일어남을 확인하였다. 그러나 최근 연구들은 NSP loss의 필요성의 의문을 제기하였다 ([Lample and Conneau, 2019](https://arxiv.org/abs/1901.07291); [Yang et al., 2019 (XLNet)](https://arxiv.org/abs/1906.08237); [Joshi et al., 2019 (SpanBERT)](https://arxiv.org/abs/1907.10529)).

이러한 상반된 의견을 검증하기 위해 다음과 같은 학습을 비교한다.

- `SEGMENT-PAIR + NSP`: BERT와 같은 형태. 인풋은 segment쌍으로 이루어지며, 각 segment는 여러 문장을 포함하고 있다.
- `SENTENCE-PAIR + NSP`: BERT가 segment의 결합인 것과는 달리 **문장**의 결합으로 이루어짐. 문장이기 때문에 토큰의 수가 작으므로 배치를 늘려 전체적인 토큰의 수를 `SEGMENT-PAIR + NSP`와 비슷하게 만듬.
- `FULL-SENTENCES`: 하나 이상의 문서에서 모은 문장을 사용. 문서 내 문장을 다 사용하면 다음 문서의 문장에서 채움. NSP는 제거.
- `DOC-SENTENCES`:  `FULL-SENTENCES`와 비슷하나 한 문서만을 사용하여 인풋을 구성. `SENTENCE-PAIR + NSP`와 마찬가지로 배치를 키움.  NSP는 제거.

**Results:**  
아래 Table 2는 본 실험의 결과를 보여준다.

![image](https://user-images.githubusercontent.com/47516855/136082332-141f69b0-0635-4399-aa56-6e8df4ab56ab.png){: .align-center}{: width="600"}

우선 BERT와 세팅이 가장 유사한 `SEGMENT-PAIR + NSP`와 `SENTENCE-PAIR + NSP`를 비교해보자. 이 둘은 segment/sentence 사용 외에는 전부 같다. 결과를 보면 문장을 사용할 경우 downstream 태스크에서 성능을 해치는 것으로 나타났다. 이는 모델이 long-range dependencies를 학습하지 못하기 때문인 것으로 보인다.

그 다음은 NSP loss를 제거한 `DOC-SENTENCES`를 살펴보자. 이는 BERT base의 성능을 능가하였으며, NSP loss를 제거할 경우 downstream 태스크 성능과 맞먹거나 능가하는 것을 확인하였다. 이는 BERT 논문에서 공개한 것과 반대되는 결과인데, 아마도 BERT의 실험에서는 loss term만 제거하고 input format은 그대로 남겨놨기 때문인 것으로 보인다.

마지막으로 여러 문서에서 문장을 수집하여 인풋을 구성하는 것보단 (`FULL-SENTENCES`) 하나의 문서에서 문장을 수집하여 인풋을 구성하는게 더 좋은 성능을 보이는 것으로 나타났다 (`DOC-SENTENCES`). 그러나 `DOC-SENTENCES`의 경우 배치사이즈가 가변적이므로 `FULL-SENTENCES`를 사용하여 남은 실험을 진행한다.

### Training with large batches

NMT에서 매우 큰 미니배치와 learning rate를 적절하게 상승시켜 학습하면 최적화 속도와 end-task 성능 둘 다 향상시킬 수 있다고 알려져있다. 최근 연구에 의하면 BERT 또한 이러한 세팅을 통해 학습시킬 수 있음을 보여주었다 ([You et al., 2019](https://arxiv.org/abs/1904.00962)).

BERT는 1M step과 256 배치를 이용하여 학습했는데, 이는 *gradient accumulation*를 사용할 경우 125K step과 2K 배치, 31K step과 8K 배치를 사용한 것과 동일한 computational cost가 된다.

gradient accumulation에 대해서는 [다음](https://velog.io/@nawnoes/Pytorch%EB%A1%9C-%ED%81%B0-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%EC%8B%9C-%EC%96%B4%EB%96%BB%EA%B2%8C-%EB%B0%B0%EC%B9%98-%EC%82%AC%EC%9D%B4%EC%A6%88%EB%A5%BC-%EB%8A%98%EB%A6%B4%EC%88%98-%EC%9E%88%EC%9D%84%EA%B9%8C)을 참고.
{: .notice--info}

![image](https://user-images.githubusercontent.com/47516855/136089171-f1a5b849-3963-4cf4-a931-386d41f26d13.png){: .align-center}{: width="400"}

Table 3에서는 step수는 유지한 채 배치사이즈를 조절함에 따른 PPL과 end-task 성능을 비교한 것이다. 이를 통해 배치 사이즈를 키울수록 성능이 올라감을 확인할 수 있다. 또한, 배치사이즈를 키우면 병렬화하기 쉽다는 장점도 있다. 큰 배치 사이즈는 하드웨어가 받쳐주지 않더라도 gradient accumulation을 사용하여 달성가능하다. 이는 FAIRSEQ에도 구현되어 있다. 따라서 앞으로의 실험은 8K 배치사이즈로 진행한다.

특히 [You et al., 2019](https://arxiv.org/abs/1904.00962)의 경우에는 배치사이즈를 32K까지 늘렸지만, RoBERTa에서는 배치 사이즈에 대한 한계를 후속 연구로 남겨놓았다.

### Text Encoding

BERT의 경우 BPE를 사용하여 subword segmentation을 진행한다. BPE는 통상적으로 10K-100K의 subword를 갖지만, RoBERTa에서 다루는 것과 같이 크고 다양한 corpus를 다루는 경우에는 사전의 상당수가 유니코드 문자로 이루어지게 된다. GPT-2에서는 BPE를 byte-level로 이용하였는데, 이 덕에 50K의 사전 크기로 [UNK] 토큰 없이 모든 인풋을 인코딩할 수 있게 되었다.

BERT는 휴리스틱한 알고리즘을 통해 전처리하여 30K의 사전으로 구성한다. RoBERTa의 경우에는 GPT-2를 따라서 byte-level BPE를 사용한다. 이를 통해 어떠한 추가적인 전처리/tokenization 없이 50K의 사전을 구성하였고, BERT base, BERT large에 대해 약 15M/20M의 파라미터를 추가하게 된다.

BPE와 byte-level BPE가 성능상엔 큰 차이를 보이지 않고, 오히려 byte-level BPE가 더 안 좋은 모습을 보여주지만 그럼에도 불구하고 BBPE의 universal encoding scheme이 이러한 성능 차이를 상쇄할만큼의 장점이 있기 때문에 이를 이용한다고 한다.

## RoBERTa

이러한 방법론을 묶어 **RoBERTa** (**Ro**bustly optimized **BERT** **a**pproach)라고 부른다. RoBERTa는 앞서 설명한 dynamic masking, NSP loss를 제거한 `FULL-SENTENCES`, 큰 mini-batch, 기존보다 더 큰 BBPE를 사용하여 학습한다.

추가적으로 논문에서 덜 부곽된 두 가지 주요 요소가 있는데, 이는 학습에 사용된 데이터의 크기와 학습과정에서 집어넣은 데이터의 양이다 (즉, epoch * # data). 에를들어 XLNet의 경우 BERT보다 10배나 큰 데이터를 사용하여 학습하였고, 배치사이즈는 8배 더 많고 optimization step의 경우 절반이므로 총 4배 더 많은 데이터를 학습하였다.

이러한 데이터의 중요성과 모델의 design choice (e.g. pretraining objective)의 영향을 분리하여 비교하기 위해 RoBERTa를 BERT Large와 같은 구조 ($L=24, H=1024, A=16, 335M \text{parameters}$)와 같은 구조로 실험을 진행하였다.  100K steps까지는 BookCorpus와 위키피디아 데이터를 이용하여 학습시켰고, 1024개의 V100 GPU를 이용하여 학습하였다.

**Results**  

![image](https://user-images.githubusercontent.com/47516855/136686079-de9d0cd3-f6d3-40f9-a77b-f47a84f9a179.png){: .align-center}{: width="600"}

위 Table 4는 실험결과이다. 학습데이터를 통제하였을 때에는 RoBERTa가 BERT Large에 비해 큰 폭으로 성능이 향상됨을 확인할 수 있다. 이는 앞서 Section 4에서 탐구한 design choice의 중요성을 재확인 시켜주는 것이라 할 수 있다.

그 다음은 Section 3.2에서 탐색한 세 개의 데이터를 추가로 이용한 결과이다. 학습 step의 수는 이전 (100K)과 그대로 유지하였고, 총 160GB의 데이터를 이용하여 학습하였다. 그 결과 성능의 향상을 확인할 수 있었으며, pretraining에 있어 데이터의 크기와 다양성의 중요성을 검증하였다 (단, 엄밀히 말해 이 둘을 분리하진 않았으므로, 이에 대한 분석을 후속연구로 남겨놓는다고 한다).

마지막으로, RoBERTa를 현저하게 많이 학습시킨 결과를 보자. 기존의 100K에서 300K, 500K까지 늘려서 실험하였다. 이 또한 성능향상이 있었음을 확인하였고, 300K와 500K는 대부분의 task에서 XLNet Large보다 더 나은 결과를 보여주었다.

다음은 세가지 벤치마크 데이터셋에 대한 실험결과이다. 평가는 500K step으로 진행하였다.

### GLUE Results

GLUE에서는 두 가지의 fine-tuning으로 진행한다. 

첫번째 세팅인 (*single-task, dev*)에서는 각 GLUE 태스크에 대해 독립적으로 학습한다. 하이퍼파라미터의 경우 제한적으로만 사용하며, 아래와 같다.

- $\text{learning rate} \in \{1e−5, 2e−5, 3e−5\}$
- $\text{batch sizes} \in \{16, 32\}$
- 스텝의 첫 6%만 linear warmup, 그 후 linear decay 0

그리고 Dev set에 대해 10 에폭만큼 학습시키고 early stopping을 이용하여 학습하였다. 나머지 하이퍼파라미터는 pretraining과 똑같이 유지한다. 성능평가는 5개의 모델을 random initialization을 통해 학습시킨 후, 이들의 중앙값으로 보고한다. 앙상블은 사용하지 않는다.

두번째 세팅은 (*ensembles, test*)로, test set에 학습시킨 결과를 GLUE 리더보드에 있는 결과들과 비교한다. 대부분의 리더보드 제출결과가 multi-
task finetuning을 쓴 것과는 다르게 여기서는 오로지 single-task finetuning만 사용한다. RTE, STS, MRPC의 경우엔 MNLI single-task model를 에서 fine-tuning하는게 RoBERTa를 fine-tuning하는 것보다 더 좋은 것으로 나타났다. 여기서는 좀 더 넓은 범위의 hyperparameter search를 진행하였고, 한 태스크 당 5-7개의 모델에 대해 앙상블을 사용하였다.

![image](https://user-images.githubusercontent.com/47516855/136688036-df547cae-5e0b-4938-afc0-cf1af15e82fd.png){: .align-center}{: width="600"}

**Task-specific modifications**  
GLUE 태스크 중 QNLI와 WNLI는 경쟁력을 갖추기 위해 task-specific finetuning을 사용하였다.

QNLI:  
최근 GLUE 리더보드 결과들을 살펴보면 pairwise ranking formulation을 사용한 것을 볼 수 있다. pairwise ranking formulation는 학습셋으로부터 후보 답변들을 모은 뒤 이들끼리 비교, 하나의 (question, candidate)를 positive로 분류하는 기법이다 ([Liu et al., 2019b](https://arxiv.org/abs/1901.11504), [a](https://arxiv.org/abs/1904.09482); [Yang et al., 2019 (XLNet)](https://arxiv.org/abs/1906.08237)). 그러나 이러한 방법은 태스크를 매우 간단하게 만들 순 있지만 BERT와의 직접적인 비교는 어려워진다. 따라서 test set에 대해 pairwise ranking formulation를 사용하고, BERT와의 공정한 비교를 위해 dev set은 순수한 classification 문제로 만들어 푼다.

WNLI:  
NLI-format의 데이터는 작업하기가 까다로우므로 이 대신 SuperGLUE의 WNLI를 사용한다. WNLI는 query 대명사와 참조의 span을 제공해준다. [Kocijan et al. (2019)](https://arxiv.org/abs/1905.06290)에서 사용한 margin ranking loss을 이용하며, 인풋 문장은 spaCy를 사용하여 명사구를 추출한다. 그러나 이런 방법을 사용할 경우 데이터의 절반 가까이 사용할 수 없다는 단점이 생긴다.

**Results**  
아래는 본 실험에 대한 결과이다.

![image](https://user-images.githubusercontent.com/47516855/136696586-fcd933b2-39bc-4821-9444-ab2cffc3a344.png){: .align-center}{: width="800"}

첫번째 세팅인 (*single-task, dev*) 에서는 GLUE의 9개 태스크에 대해 모두 SOTA를 달성하였다. BERT와 동일한 MLM objective를 사용했음에도 BERT LARGE와 XLNet LARGE를 모두 뛰어넘는 성과를 보여준다. 이를 통해 모델 구조나 objective이 데이터 사이즈나 학습시간과 같은 사소한 디테일보다 더 중요한 것인지에 대한 의문을 보여준다.

두번째 세팅인 (*ensembles, test*)에선 9개 중 4개의 태스크에서 SOTA를 달성하였다. multitask learning에 의존하지 않고도 달성했다는 점이 상당히 주목할만하다고 볼 수 있다.

### SQuAD Results

BERT와 XLNet이 추가적인 QA 데이터셋을 이용한 반면, 여기서는 오직 SQuAD 학습데이터만 사용하여 진행한다. XLNet이 custom layer-wise learning rate schedule를 이용한 반면 여기서는 모든 레이어를 똑같은 learning rate를 사용하여 fine-tuning한다.

SQuAD v1.1은 BERT와 같은 절차로 fine-tuning을 진행하고, SQuAD v2.0의 경우 이 질문이 대답가능한지 분류하는 기능을 추가한다. 그리고 이를 jointly 학습한다.

**Results:**  
SQuAD에 대한 실험결과는 아래 Table 6에 나타나있다. 

![image](https://user-images.githubusercontent.com/47516855/136798626-7d84a5e0-2a05-4f77-ac51-b7003d336c79.png){: .align-center}{: width="400"}

SQuAD v1.1의 dev set에서 RoBERTa는 XLNet과 같은 성능을 보였으며, SQuAD v2.0의 dev set에서는 SOTA를 달성하였다.

SQuAD v2.0 리더보드를 살펴보면 BERT와 XLNet 기반의 시스템이 상위권을 차지하고 있는데, 이 둘 모두 외부데이터를 이용한 모델들이다. single model 중에서는 단 하나만 제외하고 나머지는 능가하였으며, data augmentation를 사용하지 않은 모델들 중에선 최고점을 달성하였다.

### RACE Results

RACE는 지문과 질문, 그리고 네개의 보기를 주고 정답을 맞추게끔 되어있다. 이를 위해 각각의 보기를 지문, 질문과 concat하고, `[CLS]` token을 이용하여 정답인지 아닌지를 예측하도록 하였다.질문, 정답의 경우 128이하로 맞춰주어 지문과 합쳤을 때 512이하의 길이를 갖게끔 하였다. 아래는 이에 대한 실험 결과이다.

![image](https://user-images.githubusercontent.com/47516855/136814238-2a3a97d3-ef2f-4e8d-b352-b635ead97ab7.png){: .align-center}{: width="400"}


## Conclusion

본 논문을 통해 BERT를 학습하는 과정에서의 다양한 design choice를 실험하고 연구하였다. 이를 통해 BERT의 성능이 모델을 더 오래, 더 많은 배치를, 더 많은 데이터를 이용할 때 향상할 수 있음을 확인하였다. 또한, NSP 제거, 긴 문장에 대한 학습, dynamic masking과 같이 사전 연구들에서 무시되었던 design choice들 또한 모델의 성능을 높일 수 있음을 확인하였다. 이를 이용해 학습한 RoBERTa는 GLUE, RACE, SQuAD에서 SOTA를 달성하였다.

{: .align-center}{: width="600"}