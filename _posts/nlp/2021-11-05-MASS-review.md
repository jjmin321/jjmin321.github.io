---
title:  "MASS:Masked Sequence to Sequence Pre-training for Language Generation review"
toc: true
toc_sticky: true
permalink: /project/nlp/MASS-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-11-27
---

## 들어가며

MASS는 난징대학교와 Microsoft에서 개발한 lanuge model로, 2019년 ICML에 소개되었다. MASS는 기존의 language model이 encoder/decoder only 모델을 쓰는 것과는 달리, seq2seq 구조를 이용하여 효율적인 language generation task를 진행할 수 있게 해준다. MASS를 이용하여 논문의 저자들은 low-resource 환경에서의 번역, 요약, 대화의 세가지 generation task에서 SOTA를 달성하였다. 

- [원문 보러가기](https://arxiv.org/pdf/1905.02450.pdf)
- [MASS repository 보러가기](https://github.com/microsoft/MASS)

## 1. Introduction

우리가 잘 알고있는 pre-training/fine-tuning 구조는 target에 대한 학습 데이터가 부족한 경우 널리 사용된다. Computer vision의 경우 매우 큰 스케일의 ImageNet으로부터 pre-train하고, object detection, segmantation 등등의 downstream task에 적용하기도 한다. 이로부터 영향을 받아 자연어처리 영역에서도 ELMo, GPT, BERT등의 방법론들이 인기를 끌었고, SOTA를 달성하였다.

그러나 language understanding과는 달리 language generation은 주어진 인풋에 조건부인 자연어 문장을 생성하는 것이 목적이다. 이에는 neural machine translation (NMT), conversational response generation, text summarization 등의 작업이 있다. Language generation tasks는 **일반적으로 데이터가 부족하며(data-hungry), low-resource이거나 심지어 zero-resource**인 경우도 흔하다. BERT와 같은 pre-training 방법론을 language generation에 바로 적용하는 것은 불가능한데, 이는 BERT가 NLU를 위해 설계되어있기 때문에 **오직 하나의 encoder/decoder**만 갖기 때문이다. 그러므로 seq2seq 구조를 갖는 generation task에는 pre-training 방법을 디자인하는 것이 매우 중요한 작업이라 할 수 있다.

이러한 이유들로 인해 본 저자들은 MASS(MAsked Sequence to Sequence learning)를 통해, generation에 어울리는 pre-training objective를 제안하고 있다. MASS는 앞서 언급했듯 seq2seq 구조를 갖는다. Encoder에서는 문장의 segment(연속적인 토큰)를 마스킹한 것을 입력으로 받고, decoder에서는 encoder에 attention하여 이를 예측하도록 한다.

BERT와 달리 encoder와 decoder 모두가 존재하기 때문에 다음의 두 단계를 통해 주의를 기울여 encoder와 decoder를 동시에 학습하도록 한다.
1. Encoder에서 마스킹한 부분을 예측해야 한다. Decoder가 이를 예측하도록 만들려면 MASS는 encoder로 하여금 **마스킹되지 않은 토큰을 이해하는 능력**을 부여해야 한다.
2. Encoder에서 마스킹되지 않은 부분은 decoder에서 마스킹하게 된다. 따라서 MASS는 decoder가 **이전 토큰**에 의존하기보다는 encoder에서 **정보를 뽑아**다 쓰게끔 만든다. 이를 통해 encoder와 decoder가 동시에 학습하도록 만든다.

저자들이 밝히는 주요 contribution은 다음과 같다.
1. language generation task에 효과적인 MASS를 제안함
2. NMT, 요약, 대화와 같은 language generation에서 상당한 성능 향상을 이끔

## 2. Related work

대부분의 language model들이 하는 이야기는 비슷하므로 (self-supervised, 데이터 양, etc.) MASS가 이전의 다른 모델과 갖는 상이함에 중점을 맞춰 리뷰해보자.

### 2.1. Sequence to Sequence Learning

Seq2seq은 AI분야에서 어려운 태스크로 여겨지며, NMT, 요약, 대화, QA 등의 다양한 genetation task를 다뤄왔다. Seq2seq은 딥러닝의 발전으로 최근 여러 연구자들의 관심을 받고 있으나 사용 가능한 데이터가 매우 적다는 단점이 있다. 따라서 그 무엇보다도 pre-training/fine-tuning 구조가 절실하며, 본 논문에서 초점을 맞추고 있는 것과 정확히 일치한다.

### 2.2. Pre-training for NLP tasks

앞서 언급했듯 이 부분은 최근 language model관련 논문이라면 한번쯤은 들을만한 이야기들로 적혀져있다. 자연어처리 분야에선 pre-training을 사용하여 더 나은 language representation을 얻는 것을 목표로 한다. NLU에 경우 크게 **feature-based**와 **fine-tuning-based**로 나뉘어진다. 

Feature-based의 경우 pre-training을 이용하여 downstream task에서 사용할 representation과 feature를 얻는데 초점이 맞춰져 있다. 대표적으로는 word2vec과 같은 word-level, doc2vec, Skip-thought vectors, Quick Thought과 같은 sentence-level의 representation, 마지막으로 ELMo, CoVe와 같이 context가 잘 반영된 feature를 잡는 representation이 있다.

Fine-tuning-based의 경우 우리가 잘 알고, 주류가 된 방법론으로, 모델을 pre-training한 후 supervised data에 대해 fine-tuning하는 형태로 되어있다.

또한, MASS와 같이 generation을 위해 pre-training encoder-decoder을 사용하는 연구들도 존재한다. Pre-training에 대한 연구의 서막을 열었던 Dai & Le (2015)와 [Ramachandran et al. (2016)](https://www.semanticscholar.org/paper/Unsupervised-Pretraining-for-Sequence-to-Sequence-Ramachandran-Liu/85f94d8098322f8130512b4c6c4627548ce4a6cc?p2df)의 연구같은 경우 auto-encoder를 사용하여 pre-training encoder-decoder를 학습하였다. 이를 통해 성능의 향상은 입증할 수 있었으나, 이는 제한되고, 일반화하기 어려웠으며, BERT처럼 가시적인 성과를 보이지는 못하였다. 최근들어서는 XLM이 unsupervised NMT에서 성능향상을 이끌어 내었으나 encoder와 decoder가 분리되어 학습되기 때문에 이 둘 사이의 cross-attention을 이끌어 내지 못하였으며, 그 결과 seq2seq기반의 language generation에선 sub-optimal하다고 말할 수 있다.

이전 연구들과는 다르게, MASS는 이 둘을 동시에 학습하기 위해 심혈을 기울여 모델을 디자인했고, 대부분의 language generation task에 적용이 가능하였다.

## 3. MASS


{: .notice--info}
{: .align-center}{: width="500"}
