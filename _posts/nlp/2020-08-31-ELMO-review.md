---
title:  "Deep contextualized word representations (ELMO) review"
excerpt: "ELMO 정리"
toc: true
toc_sticky: true
permalink: /project/nlp/elmo-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling

use_math: true
last_modified_at: 2020-08-31
---

# Intro.

NLP 스터디에서 ELMo를 공부할 차례가 되서 간략하게 정리해보았다. 원 논문은 [다음](https://arxiv.org/abs/1802.05365)에서 확인할 수 있다.

---

# 3. ELMo: Embeddings from Language Models

ELMo word representations
- input sentence 전체에 대한 functions
- character convolutions

## 3.1. Bidirectional language models

N tokens의 sequence ($t _1, t _2, ..., t _N$)에 대해, 
- 여태까지의 SOTA는 forward language model로 sequence의 확률을 계산
- context-independent
- 즉, 다음과 같음:
$$
\begin{align}
p(t _1, t _2, ..., t _N) = \Pi^{N} _{k=1} p(t _k \lvert t _1, t _2, ..., t _{k-1})
\end{align}
$$

- 각 position $k$에서는 context-dependent representation $\overrightarrow {h^{LM} _{k, j}}$를 계산
  - $j=1, ..., L$
- top layer LSTM의 결과 $\vec {h^{LM} _{k, j}}$는 Softmax layer를 통해 다음 토큰 $t _{k+1}$을 예측하는데 사용된다.

- Backward LM은 sequence를 반대순서로 돌고, 앞선 시점의 문맥을 통해 이전 토큰을 예측한다는 점만 빼면 정방향 LM과 비슷
- 즉,  

$$
\begin{align}
p(t _1, t _2, ..., t _N) = \Pi^{N} _{k=1} p(t _k \lvert t _{k+1}, t _{k+2}, ..., t _{N})
\end{align}
$$

- BiLM은 정방향과 역방향 LM 두개 모두를 combine
- 그리고 정방향과 역방향의 log likelihood를 jointly maximize  

$$
\begin{align}
\sum^N _{k=1} (\log p(t _k \lvert t_1, ..., t _{k-1}; \Theta _x, \overrightarrow \Theta _{LSTM}, \Theta _s) \\
+(\log p(t _k \lvert t _{k+1}, ..., t _{N}; \Theta _x, \leftrightarrow \Theta _{LSTM}, \Theta _s))
\end{align}
$$

- token representation $\Theta _x$와 Softmax layer $\Theta _s$는 LSTM끼리 공유 
- LSTM의 parameter ($\overrightarrow \Theta _{LSTM}, \leftrightarrow \Theta _{LSTM}$)는 분리를 유지
- 방향사이의 weight를 공유한다는 점을 제외하면 전체적으로 Peters et al. (2017)와 유사함

## 3.2. ELMo

ELMo는 biLM의 중간 레이어의 표현의 *task-specific*한 조합이다. 
- 즉, 중간 중간 layer결과물을 combine한다는 뜻

각 토큰 $t _k$에 대해, *L*-layer biLM은 $2L+1$개의 표현을 배운다.  

$$
\begin{align}
R _k &= \{ \mathbf{x}^{LM} _k, \overrightarrow {\mathbf{h}^{LM} _{k, j}}, \overleftarrow {\mathbf{h}^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ \mathbf{h}^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

- $\mathbf{h}^{LM} _{k, 0}$: token layer (embedding) 
- $\mathbf{h}^{LM} _{k, j} = [\overrightarrow {\mathbf{h}^{LM} _{k, j}} ; \overleftarrow {\mathbf{h}^{LM} _{k, j}}]$이다.

이후 ELMo는 *R*에 있는 모든 레이어를 하나의 벡터($\textrm{ELMo} _k = E(R _k; \mathbf{\Theta} _e$)로 collapse

$$
\textrm{ELMo}^{task} _k = E(R_k; \Theta^{task}) = \gamma ^{task} \sum^L _{j=0} s^{task} _j \mathbf h^{LM} -{k, j}
$$

- $s^{task}$: Softmax-normalized weights
- $\gamma ^{task}$: scaler
- LayerNorm을 적용하는 것도 고려해볼만함

## 3.3 Using biLMs for supervised NLP tasks

target NLP task을 위한 supervised구조와 pre-trained biLM이 주어졌을 때, biLM이 task model을 향상시키도록 만드는 것은 간단하다. 그냥 biLM을 돌린다음에, 각 단어에 대한 레이어의 모든 표현을 기록하기만 하면 된다. 그 후, end task model로 하여금 이러한 표현을 통해 linear combination을 배우게끔 만들면 된다. 그 과정은 다음과 같다.
- 그냥 biLM을 LM으로 쓰겠다는 이야기
- biLM을 freeze하고, ELMo vector $\textrm{ELMo}^{task} _k$를 $x _k$와 concat
- ELMo$[\mathbf x _k; \textrm{ELMo}^{task} _k]$은 새로운 representation이 됨
- $\mathbf h _k$을 $[\mathbf h _k; \textrm{ELMo}^{task} _k]$
- 적당한 양의 드랍아웃을 추가하는 것이 도움이 됨
- $\lambda {\lvert \lvert \mathbf w \rvert \rvert}^2 _2 $를 로스에 regularize하는 것도 고려 
  - *inductive bias*를 모든 biLM 레이어의 평균에 가깝게 만듬

## 3.4 Pre-trained bidirectional language model architecture

- Jozefowicz et al. (2016)과 Kim et al. (2015)의 구조와 비슷하지만, bi-directional에 대해 joint training 지원하도록 수정되었고, LSTM 레이어 사이에 residual connection을 추가
- character input representation 사용
- Jozefowicz et al. (2016)의 **CNN-BIG-LSTM**로부터 모든 임베딩과 hidden dimension을 이등분
- 마지막 모델은 
  - $L=2$인 biLSTM
  - 4096 유닛과 512 차원의 projection
  - 첫 번째 레이어부터 두 번째 레이어까지 residual connection
- context insensitive
  - 2048 character n-gram convolutional filters
  - 두 개의 highway 레이어 (Srivastava et al. 2015)
  - 512 차원의 projection 
- *결과적으로, biLM은 character와 각 토큰에 대해 세 개의 레이어를 통해 embedding 
- 1B Benchmark (Chelba et al., 2014)에 대해 10 epochs 학습을 실시한 후 perplexity에 대한 평균은 39.7로 측정
- 일반적으로, 정방향과 역방향 perplexity는 비슷하고, backward의 경우가 약간 더 낮음
- domain-specific data에 대한 fine tuning biLM은 perplexity를 크게 낮추는 경향
- 다운스트림 task에선 증가
- 이는 domain trasnfer의 일종으로 간주 가능
- 결과적으로, 대부분의 다운스트림 task에서 fine-tuned biLM을 사용

# 추가사항

## Pre-trained Language Representation 종류

- feature-based
  - 특정 task를 수행하는 network에 pre-trained LM을 추가적인 feature로 제공하는 방식 (ELMo)
  - ELMo가 pre-trained LM에 다른 word vector를 concat해서 사용하는 것을 생각하면 될듯
- fine-tuning
  - task-specific한 parameter를 최대한 줄이고, pre-trained된 parameter들을 downstream task 학습을 통해 fine-tuning하는 방식 (BERT)
  - BERT는 ELMo와는 다르게 END-TO-END로 사용

## Fine-tuning vs. Transfer Learning

비슷한 맥락에서 비슷하게 사용하는 개념인데, 이 둘의 차이를 구체적으로 비교하고, 이에는 어떤 종류가 있는지 살펴보지.

- Transfer Learning (Domain Adaptation)
  - pre-trained model $f(x)$가 있고, 새롭게 학습할 도메인 $g()$가 있을 때, $g(f(x))$를 학습
  - train the same model with another dataset that has a different distribution of classes

  - Fine-tuning
    - **an approach of Transfer Learning**
    - making some fine adjustments to further improve performance
    - For train data set, 90% for training, train the same model with **the remaining 10%**