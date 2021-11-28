---
title:  "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
toc: true
toc_sticky: true
permalink: /project/nlp/BART-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-11-06
---

## 들어가며

BART는 FAIR에서 연구한 seq2seq 형태의 language model로, 기존 BERT/GPT로 양분되어 있던 language model과 상이한 특성을 보인다. 난징대학교와 MS에서 개발한 MASS와 상당히 유사한 형태로 구성되어 있지만, MASS가 generation task에 특화되어 있는 반면, BART는 NLU를 비롯한 다양한 downstream task에도 적용할 수 있다는 장점이 있다. FAIR에서 연구한 것 답게 `torch`와 `fairseq`에서 이용할 수 있다.

- [원문 보러가기](https://arxiv.org/pdf/1910.13461.pdf)
- [BART repository 보러가기](https://github.com/pytorch/fairseq/tree/main/examples/bart)

## 1 Introduction

모두가 다 알다시피, self-supervised learning은 사실상의 표준이 되어왔다. 이 중 특히 BERT와 같은 denoising auto encoder를 사용하는 모델이 매우 큰 성공을 거두게 되었다. 최근 연구 중에는 BERT와 비슷하면서도 masking scheme을 조금씩 응용하고 바꾼 연구들이 존재하는데, 이에는

[SpanBERT (Joshi et al., 2019)](https://arxiv.org/pdf/1907.10529.pdf)와 같이 마스크 토큰의 분포를 바꾼다던가,

SpanBERT
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/140639143-ff9cd8c9-dd3d-461c-83d7-680b8520a765.png){: .align-center}{: width="800"}

[XLNet (Yang et al., 2019)](https://arxiv.org/abs/1906.08237)처럼 마스크 토큰의 순서를 바꾸어 버리던가,

XLNet ([리뷰보기](/project/nlp/XLNet-review/))
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/134289037-8573a4a3-b7b5-4804-b958-a8fde3fc82a1.png){: .align-center}{: width="700"}

[UNILM (Dong et al., 2019)](https://arxiv.org/pdf/1905.03197.pdf)과 같이 마스킹된 토큰이 이용하는 context를 다양한 형태로 구성하는 방법이 있다.

UNILM
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/140639448-c7d9b879-6697-4c0d-8ade-14fdd0e5f3c9.png){: .align-center}{: width="700"}


그러나 이러한 방법들은 특정한 종류의 end task(e.g. generation, span prediction)에만 집중하는 경향이 있다.

BART(**B**idirectional and **A**uto-**R**egressive **T**ransformers)는 DAE에 seq2seq을 결합하여 다양한 종류의 end task에 적용가능하게끔 한 모델로, 다음의 두 단계를 거쳐 pre-training을 진행하게 된다.

1. 임의의 noise function을 통해 텍스트를 오염
2. seq2seq을 통해 복원

BART는 BERT, GPT 등과 같이 최근 유행하는 pre-training scheme을 사용하였다. 이들의 차이점은 아래 Figure 1에 나타나있다.

![image](https://user-images.githubusercontent.com/47516855/140639555-bace73fc-de05-490a-8ef8-0e2581ae61b3.png){: .align-center}{: width="700"}

BART의 최대 장점으로는 문서를 오염시키는 noising function에 유연하게 적용가능하다는 것이다. 임의의 transformation이 원본 문서에 적용가능하며, 실험을 통해 랜덤 셔플링, novel in-filling scheme (임의의 길이의 span을 하나의 mask token으로 치환) 등 다양한 noise function을 평가했다고 한다. In-filling scheme의 경우 특히 BERT의 일반화로 생각할 수 있다.

BART는 생성 태스크에 특히 유용하지만 comprehension에도 잘 동작하는 것을 확인하였다. GLUE, SQuAD에서는 RoBERTa와 비슷한 training resource를 사용하여 비슷한 성능을 내었고, abstractive dialogue, question answering, summarization tasks에서는 SOTA를 달성하였다.

NMT에서는 Transformer 위에 BART를 stack하여 학습하였고, 이를 통해 Transformer는 외국어를 오염된 영어로 번역하게끔 학습되고, BART를 pre-trained target-side language model로 사용하는 식의 새로운 학습 scheme을 선보였다.

## 2. Model

### 2.1 Architecture

BART의 구조는 기존 Transformer와 똑같지만, GPT를 따라서 GELU를 이용한다는 점이 다르며, GELU의 initial parameter는 $\mathcal N (0, 0.02)$를 채용하였다. 모델은 인코더/디코더가 각각 6/6씩 stack되어 있으며, large에선 12/12로 구현하였다. 구조는 BERT와 거의 똑같지만 인코더의 마지막 레이어에 대해 각 디코더가 추가적으로 cross-attention을 수행한다는 것과 (Transformer처럼), BERT에선 word prediction 이전에 linear layer를 추가하였는데, BART는 없다는 것이다. 대략 비슷한 구조의 BERT에 비해 10% 더 많은 파라미터를 사용한다.

### 2.2 Pre-training BART

BART의 학습은 문서를 오염시킨 뒤 이를 복원하는 reconstruction loss를 최적화함으로 이루어진다. 이는 디코더의 인풋과 아웃풋에 대한 cross entropy가 된다. 기존의 DAE를 이용한 모델들은 특정 noising scheme에 맞춤형(tailored) 모델인 것에 반해 BART는 어떠한 형태의 noising function을 허용한다. 아주 극단적으로 모든 source input을 없앨 경우 BART는 일반적인 language model이 된다.

BART에선 이전 연구들과 새로운 형태의 transformation을 사용하였는데, 저자들은 이 부분에서 개선의 여지가 충분히 있을 거라고 보고 있다. 논문에서 사용한 noise function의 종류는 아래의 그림과 그 밑에 기술해두었다.

![image](https://user-images.githubusercontent.com/47516855/140638811-eeacd8c5-655b-4272-8339-c690f3b62868.png){: .align-center}{: width="800"}

**Token Masking**  
BERT와 똑같은 형태로, 임의의 토큰을 샘플링하고, 이를 `[MASK]`로 변경한다.

**Token Deletion**  
임의의 토큰이 인풋으로부터 지워진다. Token masking과는 다르게 모델이 어떤 위치가 지워졌는지 결정해야 한다.

**Text Infilling**  
포아송 분포($\lambda=3$)를 따르는 길이의 연속된 토큰(text span)을 샘플링하고, 이를 하나의 `[MASK]` 토큰으로 변경한다. 길이 0의 span은 `[MASK]` 토큰을 삽입하는 것과 같다. Text infilling 방법은 SpanBERT에서 아이디어를 얻은 것이지만, SpanBERT는 다른 고정된 기하분포(clamped geometric distribution)로부터 길이를 샘플링하며, 각 span의 길이와 똑같은 갯수의 `[MASK]`으로 변경한다는 차이점이 있다. Text infilling은 모델로 하여금 얼마나 많은 토큰이 사라졌는지 예상하도록 학습시킨다. 

**Sentence Permutation**  
문서가 마침표(full stops)를 기준으로 나눠지며, 나눠진 문장은 임의의 순서로 섞는다.

**Document Rotation**  
임의로 토큰을 선택한 후, 선택된 토큰이 문서의 첫 단어가 되도록 문서를 회전시킨다. 이러한 방법은 모델로 하여금 문서의 시작점을 구분하도록 학습시킨다.

## 3 Fine-tuning BART

BART를 통해 생성된 representation은 downstream task에서 여러 방법으로 이용할 수 있다.

### 3.1 Sequence Classification Tasks

Sequence classification task의 경우 똑같은 인풋이 인코더와 디코더에 삽입되게 되고, 디코더의 마지막 hidden state가 새로운 multi-class linear classifier로 들어가게 된다. 이러한 방법은 BERT의 `[CLS]` 토큰과 연관이 있다. 그러나 여기서는 **마지막**에 추가적인 토큰을 넣는다. 이는 디코더 단의 마지막 단어($t=T$)는 마스킹 없이 인풋 문장 전체에 attention할 수 있기 때문이다.
 
![image](https://user-images.githubusercontent.com/47516855/140638778-ff37b0e1-6e02-4355-b55b-c727e956a933.png){: .align-center}{: width="500"}


### 3.2 Token Classification Tasks

SQuAD와 같은 token classification의 경우 모든 document를 인코더와 디코더에 넣고, 디코더의 마지막 representation을 각 단어의 representation으로 사용한다. 이러한 방법은 token을 분류하는데 사용한다,

### 3.3 Sequence Generation Tasks

BART는 autoregressive 디코더도 포함하고 있기 때문에, abstractive question answering, summarization와 같은 생성 태스크에도 바로 fine-tuning할 수 있다. 두 태스크 모두 인풋으로부터 정보가 복사되지만 denoising pre-training objective로 인해 정보가 변형된다. 여기서는 인코더의 인풋에 문서를 넣게되고, 디코더의 아웃풋에서 autoregresively 문서를 생성하게 된다.

### 3.4 Machine Translation

본 논문에선 NMT에 대해서도 실험하였다. 디코더가 영어로 번역하는 성능을 향상시키기 위해 BART를 이용하였다. 이전 연구인 [Edunov et al. (2019)](https://arxiv.org/abs/1903.09722)에서는 *pre-trained encoder들을 통합(incorporating)하여* 모델 성능을 향상시킬수는 있지만, language model의 decoder로부터 얻는 이점은 제한된다고 밝힌바 있다.

여기서 통합(incorporating)이란, ELMo 스타일이나 LM의 아웃풋을 fine-tuning하는 전략을 의미한다.
{: .notice--info}

그러나 BART의 경우 BART 전체를 하나의 디코더로 세팅하고 새로운 인코더를 도입하여 bitext(병렬말뭉치)를 학습하는 것이 가능함을 보였다고 한다 (아래 그림 참고)

![image](https://user-images.githubusercontent.com/47516855/140638747-3d2dc41c-254f-4cb5-9df9-6390708e88e3.png){: .align-center}{: width="500"}


구체적으로 설명하면, BART의 인코더단의 임베딩 레이어는 랜덤으로 초기화한 인코더로 교체하고, 새로운 인코더는 외국어를 BART가 영어로 de-noise하는 형태로 집어넣어 end-to-end로 학습하는 형태이다. 새로운 인코더는 BART의 vocab과 다른 형태의 vocab을 가질 수 있다.

Source encoder의 경우 학습이 두 단계로 나뉘게 된다. 둘 모두 BART의 아웃풋으로부터 cross-entropy를 통해 얻어진다. 

1. 대부분의 BART 파라미터를 고정한 채로 새로운 인코더와 BART의 positional embedding, BART 인코더의 첫번째 레이어의 self-attention input projection matrix를 학습.
2. 작은 수의 iteration으로 BART 전체를 학습.

## 4 Comparing Pre-training Objectives




{: .notice--info}
{: .align-center}{: width="500"}
