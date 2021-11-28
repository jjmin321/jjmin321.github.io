---
title:  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding review"
excerpt: "맥락과 함께 자세히 살펴보는 BERT 논문 리뷰/설명"
toc: true
toc_sticky: true
permalink: /project/nlp/bert-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2020-12-26
---

이번 시간엔 NLP pre-trained model의 서막을 알린 BERT에 대해 알아보자. 원문은 [다음 링크](https://arxiv.org/abs/1810.04805)에서 확인할 수 있다.

기존의 language model은 AR(Auto-Regressive)을 통해 다음 단어를 예측한다. 
그러나 BERT의 등장과 함께 AR의 시대는 저물고 AE(Auto-Encoder) 시대가 오게되었다.

우선 논문을 읽고 궁금증과 이에 대한 설명/해석을 적었다. 다음은 논문을 읽으면서 생긴 궁금증이다.

- 왜 BERT가 대단한가?
    - bidirectional이 그렇게 대단한가?
        - unidirectional은 안 좋은가?
        - 이전 논문들은 왜 시도를 안(못) 했지?
- 왜 GPT는 decoder를 썻지?
- encoder-decoder 둘 다는 안 되나?
- GPT와의 성능 차이: encoder vs. MLM vs. NSP ?

# Pre-trained Language Model

Language modeling pre-training은 다양한 NLP task에서 효과적임을 보여왔다. 이에 대표적인 예로는,
- GPT-1 ([Review 보러가기)](/project/nlp/gpt1/)
- ELMo ([Review 보러가기)](/project/nlp/elmo-review/)
- Semi-supervised sequence learning
- ULMFiT ([다음](https://github.com/InhyeokYoo/CS224N/blob/main/week7/Modeling-contexts-of-use-Contextual-Representations-and-Pretraining.md#3-ulmfit-and-on-ward)을 참고)
등이 있다. 

이는 language inference, *paraphrasing*과 같이 문장들을 일부분이 아닌 전체적으로 분석하여 이들 사이의 관계를 예측하는 **sentence-level task**와, name entity recognition (NER), question answering(QA)와 같이 모델이 token 단위로 fine-grained output을 내는 **token-level task**을 포함한다.

paraphrasing: paraphrase를 생성하거나 탐지하는 작업을 말한다. 위키피디아에서는 QA도 paraphrasing이라고 설명하고 있는데도 불구하고 BERT에서 QA가 token-level task인 이유는 각 token-level에 대해서 정답의 시작점과 끝점의 index를 찾기 때문이다.
{: .notice--info}

pre-trained language representation을 downstream task에 적용하는 방법은 두 가지가 있다.

**1. feature-based**

pre-trained representation을 **추가적인 feature**로 이용하는 **task-specific architecture**를 사용하는 방법이다. ELMo가 그 대표적인 예시이다. 다음은 ELMo 논문의 3.3장에서 발췌한 내용이다.

> Given a pre-trained biLM and a supervised architecture for a target NLP task, it is a simple processto use the biLM to improve the task model.  Wesimply  run  the  biLM  and  record  all  of  the  layerrepresentations  for  each  word.   Then,  we  **let the end task model learn a linear combination of these representations**, as described below.

즉, 모델을 학습하여 embedding을 얻고, 이를 다른 모델에 집어넣는 형태를 의미한다.

**2. fine-tuning approach**

fine-tuning approach는 최소한의 task-specific parameter만을 도입하는 모델이다. downstream에 대해 단순히 모든 parameter를 fine-tuning한다. 이에는 GPT-1이 대표적이다. ([리뷰 보러가기](/project/nlp/gpt1/)) 

이 두 가지 방법은 pre-training에서 같은 objective function ($p(x _t \rvert x _1, ..., x _{t-1})$)을 공유하고, unidirectional language model을 통해 일반적인 language representation을 학습한다.

그러나 논문에서는 이러한 방법이 pre-trained representation의 능력을, 특히 fine-tuning 방법에서, **제약**시킨다고 보고 있다 (We argue). 가장 큰 제약은 일반적인 LM이 **단방향**이라는 것이고, 이로 인해 pre-training에서 사용하는 **모델의 선택지를 제한**한다고 본다. 예를 들어 GPT의 경우엔 left-to-right architecture (transformer decoder)를 사용하였는데, 이는 subsquence mask로 인해 오직 **이전의 token**에만 self-attnetion이 가능하다. 이는 언어가 갖는 직관과 상반된다. 또한, sentence-level에선 sub-optimal이고, token-level task (e.g. QA)와 같이 양방향 (bidirectional)의 정보를 이용하는 것이 중요한 task에선 매우 치명적이다.

본 논문에서는 BERT를 제안하여 fine-tuning based approach를 향상시킨다. BERT는 Cloze task에서 영감받은 **"masked language model" (MLM)**  pre-training objective을 통해 앞서 언급한 unidirectionality constraint를 해결한다.

# Masked Language Model in BERT

BERT는 Masked Language Model (MLM) pretraining objective를 사용하여 Transformer encoder를 학습시킨다. 
이는 [Cloze task (test)](https://en.wikipedia.org/wiki/Cloze_test)에서 영감을 받은 것으로, 이 cloze task는 학생들에게 빈 칸을 채우도록 한 다음 이들의 언어 능력을 평가하는 테스트이다. Cloze test를 잘 수행하기 위해서는 context를 이해하고 단어를 잘 이해하는 능력이 필요하다. 따라서 머신이 인간처럼 잘 학습하기 위해서는 **인간이 언어를 이해하는 방법**을 잘 따라하는 것이 필요하고, 이는 Cloze task의 목표와도 잘 부합한다고 볼 수도 있을 것 같다.

![Cloze task](https://miro.medium.com/max/620/1*2X0uYNinK7KOQLtNknQPsg.png){: .align-center}{: width="400"}

Masked language model은 기존 unidirectional language model을 bidirectional로 변경하면서 갖는 문제를 해결하기 위해 등장하게 되었다. language model은 문장의 단어를 왼쪽에서 오른쪽으로 읽으며 학습하는데, 실제로 언어를 이해하기 위해서는 역방향(backward) 또한 고려를 해야한다 (bidirectional). 그러나 대부분의 연구는 unidirectional에 치중되어 있고, 설령 ELMo처럼 bidirectional하더라도 shallow concatenation을 사용할 정도로 소극적인데, 이에는 다음과 같은 문제가 있다.
- 두 방향 모두를 고려하며 생성할 순 없으므로 어차피 단어의 분포를 생성하려면 unidirectional 해야 한다
- **단어를 예측할 때 자기 자신을 볼 수 있다** 

특히 중요한 것은 두 번째인데, 본문에서는 다음과 같이 표현하고 있다.

> Unfortunately, standard conditional language models can **only be trained left-to-right or right-to-left**, since bidirectional conditioning would allow each word to indirectly **“see itself”**, and the model could trivially predict the target word in a multi-layered context.

![bidirectional conditioning would allow each word to indirectly “see itself”](https://qph.fs.quoracdn.net/main-qimg-514d3773b91bbd2c1a72ca4e3d83f707){: .align-center}{: width="800"}

이에 대한 솔루션으로 단어의 15%를 [mask] 토큰으로 대체하게 된다. 15%에 대한 근거는 찾기가 힘들었는데, 다만 너무 적게할 경우 학습하는데 비용이 많이 들고 (즉, 충분히 학습하기가 힘들어서 더 많은 학습이 필요함), 너무 많이 할 경우 문맥(context)를 충분히 주기가 힘들어 학습하기가 어렵다고 한다.

이를 deep learning 구조에서 생각해보면 denoising auto-encoder를 사용하는 것과 같다. denoising auto-encoder에서 collapse된 부분이 mask를 씌운 것과 동일하다고 보는 것이다. 아래는 BART논문에서 발췌한 내용이다.

> The most successful approaches have been variants of masked language models, which are denoising auto encoders that are trained to reconstruct text where a random subset of the words has been masked out.

BERT에서 (denoising) auto-encoder를 언급하는 부분은 아래와 같다.

> **2.1  Unsupervised Feature-based Approaches**
......
To  train  sentence  representations,  prior work  has  used  objectives  to  rank  candidate  next sentences  (Jernite  et  al.,  2017;  Logeswaran  and Lee, 2018),  left-to-right  generation  of  next  sentence words given a representation of the previous sentence  (Kiros  et  al.,  2015),  or  **denoising  auto-encoder derived objectives [(Hill et al., 2016)](https://www.aclweb.org/anthology/N16-1162/)**.

> **2.2 Unsupervised Fine-tuning Approaches**
......
Left-to-right  language  modeling and auto-encoder  objectives  have  been  used for pre-training such models Howard and Ruder, 2018; Radford et al., 2018; **[Dai and Le, 2015](https://arxiv.org/pdf/1511.01432.pdf)**)

> **3.1  Pre-training BERT - Task #1: Masked LM**
......
In contrast to denoising auto-encoders (Vincent et al., 2008), we only **predict the masked words rather than reconstructing the entire input**.

이 중에서 (denoising) auto encoder를 사용한 것은 Dai and Le, 2015와 Hill et al., 2016의 연구이다. 간략하게 한 번 살펴보자.

## Semi-supervised Sequence Learning (Dai and Le, 2015)

본 논문은 auto encoder를 활용하여 pre-trained model을 만들고 성능을 평가했다. 본 논문에선 unlabeled data를 이용하여 RNN sequence learning을 향상시키기 위한 두 가지 방법을 비교하고 있다. 이 두 알고리즘은 일종의 **pre-trained** 단계에서 사용되어 supervised learning을 거치게 된다. 두 방법론은 다음과 같다.
- 일반적인 LM: $p(x _t \rvert x _1, ..., x _{t-1})$
- sequence autoencoder: input sequence를 읽어 이를 벡터로 만든 후에 다시 input sequence를 생성

이 두 가지 방법은 random initializing을 통해 모델을 end-to-end로 학습하는 것보다 더 좋았다고 한다. 또 한가지 중요한 결과는 관련된 task의 unlabeled data를 활용했을 때 generalization이 더 좋았다는 점이다.

위에서 언급한 sequence autoencoder는 seq2seq과 비슷한 구조를 갖고 있지만 unsupervised learning이라는 점에서 차이점이 있다. Sequence autoencoder는 데이터를 입력받고 encoding한 후, decoding과정에서 **원본 데이터의 복원**을 objective로 삼는다. 

![Sequence autoencoder](https://user-images.githubusercontent.com/47516855/100340636-d83d9a00-301e-11eb-9b74-00b719c46d87.png){: .align-center}{: width="800"}

이렇게 얻은 weight는 다른 supervision task에서 initialization으로 사용할 수 있다. 

## Sequential (Denoising) Autoencoder, Hill et al., 2016

본 논문에서는 DAE를 활용한 representation learning objective를 사용한다. 본래 DAE는 고정된 길이를 갖는 이미지에 적용하는데, 여기서는 가변 길이를 갖는 문장에 적용하도록 noise function $N(S \rvert p _0, p _x)$의 평균값을 이용한다. $p _0, p _x$는 0과 1사이의 값는 확률 값으로, 각 단어 $w$는 독립 확률 $p _0$를 통해 삭제된다. 그리고 문장 안에 서로 겹치지 않는 bigram $w _iw _{i+1}$에 대해 $N$은 $p _x$확률로 이를 $w _i$와 $w _{i+1}$로 바꾼다. 그후 LSTM 기반의 encoder-decoder를 통해 원래 문장을 예측하도록 한다. 즉, 원래의 source sentence는 ground truth로, input은 $N(S \rvert p _0, p _x)$가 되는 것이다. 이러면 novel word sequence를 distributed representation으로 표현할 수 있게 된다. 만일 $p _0, p _x$가 0이 되면, 앞서 언급한 sequence autoencoder와 동일한 objective가 된다.

BERT에서의 MLM은 임의로 input token 일부를 masking하고, objective는 context를 통해 이러한 masked token을 예측한다. 

# BERT

BERT에는 두 가지 step을 통해 downstream task를 수행한다.

첫 번째는 **pre-train**으로, pre-training task에 대해 unlabeled data를 사용하여 학습한다. 일반적으로는 semi-supervised/unsupervised learning이라 불리는 것을 의미한다. 이에 대한 개념은 [다음](/project/nlp/gpt1/#unsupervised-pre-training)을 참고해보자.

두 번째 step은 **fine-tuning**이다. 앞서 수행한 pre-trained parameter로 initialize한 후, downstream task 수행하여 parameter를 fine-tuning한다. 아래의 Figure 1은 본 모델에서의 QA task의 예시이다.

![Figure 1](https://user-images.githubusercontent.com/47516855/98634518-c0c69780-2323-11eb-820b-20663a1ac8fa.png)
{: .align-center}

여기서 한 가지 명심할 점은 BERT는 다른 task에 대해서도 같은 구조를 유지하고 있다는 것이다. 이는 ULMfit의 구조와도 동일하다.
이러한 구조적 이점으로 인해 매번 다른 task에 대해 새로운 구조를 고려할 필요가 없고, BERT 단일구조를 통해 매우 높은 성능을 낸다는 장점이 생긴다.

## Model Architecture

BERT의 구조는 Transformer의 encoder와 같다. 우선 논문에서 표현하는 notation을 살펴보자.
- $L$: # layers
- $H$: hidden size
- $A$: # heads
- 이에 더해 Feed-forward는 4H를 사용

BERT_BASE (BASELINE)의 스펙은 다음과 같다. 이는 OpenAI GPT와의 비교를 위해 의도적으로 같은 사이즈로 만들었다.
- $L$: 12
- $H$: 768
- $A$: 12

저자들은 BERT_BASE 이외에도 BERT_LARGE라는 더 큰 모델을 학습시켰다. 이에 대한 스펙은 다음과 같다.
- $L$: 24
- $H$: 1024
- $A$: 16

## Input/Output Representation

앞서 언급했던 것처럼 BERT로 하여금 다양한 downstream task를 다루게 하기 위해, input representation이 하나의 token sequence 안에 한 문장/문장 쌍 (e.g., <Qustion, Answer>) 모두를 표현할 수 있게 한다. 본 논문에서 "sentence"이란 실제 문장이라기보단 연속된 text의 모음이라고 이야기하고 있는데, 이게 실제로는 어떤 차이인지 모르겠다. 반면 "sequence"는 BERT에 넣을 input token sequence를 의미한다. 앞서 밝힌 바와 같이, 하나의 문장이 될수도 있고, 두 개의 문장 쌍이 될 수도 있다. 본 논문에서는 30000개의 vocab을 갖는 WordPiece를 사용했다. 

모든 input sequence의 첫 번째 token은 항상 special classification token인 [CLS]가 되고, 이 토큰에 해당하는 마지막 hidden state는 classification task를 위한 representation이 된다. 문장 쌍의 경우에는 하나의 sequence로 합쳐서 들어가게 된다. 본 논문은 이러한 문장을 구분짓기 위해 special token [SEP]를 통해 분리하고, 각 토큰이 문장 A에 속하는지, B에 속하는지를 학습한 embedding을 더하게 된다. 이는 앞선 Figure 1에서 볼 수 있듯, input embedding을 $E$, special token [CLS]의 마지막 hidden state를 $ C \in \mathbb R^H$로, i번째 인풋 토큰을 $ T _i \in \mathbb R^H$로 표현한다. 전체적인 토큰의 표현은 token, segment, position embedding의 합으로 구성되어 있다.

![Figure 2](https://user-images.githubusercontent.com/47516855/98550274-3fe5ac80-22df-11eb-9b2e-a18b49868953.png)

## Pre-training BERT

BERT는 ELMo나 GPT-1과는 다르게 left-to-right나 right-to-left LM같은 전통적인 모델을 사용하진 않는다. 대신 두 개의 unsupervised task를 이용하여 pre-train을 진행한다.

### Task #1: Masked LM

앞서 설명했듯 MLM은 bidirectional language model을 만들기 위한 작업이다. 전체 sequence 중 일부에 MASK를 씌워 정보를 손실한 후 (DAE), 
이를 다시 복원하는 것을 목표로 한다. 실험에서는 각 sequence의 15%가 (WordPiece token) 랜덤으로 masking된다.
그러나 이런 방식은 pre-train과 fine-tuning 사이에 mismatch를 만들게 되는데, [MASK] 토큰은 fine-tuning 단계에서 등장하지 않기 때문이다.
즉, 모델은 다른 것을 학습하기 보단 [MASK] 토큰만을 복원하는데 주력할 것이고, [MASK]가 없는 fine-tuning에서는 토큰이 없기 때문에 문제가 발생한다.

이를 보완하기 위해 항상 mask된 단어를 [MASK] 토큰으로 변경하는 대신,
1. 80%의 확률로 [MASK]로 변경
2. 10%의 확률로 임의의 토큰으로 변경
3. 10%의 확률로 변경 X

그 후 $T _i$는 cross entropy loss를 통해 원래의 token을 예측하는데 사용된다.

![image](https://user-images.githubusercontent.com/47516855/103149380-283d8700-47ac-11eb-8147-6a06245c2c03.png){: .align-center}


**C.2 Ablation for Different Masking Procedures**

본 논문에서는 이에 대한 추가 실험 (ablantion study)을 appendix에 정리해놓았다.

Masking의 효과를 알아보기 위해 **비율을 아래의 표와 같이 각각 달리 적용하여** MNLI (entailment classification task), NER에 대해 실험한 결과이다.
NER의 경우 fine-tuning approach와 feature-based approach를 둘 다 실험하였는데, feature-based의 경우 모델이 representation을 수정할 기회가 없기 때문에 앞서 언급한 **mismatch가 더욱 증폭**될 것이라 생각했기 때문이다.

feature-base의 경우에는 Section 5.3에 표현된 바와 같이 최적의 approach를 선정하여 마지막 4개 layer의 concat으로 사용했다고 한다.

![image](https://user-images.githubusercontent.com/47516855/103097901-3379a100-464c-11eb-9dd8-c27e0f4af7ca.png){: .align-center}

테이블을 보면 fine-tuning의 경우 놀라울 정도로 robust한 것을 볼 수가 있다 (dev set에 대해 좋은 성능을 보였으므로).
그러나 예상했던데로 MASK만 쓰는 경우 feature-based에서 문제가 생기는 것을 확인할 수 있다.
흥미로운 점은 아예 랜덤하게 한 경우가 더욱 나쁜 결과를 보인다는 것이다.

### TASK #2: Next Sentence Prediction (NSP)

NLI나 QA 같은 task는 두 문장의 **관계**를 이해해야 한다. 그러나 language modeling을 이용해 이러한 관계를 직접적으로 파악할 순 **없다**.
문장 사이의 관계를 이해하기 위해서는 어떠한 monolingual corpus에서도 생성 가능한 binarized next sentence prediction task를 학습하는 것이 필요하다. 

각 pre-training example에 대해 sentence A와 B를 선택할 때, 절반의 확률로 B가 실제로 A 다음에 오는 경우이고 (*IsNext*로 label), 나머지 절반은 corpus로부터 추출한 random sentence가 된다 (*NotNext*로 label).
Figure 1에서 $C$는 (빨간색 네모) 이러한 next sentence prediction을 위해 사용한다.

![Figure 1](https://user-images.githubusercontent.com/47516855/98634518-c0c69780-2323-11eb-820b-20663a1ac8fa.png)
{: .align-center}

최종 모델은 NSP에서 97%-98%의 정확도를 달성했고, 굉장히 단순한 작업임에도 불구하고 QA와 NLI에서 매우 유용하다.
그러나 vector $C$는 fine-tuning이 아닐 경우 유용하지 않은데, 이는 NSP를 위해 학습되는 요소이기 때문이다.

NSP task는 Jernite et al. (2017)와 Logeswaran  and  Lee  (2018)의 representation-learning objective와 관련이 깊다.
그러나 이전 연구들은 오직 sentence embedding만 transfer하여 downstream task로 진행한 반면, BERT는 end-task model paramter를 초기화하기 위해 모든 parameter를 transfer했다.

**A.2 Pre-training Procedure**

input sequence를 생성하기 위해서 두 개의 text span을 추출하고 이를 **sentences**라 한다. 첫 번째 sentence는 문장 A를 embedding하고, 두 번째 sentence는 문장 B를 임베딩한다. B는 50%의 확률로 진짜 다음 문장이 되거나, 임의의 문장이 된다.

이러한 문장은 더해서 512 이하만 추출하고, WordPiece tokenization을 적용한 이후에 15%만 마스킹을 하게 된다.

256 문장의 배치와 (256 문장 * 512 토큰 = 128,000토큰/배치), 에폭은 40, 스텝은 약 1,000,000이 된다.

Adam과 함께 1e-4의 learning rate와 $\beta _1=0.9, \beta _2 = 0.999$, L2 decay는 0.01, 10,000번째까지 learning rate warmup을 해주고 linear decay with learning rate를 한다. dropout은 0.1, gelu activation.

loss는 MLM likelihood의 sum과 mean NSP likelihood가 된다.

긴 문장은 attention을 계산할 때 quadratic하니까 비효율적이다. 따라서 90%는 128개 문장으로, 나머지 10%는 512개를 넣어 positional embedding을 학습하도록 한다.

## Pre-training data

Pre-training data는 Zhu  et  al.,2015의 BookCorpus (8억 단어)와 영어 위키피디아 (25억 단어)를 이용한다. 
위키피디아의 경우 테이블, 리스트, 헤더를 제외한 텍스트만 추출한다. 긴 문장을 뽑아내기 위해 단순하게 섞인 문장 단위의 corpus보단 이런 document 단위의 corpus가 더 좋다.

## Fine-tuning BERT

BERT의 fine-tuning은 다른 모델과는 달리 쉽다. 이는 transformer의 self-attention이 BERT로 하여금 다양한 downstream task를 하도록 돕기 때문이다.
특히나 single text/text pair에 무관하게 적절한 input/output을 바꿔가며 진행한다.

Text pair의 경우 일반적인 방법은 텍스트 쌍에 대해 독립적으로 인코딩한 후 bidirectional cross attention과 같은 구조에 적용한다.
그러나 BERT는 self-attention을 활용, 이 두개의 스텝을 통합한다. 
이는 concat된 텍스트 쌍을 self-attention을 통해 encoding하면 효과적으로 bidirectional cross attention을 포함할 수 있기 때문이다. 

각 task에 대해, BERT에다가 task specific input/output을 삽입하고, 모든 파라미터를 end-to-end로 fine-tuning한다.
이 단계에서 Input의 문장 A와 B는 (1) paraphrasing의 sentence 쌍, (2) entailment에서의 hypothesis-premise 쌍, (3) QA에서의 question-passage 쌍, (4) text classification, sequence tagging에서의 degenerate text-$\varnothing$ 과 같다.

Output에서의 token representation은 sequence tagging이나 QA와 같은 token-level task를 위한 output layer를 통과한다.
그리고 [CLS] representation은 entailment나 sentiment analysis와 같은 classification을 위한 layer를 통과한다.

Pre-training에 비하면 fine-tuning은 상대적으로 저렴하다.
논문에 나오는 모든 결과는 single Cloud TPU를 통해 최대 1시간, GPU로는 몇 시간내 재현 가능하다.

**A.3  Fine-tuning Procedure**

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4

큰 데이터 셋의 경우 하이퍼 파라미터 선택에 대해 덜 민감하다. 파인튜닝은 굉장히 빠르므로 위의 하이퍼 파라미터 셋에 대해 exhasutive search를 적용하여 최고의 하이퍼 파라미터를 적용한다. 나머지는 pre-train과 똑같이 유지한다.

### Illustrations of Fine-tuning on Different Tasks

BERT를 통한 fine-tuning 과정은 아래 그림을 통해 확인할 수 있다. 
BERT의 task specific 구조는 BERT위에 레이어를 하나 추가하는 것으로 이루어진다.
이는 최소한의 파라미터를 통해 밑바닥(scratch)부터 학습할 수가 있다.

![image](https://user-images.githubusercontent.com/47516855/103151896-89be1f80-47c5-11eb-92db-f6bf8ece6d3b.png)


(a)와 (b)의 경우 sequence-level task인 반면, (c)와 (d)는 token-level task이다.

그림에서 $E$는 input embedding을 의미하고, $T _i$는 token $i$에 대한 contextual representation, [CLS]의 경우 classification output을 위한 special token을, [SEP]는 연속적이지 않은(non-consecutive) 토큰 시퀀스를 분리하기 위한 스페셜 토큰을 의미한다.



# Experiment

[General Language Understanding Evaluation (GLUE)](https://arxiv.org/pdf/1804.07461.pdf)는 다양한 natural language understanding task를 위한 데이터셋이다.

GLUE에는 다음과 같은 데이터가 포함되어 있다.
- NLI: 가설(hypothesis)이 전제(premise)로부터 도출될 수 있는지 판단하는 과제
- Textual entailment: text와 hypothesis를 읽고, text를 통해 hypothesis가 진실임을 추론하면 entailment, 아니라면 contradiction, 둘 다 추론할 수 없으면 neutral이 됨

| task | 설명 |
| :------ | :--- |
| [Multi-Genre Natural Language Inference (MNLI)](https://nlp.stanford.edu/projects/snli/) | 다양한 장르에 대해 크라우드 소싱을 통해 얻은 라지스케일의 entailment classification task이다. 한 쌍의 문장에 대해, 두 번째 문장이 entailment인지, contradiction인지, neutral인지 분류하는 문제 |
| QQP (Quora Question Pairs) | Quora에 올라온 질문들이 의미적으로 비슷한지 분류하는 문제 |
| QNLI (Question Natural Language Inference) |  Stanford  Question  Answering Dataset의 이진분류 버전으로, positive example (question, sentence) 쌍은 올바른 정답을 포함하고 있고, negative example (question, sentence) 쌍은 정답이 없다. 원래 Stanford  Question  Answering Dataset은 question-paragraph 쌍으로, paragraph내 sentence 중 하나는 이에 대한 정답을 갖고 있다. QNLI는 이 paragraph를 여러 sentence로 나눈 버전 |
| SST-2 (Stanford Sentiment Treebank 2) | 영화 리뷰에 대한 문장의 긍/부정을 판별하는 sentiment analysis |
| CoLA (The Corpus of Linguistic Acceptability) | 문장에 대한 이진분류로, 영어 문장이 언어적(linguistically)으로 문법적으로 옳은지 판단 |
| STS-B (Semantic Textual Similarity Bench-mark) | 뉴스 헤드라인과 그 외 소스에서 뽑은 문장 쌍으로, 두 문장의 유사도에 따라 1부터 5까지 판단 |
| MRPC (Microsoft  Research  Paraphrase  Corpus) | 온라인 뉴스로부터 뽑은 문장 쌍으로, 문장이 의미적으로 유사한지를 판단 |
| RTE (Recognizing  Textual  Entailment) | 이진 entailment task로, MNLI와 비슷. 그러나 학습 데이터량이 훨씬 작음 |
| WNLI (Winograd  NLI) | 작은 NLI 데이터 셋으로, 대명사가 포함된 문장을 읽고 대명사가 무엇인지 파악하는 task |

GLUE 데이터 셋에 대해 fine-tune하기 위해서 input sequence를 BERT에 넣고 첫 번째 input token [CLS]에 해당하는 마지막 hidden vector $C \in \mathbb R^H$를 이용한다.

파인튜닝과정에서 필요로하는 새로운 파라미터는 오직 classification layer weight $W \in \mathbb R^{K \times H}$이다. 여기서 $K$는 # of classes이다. $C$와 $W$에 대해 일반적인 classification loss를 사용한다. (i.e. $\log ( softmax(CW^T))$)

배치사이즈는 32, 에폭은 3을 사용한다. Dev set에 대해, 각 task에 대해 최고의 fine-tuning learning rate를 후보(5e-5, 4e-5, 3e-5, and 2e-5) 중에 고른다.

추가적으로 BERT_LARGE 모델에 대해서는 파인튜닝과정이 가끔 작은 데이터셋에 대해서는 불안정하다는 사실을 발견했다. 따라서 몇 번의 random start를 통해 최고의 모델을 선택한다 (Dev set). 
Random start에 대해서는 같은 pre-train모델을 사용하나, 데이터를 다르게 섞거나 classification layer의 초기화를 매번 새롭게 해준다.

이에 대한 결과는 아래의 표에 있다.
BERT_BASE와 BERT_LARGE 모두 상당한 격차 (substantial margin)로 모든 테스크에 대해 모든 시스템을 압도하였다. 이는 이전의 SOTA보다 각기 4.5%, 7%의 평균 정확도의 향상이다.

![image](https://user-images.githubusercontent.com/47516855/103152119-25508f80-47c8-11eb-95c2-5409f117d101.png){: .align-center}

여기서 주목할점은 BERT_BASE와 OpenAI GPT의 경우 attention masking을 제외하고는 모델적인 측면에서 동일하다는 것이다 (encoder와 decoder의 차이). 가장 크고 널리 보고된 GLUE 테스크인 MNLI의 경우 4.6%의 정확도 상승을 이뤄냈다.
GLUE 공식 리더보드에서는 BERT_LARGE의 경우 80.5인 반면 GPT는 72.8이다.

BERT_LARGE는 모든 테스크에 대해 BERT_BASE를 굉장히 압도하는 것으로 나타났으며, 특히나 작은 트레인 셋에 대해서 그런 경향이 있음을 발견했다. 모델 사이즈에 대한 실험은 [Effect of Pre-training Tasks](/project/nlp/bert-review/##Effect of Pre-training Tasks)에서 추가로 확인할 수 있다.

## SQuAD v1.1

앞서 그림에서 확인했듯, QA의 question (sentence A)과 passage (sentence B)는 하나의 sequence로 묶는다. 여기서는 오직 start vector $S \in \mathbb R^H$와 end vector $E \in \mathbb R^H$만을 도입한다.

i번째 단어가 answer span이 될 확률은 $T _i$와 $S$, $E$의 dot product와 softmax를 통해 결정되며 ($P _i = \frac{e^{S \cdot T _i}}{\sum _j e^{S \cdot T _j}}$), 따라서 i번째부터 j번째까지의 candidate span의 점수는 $S \cdot T _i + E \cdot T _j $가 된다. 그리고 이 중 $ j \geq i$를 만족하는 maximum score가 prediction으로 사용된다.

3번의 에폭과 5e-5의 learning rate, 32의 배치사이즈를 통해 파인튜닝이 실시된다.

아래의 표는 탑 리더보드로, 다른 데이터를 통해 학습하는 것이 허용되어있기 때문에 TriviaQA 데이터를 이용하여 먼저 파인튜닝한 후 SQuAD 데이터를 파인튜닝했다.

![image](https://user-images.githubusercontent.com/47516855/103153098-49fd3500-47d1-11eb-9c77-9bd37efb842b.png){: .align-center}

BERT는 탑 리더보드에 대해 앙상블의 경우 1.5 (초록색), 단일 모델 (single)의 경우 1.3(노란색) 높은 F1 score를 기록했다. 단일 모델의 경우에는 가장 높은 앙상블 시스템보다도 더 좋은 성능을 기록하였다 (0.1, 0.4 차이).

## SQuAD v2.0

SQuAD v2.0에서는 v1.1를 확장시킨 방법을 사용한다. 2.0 버전에서는 정답이 없는 것도 포함되어 있으므로, 정답이 없는 문제의 answer span의 시작과 끝을 [CLS] 토큰에 할당한다. 따라서 probability space는 [CLS] 토큰의 위치를 포함하도록 확장된다.

예측의 경우 no-answer span $ s _{null} = S \cdot C + E \cdot C $과 가장 높은 non-null span $\hat{s} _{i, j} = \max _{j \geq i}  S \cdot T _i + E \cdot T _j $를 비교한다. $ \hat{s} _{i, j} > s _{null} + \tau $인 경우에만 non-null로 예측하고, $\tau$는 Dev set에 대해 F1 score를 최고로 하는 threshold가 된다.

이 경우 TriviaQA는 사용하지 않고, 2에폭과 48 배치, 5e-5의 learning rate를 사용한다.

![image](https://user-images.githubusercontent.com/47516855/103153473-14a61680-47d4-11eb-8935-39a688f854d5.png){: .align-center}

그 결과 이전 시점의 SOTA보다 5.1높은 F1 score를 달성했다.

## SWAG

The Situations With Adversarial Generations (SWAG)은 113k의 문장쌍으로 이루어진 데이터셋으로, common sense inference에 대해 검사한다. 예를 들어 "그녀는 차 덮개를 열었다" 다음에 나올 문장으로 사람은 적당한 문장을 찾을 수 있다. ("그리고는 엔진을 검사하기 시작했다") SWAG은 natural language inference와 physically grounded reasoning을 합쳐 평가하는 데이터 셋이다.

파인튜닝과정에서 4개의 시퀀스를 구성했고, 각기 주어진 문장 (sentence A)과 가능한 continuation (sentence B)의 concat으로 구성되어 있다.

Task specific parameter는 오직 [CLS] 토큰에 곱해지는 벡터 하나로, $C$와의 곱과 softmax를 통해 각 선택에 대한 score를 나타내게 된다.

파인튜닝에서 3개의 에폭, 2e-5의 learning rate, 16의 배치를 사용했고, 저자들의 baseline인 ESIM+ELMo를 27.1%, GPT를 8.3% 상회하였다.

![image](https://user-images.githubusercontent.com/47516855/103153713-e3c6e100-47d5-11eb-8ad2-d9a9ba0d23fc.png){: .align-center}

우선 SWAG은 하나의 문장과 네 개의 보기로 되어 있는 것으로 보인다. 이에 대해 matrix가 아닌 vector를 곱하여 4 x 1의 logit을 얻는다. 이에 대해 softmax를 통과시키고, 정답을 고른다. [출처](https://github.com/google-research/bert/issues/38)
{: .notice--info}

# Ablation Studies

여기서는 BERT에 대해 더 나은 이해를 위해 ablation study를 진행한다. 추가적인 정보는 Appendix C에서 확인할 수 있다.

## Effect of Pre-training Tasks

여기서는 BERT_BASE와 동일한 데이터 셋, 파인튜닝 구조, 하이퍼파라미터를 갖는 pre-training objective를 이용하여 deep bidirectionality에 대해 살펴본다.

**No NSP:** MLM without NSP
**LTR & No NSP**: 전통적인 left-to-right LM과 NSP없이 사용. GPT와 직접 비교가 가능하지만, 논문의 데이터 셋과 input representation, 파인튜닝 scheme을 사용

우선 NSP가 가져오는 이점을 보도록 하자. 아래의 표는 NSP를 제거한 경우 QNLI, MNLI, SQuAD v1.1에서 상당한 성능의 저하를 가져오는 것으로 나타났다 (노랑색).

그 다음은 No NSP와 LTR & No NSP를 통해 bidirectional representation을 비교해보았다. LTR은 NSP에 비해 전체적으로 낮은 성능을 보였고, MRPC와 SQuAD에서는 큰 폭으로 성능 하락이 있었다 (초록).

SQuAD의 경우 직관적으로 LTR이 더 낮은 성능을 보일 것이라고 예상할 수 있다. 이는 token-level hidden state가 오른쪽의 문맥을 볼 수 없기 때문이다 (LTR).
이를 확인하기 위해 추가적으로 BiLSTM을 쌓은 모델을 비교해보았고, SQuAD를 향상시키는 결과를 가져왔다 (파랑). 그러나 여전히 bidirectional 보다는 낮은 것을 확인할 수 있다. BiLSTM은 오히려 GLUE에 대해 성능 저하를 가져왔다.

![image](https://user-images.githubusercontent.com/47516855/103154554-5e462f80-47db-11eb-94df-3d5ac290df74.png){: .align-center}

또한, ELMO처럼 LTR과 RTL 따로 학습시킬 수는 있으나 이는 단점이 있는데,
- 두배 더 expensive하고,
- QA와 같은 작업엔 비직관적. 왜냐하면 RTL은 answer을 보고 question을 이해하기 때문.
- deep bidirectional 보단 덜 강력함. 왜냐하면 매 레이어에서 양방향을 보기 힘들기 때문
과 같은 단점이 있다.

## Effect of Model Size

여기서는 model size에 따른 정확도를 살펴보도록 하자. 본 실험에서는 다양한 레이어, 사이즈, 어텐션 헤드 등을 다르게 한 BERT를 준비했고, 나머지 하이퍼 파라미터나 학습과정은 동일하게 유지했다.

아래 표는 이에 대한 결과이다. 보면 모델이 커질수록 모든 실험에 대해 좋은 결과를 보임을 알 수 있고, MRPC와 같이 적은 데이터 셋에도 잘 동작한다. 또한, 이전 연구들에 비해 상대적으로 큼에도 불구하고 상당한 향상을 이뤄냈다.

![image](https://user-images.githubusercontent.com/47516855/103154666-2d1a2f00-47dc-11eb-8922-64c9ba0f82d2.png){: .align-center}

트랜스포머는 (L=6,  H=1024,  A=16)의 100M 파라미터를 갖고 있고, 문헌에서 찾아낸 가장 큰 트랜스포머는 (L=64,  H=512,  A=2)의 235M 파라미터이다. 반면 BERT_BASE는 110M 파라미터이고, BERT_LARGE는 340M 파라미터를 갖는다.

NMT나 LM과 같이 lage scale task에서 모델 사이즈를 늘릴수록 성능이 좋아진다는 것은 오랫동안 알려진 사실이었다. 여기서는 pre-train이 잘 되어 있다는 가정하에 모델 사이즈를 매우 크게 늘리고 task scale은 작게 유지해도 좋은 성능을 보임을 확인했다.

Peters et al. (2018b)의 경우 레이어를 2에서 4로 늘림에 따라 서로 다른 결과를 보였고, Melamud  et  al. (2016)의 경우 hidden dimension을 200에서 600까지 올렸을 땐 좋은 성능을 보였지만, 1000까지 올렸을 때는 성능 개선을 보이지 못했다.

이 둘 모두 feature-based 이므로 여기서 가설을 하나 내릴 수 있는데, 모델이 downstream task에 대해 직접적으로 파인튜닝이 적용되어 있고, 매우 작은 수의 추가적인 파라미터를 학습시킨다고 할 때, downstream task의 양이 적더라도, 크고 더욱 expresive한 pre-trained representation을 통해 task specific model은 효과를 볼 수있다는 것이다.

## Feature-based Approach with BERT

이전까지 BERT는 fine-tuning approach를 사용하고 있었다. fixed  features 가 pre-trained model에서 추출되는 feature-based approach는 몇 가지 분명한 장점을 갖고 있다.
- 모든 task가 쉽게 트랜스포머 인코더 구조로 변환되는 것이 아님. 따라서 task-specific architecture가 필요함
- 표현을 미리 계산하기만 하면 그 다음에도 그냥 주기만 하면 되므로 연산적인 면에서 이득

따라서 이번에는 BERT를 두 가지 방법에 적용하여  CoNLL-2003 NER에 적용해보도록 하겠다.

BERT의 인풋은 대소문자 상관있는 (case-preserving) WordPiece 모델을 사용하고, 최대한 데이터를 많이 학습시켰다 (we include  the  maximal  document  context  provided by the data).

이는 일반적으로 tagging 작업으로 간주하므로 본 실험에서도 tagging으로 간주하여 작업했으나 CRF는 사용하지 않았다. 첫 번째 sub-token에 대한 representation을 token-level classifier의 input으로 사용했다.

또한, 파인튜닝은 전혀 하지 않았으며, BERT의 embedding은 초기화된 biLSTM에 넣고 classification을 진행했다 (2 layers, 768 dim).

다음은 이에 대한 결과로, BERT_LARGE (fine-tuning)가 SOTA와 비슷한 성능을 낸 것을 확인할 수 있다. feature-based에서 최고의 결과는 트랜스포머 상의 네 개의 레이어를 concat한 결과로, fine-tuning보다 0.3 뒤쳐진 결과이다.

이로서 BERT는 두 가지 방법 모두에 효과적임을 파악할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/103155413-08758580-47e3-11eb-91b8-04f03a872be6.png){: .align-center}


# Appendix

앞서 중간중간 필요한 내용은 삽입해놨으므로 안 다룬 부분만 다두도록 하겠다.

## A.4  Comparison of BERT, ELMo ,and OpenAI GPT

여기서는 BERT, ELMo, GPT의 차이에 대해 다루도록 하겠다. 이들 구조의 차이점은 아래 그림에 잘 나타나있다. 한 가지 짚고 넘어갈 점은, 아키텍쳐 외에도 BERT와 GPT는 fine-tuning approach지만, ELMo는 feature-based approach라는 차이가 있다.

![image](https://user-images.githubusercontent.com/47516855/103156160-49709880-47e9-11eb-89e6-886dda9f86ff.png){: .align-center}

BERT와 가장 비교할만한 것은 GPT이다. GPT는 LTR transformer LM으로 큰 데이터 셋에 대해 학습했다. BERT의 디자인은 의도적으로 GPT를 참고하였는데, 이는 두 개의 차이를 최소화하여 비교하기 위함이다.

가장 큰 차이점은 bi-directionality와 pre-training과정이지만, GPT와 BERT가 학습하는 것에도 차이가 있다.
- GPT는 BookCorpus를 학습한 반면 BERT는 위키피디아도 추가로 학습
- GPT는 [SEP]와 [CLS]를 fine-tuning시에만 사용하는 반면 BERT는 둘 모두와 sentence A/B embedding 모두 pre-train 부터 사용
- GPT는 32,000 단어에 대해 1M step을 학습하지만, BERT는 128,000 단어에 대해 1M step을 학습
- GPT의 learning rate는 파인튜닝시 5e-5로 고정되어 있지만 BERT는 task마다 다르게 적용

이러한 영향을 분리하여 실험하기 위해 [Effect of Pre-training Tasks](/project/nlp/bert-review/##Effect of Pre-training Tasks)에서 ablation experiment를 진행하였고, 그 결과 성능 향상의 주역은 MLM/NSP와 bidirecionality임을 확인하였다.

## C.1 Effect of Number of Training Steps

![image](https://user-images.githubusercontent.com/47516855/103097541-f234c180-464a-11eb-854b-07f9b423ec66.png){: .align-center}

위는 MNLI에 대한 결과로, k step에 대한 pre-training으로부터 fine-tuning한 결과이다. 이는 다음과 같은 질문에 대답할 수 있게 해준다.

1. 질문: BERT가 fine-tuning accuracy를 달성하기 위해 실제로 큰 pre-training (128,000words/batch  *  1,000,000  steps)가 필요한가?
- 답변: 그렇다. 500k와 1M을 비교해보면 BERT_BASE는 1.0%의 성능 향상을 이뤄냈다.
2. 질문: 모든 단어가 아닌 15%의 단어에 대해서만 MLM을 진행하여 LTR보다 더 느리게 수렴하는가?
- 답변: MLM은 LTR보다 늦게 수렴한다. 그러나 정확도 측면에서보면 MLM은 LTR을 즉시 추월하기 시작한다.
