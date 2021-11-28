---
title:  "How to Fine-Tune BERT for Text Classification? review"
excerpt: "BERT의 fine-tuning 전략을 알아보자"
toc: true
toc_sticky: true
permalink: /project/nlp/How-to-Fine-Tune-BERT-for-Text-Classification-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
  - BERT
use_math: true
last_modified_at: 2021-06-23
---

## 들어가며

최근 회사에서 BERT를 활용하여 소셜 미디어 텍스트 분석을 진행하게 되었는데, 데이터셋부터 fine-tuning까지 여러 과정에서 문제를 겪게 되었다. 사실 너무 naive하게 생각해서 BERT만 있으면 되는줄 알았는데 여러 과정에서 신경쓸 부분이 많았다. 본 논문은 이렇듯 fine-tunining 과정에서 생길 수 있는 문제들이나 전략을 다루며, 효과적인 fine-tuning을 탐색하는 논문이다. 즉, pre-trained model에 데이터를 단순하게 때려박는다고 높은 성취를 달성할 수는 없다는 뜻이다. 만일 BERT를 사용하여 downstream task를 진행한다면 본 논문을 보며 fine-tuning 전략을 구상하는 것이 바람직해보인다.

본 글을 이해하기 이전에 BERT에 대한 이해가 필수적이므로 리뷰를 보고 오도록 하자. 리뷰는 [이곳](/project/nlp/bert-review/)에서 확인할 수 있다.

본 논문의 원본은 [이곳](https://arxiv.org/abs/1905.05583)에서 확인할 수 있으며, repository는 [여기](https://github.com/xuyige/BERT4doc-Classification)서 확인할 수 있다.

## 1. Introduction

텍스트 분류는 NLP에서 다루는 클래식한 문제로, 주어진 text sequence에 대해 미리 정해둔 카테고리를 할당하는 것을 말한다. 이 과정에서 text representation이라는 중요한 중간단계가 있다. 이전 연구들은 이러한 text representation을 학습하기 위해 다양한 뉴럴 네트워크를 사용하였는데, 이에는 CNN, RNN, attention mechanism들이 있다.

또한 이와는 다르게 적지않은 수의 연구가 대용량 코퍼스에 대해 학습한 pre-trained model이 text classification이나 다른 NLP task에 도움이 된다는 점을 증명하였다. 이는 새로운 모델을 학습할 때 밑바닥부터 학습하는 것을 방지해준다. Pre-trained model의 한 종류로는 word2vec, GloVe와 같은 word embedding이나, CoVe, ELMo와 같은 contextualized word embedding이 있다. 이러한 word embedding은 main task에 대해 추가적인 feature가 된다. Pre-training model의 또 다른 종류로는 sentence-level이 있다. [Howard and Ruder (2018)](https://arxiv.org/abs/1801.06146)같은 경우는 ULMFiT을 제안하였는데, 범용적으로 연구되는 6개의 text classification dataset에 대해 pre-trained language model에 fine-tuning을 적용하여 SOTA를 달성하였다. 최근에는 OpenAI GPT [(Radford et al.,
2018)](https://openai.com/blog/language-unsupervised/)와 [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)와 같이, pre-trained language model이 많은 양의 unlabeled data를 활용하여 일반적인 language representation을 배우는데 유용하다는 것을 보여주었다. BERT는 multi-layer bidirectional Transformer에 기반하고, 일반적인 텍스트에 대해 masked word prediction과 next sentence prediction을 학습한다.

비록 BERT가 다양한 NLU task에 대해 엄청난 결과를 달성하였을지라도, 아직 잠재적으로 완전히 탐구되지는 않았다. BERT를 강화시켜 target task의 성능을 더욱 향상시키는 연구는 아직 미미하다.

본 연구에서는 텍스트 분류를 위한 BERT의 활용을 극대화하기 위한 방법을 조사하였다. BERT를 fine-tuning하여 텍스트 분류의 성능을 향상시키는 몇 가지 방법에 대해 탐색하고, BERT의 자세한 분석을 위해 exhaustive experiments을 디자인하였다.

다음은 본 연구의 contribution이다.
- Pre-trained BERT를 fine-tuning하는 일반적인 solution을 제안한다. 여기엔 다음과 같은 세 가지 스텝이 있다.
  - (1) domain data/fine-tuning data를 통해 pre-trained BERT를 further training
  - (2) 몇가지 연관된 task가 존재한다면 이를 이용하여 multi-task learning을 진행
  - (3) target task에 대한 fine-tuning
- 또한, BERT의 target task에 대한 fine-tuning method를 조사하였다. 여기에는 long text에 대한 전처리, layer selection, layer wise learning rate, catastrophic forgetting, low-shot learning problems을 포함한다.
- 범용적으로 연구되는 7개의 영어 텍스트 분류와 1개의 중국어 텍스트 분류에 대해 SOTA를 달성하였다.

## 2. Related Work

다른 task에서 학습된 지식을 빌려오는 것은 NLP 분야의 떠오르는 관심사이다. 본 장에서는 이와 연관된 language
model pre-training, multi-task Learning의 두 개의 방법론을 가볍게 살펴볼 것이다.

### 2.1 Language Model Pre-training

현대 NLP system의 중요한 구성요소인 pre-trained word embedidng (word2vec, GloVe)은 처음부터 학습한 embedding을 상당히 넘어서는 성능을 제공한다. Sentence embedding ([Kiros et al.,2015](https://arxiv.org/abs/1506.06726) (Skip-Thought Vectors), [Logeswaran and Lee, 2018](https://arxiv.org/abs/1803.02893))나 paragraph embedding ([Le and Mikolov, 2014](https://arxiv.org/abs/1405.4053) (Doc2vec))같은 word embedding의 일반화된 버전 또한 downstream model에서 feature로 사용된다.

ELMo는 language model에서 얻은 embedding을 concatnate하여 main task에 대한 추가적인 feature로 사용하였고, 이를 통해 몇몇 주요 NLP 벤치마크에 대해 SOTA를 달성하였다. Unsupervised data를 활용한 pre-training에 더해 대량의 supervised data를 통한 transfer learning 또한 natural language inference ([Conneau et al., 2017](https://arxiv.org/abs/1705.02364))와 machine translation ([McCann et al., 2017](https://arxiv.org/abs/1708.00107) (CoVe))에서 좋은 성능을 보였다. 

더욱 최근에는 큰 네트워크에 대해 대량의 unlabeled data를 활용한 pre-training language model과 downstream task에 대해 fine-tuning을 섞는 방법이 몇몇 NLU task에서 큰 발전을 이뤄냈다. 이에는 OpenAI GPT [(Radford et al.,
2018)](https://openai.com/blog/language-unsupervised/)와 [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)가 있다. [Dai and Le (2015)](https://arxiv.org/abs/1511.01432)는 language model fine-tuning를 사용하였는데 10k labeled example에 대해 overfitting이 발생하였다. 반면 [Howard and Ruder (2018)](https://arxiv.org/abs/1801.06146)는 ULMFiT을 통해 텍스트 분류에서 SOTA를 달성하였다. BERT는 대량의 corpus에 대해 *Masked Language Model Task*과 *Next Sentence Prediction Task*를 pre-train한 것으로, 기존의 bidirectional language model (biLM)이 두 개의 unidirectional language model (왼-오, 오-왼)의 조합에 제한이 있던것과는 다르게, BERT는 Masked Language Model을 통해 임의로 가려지거나 바뀐 단어를 예측하는 작업을 수행한다. BERT는 처음으로 제안된 fine-tuning based representation model로, 다양한 NLP task에서 SOTA를 달성했으며, fine-tuning method의 엄청난 잠재력을 증명하였다. 본 논문에서는 텍스트 분류를 위한 BERT fine-tuning method를 더 탐구해보도록 한다.

### 2.2 Multi-task learning

Multi-task learning은 이와 관련된 또 다른 갈레길로 볼 수 있다. [Rei (2017)](https://arxiv.org/abs/1704.07156)와 [Liu et al. (2018)](https://arxiv.org/abs/1709.04109)는 multi-task learning을 통해 language model과 main task model을 동시에 학습하였다.

**[Rei (2017)](https://arxiv.org/abs/1704.07156)의 architecture**
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/122870245-61743f80-d368-11eb-83d8-2e26ff956198.png){: .align-center}{: width="900"}

**[Liu et al. (2018)](https://arxiv.org/abs/1709.04109)의 architecture**
{: .text-center}

![image](https://user-images.githubusercontent.com/47516855/122870892-3807e380-d369-11eb-877f-dd143cb3b9b0.png){: .align-center}{: width="900"}

[Liu et al. (2019)](https://arxiv.org/abs/1901.11504)은 [Liu et al. (2015)](https://www.aclweb.org/anthology/N15-1092/)의 모델에 BERT를 합친 MT-DNN을 제안하였다. MT-DNN은 아래 그림의 shared layer를 shared text encoding layer로 사용한 모델이다.

**MT-DNN**
{: .text-center}
![image](https://user-images.githubusercontent.com/47516855/122871536-2246ee00-d36a-11eb-9b16-98cf5e393ca4.png){: .align-center}{: width="900"}

Multi-task learning은 매번 처음부터 학습해야되기 때문에 비효율적이고, 대부분 task-specific objective functions를 신중하게 설정해야 한다 ([Chen et al., 2017](https://arxiv.org/abs/1711.02257)). 그러나 공유되는 pretrained model을 전부 사용하는 multi-task BERT fine-tuning을 사용하여 이러한 문제를 해결할 수 있다.

## 3. BERT for Text Classification
BERT-base model은 12개의 transformer encoder 블록과 self-attention head, 768의 hidden size를 갖는다. BERT는 512개의 토큰을 input으로 취하고, sequence의 representation을 내뱉는다. Sequence는 한 개 이상의 segment로 구성되어 있고, squence의 첫번째 token은 항상 <span style="font-family:Courier New">[CLS]</span>로 이는 classification embedding을 포함한다. 또 다른 special token인 <span style="font-family:Courier New">[SEP]</span>는 segment를 분리하는데 이용한다.

BERT는 텍스트 분류에서 첫번째 토큰인 <span style="font-family:Courier New">[CLS]</span>의 마지막 hidden state $\boldsymbol h$를 전체 sequence에 대한 representation으로 취급한다. 그리고 간단한 softmax classifier가 BERT의 맨 위에 추가되고, label $c$일 확률을 계산한다.

$$
p(c \rvert \boldsymbol h) = \text{softmax}(W) \boldsymbol h \tag{1}
$$

$W$는 task specific parameter matrix가 된다. 여기서 정답 label에 대한 log-probability를 maximization하는 방식으로 BERT의 모든 파라미터와 $W$를 fine-tune하게 된다.

## 4. Methodology

Target domain의 NLP task에 BERT를 적용할 때 올바른 fine-tuning 전략이 요구된다. 본 연구에서는 적절한 fine-tuning method를 다음과 같은 세 가지 방법으로 살펴볼 것이다.

**1) Fine-Tuning Strategies:** target task에 대해 fine-tuning을 진행할 때 BERT를 활용할 수 있는 방법이 무궁무진하다. 예를 들어 다른 레이어는 다른 의미적/구문적 정보를 갖게 되므로 어떠한 레이어가 target task에 좋은지 궁금할 수 있다. 또한 어떻게 optimization과 learning rate를 구성할지에 대해서도 궁금해 할 수 있다.

**2) Further Pre-training:** BERT는 일반적인 도메인에 학습되므로 target domain과 다른 분포를 갖을 수 있다. 자연스럽게 떠오르는 생각은 target domain에 BERT를 further training하는 것이다.

**3) Multi-Task Fine-Tuning:** Pre-trained LM model없이 multi-task learning만으로 여러 task에 대해 지식 공유의 활용성을 증명해왔다. 만일 target domain에 대해 몇몇 개의 task만 수행가능하다면, 그래도 여전히 모든 task에 대해 fine-tuning하는 것이 유용할 것인지 의문이 들게될 것이다.

본 연구의 fine-tuning 방법은 아래 그림에 소개되어있다.

![image](https://user-images.githubusercontent.com/47516855/122883325-09453980-d378-11eb-832c-05428859fd81.png){: .align-center}{: width="500"}

### 4.1 Fine-Tuning Strategies

뉴럴 네트워크의 다른 레이어는 다른 수준의 의미적/구문적 정보를 갖고 있다 ([Yosinski et al., 2014](https://arxiv.org/abs/1411.1792); [Howard and Ruder, 2018](https://www.aclweb.org/anthology/P18-1031/)).

BERT를 target task에 적용하기 위해 본 연구는 다음과 같은 여러 요소들을 고려할 필요가 있다.
1. BERT의 maximum length가 512이므로 long text에 대한 전처리 요소
2. Layer selection. 공식 BERT-base model은 embedding layer, 12-layer encoder, pooling layer를 포함한다. 과연 텍스트 분류를 위한 최고의 효율을 갖는 구조는 어떤것인가?
3. Overfitting. 적절한 learning rate와 optimizer가 필요.

직관적으로 하위의 레이어는 좀 더 일반적인 정보를 갖고 있다. 우리는 이를 다른 learning rate로 fine-tune할 수 있다.

[Howard and Ruder (2018)](https://www.aclweb.org/anthology/P18-1031/) (ULMFiT)처럼 parameter $\theta$를 $\\{\theta^1, \cdots, \theta^L \\}$로 나누고, 다음과 같이 parameter를 업데이트 한다.

$$
\theta^l _t = \theta^l _{t-1} - \eta^l \cdot \nabla _{\theta^l} J(\theta) \tag{2}
$$

여기서 $\eta^l$은 $l$번째 레이어의 learning rate이다.

본 연구는 base learning rate를 $\eta^L$로 설정하고, $\eta^{k-1} = \xi \cdot \eta^k$로 설정한다. 여기서 $\xi$는 decay factor로 1 이하의 값을 갖는다.  $\xi < 1$일 경우 하위 레이어는 상위 레이어보다 낮은 learning rate를 갖는다.  $\xi =1$일 경우 모든 레이어는 같은 learning rate를 갖는다. 이는 일반적은 SGD와 같아진다. 이에 대해서는 추후에 Sec 5.3에서 다시 보도록 한다.

### 4.2 Further Pre-training

BERT 모델은 일반적인 도메인 corpus에 대해 pre-train 된다. 영화 리뷰와 같은 특정한 도메인의 텍스트 분류에서는 BERT의 분포와 데이터 분포가 달라질 수 있다. 그러므로 domain-specific data에 대해 MLM/NSP를 further pre-train해야한다. 여기서 세 가지의 pre-training 접근법이 사용된다.
1. With-in task pre-training. BERT가 target task data에 대해 further training 된다.
2. In-domain pre-training. Pre-training data가 target task와 같은 도메인에서 얻어진다. 예를들어 여러가지 종류의 감성분석 작업이 있고, 이들의 분포가 비슷하다고 하자. 이 경우 우리는 이러한 데이터로부터 학습 데이터를 조합하여 BERT의 further pre-training을 수행할 수 있다.
3. Cross-domain pre-training. Pre-training data가 target task의 도메인과 상관없이 얻어지는 것이다.

이에 대해서는 Sec 5.4에서 자세히 살펴보도록 한다.

### 4.3 Multi-Task Fine-Tuning

Multi-task Learning는 여러개의 연관있는 supervised task로부터 얻은 지식을 공유하는 방법으로 이 또한 효과적이다. [Liu et al. (2019)](https://arxiv.org/abs/1901.11504)과 비슷하게 텍스트 분류를 위해 multi-task learning으로 BERT를 fine-tuning한다. 

모든 task는 BERT layer와 embedding layer를 공유한다. 공유되지 않는 레이어는 오로지 마지막의 분류를 위한 레이어뿐이다. 이는 Sec 5.5에서 더욱 자세히 살펴보도록 한다.

## 5. Experiments

앞서 살펴본 fine-tuning method를 7개의 영어 데이터와 1개의 중국어 데이터에 실험해보았다. 여기서 base BERT 모델을 사용하였다.

### 5.1 Datasets

연구의 방법론을 평가하기 위해 널리 사용되는 8개의 데이터셋을 사용하였다. 데이터셋은 다양한 수의 문서와 길이, 그리고 감성분석, 질문 분류, 토픽 분류와 같은 일반적인 텍스트 분류 작업을 포함한다. 이에 대한 설명은 아래 테이블에 소개되어있다.

![image](https://user-images.githubusercontent.com/47516855/122887629-0ea48300-d37c-11eb-90b5-b9cef8dd10f9.png){: .align-center}{: width="900"}

**Sentiment analysis:** binary film review IMDb dataset ([Maas et al., 2011](https://www.aclweb.org/anthology/P11-1015/))와  [Zhang et al.(2015)](https://dl.acm.org/doi/10.5555/2969239.2969312)이 사용한 Yelp review dataset의 binary and five-class version을 사용하였다.

**Question classification**:  six-class version of the TREC dataset ([Voorhees and Tice,
1999](https://www.researchgate.net/publication/2888359_The_TREC-8_Question_Answering_Track_Evaluation))와 [Zhang et al.(2015)](https://dl.acm.org/doi/10.5555/2969239.2969312)에 의해 만들어진 Yahoo! Answers dataset을 이용한다. TREC dataset은 open domain에 대한 질의 분류 데이터 셋으로, fact-based questions이 여러 의미적 카테리고리로 분리되어 있다. 다른 document-level의 데이터셋과 비교하여 TREC는 sentence-level이며, 더욱 작은 수의 샘플로 이루어져있다. Yahoo! Answers dataset은 큰 데이터셋으로 1400K의 train sample로 이루어져있다.

> Fact-based question (factoid question)에 대한 설명은 아래와 같다.
>
> A non-factoid question answering (QA) is an umbrella term that covers all question-answering topics beyond factoid question answering. As a quick reminder: a factoid QA is about providing concise facts. For example, "who is the headmaster of Hogwarts?", "What is the population of Mars", and so on, so forth.
> 
> In constrast, a non-factoid question can be about anything. You can be asked to provide an answer to a math problem, to explain how to fix a specific model of a car, and so on, so forth. Answering multiple-choice questions also belongs to the area of non-factoid QA, though, there might be some overlap with factoid QA in this task.
> 
> An important sub-problem of non-factoid QA consists in finding already existing answers posted on community question answering sites such as Quora! This is very much an IR task. [출처: Quora-Natural Language Processing: What is "Non-factoid question answering"?](https://www.quora.com/Natural-Language-Processing-What-is-Non-factoid-question-answering)
> 
> Not surprisingly, the most common factbased question words are who, what, when, where, why, and how. It is assumed that the set of question words is fixed and known a *priori*, although it can also be learned automatically. [Metzler and Croft, 2004](https://scholarworks.umass.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1082&context=cs_faculty_pubs)

**Topic classification:** [Zhang et al.(2015)](https://dl.acm.org/doi/10.5555/2969239.2969312)에 의해 만들어진 large-scale AG’s News and DBPedia 데이터를 이용한다. 중국어의 경우엔 Sogou news corpus를 만들어서 학습과 테스트를 진행한다. [Zhang et al.(2015)](https://dl.acm.org/doi/10.5555/2969239.2969312)과는 다르게 병음을 이용하기보단 중국어를 그대로 이용한다. 데이터셋은 SogouCA and SogouCS news corpora (Wang et al., 2008)를 이용한다. 뉴스의 카테고리는 URL을 기반으로 결정한다. 예를들어 스포츠의 경우 "http://sports.sohu.com”와 같이 나오게 된다. 전체적으로 "sports", "house”, "business”, "entertainment”, "women”, "technology”의 6개의 카테고리를 이용한다. 각 클래스 당 trainig example의 수는 9K이며, 테스트의 경우 1K이다.

**Data preprocessing**: [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)와 같이 WordPiece embeddings [(Wu et al.,
2016)](https://arxiv.org/abs/1609.08144)을 사용하여 30,000개의 토큰 사전과 단어를 분리하는 것을 ##으로 표현한다. 따라서 데이터셋 내의 문서의 길이는 word piece를 기준으로 한다. BERT에 대한 further pre-training을 위해서는 spaCy를 사용하여 영어 데이터셋에 대해 문장 분리를 진행하며, 중국어를 대상으로 할 때는 ".", "?", "!"를 기준으로 분리한다.

### 5.2 Hyperparameters

768의 hidden size, 12 Transformer blocks, 12 self-attention heads를 갖는 BERT-base model을 사용한다. Further pre-train은 한 개의 TITAN Xp GPU에 batch size 32를 이용하고, 문장의 최대길이는 128, learning rate는 5e-5, training step은 100,000, warp-up step은 10,000으로 한다.

Fine-tuning에서는 4개의 TITAN Xp GPU와 24 batch를 사용하여 GPU 메모리 전체가 사용될 수 있도록 한다. Dropout의 경우 항상 0.1로 세팅한다. Adam optimizer를 사용하고, $\alpha=0.9, \beta=0.999$를 사용한다. [Howard and Ruder (2018)](https://www.aclweb.org/anthology/P18-1031/) (ULMFiT)에서 소개된 것처럼 *slanted triangular learning rates*를 사용하고, 기본 learning rate는 2e-5, warm-up 비율은 0.1로 한다. 최대 에폭은 4로 경험적으로 세팅하였고, validation set을 이용하여 test하여 최고의 모델을 찾아내었다.

> Slanted Triangular Learning Rates (STLR) is a learning rate schedule which first linearly increases the learning rate and then linearly decays it, which can be seen in Figure to the right. It is a modification of Triangular Learning Rates, with a short increase and a long decay period.
>
> ![image](https://paperswithcode.com/media/methods/new_lr_plot_tNtxBIM.jpg){: .align-center}{: width="600"}
>
> [Papers with code - Slanted Triangular Learning Rates](https://paperswithcode.com/method/slanted-triangular-learning-rates#)


### 5.3 Exp-I: Investigating Different Fine-Tuning Strategies

이번 subsection에서는 IMDb dataset을 이용하여 다른 fine-tuning 전략의 효과를 살펴본다. 구글에서 개발한 [official pre-trained model](https://github.com/google-research/bert)을 기본 encoder로 사용한다.

#### 5.3.1 Dealing with long texts

BERT에서 다루는 최대 문장 길이는 512이다. BERT에 text classification 문제를 적용할 때 생기는 첫번째 문제는 512가 넘는 문장을 어떻게 전처리할 것인가이다. 긴 article을 다루기 위해 다음과 같은 방법들을 시도하였다.

**Truncation methods:**  article의 핵심 정보는 주로 문장의 시작과 끝에 존재한다. BERT fine-tuning을 수행하기 앞서 다음과 같은 세 가지 방법으로 텍스트를 잘라내었다.

1. **head-only**: 첫 510 토큰
2. **tail-only:**: 마지막 510 토큰
3. **head+tail:**: 경험적으로 선택한 첫 128 토큰과 마지막 382 토큰

**Hierarchical methods**: 인풋 텍스트를 $k= L/510$개로 나누어 BERT가 $k$개의 text에 대한 representation을 얻도록한다. 각 토큰에 대한 표현은 마지막 레이어의 <span style="font-family:Courier New">[CLS]</span> 토큰의 hidden state가 될 것이다. 그후 mean pooling, max pooling, self-attention을 사용하여 모든 토큰에 대한 representation을 결합한다.

Table 2는 위 방법에 대한 효율성을 나타낸 것이다. Truncation methods 중 **head+tail**이 제일 좋은 것으로 나타났다. 따라서 앞으로의 실험에서도 이 방법을 통해 문장을 자르도록 한다.

![image](https://user-images.githubusercontent.com/47516855/123041037-bb8c0800-d42f-11eb-941d-d2ab83f7e83b.png){: .align-center}{: width="500"}

#### 5.3.2 Features from Different layers

각 레이어는 인풋 텍스트에 대한 서로 다른 정보를 갖게된다. 여기서는 다른 레이어들간의 정보의 유용성을 평가하도록 한다. 그 후 fine-tuning하여 test error rate에 대한 성능을 측정하도록 한다.

아래 Table 3은 레이어 별 fine-tuning의 성능을 나타낸다. BERT의 마지막 레이어의 성능이 제일 좋으므로, 앞으로의 실험에서도 이를 사용한다.

![image](https://user-images.githubusercontent.com/47516855/123041277-1291dd00-d430-11eb-9bb1-582d79325b57.png){: .align-center}{: width="500"}


#### 5.3.3 Catastrophic Forgetting

Catastrophic forgetting (McCloskey and Cohen, 1989)은 새로운 지식을 학습하는 과정에서 pre-train 지식이 사라지는 현상으로, transfer learning에서 흔하게 볼 수 있는 문제이다. 그러므로 BERT 또한 이러한 문제가 있는지를 살펴보았다.

다른 learning rate를 이용하여 BERT를 fine-tuning하였고, error rate에 대한 learning curve는 Figure 2에 나와있다.

![image](https://user-images.githubusercontent.com/47516855/123043306-0fe4b700-d433-11eb-8450-921341d7fa12.png){: .align-center}{: width="900"}

2e-5와 같은 낮은 learning rate에서는 catastrophic forgetting이 일어나는 것을 확인하였다. 반대로 높은 learning rate에서는 수렴하는데 실패한다.

#### 5.3.4 Layer-wise Decreasing Layer Rate

Table 4는 다른 learning rate와 decay factor에 대한 성능을 나타낸 표이다 (식 (2) 참고). 하위 레이어에 낮은 learning rate를 할당하는 것은 BERT를 fine-tuning하는데 효과적이며, 적절한 세팅값은 $\xi=0.95, \text{lr}=2.0-5$로 밝혀졌다.

![image](https://user-images.githubusercontent.com/47516855/123044096-30614100-d434-11eb-9340-09337dbfef05.png){: .align-center}{: width="400"}

### 5.4 Exp-II: Investigating the Further Pretraining

Supervised learning을 통해 BERT를 fine-tuning하는 것 외에도 BERT의 MLM/NSP에 대해 further training을 진행할 수도 있다. 본 장에서는 이에 대한 효율성을 살펴본다. 앞서 fine-tuning 전략에서 도출된 최적의 세팅을 여기서도 사용하도록 하겠다.

#### 5.4.1 Within-Task Further Pre-Training

따라서 with-in task further training에 대해 먼저 진행하도록 한다. 여러 스텝으로 나누어 학습을 진행하고, 텍스트 분류 작업을 진행한다. 

![image](https://user-images.githubusercontent.com/47516855/123045932-76b79f80-d436-11eb-9d81-434200d4695c.png){: .align-center}{: width="500"}

Figure 3에서 볼 수 있듯, further pre-training이 BERT의 target task에 유용한 것을 알 수 있다. 가장 좋은 성능은 100K training step에서 달성하였다.


#### 5.4.2 In-Domain and Cross-Domain Further Pre-Training

target task에 대한 training data에다가, 추가적으로 같은 도메인에서 얻은 데이터를 이용하여 further pre-train을 진행한다. 완벽하게 정확하진 않지만 얼추 7개의 영어 데이터를 topic, sentiment, question의 세 가지 도메인으로 분리하였다. 그러므로 cross task pre-training에 대해 광범위한 실험을 진행하였다. 각 task는 다른 도메인으로 간주한다.

Table 5에서 볼 수 있듯, 거의 모든  further pre-training 모델이 원본 BERT(w/o pretrain)보다 더 좋은 성능을 보이는 것을 확인하였다.

![image](https://user-images.githubusercontent.com/47516855/123046546-3278cf00-d437-11eb-8ff3-a60e8314bc94.png){: .align-center}{: width="900"}

일반적으로  in-domain pretraining은 within-task pretraining보다 더 나은 성능을 보인다. 작은 샘플 수를 갖는 sentence-level TREC dataset에서는 within-task pre-training가 성능을 저하시키는 것으로 나타났으며, Yah. A. corpus을 활용한 in-domain pre-training에서는 TREC에서 좋은 결과를 보였다.

Cross-domain pre-training (Table 5의 all)은 명확한 이점을 보이지 않았다. BERT가 일반적인 도메인의 데이터로 pre-train되었다는 것을 생각하면 납득이 되는 결과다.

또한, IMDb와 Yelp가 sentiment domain에서는 서로 도움이 되지 않는 것을 확인하였다. 이 둘의 도메인이 영화와 음식이기 때문인 것으로 추정된다. 이들의 데이터 분포가 확연히 다른 점도 확인된다.

#### 5.4.3 Comparisons to Previous Models

본 연구의 모델을 다음과 같은 텍스트 분류에서 SOTA를 달성한 모델들과 비교해보았다.
- CNN-based methods: Char-level CNN ([Zhang et al.(2015)](https://dl.acm.org/doi/10.5555/2969239.2969312)), VDCNN ([Conneau et al., 2016](https://www.aclweb.org/anthology/E17-1104/)), DPCNN ([Johnson and Zhang, 2017](https://www.aclweb.org/anthology/P17-1052/))
- RNN-based models: D-LSTM ([Yogatama et al., 2017](https://arxiv.org/abs/1703.01898)), Skim-LSTM ([Seo et al., 2017](https://arxiv.org/abs/1711.02085)), hierarchical attention networks ([Yang et al., 2016](https://www.aclweb.org/anthology/N16-1174/))
- feature-based transfer learning methods: rigion embedding ([Qiao et al., 2018](https://openreview.net/forum?id=BkSDMA36Z)), CoVe ([McCann et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/20c86a628232a67e7bd46f76fba7ce12-Abstract.html))
- Language model fine-tuning method: ULMFit [Howard and Ruder (2018)](https://arxiv.org/abs/1801.06146)

biLSTM에 self-attention을 활용한 모델([Lin et al., 2017](https://arxiv.org/abs/1703.03130))에 input embedding으로 BERT를 사용하는 BERT-Feat을 구현하였다. BERT-IDPT-FiT의 결과는 Table 5의 'all sentiment', 'all question', 'all topic'에 나와있고, BERT-CDPT-FiT의 결과는 'all'에 나와있다.

아래 Table 6에 나와있듯 BERT-Feat은 ULMFiT을 제외한 모든 baseline을 넘어서는 성능을 보인다. DBpedia에서 BERTFeat보다 약간 낮은 결과를 보이는걸 제외하면 BERT-FiT은 다른 7개의 task에서 BERT-Feat보다 좋은 성능을 낸다.

![image](https://user-images.githubusercontent.com/47516855/123051059-525ec180-d43c-11eb-880d-ef8bd18ac569.png){: .align-center}{: width="800"}

또한, 세가지 further pre-training 모두 다 BERT-FiT보다 좋은 것으로 나타났다. BERT-Feat을 기준으로 각 task에 대해 다른 BERT-FiT 모델의 성능 평균 상승치를 계산하였다. BERT-IDPT-FiT이 평균 error test를 18.75%낮춤으로서 제일 좋은 성능을 보이는 것으로 밝혀졌다.

### 5.5 Exp-III: Multi-task Fine-Tuning

텍스트 분류를 위한 몇몇 데이터 셋이 이미 존재하지만, 가용한 모든 데이터를 활용하기 위해 multi-task learning을 통한 fine-tuning을 진행하였다. IMDb, Yelp P., AG, DBP의 4가지 종류의 영어 텍스트 분류 데이터셋을 사용하였다. Yelp F.의 테스트셋에 Yelp P.의 학습셋이 섞여있기 때문에 이는 제외하였다.

실험은 official uncased BERT-base에다가 7개 데이터셋 각각에 대해  further pre-trained를 진행한 웨이트에 실험하였다. 각 분류 subtask에 대해 좋은 결과를 달성하기 위해 fine-tuning이후에 낮은 learning rate로 각 데이터셋에 fine-tuning을 추가로 진행하였다.

Table 7은 multi-task fine-tuning에 대한 결과를 보여주며, 그 효능을 증명한다. 그러나 multi-task fine-tuning이 BERT-CDPT의 Yelp P.와 AG에는 큰 도움이 되지 않는 것으로 보인다 (BERT-CDPT-MFiT-FiT).

![image](https://user-images.githubusercontent.com/47516855/123054399-c64e9900-d43f-11eb-936d-b1b1f83418a3.png){: .align-center}{: width="500"}

BERT-CDPT가 이미 풍부한 domain-specific information를 포함하고 있기 때문에 Multi-task fine-tuning과 cross-domain pre-training은 서로 대체할 수 있는 방법으로 보인다. 또한, multi-task learning는 연관된 텍스트 분류 subtask의 일반성을 향상시키기에 필수적이지는 않은 것으로 보인다.

### 5.6 Exp-IV: Few-Shot Learning

pre-trained 모델의 장점 중 하나는 작은 데이터를 갖고있는 task도 학습할 수 있다는 점이다. 본 연구에서 다른 데이터셋에 대한 BERT-FiT과 BERT-ITPT-FiT의 성능을 평가하였으며, IMDb training data의 일부를 선택하여  BERT-FiT와 BERTITPT-FiT를 학습하였다. 이 결과는 Figure 4에 나와있다.

![image](https://user-images.githubusercontent.com/47516855/123055285-af5c7680-d440-11eb-9668-44ba1073c7b5.png){: .align-center}{: width="500"}

본 실험결과를 통해 BERT가 작은 데이터에도 매우 좋은 성능향상을 이끌어낸다는 것을 증명하였다. Further pre-trained BERT는 오로지 0.4%의 학습 데이터를 이용하여 성능을 더 이끌어낼 수 있었다 (error rate 17.29% → 9.23%).

### 5.7 Exp-V: Further Pre-Training on BERT Large

이번에는 BERT<sub><span style="font-family:Courier New">LARGE</span></sub>에서도 비슷한 결과를 낼 수 있는지 실험하였다. BERT<sub><span style="font-family:Courier New">LARGE</span></sub>에 대해 Tesla-V100-PCIE 32G GPU, 24 batch, 128의 최대 문장길이, 120K training steps을 통해 further pre-train을 진행하였고, target task classifier BERT fine-tuning에 대해서는 24 batch, 4대의 Tesla-V100-PCIE 32G GPUs, 512 최대 문장 길이를 통해 BERT<sub><span style="font-family:Courier New">LARGE</span></sub>를 사용하였다.

![image](https://user-images.githubusercontent.com/47516855/123060877-1597c800-d446-11eb-9b51-4d37a1989d9a.png){: .align-center}{: width="500"}

Table 8에서 볼 수 있듯, ULMFiT은 BERT<sub><span style="font-family:Courier New">BASE</span></sub>에 비해 대부분 좋은 성능을 보였지만, BERT<sub><span style="font-family:Courier New">LARGE</span></sub>에 대해서는 그렇지 않았다. 그러나 the task-specific further pre-training에 대해서는 BERT<sub><span style="font-family:Courier New">BASE</span></sub>가 ULMFiT보다 좋은 성능을 내었다.  task-specific further pre-training에 대해 BERT<sub><span style="font-family:Courier New">LARGE</span></sub>를 fine-tuning한 결과 SOTA를 달성할 수 있었다.

## 6. Conclusion

본 연구에서는 텍스트 분류를 위해 광범위한 실험을 수행하였다. 이를 통해 몇가지 실험결과를 도출하였다. 1) 텍스트 분류를 위해서는 상위 레이어가 더 유용한 정보를 갖고 있다는 점, 2) 적절한 layer-wise로 learning rate를 decay할 경우 BERT는 catastrophic forgetting을 겪지 않는다는 점, 3) Within-task further pre-training/in-domain further pre-training은 성능을 극대로 향상시킬 수 있다는 점, 4) multi-task fine-tuning를 single task fine-tuning에 적용하면 도움이 되지만 further pre-training보단 효능이 작다는 점, 5) BERT는 작은 데이터셋에 대해서도 성능을 향상시킬 수 있다는 점이다.

이러한 발견들을 토대로 8개의 데이터셋에서 SOTA를 달성할 수 있었다.

