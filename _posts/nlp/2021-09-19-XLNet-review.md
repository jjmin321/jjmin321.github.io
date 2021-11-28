---
title:  "XLNet: Generalized Autoregressive Pretraining for Language Understanding review"
toc: true
toc_sticky: true
permalink: /project/nlp/XLNet-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-09-26
---

## 들어가며

XLNet은 카네기 멜론 대학교와 구글 브레인에서 나온 논문으로, 기존에 BERT가 bidirectionality를 얻는 과정에서 잃어버리는 정보를 autoregressive 방법을 이용하여 만회함과 동시에 bidirectionality를 달성한다.

- [원문 보러가기](https://arxiv.org/pdf/1906.08237.pdf)
- [XLNet repository 보러가기](https://github.com/zihangdai/xlnet)

## 1 Introduction

Unsupervised representation learning은 NLP 분야에서 엄청난 성공을 거둬왔다. 이러한 모델은 일반적으로 대량의 unlabeled text corpora로부터 pre-training을 통해 네트워크를 학습하고, downstream task에 대해  models/representations을 fine-tuning하게 된다. 이러한 고수준의 아이디어 하에 서로 다른 unsupervised pretraining objectives가 연구되어 왔다. 이 중 autoregressive (AR) language modeling와 autoencoding (AE)이 가장 큰 성공을 거둔 pretraining objective이다.

### AR Model

AR LM은 corpus의 **probability density**, 즉, **문장이 완성될 확률**을 autoregressive model을 통하여 추정한다.

수식을 사용하여 표기하면, 주어진 텍스트 열 $\mathbf x = (x _1 , \cdots, x _T)$에 대해, likelihood를 forward product $p(\mathbf x) = \prod ^T _{t=1} p(x _t, \rvert \mathbf x _{< t})$ 혹은 backward product $p(\mathbf x) = \prod _{t=1} ^T p(x _t, \rvert \mathbf x _{> t})$ 를 factorize한다.

그리고 다음과 같이 likelihood를 최대화하는 방식으로 pre-training을 진행한다.

$$
\max _\theta \log ~p _\theta (\mathbf x) = \sum^T _{t=1} \log ~p _\theta (x _t \rvert \mathbf x _{< t}) = \sum^T _{t=1} \log ~ \frac{\exp(h _\theta (\mathbf x _{1:t-1})^\intercal e(x _t))}{\sum _{x'} \exp (h _\theta (\mathbf x _{1:t-1})^\intercal e(x'))} \tag{1}
$$

여기서 $h _\theta (\mathbf x _{1:t-1})$는 RNN/Transformer와 같은 네트워크에 의해 생성되는 context representation이고, $e(x)$는 $x$의 임베딩이 된다. 

아무래도 $\max$가 각 식마다 들어가야 할 것 같지만, 우선은 논문에 소개된대로 적는다.
{: .notice--info}

AR 모델은 오직 단방향의 문맥만을 파악할 수 있으므로, deep bidirectional context를 모델링하는데는 효과적이지 않다.

### AE Model

AR 모델과는 다르게 AE에 기반한 pre-training 모델은 density estimation, 즉, **문장이 완성될 확률을 정확히 구하기**보단 손상시킨 데이터를 원본 데이터로 복원하는걸 중점으로 둔다. 따라서 전통적인 langauge model과는 다소 괴리감이 있다. 

이의 대표적인 예가 BERT로, 일부로 원본 데이터에 마스킹을 씌워 `[MASK]`토큰으로 변환한 후 원본 데이터를 예측하도록 한다. Density estimation이 objective에 포함되지 않으므로 복원과정에서 bidirectional context를 활용하게 된다.

수식으로 표현하면, 주어진 text sequence $\mathbf x = [x _1, \cdots, x _T]$에 대해 임의의 단어를 일정 확률 (15%)로 `[MASK]` 토큰으로 변경하여 오염된 버전의 sequence $\hat {\mathbf x}$를 생선한다. 마스킹된 토큰을 $\bar{\mathbf x}$라 하자. Training objective는  $\hat{\mathbf x}$으로부터 $\bar{\mathbf x}$를 복구하는 것이 된다.

$$
\max _\theta \log ~p _\theta (\hat{\mathbf x} \rvert \bar{\mathbf x}) 
\approx \sum^T _{t=1} m _t \log ~p _\theta (x _t \rvert \hat{\mathbf x}) 
= \sum^T _{t=1} m _t \log ~ \frac{\exp(H _\theta (\hat{\mathbf x} _t)^\intercal e(x _t))}{\sum _{x'} \exp (H _\theta (\hat{\mathbf x _t} _t)^\intercal e(x'))} \tag{2}
$$

$m _t=1$은 $x _t$가 마스킹되었음을 뜻하고, $H _\theta$는 Transformer로, $T$길이의 text sequence $\mathbf x$를 hidden vector의 sequence $H _\theta(\mathbf x) = [H _\theta(\mathbf x) _1, H _\theta(\mathbf x) _2, \cdots, H _\theta(\mathbf x) _T]$로 변환하는 역할을 한다.

AR 모델과 AE 모델의 장단점은 다음과 같다.

### Independence Assumption

식 (2)을 자세히 보면 같다($=$)가 아니고 근사한다($\approx$)이다. BERT는 joint conditional probability $p (\hat{\mathbf x} \rvert \bar{\mathbf x})$를 factorize하는데, 이는 마스크된 모든 토큰이 **각자 복원**된다는 뜻이다. 즉, 마스크된 토큰끼리는 **서로 독립적**이라는 가정하에 복원을 진행하게 된다.

예를들어 *New York is a ccity*라는 단어가 있다고 가정하자. BERT와 같은 AE 모델이 *New*와 *York*를 마스킹했다고 가정하자.

- Original data: *New York is a city*
- Corrupted data: *`[MASK]` `[MASK]` is a city*

우리는 *New*와 *York*가 긴밀한 관계에 있음을 알고있다. 그러나 BERT는 이런 관계를 무시하고(독립적) 복원을 하게 된다.

$$
\mathcal J _{BERT} = \log p (\text{New} \rvert \text{is a city}) + \log p (\text{York} | \text{is a city}) 
$$

그러나 AR 모델의 경우 *New*와 *York*의 의존성에 대해 파악할 수 있다.

$$
\mathcal J _{AE} = \log p (\text{New} \rvert \text{is a city}) + \log p (\text{York} |\text{New},  \text{is a city}) 
$$

이에 대한 예시는 *Appendix 5.1 Comparison with BERT*에 잘 나와있다.

### Input noise

BERT는 인공적인 심볼 `[MASK]`를 통해 pre-training을 진행하게 되는데, 이는 downstream task에서는 **절대 등장하지 않아** pre-training과 fine-tuning사이의 **불일치**를 만들게 된다. 이를 막기위해서 BERT는 마스크된 토큰 중 80%는 그대로, 10%는 랜덤하게, 10%는 원래 단어로 다시 되돌리게 되는데, 이 확률자체가 너무 작기 때문에 이러한 불일치 문제를 해결하기가 어렵다. 그렇다고 이 확률을 키우게되면 최적화하기 trivial한 문제가 된다. AR 모델은 다행히 이러한 문제를 겪지 않는다.

### Context dependency

AR은 왼쪽방향에 있는 토큰에만 의지하게된다. 이는 **bidirectional context를 잡지못하는 문제**가 발생한다.

## 2. Proposed method

### Objective: Permutation Language Modeling

XLNet은 permutation language modeling objective를 통해 AR의 장점은 유지하면서 bidirectional context도 잡을 수 있게한다. 길이 $T$를 갖는 sequence $\mathbf x$는 $T!$의 순서를 갖을 수 있다. 이러한 factorization order에 상관없이 모델을 학습시키면 양쪽 모두의 방향에서의 정보를 얻을 수 있을 것이다.

이러한 아이디어를 정리해보자. $T$길이의 index sequence $[1, \cdots, T]$의 모든 permutation 집합을 $\mathcal Z _T$라 해보자. 어떤 permutation $\mathbf z \in \mathcal Z$의 $t$번째 element를 $z _t$, $t-1$까지의 element들을 $\mathbf z _{<t}$라 하자.

![image](https://user-images.githubusercontent.com/47516855/133965319-63a089ec-5ae5-4c09-810f-c0aa43f27946.png){: .align-center}{: width="500"}

그러면 permutation language modeling objective는 다음과 같이 적을 수 있다.

$$
\max \mathbb E _{\mathbf z \sim \mathcal Z _T} \left [ \sum ^T _{t=1} \log p _\theta (x _{z _t} \rvert \mathbf x _{\mathbf z _{<t}})  \right ] \tag{3}
$$

Text sequence $\mathbf x$에 대해 fatorization order $\mathbf z$를 sampling하고 likelihood $p _\theta (\mathbf x)$를 $\mathbf z$에 따라 분해한다. 학습과정에서 모든 factorization order에 대해 같은 파라미터 $\theta$를 공유하게 되므로 bidirectional context를 모델이 학습할 수 있게된다. 뿐만 아니라 이는 AR 모델의 구조를 따르고 있으므로 앞서 언급했던 independence assumption과 pre-training - fine-tuning 불일치 문제를 해결할 수 있게된다.

여기서 한 가지 명심할 점은 factorization order만을 permute할 뿐, 원래의 단어 순서는 그대로 유지한다는 것이다. 즉, 원본 단어의 순서대로 positional encoding은 유지한채 permutation order에 따라 적절한 마스킹을 통해 self-attention을 유지한다. 그 이유는 fine-tuning과정에서 모델로 전달되는 데이터는 실제 문장 그대로 들어오기 때문이다. 아래 그림은 Appendix A.7의 Figure 4로, 다양한 factorization order에 따라 $x _3$를 예측하는 과정을 나타낸 것이다.

![image](https://user-images.githubusercontent.com/47516855/133968601-80d06128-7fce-45a9-b818-6a6111acf3ed.png){: .align-center}{: width="700"}

### Architecture: Two-Stream Self-Attention for Target-Aware Representations

앞서 언급했던 문제를 해결하기 위해 XLNet은 permutation language model을 제안하였다. 이를 통해 bidirectional context를 학습하는 것은 물론 pre-train - fine-tuning 불일치 문제를 해결할 수 있는 것은 맞지만 이를 직접적으로 transformer에 적용할 경우 또 다른 문제가 발생한다.

XLNet을 통해 next-token distribution $p _\theta (X _{z _t} \rvert \mathbf x _{\mathbf z _ {< t}})$를 parameterize한다고 가정해보자. 이 경우 softmax를 이용하게 되므로 다음과 같은 수식이 나오게 된다.

$$
p _\theta (X _{z _t} \rvert \mathbf x _{\mathbf z _ {< t}}) = \frac{\exp (e(x)^\intercal h _\theta (\mathbf x _{\mathbf z _{<t}}))}{\sum _{x'}^\intercal h _\theta (\mathbf x _{\mathbf z _{<t}})}
$$

여기서 $h _\theta (\mathbf x _{\mathbf z _{<t}})$는 $\mathbf x _{\mathbf z _{<t}}$의 hidden representation으로, transformer에 적절한 masking을 씌어 얻을 수 있다.

이 경우 문제가 되는 것은 representation이 이를 예측할 위치 $z _t$에 대해 의존하지 않는다는 점이다. 좀 더 쉬운 이해를 위해 *A.1 A Concrete Example of How Standard LM Parameterization Fails*에서 소개하는 예제를 한번 살펴보자.

두 개의 permutation order $\mathbf z^{(1)}, \mathbf z^{(2)}$가 다음과 같은 관계를 만족한다고 해보자.

$$
\mathbf z^{(1)} _{< t} = \mathbf z^{(2)} _{< t} = \mathbf z _{< t} ~~ \text{but} ~~ z^{(1)} _t = i \neq j = z^{(2)} _t
$$

즉, t시점 이전($\mathbf z _{< t}$)까지의 permutation order은 같고 t시점 ($z _t$)만 다른 경우이다. 이를 parametrization해보면,

$$
\underbrace{p _\theta (X _i = x \rvert \mathbf x _{\mathbf z _{<t}})} _{z^{(1)} _t =i , \mathbf z^{(1)} _{<t} = \mathbf z _{<t}}
=\underbrace{p _\theta (X _j = x \rvert \mathbf x _{\mathbf z _{<t}})} _{z^{(1)} _t =i , \mathbf z^{(2)} _{<t} = \mathbf z _{<t}}
= \frac{\exp (e(x)^\intercal h(\mathbf x _{\mathbf z _{< t}}))}{\sum _{x'} \exp (e(x')^\intercal h(\mathbf x _{\mathbf z <t}))}
$$

보다시피 서로 다른 위치 $i, j$를 예측하기 위해 똑같은 parameter를 사용하는 것을 볼 수 있다.

이러한 문제를 피하기 위해, XLNet은 next-token prediction이 target position을 인식하도록 re-parameterization해준다.

$$
p _\theta (X _{z _t} \rvert \mathbf x _{\mathbf z _ {< t}}) = \frac{\exp (e(x)^\intercal g _\theta (\mathbf x _{\mathbf z _{<t}}, z _t))}{\sum _{x'}^\intercal g _\theta (\mathbf x _{\mathbf z _{<t}}, z _t)} \tag{4}
$$

여기서 새롭게 도입된 $g _\theta (\mathbf x _{\mathbf z _{<t}}, z _t)$는 새로운 형태의 representation으로, target position $z _t$을 input으로 포함하게 된다.

**Two-Stream Self-Attention**  
앞서 살펴봤던 것과 같이, 새로운 representation은 t시점 이전의 정보와 t에 대한 위치 정보 둘 다를 포함해야 한다. 그러나 이 둘은 일반적은 transformer에서는 서로 대조되는 개념이라 달성하기가 까다롭다.

1. $x _{z _t}$를 예측하기 위해서 새로운 representation $g _\theta (\mathbf x _{\mathbf z _{< t}}, z _t)$는 $z _t$의 **위치정보** $z _t$는 이용해야 하지만 **내용 (content)** $x _{z _t}$는 이용해선 안된다. 만약 $x _{z _t}$까지 이용하게 될 경우 문제가 너무 단순해지게 된다.
2. $t$보다 뒤에 위치에 있는 $j$ 토큰 $x _{z _j}$를 예측하게 될 경우, 새로운 representation $g _\theta (\mathbf x _{\mathbf z _{< t}}, z _t)$는 위치정보뿐만 아니라 $x _{z _t}$까지 활용하여 문맥에 대한 정보를 반영해야 한다.

이 부분은 논문을 읽으면서 잘 이해가 안되었던 부분이라 아래 그림을 첨부한다.

![image](https://user-images.githubusercontent.com/47516855/134291222-6f29c11a-1865-4e9e-b904-10717b2b0297.png){: .align-center}{: width="500"}

$t=3$으로, permutation order는 `[2, 4, 3, 1]`으로 가정하자. 새로운 representation $g _\theta$는 각 시점(token)에 대해 존재할 것이다.

![image](https://user-images.githubusercontent.com/47516855/134291258-20fd579c-f76b-463d-bb4e-95354339b47f.png){: .align-center}{: width="500"}

앞서 설명한 1번과 같은 케이스가 바로 위의 그림이다. $g _{\theta}$는 $t=3$일 때의 위치정보(파랑색 점선)만 포함할 뿐 내용(검은색 실선)은 포함해선 안된다.

![image](https://user-images.githubusercontent.com/47516855/134291286-79cfe0b5-3a5b-45a2-bcf2-cee1ca79e393.png){: .align-center}{: width="500"}

이번엔 2번과 같은 경우를 보자. 이번에는 $j=1>3=t$로 가정하였다. $g _\theta$는 context를 반영해야 되기 때문에 이번에는 full context information, 즉, 내용을 전달해야 한다. 이번에 포함되는 위치정보는 $z _j$가 된다. $g _theta$는 token을 예측하는 시점에 따라 표현하는 방법이 파랑색/검은색 두 가지로 나눠지게 된다. Transformer는 이 두가지 제한사항을 동시에 만족할 수 없다.

이러한 모순을 해결하기 위해 하나의 representation 대신 두 개의 representation인 **two-stream self-attention**을 사용한다.

**Content representation**: $h _\theta (\mathbf x _{\mathbf z _{< t}})$ or simply $h _{z _t}$  
Content representation은 Transformer의 hidden state와 유사한 역할을 하는 것으로, context와 위치 $x _{z _t}$ 둘 다 encoding한다. 첫번째 레이어는 embedding vector $h^{(0)} _i = e(x _i)$가 된다.

![image](https://user-images.githubusercontent.com/47516855/134290146-1ea33e91-d6a3-4988-a759-2b8d423f3672.png){: .align-center}{: width="700"}

**Query representation**: $g _\theta (\mathbf x _{\mathbf z _{< t}}, z _t)$, or simply $g _{z _t}$  
Query representation은 contextual information $\mathbf x _{\mathbf z _{< t}}$와 위치 $z _t$에 대한 정보를 포함한다. (단 여기서 $x _{z _t}$의 대한 content는 포함하지 않는다.) Query stream의 첫번째 레이어는 trainable vector $g^{(0)} _i = w$로 초기화한다.

![image](https://user-images.githubusercontent.com/47516855/134290435-593ff254-27bd-4929-982a-b227fd46f374.png){: .align-center}{: width="700"}


두 레이어는 아래와 같은 방법으로 업데이트 된다.

- $g^{(m)} _{z _t} ~ \gets ~ \text{Attention}(\mathbf Q = g ^{(m-1)} _{z _t}, \mathbf{KV} = \mathbf h^{(m-1)} _{\color{red}{\mathbf z _{<t}}}; \theta)$
- $h^{(m)} _{z _t} ~ \gets ~ \text{Attention}(\mathbf Q = h ^{(m-1)} _{z _t}, \mathbf{KV} = \mathbf h^{(m-1)} _{\color{red}{\mathbf z _{\leq t}}}; \theta)$

![image](https://user-images.githubusercontent.com/47516855/134289037-8573a4a3-b7b5-4804-b958-a8fde3fc82a1.png){: .align-center}{: width="700"}

$\mathbf Q, \mathbf K, \mathbf V$는 query, key, value를 의미한다. Content representation은 일반적인 Transformer와 정확하게 일치하므로, fine-tuning 단계에서는 query stream을 떼어놓고 content stream만 Transformer-XL처럼 사용하게 된다. 마지막으로 가장 끝에 있는 query representation $g^{(M)} _{z _t}$를 사용하여 Eq. (4)를 계산하게 된다.

**Partial Prediction**  
permutation language modeling objective (Eq. (3))에 이점이 있다고 하더라도 permutation으로 인해 최적화가 어렵게 된다. 이러한 최적화 난이도를 줄이기 위해 factorization order의 마지막 토큰만을 예측하도록 조절했다.

$\mathbf z$의 토큰을 cutting point $c$를 통해 non-target sequence $\mathbf z _{\leq c}$와 target subsequence $\mathbf z _{> c}$로 나누게 된다. 그리고 non-target sequence에 조건부인 target susbsequence를 예측하도록 한다.

$$
\max _{\theta} \mathbb E _{\mathbf z \sim \mathcal Z _T} 
[\log p _{\theta} (\mathbf x _{\mathbf z _{> c}} \rvert \mathbf x _{\mathbf z _{\leq c}})]
= \mathbb E _{\mathbf z \sim \mathcal Z _T} \left [ \sum^{\lvert \mathbf z \rvert} _{t=c+1} \log p _{\theta} (x _{z _t} \rvert \mathbf x _{\mathbf z _{< t}}) \right ] \tag{5}
$$

여기서 $\mathbf z _{> c}$를 예측하도록 한 것은 factorization order에 대해 가장 긴 context를 포함하고 있기 때문이다. 이를 위해 hyperparameter $K$를 도입하여 $\lvert \mathbf z \rvert / (\lvert \mathbf z \rvert - c) \approx K$인 $c$를 선택하도록 한다.

이 부분은 논문에서 애매하게 소개되어있다. 약 $1/K$개의 토큰을 예측하도록 한다는데, 이렇게 될 경우 $1/K$는 1보다 작게 된다. [공식 repo](https://github.com/zihangdai/xlnet/blob/master/data_utils.py#L331)를 보면 특정한 알고리즘을 통해 적용하는 것으로 보인다.
{: .notice--info}

### Incorporating Ideas from Transformer-XL

XLNet은 AR 모델이므로, SOTA AR 모델인 Transformer-XL을 pre-training framework에 포함하도록 한다. XLNet은 Transformer-XL의 두 가지 중요한 테크닉을 가져온다.

**Relative positional encoding**  
Transformer-XL에선 기존의 Transformer가 갖는 positional encoding의 문제점을 해결하기 위해 상대적인 위치를 고려하는 positional encoding을 수행한다. 이는 segment가 나눠지는 경우 각 segment에서의 positional encoding이 같은 경우가 발생하는데 (`[1, 2, 3, 4] -> [1, 2, 1, 2]`) 이는 실제 단어의 위치를 정확하게 반영하지 못한다는 단점이 있다. 따라서 상대적인 거리를 통해 위치정보를 encode하게 된다.

[여기](https://baekyeongmin.github.io/paper-review/transformer-xl-review/)서 자세하게 설명하고 있으니 이를 참고하는 것이 좋아보인다.
{: .notice--info}

**Recurrence mechanism**  
긴 문장 $\mathbf s$에서 얻어진 segments $\mathbf {\tilde x} = \mathbf s _{1:T}$와 $\mathbf x = \mathbf s _{T+1:2T}$가 있고, 이에서 permutation한 것을 각각 $\tilde{\mathbf z}$와 $\mathbf z$라 하자. Permutation $\tilde{\mathbf z}$에 대해 first segment를 진행한 뒤 각 layer $m$에 대해 content representation $\tilde{\mathbf h}^{(m)}$를 얻을 수 있다. 그 후 다음 segment $\mathbf x$에 대해 다음과 같은 식으로 attention update를 진행한다.

$$
h^{(m)} _{z _t} ~ \gets ~ \text{Attention}(\mathbf Q = h^{(m-1)} _{z _t}, \mathbf K \mathbf V = \left [ \tilde{\mathbf h}^{(m-1)}, \mathbf h ^{(m-1)} _{\mathbf z _{\leq t}} \right ]; \theta)
$$

$[.,.]$는 sequence dimension으로 concat하는 것을 의미한다. Positional encoding은 실제 위치에만 의존하므로 $\tilde{\mathbf h}^{(m)}$이 얻어지게 되면 $\tilde{\mathbf z}$와는 무관하게되고, 이를 통해 이전 factorization order를 저장하지 않고도 caching/reusing이 가능하게 된다. 기댓값으로 보면 모델은 이전 segment의 모든 factorization order를 활용할 수 있다. 

![image](https://user-images.githubusercontent.com/47516855/134307547-80dc70e0-2b13-406e-819c-092ebe823086.png){: .align-center}{: width="600"}

이를 실제로 구현할 때는 위의 그림처럼 마스킹을 통해 attention을 진행한다. Transformer의 decoder를 학습할 때는 마스킹을 통해 학습하고, inference시에는 auto-regressive하게 진행하는 것과 같다. 그림을 자세히 보면 가장 첫번째 permutation인 3의 row에서는 어떠한 토큰도 attention하지 않는 것을 확인할 수 있다 (content에서 자기자신 제외).

따라서 permutation order 3 → 2 → 4 → 1에 대해,
- row 3: 어떠한 것도 attention하지 않는다 (분홍색이 attention).
- row 2: 이전 시점인 3만 attention한다.
- row 4: 이전 시점인 3, 2만 attention한다.
- row 1: 이전 시점인 3, 2, 4를 attention한다.

### Modeling Multiple Segments

QA와 같이 NLP의 태스크 중에는 여러 segment를 필요로 하는 경우가 있다. XLNet의 경우 pre-training 과정에서 BERT와 동일하게 임의의 두 샘플 segment들을 가져와 이를 concat한 뒤 하나의 segment처럼 취급하여 permutation languege model에 넣게 된다. 그리고 두 segments가 같은 context인 경우에만 memory를 사용하게 된다. 인풋의 형태는 BERT와 동일하게 `[CLS], [A], [SEP], [B], [SEP]`를 따르게 된다.

비록 BERT와 동일한 데이터 포맷을 따르게 되었을지언정, XLNet-Large는 NSP를 수행하진 않는다. 이는 ablation study (Section 3.4)에서 지속적인 성능향상을 보이지 못했기 때문이다.

**Relative Segment Encodings**  
BERT는 각 segment에 대해 절대적인 위치 정보인 Segment embedding A/B를 추가하지만, 여기서는 이를 확장하여 상대적인 위치정보를 포함하게 된다.

Segment $i, j$에 대해, 이들이 같은 segment에서 나오게 되면 segment embedding $\mathbf s _{ij} = \mathbf s _{+}$를, 그렇지 않은 경우 $\mathbf s _{ij} = \mathbf s _{-}$를 이용한다. 즉, 여기서 사용하는 핵심 아이디어는 이들이 **같은 segment에서 나왔는지, 아닌지**에 대한 정보만을 추가로 이용한다는 것이다. 이들은 학습가능한 파라미터로, attention head에서 weight를 계산하는데 이용된다.

$$
a _{ij} = (\mathbf q _i + \mathbf b)^\intercal \mathbf s _{ij}
$$

여기서 $\mathbf q$는 query vector이다. 그리고 $a _{ij}$는 일반적인 attention weight에 더하게 된다.

이를 통해 얻을 수 있는 이점으로는 두 가지가 있다. 하나는 상대정보를 encoding하는 inductive bias가 generalization을 향상시킬 수 있다는 것이고, 나머지 하나는 두 개 이상의 segment를 사용하는 fine-tuning 태스크에 적용할 수 있는 가능성을 열어줬다는 점이다.

## 3 Experiments

BERT를 따라서 BookCorpus, Wikipedia를 pre-training에서 사용한다 (13GB). 이에 추가로 Giga5 (16GB), ClueWeb 2012-B, Common Crawl을 사용하였다. ClueWeb 2012-B과 Common Crawl에 대해서는 엄격한 필터링을 통해 짧거나 낮은 퀄리티를 보이는 문서를 제거하였고, 그 결과 각각 19GB, 110GB의 텍스트 데이터를 얻었다.

SentencePiece를 이용하여 2.78B, 1.09B, 4.75B, 4.30B, 19.97B subword pieces를 얻었고, 총 32.89B의 subword를 얻게 되었다.

XLNet-Large는 BERT-Large와 같은 구조를 갖게하여 비슷한 파라미터를 유지하도록 하였다. Pre-training에서 sequence length는 512로 맞췄다. 또한, BERT와의 공정한 비교를 위해 XLNet-Large-wikibooks는 BookCorpus와 Wikipedia만 사용하여 학습했고, 앞서 언급한 데이터를 이용하여 추가로 학습하였다. 학습의 마지막 단계에서 모델에 데이터를 underfit하는 것을 관측하였다.

Reccurence 구조가 추가되었으므로, 정방향/역방향 모두를 이용할 수 있도록 배치의 절반씩을 할당하였고, $k$는 6으로 세팅하였다. 또한, $L \in [1, \cdots, 5]$를 통해 $(KL)$ 토큰의 context내에서 임의로 선택된 $L$개의 연속적인 span을 예측하도록하는 **span-based prediction**을 사용하였다.

이하 다양한 NLP 태스크에 대한 설명은 직접 논문을 살펴보도록 하자.

### Ablation Study

다양한 특성을 갖는 4개의 데이터셋에 대해 XLNet의 design choice에 대한 ablation study를 수행하였다. 특히 다음의 세 가지 측면에 대해 초점을 맞추어 진행하였다.
- Permutation language model의 objective의 효율성을 증명. 특히 DAE방법을 사용하는 BERT랑 비교를 진행.
- backbone으로서의 Transformer-XL의 중요성
- span-based prediction, bidirectional input pipeline, next-sentence prediction와 같은 implementation details의 필요성

아래의 table 6은  XLNet-Base와 이에 implementation details을 추가한 것을 나타낸 것이다. 공정한 비교를 위해 모든 모델은 BERT의 하이퍼 파라미터와 같은 12-layer architecture와 Wikipedia BooksCorpus를 이용하여 학습하였다. 실험 결과는 5번의 실행값의 중앙값으로 보고하였다.

![image](https://user-images.githubusercontent.com/47516855/134807561-410595ca-fb9f-4079-923d-3968a1c3488b.png){: .align-center}{: width="400"}

1-4 행을 보면 Transformer-XL과 permutation LM이 성능에 큰 영향을 주는 것을 확인할 수 있다. 또한, 행 5의 memory caching mechanism을 제거할 경우 성능이 급격하게 떨어지는 것을 확인하였으며, 행 6-7은 span-based prediction, bidirectional input pipeline 모두 성능에 영향을 미치는 것을 보여주었다. 마지막으로 BERT에서 제안한 next-sentence prediction이 성능 향상으로 연결되지는 않음을 확인하였다. 

### Qualitative Analysis of Attention Patterns (Appendix A.6)

이번에는 파인튜닝 없이 BERT와 XLNet의 attention pattern을 파악해보았다. 우선 아래의 Fig. 2와 같이 BERT와 XLNet 모두에서 관찰할 수 있는 공통적인 패턴을 발견하였다.

![image](https://user-images.githubusercontent.com/47516855/134808336-72528305-06bc-4cf2-af83-e5de67d4f500.png){: .align-center}{: width="600"}

아래는 XLNet에서만 발견한 3개의 패턴이다.

![image](https://user-images.githubusercontent.com/47516855/134808733-ad36b54d-3c7e-4c31-b296-f38232a321da.png){: .align-center}{: width="600"}


- (a). self-exclusion pattern: 자기 자신을 제외하고 나머지 토큰에 attention. Global information을 빠르게 모으기 위함으로 보임.
- (b). Relative stride: query 위치에 **상대적으로** 떨어진 stride 간격에 attention함 (attends to positions every a few stride apart relative to the query position).
- (c). One-side masked: lower-left masking과 유사하게 **상대적인** 오른쪽 절반(relative right half)에 attention하지 못함.

XLNet만의 고유한 특성 모두 절대적인 위치보단 **상대적** 위치를 포함하고 있으며, XLNet의 **relative attention**의 개념을 잘 보여주고 있다. 이러한 고유한 특성이 XLNet의 성능에 이점이 되었으리라 추측하고, permutation LM objective은 비록 qualitative visualization은 불가능하지만 대부분의 contribution을 차지하고 있을 것으로 본다.
