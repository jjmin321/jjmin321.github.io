---
title:  "[미완성] Bias Variance Trade Off"
toc: true
toc_sticky: true
categories:
  - Machine Learning
tags:
  - statistics
  - ensemble
  - bagging
  - boosting
  - TODO
use_math: true
last_modified_at: 2021-03-21
---

머신러닝에서 흔히 다루는 error는 bias와 variance로 구성된다. 이는 흔히 bias-variance trade-off (dilemma)라고 부른다. 이번 글에서는 우리의 모델이 갖는 에러가 어떻게 bias와 variance로 분해되는지 살펴보고, 이를 줄이기 위한 방법을 살펴보도록 하자.

## MSE에 대한 Bias-Variance Decomposition

우리의 데이터셋이 $x _1, ..., x _n$으로 구성되어 있고, 각각의 $x _i$에 대해 실수 $y _i$가 쌍으로 있다고 하자. 그리고 $y=f(x) + \epsilon$으로 표현가능하다고 하자. 이 때 $\epsilon$은 노이즈로, 평균값 0과 특정한 분산을 따른다고 하자.

머신러닝의 목적은 훈련셋 (샘플) $D = \{ (x _1, y _1), ..., (x _n, y _n) \} $을 통해 어떤 함수 $\hat f(x; D)$를 찾는 것으로, 이는 true function $f(x)$를 잘 근사한다.

여기서 $y$와 $\hat f(x; D)$간의 MSE를 활용하여 가능한 한 정확하게 만들어보자. 우리는 $x _1, ..., x _n$ 뿐만 아니라 **샘플 밖의 데이터에 대해서도** $\hat f(x; D)$를 최소화하고 싶다. 물론 $\epsilon$으로 인해 완벽하게 할 수는 없다. 이는 데이터가 갖는 노이즈이기 때문이다. 즉, 우리는 이를 **줄일 수 없는 오차 (irreducible error)**로 간주한다. 모델의 문제가 아닌 데이터의 문제이므로 이러한 오차는 어떠한 함수로 근사하더라도 항상 일어난다.

학습셋 밖에 있는 데이터에 대해서도 일반화하는 $\hat f$을 찾는 것은 어떠한 지도학습 모델로도 수행이 가능하다. 따라서 어떠한 함수 $\hat f$을 선택하더라도, 한번도 등장한 적 없는 $x$에 대해 다음과 같이 에러의 기댓값을 구할 수 있다.

$$
E _D[(y - \hat f(x; D))^2] = (E _D [\hat f(x; D)] - f(x))^2 + E _D [(E _D [\hat f(x; D)] - \hat f(x; D))^2] + \sigma ^2
$$

여기서 첫번째 항 제곱 안의 $E _D [\hat f(x; D)] - f(x))$는 **bias**를, 두번째 항 $E _D [(E _D [\hat f(x; D)] - \hat f(x; D))^2]$은 **variance**를 의미하게 된다.

이에 대한 유도는 다음과 같다. $y$의 정의에 따라 다시 쓰면,

$$
E[(y - \hat f)^2] = E[(f + \epsilon - \hat f)^2]
$$

여기서 식의 유도를 위해 $E[\hat f]$를 더하고 빼준다.

$$
= E[(f + \epsilon - \hat f + E[\hat f] - E[\hat f])^2] 
$$

이를 전개하고 다시 묶어보면 다음과 같은 식이 나오게된다.

$$
= E[(f - E [\hat f])^2] + E[\epsilon ^2] + E[(E[\hat f] - \hat f)^2] + 2E[(f - E[\hat f]) \epsilon] + 2E[\epsilon (E[\hat f] - \hat f)] + 2E[(E[\hat f] - \hat f)](f - E[\hat f])
$$

앞서 $E[\epsilon] = 0$으로 정의하였으므로 $E[\epsilon]$을 0으로 바꿀 수 있다. $2E[(E[\hat f] - \hat f)](f - E[\hat f])$은 0이 되므로 삭제한다.

$$
= E[(f - E [\hat f])^2] + E[\epsilon ^2] + E[(E[\hat f] - \hat f)^2]
$$

$f$는 deterministic하므로 (즉, D에 독립이므로), $E[y] = E[f + \epsilon] = E[f] = f$가 된다. 따라서 $E[(f - E [\hat f])^2] = (f - E [\hat f])^2$가 된다. 이는 앞서 봤던 **bias**이다. 

$E[\epsilon ^2]$은 분산-평균과의 관계 $Var[\epsilon] = E[\epsilon ^2] - E[\epsilon]^2$에 따라 $Var[\epsilon] + E[\epsilon]^2 = Var[\epsilon]$이 된다. 또한 정의에 따라 $Var[\epsilon] = \sigma^2$가 된다.

마지막 항은 분산으로 유도할 수 있으므로 **variance**가 된다.

이는 bias-variance tradeoff라 불리는 것으로, bias와 variance간의 관계를 나타낸다. Tradeoff라 불리는 이유는 bias와 variance가 양수로 이루어져 있기 때문이다. MSE가 고정되어 있고, bias variance 둘 다 square term의 양수이기 때문에 어느 하나가 증가하면 어느 하나는 감소해야 한다.

## 해석

우리의 에러가 각각 bias, variance로 나눌 수 있다는 것을 알았다 여기서 irreducible error는 물론 제외한다. 이는 우리가 어떻게 할 수 있는게 아니기 때문이다. 과연 bias와 variance가 의미하는 바는 무엇인가?

우선 **bias** $f - E _D [\hat f]$부터 살펴보자. Bias는 학습 알고리즘의 잘못된 가정으로 인한 오류이다. 즉, bias가 높다는 의미는 학습한 머신의 **예측력이 떨어진다**는 의미이다 (i.e. underfitting). 그렇기때문에 이는 true function $f$와 우리의 모델 $\hat f$의 차이로 표현되는 것이다.
 기댓값이 붙어있는 이유는 예측값들이 데이터에 따라 달라질 수 있기 때문이다. 당장 train, dev, test set을 생각해보자. train set에서 높은 정확도를 보인다고 dev, test set에서 좋은 성능을 낸다는 보장이 없다. 
 
따라서 bias는 **데이터를 infinite sampling 하여 모델을 만들었는데 ($E _D [\hat f]$), 정작 true function ($f$)이 우리의 모델과 달라서 생기는 오차이다.**

**vairance** $E _D [(E _D [\hat f(x; D)] - \hat f(x; D))^2]$를 살펴보자. 이번에는 모델들의 기댓값과 모델의 차이로 나왔다. 즉, 우리 모델이 갖는 **변동성**을 의미한다. 변동성은 왜 생기는걸까? 이는 실제로는 **데이터가 다양하게 있는데(train, test) ($E _D [\hat f(x; D)]$), 우리의 데이터는 한정($\hat f(x; D)$)**되어 있어서 생기게 된다.

![](https://modulabs-biomedical.github.io/assets/images/posts/2018-01-25-Bias_vs_Variance/3.jpg){.align-center}

위의 그림은 variance와 bias를 도식화한 것이다. 
- a: best fit이지만, 실제론 일어나지 않는다
- b: bias는 낮지만 variance가 높다. 즉, 데이터 $D$마다 fitting하는 결과($\hat f(x; D)$)가 다르지만, 각 각의 모델은 실제 데이터 ($f(x)$)와 유사함을 볼 수 있다
- c: bias는 높고 variance는 낮다. 우리 모델이 데이터 $D$에 대해 변동성이 크진 않지만 ($\hat f(x; D)$이 비슷비슷), 정답($f(x)$)과는 차이가 있다고 볼 수 있다
- d: 둘 다 높다

이를 해결하는 방법은 다음과 같다. Bias가 낮은 이유는 실제 값과 우리의 학습셋이 잘 맞지 않는 것이므로 모델 복잡도를 증가시키는 것이다. 기존에는 linear regression을 사용했다면, 이번에는 polynomial한 케이스로 fitting해보는 것이다. Bias는 underfitting이므로, 더욱 fitting하게 만들면 되기 때문이다.

Variance가 낮을 때는 데이터를 더 수집하면 된다.

## N-fold cross validation

이를 위해서 

## Regularization

이번엔 regularization을 


여러 모델로부터 평균을 내는 것이 이득을 주는 과정

## Bagging과 boosting



정규화/배깅 관점



# 출처

[한 페이지 머신러닝 - Bias and Variance (편향과 분산)](https://opentutorials.org/module/3653/22071)

https://nittaku.tistory.com/289

https://opentutorials.org/module/3653/22071

https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff

https://untitledtblog.tistory.com/143

https://towardsdatascience.com/what-to-do-when-your-model-has-a-non-normal-error-distribution-f7c3862e475f

https://www.slideshare.net/freepsw/boosting-bagging-vs-boosting
