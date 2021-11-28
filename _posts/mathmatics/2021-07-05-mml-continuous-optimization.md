---
title:  "머신러닝을 위한 수학 정리: Continuous Optimization"
toc: true
toc_sticky: true
permalink: /project/mml/Continuous-Optimization/
categories:
  - Mathmatics
  - Machine Learning
tags:
  - probability
  - distributions
use_math: true
last_modified_at: 2021-07-08
---

## Introduction

머신러닝 알고리즘은 컴퓨터상에 구현되어있기 때문에, 수학 공식들은 모두 수치 최적화 기법을 통해 표현된다. 이번 챕터에서는 머신러닝 모델을 학습시키기 위한 기본적인 수치 해석에 대해 살펴본다. 모델을 학습시키는 것은 종종 좋은 파라미터를 찾는 것으로 요약된다(boils down). 여기서 "좋다"는 것은 목적함수나 확률 모델에 의해서 결정되는 것으로, 이 책의 두번째 파트에서 살펴볼 것이다. 주어진 목적함수에 대해, 최적의 값을 찾는 것은 최적화 기법을 사용하여 이루어진다.

이 책에선 continuous optimization을 unconstrained/constrained optimization의 두 가지 갈래로 나누어 다룬다 (Figure 7.1 참고).

![image](https://user-images.githubusercontent.com/47516855/124471285-a754ed00-ddd7-11eb-8cae-7096cff5a48e.png){: .align-center}{:width="600"}

이 챕터에선 우리의 목적 함수가 미분 가능할 것이라고 가정할 것이다 ([5. Vector Calculus](/project/mml/Vector-Calculus/)). 따라서 공간 내의 각 지역에 대한 gradient를 계산하여 최적값을 구할 수 있게된다. 관례에 따라 머신러닝의 대부분의 목적함수는 최소화한다. 즉, 최적값이 최솟값이 된다. 직관적으로 좋은 값을 찾는 것은 목적 함수 내의 계곡을 찾는 것이며, 그레디언트값은 위를 가르킬 것이다. 이는 그레디언트의 반대방향으로 내려가서 가장 깊은 점을 찾는 것이다. Section 7.1에서 다룰 몇몇 개념을 제외하면 unconstrained optimization에서 이것이 우리에게 필요한 전부가 된다. constrained optimization의 경우 제약조건을 다룰 개념이 필요해진다 (Section 7.2). 또한 global minimum에 닿게 해주는 특수한 경우를 살펴보도록 할 것이다 (convex optimization problem in Section 7.3).

Figure 7.2의 함수를 생각해보자. 이 함수는 **global minimum**을 $x=-4.5$ 근방에서 $-47$로 갖게 된다. 이 함수는 **smooth(매끄러운)**이므로, 그레디언트를 통해 오른쪽으로 갈지, 왼쪽으로 갈지 정할 수 있으며, 최솟값을 갖게 해준다.

![image](https://user-images.githubusercontent.com/47516855/124472602-47f7dc80-ddd9-11eb-9fd5-77ac53d3f112.png){: .align-center}{:width="500"}

이는 그릇과도 같은 모양의 지역에 있다는 뜻이다. 그러나 이와는 다르게 $x=0.7$에서 **local minimum** 또한 존재한다. 미분값이 0인 지점을 찾으면 모든 stationary point(정류점)을 찾을 수 있다는 것을 상기시켜보자. 다음에 대해,

$$
\ell (x) = x^4 + 7x^3 + 5 x^2 - 17 x + 3 \tag{7.1}
$$

다음과 같은 그레디언트를 얻는다.

$$
\frac{\mathrm{d}\ell}{\mathrm{d}x} = 4x^3 + 21x^2 + 10x - 17 \tag{7.2}
$$

이는 3차 방정식이므로, 일반적으론 세 개의 해를 갖는다. 여기서는 두 개의 최솟값과 하나의 최댓값을 갖게된다. 정류점이 최댓값인지 최솟값인지 알기 위해서는 미분을 한 번 더 해서 양수인지 음수인지 보면 된다.

$$
\frac{\mathrm{d}^2 \ell}{\mathrm{d} x^2} = 12x^2 + 42x + 10 \tag{7.3}
$$

이를 통해 $x = -4.5, -1.4, 0.7$ 중 가운데 값이 최댓값임을 알 수 있게되었다.

우리는 여태까지 해석적으로 $x$에 대한 값을 푸는 것을 지양해왔는데, 이는 일반적으로 불가능한 경우가 많기 때문이다. 따라서 $x _0=-6$과 같은 값에서 시작하여 그레디언트의 반대방향으로 내려가야 한다. Negative gradient는 우리가 오른쪽으로 가야 한다는 것은 알려주지만, 얼마나 가야되는지는 알려주지 않는다 (이는 step-size라 부른다). 더욱이 우리가 $x _0 =0$과 같은 오른쪽에서 시작한다면 이상한 최솟점으로 우리를 이끄게 된다. Figure 7.2는 $x \geq -1$에 대해 오른쪽에 있는 최솟값으로 이끌게됨을 보여준다.

Section 7.3에서, 함수의 한 종류인 convex function(볼록함수)를 살펴볼 것이다. 볼록함수는 방금 살펴본 케이스 같이 시작점에 따라 달라지지 않는다. 여기서는 모든 local minimum이 global minimum이 된다. 머신러닝에서 사용되는 많은 목적함수들은 볼록하게 설계된다. 이는 Chapter 12에서 살펴볼 것이다.

> 본 포스트는 머신러닝에 필요한 선형대수 및 확률과 같은 수학적 개념을 정리한 포스트이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.

## Optimization Using Gradient Descent

이제 real-valued function의 최솟값을 푸는 방법에 대해 생각해보자.

$$
\min \limits _{\boldsymbol x} f(\boldsymbol x) \tag{7.4}
$$

함수 $f: \mathbb R^d \to \mathbb R$는 목적 함수이다. 함수 $f$가 미분가능하며, 닫힌 형태에 대해 해석적으로 풀 수 없다고 가정하자.

그레디언트 디센트는 first-order optimization(1차 최적화) 알고리즘이다. 이를 이용하여 local minumum을 찾기 위해서는 현재 지점에서 함수의 negative gradient 방향으로 내려가게 된다. [Differentiation of Univariate Functions](/project/mml/vector-calculus/#Differentiation-of-Univariate-Functions)에서 그레디언트는 가장 가파르게 올라가는 방향을 가르켰음을 다시 한번 떠올려보자.

이제 multivariate function에 대해 살펴보자. 어떠한 함수 $f(\boldsymbol x)$에 의해 생성되는 표면을 떠올려보자. 이 위의 한 점 $\boldsymbol x _0$에서 어떤 공이 굴러가기 시작한다. 그러면 가장 가파른 곳을 향해 공이 굴러가기 시작할 것이다. 그레디언트 디센트는 $\boldsymbol x _0$에서 negative gradient $-((\nabla f)(\boldsymbol x _0))^\intercal$의 방향으로 $f(\boldsymbol x _0)$가 가장 빨리 감소한다는 것을 이용한다. 그리고 만약 작은 step-size $\gamma \geq 0$에 대해 

$$
\boldsymbol x _1 = \boldsymbol x _0 - \gamma((\nabla f)(\boldsymbol x _0))^\intercal \tag{7.5}
$$

이면, $f(\boldsymbol x _1) \leq f (\boldsymbol x _0)$가 될것이다. 여기서 한 가지 명심할 점은 gradient에 대해 transpose를 사용하였는데, 이렇지 않을 경우 차원이 맞아 떨어지지 않기 때문이다.

이를 통해 간단한 그레디언트 알고리즘을 살펴보았다. 만일 우리가 함수 $f: \mathbb R^n \to \mathbb R, \boldsymbol x \mapsto f(\boldsymbol x)$에 대해, local optimum $f(\boldsymbol x _{*})$를 찾고 싶다면, 초깃값 $\boldsymbol x _0$에서 시작하여 다음을 반복하는 것이다.

$$
\boldsymbol x _{i+1} = \boldsymbol x _i - \gamma _i((\nabla f)(\boldsymbol x _i))^\intercal \tag{7.6}
$$

적절한 step-size $\gamma _i$에 대해 $f(\boldsymbol x _0) \geq f(\boldsymbol x _1), \dotsc$는 local minimum으로 수렴하게 된다.

이 책에서는 미분가능한 함수에 대해서만 다루고 있으며, 더 일반적인 내용에 대해서는 Section 7.4에서 다룰 것이다.

### Step-size

일찍부터 언급했듯 좋은 step-size를 고르는 것은 중요하다. 너무 작을 경우 느리게 수려하고, 너무 클 경우 overshoot하여 수렴에 실패하거나 심지어 발산하기도 한다. 다음장에선 gradient를 update가 이상하게 되거나 진동하는 것을 완화시켜주는 momentum에 대해 살펴볼 것이다.

Adaptive gradient는 함수의 국소적인 특징에 따라 매 반복마다 step-size를 rescale한다. 이에는 두 개의 간단한 휴리스틱 알고리즘이 존재한다.
- 함수값이 증가할 경우, step-size가 너무 크기 때문에 step-size를 줄여준다.
- 같은 방향으로 step을 내딛어 함수값이 더 감소한다면, step-size를 키운다.

### Gradient Descent With Momentum

Figure 7.3에서 볼 수 있듯, 최적화 표면의 곡률이 적절하게 최적화 되어 있지 않다면 그레디언트 디센트의 수렴은 매우 느릴 수 있다. 그레디언트 디센트를 이용해 계곡의 벽을 점프할 수 있다면 작은 스텝만으로도 최적화에 도달 할 수 있을 것이다. 이를 위한 적절한 방법은 그레디언트 디센트 알고리즘에 어떤 기억을 부여하는 것이다.

![image](https://user-images.githubusercontent.com/47516855/124481641-75498800-dde3-11eb-875c-a465ef5ac7ca.png){: .align-center}{:width="600"}

모멘텀은 이전 반복에서 일어났던 일에 대해 기억하는 항을 추가하는 것이다. 이 항은 그레디언트의 진동을 완화시켜주며 업데이트가 더 잘되게 만든다. 모멘텀은 무거운 공을 굴려서 방향이 잘 안바뀌게끔 만들어주는 것이다. 이는 이동편균을 이용하여 구현한다. 모멘텀 기반의 알고리즘은 $i$번째 iteration에 대한 update $\Delta \boldsymbol x _i$를 기억하고, 이전 그레디언트와 현재 그레디언트에 대한 linear combination을 통해 다음 업데이트를 결정한다.

$$
\begin{align}
\boldsymbol x _{i+1} &= \boldsymbol x _i - \gamma _i((\nabla f) (\boldsymbol x _i))^\intercal + \alpha \nabla \boldsymbol x _i \tag{7.11} \\
\nabla \boldsymbol x _i &= \boldsymbol x _i - \boldsymbol x _{i-1} = \alpha \nabla \boldsymbol x _{i-1} - \gamma _{i-1} ((\nabla f) (\boldsymbol x _{i-1}))^\intercal \tag{7.12}
\end{align}
$$

여기서 $\alpha \in [0, 1]$이다. 가끔 그레디언트를 근사적으로 알 경우가 있는데, 이때 모멘텀은 큰 힘을 발휘한다. 이는 그레디언트의 noisy estimate를 평균내기 때문이다. 그레디언트의 근사치를 얻는 또 다른 유용한 방법은 stochastic approximation을 이용하는 것이다.

### Stochastic Gradient Descent

그레디언트를 계산하는 것은 매우 시간소모적이지만, 종종 "값싸게" 그레디언트의 근사치를 구하는 것이 가능하다. 이는 여전히 그레디언트값과 적당히 비슷한 방향을 유지하고 있다는 점에서 유용하다.

**Stochastic gradient descent(SGD)**는 그레디언트 디센트의 확률적 근사법으로, 미분가능한 함수의 합으로 나타내는 목적함수를 최적화하는데 사용된다. 여기서 확률적이란 말은 그레디언트를 정확하게는 모르고 noisy approximation만 안다는 뜻이다. 그레디언트의 확률분포를 제약하여 이론적으로 SGD가 여전히 근사한다는 것을 확신할 수 있다.

머신러닝에서 주어진 $n=1, \dotsc , N$ 데이터에 대해, 종종 loss $L _n$의 합으로 이루어진 목적함수를 고려하곤 한다. 수학적 표현으로는 다음과 같다.

$$
L(\boldsymbol \theta) = \sum \limits^N _{n=1} L _n (\boldsymbol \theta) \tag{7.13}
$$

여기서 $\boldsymbol \theta$는 파라미터로, $L$을 최소화하는 파라미터를 찾는 것이 우리의 목적이다. Chapter 9의 regression에서는 negative log-likelihood가 되며, 각 example의 log-likelihood의 합으로 표현한다.

$$
L(\boldsymbol \theta) = - \sum \limits^N _{n=1} \log p(y _n \lvert \boldsymbol x _n , \boldsymbol \theta) \tag{7.14}
$$

여기서 $\boldsymbol x _n \in \mathbb R^D$는 training inputs이고, $y _n$은 training target, $\boldsymbol \theta$는 회귀 모델의 파라미터이다.

이전에 보았던 표준적인 그레디언트 디센트는 "배치" 최적화 방법으로, 즉, 다음식에 따라 전체 학습 데이터를 이용하여 파라미터 벡터의 업데이트를 통해 최적화를 수행한다.

$$
\boldsymbol \theta _{i+1} 
= \boldsymbol \theta _i - \gamma _i (\nabla L(\theta _i))^\intercal
= \boldsymbol \theta _i - \gamma _i \sum \limits^N _{n=1} (\nabla L _n(\boldsymbol \theta _i))^\intercal  \tag{7.15}
$$

만약 학습셋이 많고 간단한 표현이 불가능하다면, 그레디언트의 합을 평가하는 것은 매우 비싸질 것이다.

식 (7.15)의 $\sum^N _{n=1} (\nabla L _n(\boldsymbol \theta _i))$ 항에 대해, $L _n$의 부분만을 이용하면 계산하는데 드는 비용이 줄어들 것이다. $L _n ~ \text{for} ~ n=1, \dotsc , N$을 이용하는 배치 그레디언트 디센트와 반대로, 임의로 $L _n$의 부분집합을 선택하여 mini-batch gradient descent를 사용할 수 있다. 극단적으로 $L _n$ 내 하나씩만 임의로 선택해 gradient를 추정할 수도 있을 것이다. 데이터의 일부분만 선택하는 핵심적인 이유는 그레디언트가 수렴하기 위해서는 진짜 gradient의 불편 추정량이 필요하기 때문이다. 식 (7.15)의 $\sum^N _{n=1} (\nabla L _n(\boldsymbol \theta _i))$가 그레디언트의 기댓값에 대한 empirical estimate이기 때문이다 ([Means and Covariances](/project/mml/Probability-and-Distributions/#means-and-covariances) 참고). 따라서 다른 어떤 기댓값에 대한 unbiased empirical estimate을 하더라도 (데이터 내 어떠한 subset을 사용하더라도), 이는 그레디언트가 수렴하기에 충분하다.

어째서 그레디언트의 근사치를 사용할까? 주된 이유는 구현상의 제약으로, CPU/GPU 메모리가 제한되기 때문이다. 큰 미니배치를 사용하면 그레디언트의 추정치는 정확해질 것이며, 파라미터를 update하는데 있어 variance가 줄어들 것이다. 더욱이 큰 미니배치를 사용하면 그레디언트와 비용을 계산하는데 있어 벡터로 구현된 행렬 연산을 사용할 수 있다는 장점이 있다. 이는 최적화가 잘 되어 있기 때문이다. Variance가 감소하는 것은 수렴을 안정적으로 만들지만, 각 그레디언트의 계산이 좀 더 비싸진다.

반대로 작은 미니배치는 빠르게 추정할 수 있다. 미니배치를 작게 만들수록 그레디언트 추정치에 대한 노이즈가 커질것이며, 나쁜 local optima를 갖게 될 것이다. 머신러닝에서 최적화는 학습 데이터에 대한 목적함수를 최소화하는데 사용되지만, 최종적인 목표는 generalization performance를 높히는 것이다 (Chapter 8). 머신러닝에서의 목표는 목적함수의 최솟값을 정확히 구할 필요는 없기 때문에 미니배치를 이용하여 그레디언트의 근사를 구하는 것이 널리 이용된다. Stochastic gradient descent는 수백만장의 이미지를 학습시키는 딥러닝이나 토픽모델링, 강화학습, large-scale의 Gaussian process models을 학습시키는 등의 large-scale의 머신러닝에서 매우 효율적이다.

## Constrained Optimization and Lagrange Multipliers

이전시간에서는 다음과 같은 함수의 최솟값을 구하였었다.

$$
\min \limits _{\boldsymbol x} f(\boldsymbol x) \tag{7.16}
$$

여기서 함수는 $f: \mathbb R^D \mapsto \mathbb R$이다.

이번에는 추가적인 제약조건을 고려해보자. 즉, real-valued function $g _i : \mathbb R^D \to \mathbb R ~ \text{for} ~ i=1, \dotsc, m$에 대해, constrained optimization을 푸는 것이다 (Figure 7.4)

![image](https://user-images.githubusercontent.com/47516855/124490429-083af000-dded-11eb-8dac-8ca518200990.png){: .align-center}{:width="600"}

$$
\begin{align}
\min \limits _{\boldsymbol x} \quad & f(\boldsymbol x) \tag{7.17} \\
\text{subject to} \quad & g _i(\boldsymbol x) \leq 0 ~ \text{for all} ~ i=1, \dotsc, m
\end{align}
$$

함수 $f, g _i$가 일반적으로는 non-convex하다는 점을 일러둔다. Convex의 경우에는 다음장에서 살펴본다.

매우 실용적이지 않지만 식 (7.17)의 제약 문제를 비제약 문제로 변환하는 명확한 방법은 indicator function(지시함수)를 사용하는 것이다.

$$
J(\boldsymbol x) = f(\boldsymbol x) + \sum \limits^m _{i=1} \boldsymbol 1 (g _i (\boldsymbol x)) \tag{7.18}
$$

여기서 $\boldsymbol 1(z)$는 infinite step function으로 다음과 같다.

$$
\boldsymbol 1(z) = 
\begin{cases}
  0 & \quad \text{if } z \leq 0 \\
  \infty & \quad \text{otherwise} 
\end{cases} \tag{7.19}
$$

이는 제약조건을 만족하지 않을 경우 무한대의 패널티를 주게된다. 따라서 같은 해를 얻게 해준다. 그러나 이 함수 또한 여전히 최적화하기 어렵다. 이는 **Lagrange multiplier**를 이용하여 해결할 수 있다. Lagrange multiplier는 step function을 linear function으로 바꿀 수 있다.

각 부등 제약식에 해당하는 Lagrange multiplier $\lambda _i \geq 0$을 도입하여 식 (7.17)을 **Lagrangian**으로 바꿀 수 있다. 따라서

$$
\begin{align}
\mathfrak L (\boldsymbol x, \boldsymbol \lambda) &= f(\boldsymbol x) + \sum \limits^m _{i=1} \lambda _i g _i (\boldsymbol x) \tag{7.20a} \\
&= f(\boldsymbol x) + \boldsymbol \lambda^\intercal \boldsymbol g (\boldsymbol x) \tag{7.20b}
\end{align}
$$

마지막 줄에서 모든 제약조건 $g _i (\boldsymbol x)$는 벡터 $\boldsymbol g (\boldsymbol x)$로 concat되고, Lagrange multiplier는 벡터 $\boldsymbol \lambda \in \mathbb R^m$이 된다.

이제 Lagrangian duality에 대한 개념을 살펴보도록 하자. 일반적으로 최적화에서 duality란 변수 $\boldsymbol x$(primal variable이)에 대한 최적화 문제를 또 다른 변수 $\boldsymbol \lambda$(dual variable)에 대한 최적화 문제로 바꾸는 것을 말한다. 이 책에서는 duality에 대해 Lagrangian duality와 Legendre-Fenchel duality(Section 7.3.3)의 두 가지 방법을 살펴보도록 하겠다.

<div class="notice--success" markdown="1">

**Definition 7.1.** 식 (7.17)의 문제

$$
\begin{align}
\min \limits _{\boldsymbol x} \quad & f(\boldsymbol x) \tag{7.21} \\
\text{subject to} \quad & g _i(\boldsymbol x) \leq 0 \text{ for all } i=1, \dotsc, m
\end{align}
$$

는 **primal problem(원시문제)**로 알려져있으며, primal variable $x$에 대응한다. **Lagrangian dual problem**은 다음과 같이 주어지며,

$$
\begin{align}
\max \limits _{\boldsymbol \lambda \in \mathbb R^m} & \quad \mathfrak D (\boldsymbol \lambda) \tag{7.22} \\
\text{subject to} & \quad \boldsymbol \lambda \geq 0
\end{align}
$$

여기서 $\boldsymbol \lambda$는 dual variable이고, $\mathfrak D (\boldsymbol \lambda) = \min _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda)$이다.

</div>

<div class="notice" markdown="1">

*Remark.* 앞선 Optimization Using Gradient Descent에서 두 가지 전제가 있었다.

첫번째는 **minimax inequality**로, 두 변수를 받는 어떠한 함수 $\varphi(\boldsymbol x, \boldsymbol y)$에 대해, 이의 maximin은 minimax보다 작다는 것이다. 즉,

$$
\max \limits _{\boldsymbol y} \min \limits _{\boldsymbol x} \varphi(\boldsymbol x, \boldsymbol y) \leq \min \limits _{\boldsymbol x} \max \limits _{\boldsymbol y}  \varphi(\boldsymbol x, \boldsymbol y) \tag{7.23}
$$

이 부등식은 다음의 부등식을 통해 증명할 수 있다.

$$
\text{For all } \boldsymbol x, \boldsymbol y \quad \min \limits _{\boldsymbol x} \varphi(\boldsymbol x, \boldsymbol y) \leq \max \limits _{\boldsymbol y} \varphi(\boldsymbol x, \boldsymbol y) \tag{7.24}
$$

좌변에서 $\boldsymbol y$에 대한 최댓값을 취하더라도 부등식은 유지되는데, 이는 모든 $\boldsymbol x$에 대해 성립하기 때문이다. 비슷하게 우변도 $\boldsymbol x$에 대해 최솟값을 취할 수 있다.

두번째 개념은 **weak duality(약쌍대성)**로, 이는 식 (7.23)을 통해 primal value가 항상 dual value보다 크거나 같음을 보일 수 있다. 이는 아래 식 (7.27)에 더욱 자세히 소개되어 있다.

</div>

식 (7.18)의 $J(\boldsymbol x)$와, (7.20b)의 Lagrangian의 차이는 indicator function이 linear function이 되게끔 제약을 풀어준 것이다. 그러므로 $\lambda$가 양수일 때 Lagrangian $\mathfrak L (\boldsymbol x, \boldsymbol \lambda)$는 $J(\boldsymbol x)$의 lower bound가 된다. 따라서 $\boldsymbol \lambda$에 대한 $\mathfrak L (\boldsymbol x, \boldsymbol \lambda)$의 최댓값은 

$$
J(\boldsymbol x) = \max \limits _{\boldsymbol \lambda \geq 0} \mathfrak L (\boldsymbol x, \boldsymbol \lambda) \tag{7.25}
$$

가 된다. 원래 문제는 $J(\boldsymbol x)$를 최소화하는 것이었으므로,

$$
\min \limits _{\boldsymbol x \in \mathbb R^d} \max \limits _{\boldsymbol \lambda \geq 0} \mathfrak L (\boldsymbol x, \boldsymbol \lambda) \tag{7.26}
$$

앞서 본 minimax inequality를 통해 다음을 결론낼 수 있다.

$$
\min \limits _{\boldsymbol x \in \mathbb R^d} \max \limits _{\boldsymbol \lambda \geq 0} \mathfrak L (\boldsymbol x, \boldsymbol \lambda) \geq \max \limits _{\boldsymbol \lambda \geq 0} \min \limits _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda) \tag{7.27}
$$

이는 **weak duality(약쌍대성)**으로 알려져있다. 우변의 안쪽 항은 dual objective function $\mathfrak D(\boldsymbol \lambda)$라 한다.

제약조건을 갖던 원래의 최적화 문제와 반대로 $\min \limits _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda)$는 주어진 $\boldsymbol \lambda$에 대한  unconstrained  optimization 문제가 된다. 만일 $\min \limits _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda)$를 푸는 것이 쉽다면, 전체적인 문제도 풀기 쉬운 문제가 된다. 이는 식 (7.20b)에서 $\min \limits _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda)$이 $\boldsymbol \lambda$에 대한 affine임을 통해 알 수 있다. 그러므로 $\min \limits _{\boldsymbol x \in \mathbb R^d} \mathfrak L (\boldsymbol x, \boldsymbol \lambda)$은 $\boldsymbol \lambda$의 affine function에 대한 minimum이 되고, 따라서 $\mathfrak D (\boldsymbol \lambda)$는 $f(\cdot)$과 $g(\cdot)$의 형태에 상관없이 concave하다. 바깥쪽 항에 대한 문제(maximization over $\boldsymbol \lambda$)는 concave function에 대한 최대화 문제이므로 효율적으로 계산할 수 있다.

$f(\cdot)$과 $g(\cdot)$가 미분가능하다고 가정하면, Lagrangian을 $\boldsymbol x$에 대해 미분함으로서 Lagrange dual problem을 찾을 수 있고, 이를 0으로 놓으면 최적해를 구할 수 있다. $f(\cdot)$과 $g(\cdot)$가 convex한 형태는 Section 7.3.1과 7.3.2에서 살펴볼 것이다.

<div class="notice" markdown="1">

*Remark* (Equality Constraints). 식 (7.17)에 equality constraint를 추가해보자.

$$
\begin{align}
\min \limits _{\boldsymbol x} \quad & f(\boldsymbol x) \\
\text{subject to} \quad & g _i(\boldsymbol x) \leq 0 ~ \text{for all} ~ i=1, \dotsc, m  \tag{7.28} \\
&h _j(\boldsymbol x) = 0 ~ \text{for all}~ j=1, \dotsc, n
\end{align}
$$

equality constraint을 equality constraint $h _j (\boldsymbol x) \leq 0$과 $h _j (\boldsymbol x) \geq 0$ 두 개의 inequality constraint로 바꿔 모델링할 수 있다. 이는 Lagrange multiplier가 되고, 비제약이 된다.

그러므로 식 (7.28)의 inequality  constraints  Lagrange multiplier를 non-negative로 제약하고,  equality constraint에 해당하는 Lagrange multiplier의 제약을 없앨 수 있다.

</div>

## Convex Optimization

지금까지는 global optimum을 보장하는 최적화 문제만을 살펴보았다. $f(\cdot)$가 convex이고, $g(\cdot)$과 $h(\cdot)$를 포함하는 제약조건이 convext set일 때 이는 **convex optimization problem**이라고 부른다. 이러한 조건 하에서 우리는 **strong duality(강한 쌍대성)**을 갖는다. 강한 쌍대성은은 쌍대 문제의 최적해와 원시문제의 최적해가 갖음을 의미한다. convex function과 convex set의 차이점은 머신러닝에서 구체적으로 제시되지는 않으나 맥락을 통해 추론할 수 있을 것이다.

<div class="notice--success" markdown="1">

**Definition 7.2.** 어떠한 $x, y \in \mathcal C$, 어떠한 스칼라에 $\theta$에 대해 $0 \leq \theta \leq 1$에 대해 다음을 만족하면 집합 $\mathcal C$는 **convex set**이 된다.

$$
\theta x + (1 - \theta) y \in \mathcal C \tag{7.29}
$$

</div>

Convex set은 집합 내의 어떠한 두 점을 연결한 점이 집합에 속하는 것을 의미한다. 다음의 Figure 7.5와 7.6은 convex set과 nonconvex set을 보여준다.

![image](https://user-images.githubusercontent.com/47516855/124619592-bc06b300-deb3-11eb-9542-d45024a21fe8.png){: .align-center}{:width="150"}

<div class="notice--success" markdown="1">

**Definition 7.3.** 함수 $f: \mathbb R^D \mapsto \mathbb R$의 정의역이 convex set이라 하자. 정의역 내 어떠한 $\boldsymbol x, y$와 어떠한 스칼라 $0 \leq \theta \leq 1$에 대해 다음을 만족하면 함수 $f$는 **convex function**이라 한다.

$$
f(\theta \boldsymbol x + (1 - \theta) \boldsymbol y) \leq \theta f(\boldsymbol x) + (1 - \theta)f(\boldsymbol y) \tag{7.30}
$$

</div>

*Remarks* **Concave function**은 convex function을 뒤집은 것이다.
{: .notice}

식 (7.28)에서 제약조건 $g(\cdot)$과 $h(\cdot)$은 함수를 잘라 집합을 만든다. Convext function과 convex set의 또 다른 관계로는 convex function을 "채워서" 얻는 집합을 생각해보는 것이다. Convex function은 그릇같은 모양으로 여기다가 물을 채워넣는 것을 상상해보자. 이 결과로 채워진 집합이 나올것이며, 이는 convex function의 **epigraph**라고 하며, convex set이 된다.

만약 함수가 미분가능하다면, 이의 그레디언트 $\nabla _{\boldsymbol x}f(\boldsymbol x)$에 대한 convexity를 특정화할 수 있다. 함수 $f(\boldsymbol x)$가 convex이기 위한 필요충분조건은 어떠한 두 점$\boldsymbol x, \boldsymbol y$에 대해 다음을 만족하는 것이다.

$$
f(\boldsymbol y) \geq f(\boldsymbol x) + \nabla _{\boldsymbol x}f(\boldsymbol x)^\intercal (\boldsymbol y - \boldsymbol x) \tag{7.31}
$$

함수가 두번 미분 가능하다면, 즉, Hessian이 존재한다면, 함수가 convex이기 위한 필요충분조건은 $\nabla^2 _{\boldsymbol x}f(\boldsymbol x)$가 positive semidefinite인 것이다.

*Remark*. 식 (7.30)의 부등식은 **Jensen’s inequality**로 불린다. 뿐만 아니라 convex function의 nonnegative weighted sum형태는 전부다 Jensen’s inequality로 불리운다.
{: .notice}

요약하자면 다음을 만족하는 constrained optimization 문제는 **convex optimization problem**이라고 불린다.

$$
\begin{align}
\min \limits _{\boldsymbol x} \quad f(\boldsymbol x) \\
\text{subject to} \quad & g _i(\boldsymbol x) \leq 0 ~ \text{for all } ~ i=1, \dotsc, m  \tag{7.38} \\
&h _j(\boldsymbol x) = 0 ~ \text{for all }~ j=1, \dotsc, n
\end{align}
$$

여기서 모든 함수 $f(\boldsymbol x)$와 $g _i(\boldsymbol x)$는 convex function이고, 모든 $h _j (\boldsymbol x)=0$은 convex set이다. 다음을 통해 널리 사용되는 convex optimization 문제 두 개를 살펴보겠다.

### Linear Programming

여태까지 보았던 함수가 모두 선형이라고 해보자. 즉,

$$
\begin{align}
\min \limits _{\boldsymbol x \in \mathbb R^d} \quad & \boldsymbol c^\intercal \boldsymbol x \tag{7.39} \\
\text{subject to} \quad & \boldsymbol A \boldsymbol x \leq \boldsymbol b
\end{align}
$$

여기서 $\boldsymbol A \in \mathbb R^{m \times d}$이며, $\boldsymbol b \in \mathbb R^m$이다. 이는 **linear program(선형계획법)**이라 한다. 이는 $d$개의 변수를 갖고 있고 $m$개의 선형 제약조건을 갖고 있다. Lagrangian은 다음과 같이 주어진다.

$$
\mathfrak L (\boldsymbol x, \boldsymbol \lambda) = \boldsymbol c^\intercal \boldsymbol x + \boldsymbol \lambda^\intercal (\boldsymbol A \boldsymbol x - \boldsymbol b) \tag{7.40}
$$

여기서 $\boldsymbol \lambda \in \mathbb R^m$은 non-negative Lagrange multiplier로 이루어진 벡터이다. 이를 $\boldsymbol x$에 관하여 다시 써보면,

$$
\mathfrak L (\boldsymbol x, \boldsymbol \lambda) 
= (\boldsymbol c + \boldsymbol A^\intercal + \boldsymbol \lambda)^\intercal \boldsymbol x - \boldsymbol \lambda \boldsymbol b \tag{7.41}
$$

$\mathfrak L (\boldsymbol x, \boldsymbol \lambda)$를 $\boldsymbol x$에 대해 미분하고 0으로 두면,

$$
\boldsymbol c + \boldsymbol A^\intercal + \boldsymbol \lambda = \boldsymbol 0 \tag{7.42}
$$

그러므로 dual Lagrangian은 $\mathfrak D(\boldsymbol \lambda) = - \boldsymbol \lambda^\intercal \boldsymbol b$가 된다. 우리가 원하는 것은 $\mathfrak D(\boldsymbol \lambda)$를 최대화 하는 것이다. 미분하여 0이 되게끔 하는 제약에다가 $\boldsymbol \lambda \geq 0$인 제약을 추가하여 다음과 같은 dual optimization 문제를 얻게된다. (*primal은 minimize, dual은 maximize한다*)

$$
\begin{align}
\max _{\boldsymbol \lambda \in \mathbb R^m} \quad & -\boldsymbol b^\intercal \boldsymbol \lambda \tag{7.43} \\
\text{subject to} \quad & \boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda = \boldsymbol 0 \\
& \boldsymbol \lambda \geq 0
\end{align}
$$

이 또한 linear program이지만 이번엔 $m$개의 변수를 갖는다. 우리는 $m$이나 $d$ 중 작은 것을 택해 primal (7.39)를 풀어도 되고 dual (7.43)풀어도 된다. $d$는 변수의 갯수이고, $m$은 primal linear program의 제약조건의 수이다.

### Quadratic Programming

이번엔 제약조건이 affine인 convex quadratic objective function에 대해 고려해보자. 즉,

$$
\begin{align}
\min \limits _{\boldsymbol x \in \mathbb R^d} \quad & \frac{1}{2} \boldsymbol x^\intercal \boldsymbol Q \boldsymbol x + \boldsymbol c^\intercal \boldsymbol x \tag{7.45} \\
\text{subject to} \quad & \boldsymbol A \boldsymbol x \leq \boldsymbol b
\end{align}
$$

여기서 $\boldsymbol A \in \mathbb R^{m \times d}$이며, $\boldsymbol b \in \mathbb R^m, \boldsymbol c \in \mathbb R^d$이다. Square symmetric matrix $\boldsymbol Q \in \mathbb R^{d \times d}$ 는 positive definite하므로 목적함수는 convex이다. 이는 **Quadratic program(2차 계획법)**으로 알려져있다. 이는 $d$개의 변수와 $m$개의 선형제약조건이 있다.

이의 Lagrangian은 다음과 같이 주어진다.

$$
\begin{align}
\mathfrak L (\boldsymbol x, \boldsymbol \lambda) & = \frac{1}{2} \boldsymbol x^\intercal \boldsymbol Q \boldsymbol x + \boldsymbol c^\intercal \boldsymbol x + \boldsymbol \lambda^\intercal (\boldsymbol A \boldsymbol x - \boldsymbol b) \tag{7.48a} \\
& = \frac{1}{2} \boldsymbol x^\intercal \boldsymbol Q \boldsymbol x + (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda)^\intercal \boldsymbol x - \boldsymbol \lambda^\intercal \boldsymbol b \tag{7.48b}
\end{align}
$$

또 한번 항을 정리해보자. 미분하고 0으로 놓으면,

$$
\boldsymbol Q \boldsymbol x + (\boldsymbol c \boldsymbol A^\intercal \boldsymbol \lambda) = \boldsymbol 0 \tag{7.49}
$$

$\boldsymbol Q$가 역행렬이 존재한다고 하자.

$$
\boldsymbol x = \boldsymbol Q^\intercal (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda) \tag{7.50}
$$

식 (7.50)을 primal Lagrangian $\mathfrak L (\boldsymbol x, \boldsymbol \lambda)$로 치환하면, dual Lagrangian을 얻는다.

$$
\mathfrak D (\boldsymbol \lambda) = - \frac{1}{2} (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda)^\intercal \boldsymbol Q^{-1} (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda) - \boldsymbol \lambda^\intercal \boldsymbol b \tag{7.51}
$$

따라서 dual optimization 문제가 다음과 같이 주어진다.

$$
\begin{align}
\max \limits _{\boldsymbol x \in \mathbb R^m} \quad & - \frac{1}{2} (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda)^\intercal \boldsymbol Q^{-1} (\boldsymbol c + \boldsymbol A^\intercal \boldsymbol \lambda) - \boldsymbol \lambda^\intercal \boldsymbol b \tag{7.52} \\
\text{subject to} \quad & \boldsymbol \lambda \geq 0
\end{align}
$$

머신러닝에서 quadratic programming을 응용한 사례는 Chapter 12에서 보게 될 것이다.

### Legendre-Fenchel Transform and Convex Conjugate

Constrained Optimization and Lagrange Multipliers에서 살펴보았던 duality에 대해 제약 없이 살펴보도록 하자. Convex set에 대해 유용한 사실 중 하나는 이것이 이의 supporting hyperplane(받침 초평면)과 동일하게 표현될 수 있다는 것이다. hyperplane이 convex set을 가로지르고 한 면을 포함하면 이를 convex set의 **supporting hyperplane**이라고 부른다. 앞서 epigraph를 얻기 위해 convex function에 물을 넣었고, 그 결과가 convex set이라 하였다. 그러므로 Convex function 또한 이의 hyperplane을 통해 기술할 수 있다. 이에더해 supporting hyperplane이 convex function을 단순하게 접하기만 한다면, 이는 그 점에서의 함수의 접선이 된다. 함수에서 $\boldsymbol x _0$에서의 접선은 그레디언트 $\left. \frac{\mathrm{d} f(\boldsymbol x)}{\mathrm{d}\boldsymbol x} \right \rvert  _{\boldsymbol x = \boldsymbol x _0}$가 된다.

> 아래는 supporting hyperplane에 대한 그림이다.
>
> ![](https://glossary.informs.org/ver2/mpgwiki/images/7/72/Supporting.jpg){: .align-center}{:width="400"}

요약하면 convex set이 이의 supporting hyperplane을 통해 동일하게 표현할 수 있으므로, convex function 또한 supporting hyperplane의 그레디언트를 통해 표현할 수 있다. **Legendre transform(르장드르 변환)**은 이러한 개념을 공식으로 만든 것이다.

가장 일반적이면서도 비직관적인 정의로 시작한 다음 이에 대한 특수한 경우를 살펴볼 것이다. **Legendre transform**은 convex differentiable function $f(\boldsymbol x)$에서 접선 $s(\boldsymbol x) = \nabla _{\boldsymbol x} f(\boldsymbol x)$와 연관된 함수로 변환한다. 이는 변수에 관한 변환이 아닌 함수에 대한 변환이란 것을 염두에 두자. Legendre transform는 **convex conjugate**로도 알려져있으며 (이유는 곧 알게될 것이다), duality와 깊게 연관되어 있다.

<div class="notice--success" markdown="1">

**Definition  7.4.** 함수 $f: \mathbb R^D \mapsto \mathbb R$의 **Convex conjugate**는 다음과 같이 정의된다.

$$
f^{*} (\boldsymbol s) = \text{sup} _{\boldsymbol x \in \mathbb R^D} (\langle \boldsymbol s , \boldsymbol x \rangle - f(\boldsymbol x)) \tag{7.53}
$$

</div>

> 여기서 $\text{sup}$는 상한을 의미한다.

여기서 정의한 convex conjugate는 $f$가 convex하거나 미분가능할 필요가 없다. 정의에선 일반적인 inner product를 사용하지만 앞으로는 dot product를 사용할 것이다.

Definition 7.4를 기하학적인 관점에서 이해하려면 간단한 1차원의 미분가능한 convex 함수를 생각하면 된다. 예를 들어 $f(x) = x^2$라 해보자. 이의 hyperplane은 선이다. 어떤 선 $y = sx + c$를 생각해보자. 우리는 지금 convex function을 이의 supporting hyperplane을 통해 표현할 것이므로, $f$의 그래프 상의 각 점 $(x _0, f(x _0))$와 gradient를 고정하고, 이를 지나는 $c$의 최솟값을 찾는다. 위 점을 지나는 $c$의 최솟값은 기울기 $s$를 갖는 선이 함수 $f(x) = x^2$을 스쳐 지나가게 된다. 이를 수식으로 표현하면,

$$
y - f(x _0)  = s(x - x _0) \tag{7.54}
$$

$y$절편은 $-sx _0 +f(x _0)$가 된다. $f$의 교차하는 $y=sx +c$에서 $c$의 최솟값은 따라서

$$
\text{inf} _{x _0} - sx _0 +f(x _0) \tag{7.55}
$$

앞서 다룬 convex conjugate는 이의 negative로 정의된다. 이는 일차원의 convex나 미분가능한 function이 아닌 어떠한 nonconvex, non-differentiable $f: \mathbb R^D \mapsto \mathbb R$에도 적용 가능하다.

앞서 Lagrange multipliers를 이용하여 dual optimization 문제를 유도하였다. 또한 convex opimtization 문제에 대해 강 쌍대성이 존재하였고, 이를 통해 primal과 dual의 해가 일치함을 보였다. Legendre-Fenchel transform 역시 dual optimization으로 변환하는데 이용할 수 있다. 또한 함수가 convex하고 미분가능하면, 이의 상한은 유일하다.

Legendre-Fenchel transform은 convex optimization 문제로 표현할 수 있는 머신러닝 문제에 이용될 수 있다. 특히 각 데이터에 독립적으로 적용되는 convex loss function의 경우 conjugate loss는 dual 문제로 유도하는데 편리함을 제공한다.

## Further Reading

Continuous optimization는 활발하게 연구되는 분야이며 여기서는 최근의 고급연구들에 대해서는 다루지 않는다.

그레디언트 디센트 관점에서 두 가지 약점이 있는데, 첫번째는 이것이 first-order algorithm이므로 표면의 곡률에 대한 정보를 사용하지 않는다는 점이다. 긴 계곡이 있을 경우 그레디언트는 최솟값에 수직인 점을 가르키게 된다. 모멘텀은 가속하는 방법론 중하나로 일반화할 수 있다. Conjugate gradient는 이러한 문제를 피할 수 있다. Newton method와 같은 second-order는 Hessian을 사용하여 곡률에 대한 정보를 이용하게 된다. Step size를 계산하거나 momentum같은 아이디어는 이러한 목적함수의 곡률을 고려하는 것으로부터 시작하였다. Quasi-Newton과 같은 L-BFGS는 더 적은양의 연산을 통해 Hessian을 근사한다. 최근들어 mirror descent나 natural gradient와 같이 하강하는 방면을 계산하기 위해 다른 지표를 사용하기 시작하였다.

두번째 문제는 미분 불가능한 함수를 다루는 경우이다. 이를 위해 **subgradient method**가 이용된다. 이에 대한 더 자세한 정보는 Bertsekas (1999)의 책을 참고하자.

현대의 머신러닝은 데이터양이 너무 많아 배치 그레디언트 디센트를 사용하지 못하는 경우가 많으므로 stochastic gradient descent를 사용하는게 적절하다.

duality and convex optimization에 대해서는 Boyd and Vanden-berghe (2004)의 책을 통해 강의와 슬라이드를 제공받을 수 있다. 더욱 수학적으로 다루기 위해서는  Bertsekas  (2009)을 보는게 좋으며, 최신 책을 보고 싶다면 이 분야에서 유명한 학자 Nesterov (2018)의 책을 보는 것이 좋다. Convex optimization은 convex analysis에 기반하며, convex function에 대해 핵심적인 결론에 대해 살펴보고 싶다면 Rock-afellar (1970), Hiriart-Urruty and Lemar ́echal (2001), Borwein andLewis (2006) 를 참고하자. Legendre–Fenchel transforms 또한 이 책들에서 다루고 있지만 초보자들에겐 Zia et al. (2009)을 추천한다. Convex optimization 분석에서 Legendre–Fenchel transform의 역할에 대해서는 survey 논문 Polyak (2016)을 보면 된다.

> [Convex Conjugate에 대한 visualization](https://github.com/bikestra/bikestra.github.com/blob/master/notebooks/Convex%20Conjugates.ipynb?fbclid=IwAR2143vFu2bDYVBFSHQ-i_-YrY6NHfCaZ81o21q1ZgFKO9pj2ExU_P1EfzU)