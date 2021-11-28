---
title:  "머신러닝을 위한 수학 정리: Vector Calculus"
toc: true
toc_sticky: true
permalink: /project/mml/Vector-Calculus/
categories:
  - Mathmatics
  - Machine Learning
tags:
  - linear algebra
  - vector
  - calculus
use_math: true
last_modified_at: 2021-06-01
---

많은 머신러닝 알고리즘은 목적함수를 모델 파라미터에 대해 최적화하며, 모델이 얼마나 데이터를 잘 설명하는지를 조정한다. 좋은 파라미터를 찾는 것은 최적화 문제 (Section 8.2/8.3)이라 부른다. 이는 첫번째로는 linear regression (Chapter 9)에서 curve-fitting 문제와 선형 가중치 파라미터를 최적화하여 우도를 최대화하는데 사용한다. 두번째로는 차원 축소, 데이터 압축을 위한 auto-encoder로, reconstruction error를 최소화한다. 마지막은 데이터 분포를 모델링하는 Gaussian mixture model로 (Chapter 11), 위치와 모양에 대한 각 mixuture component의 파라미터를 최적화하여 우도를 최대화한다.

![image](https://user-images.githubusercontent.com/47516855/119262196-4d2c0e00-bc15-11eb-832e-bd821f7bff05.png){: .align-center}{:width="400"}

위 그림은 본 챕터의 마인드맵으로, 본 챕터의 개념들이 서로 어떻게 관계되고 연결되는지를 보여준다.

본 챕터의 중심은 함수의 개념이다. 어떤 함수 $f$는 어떠한 양(quantity)으로, 두 양을 서로 관련시킨다. 이 책에서 이러한 양은 일반적으로 input $\boldsymbol x \in \mathbb R^D$과 target (function value) $f(\boldsymbol x)$가 된다. 이 함수는 실함수(real-valued function)로 가정한다. 여기서 $\mathbb R^D$는 $f$의 정의역(domain)으로, 함수값(function value) $f(\boldsymbol x)$는 $f$의 상(image) 또는 공역 (codomain)이라고 한다.

> 실함수란 일반적으로 생각하는 함수로, 실수집합의 부분집합을 정의역으로 하고, 실수집합을 공역으로 하는 함수이다.

앞서 첫 챕터의 Image and Kernel에서는 linear function의 맥락에서 더욱 디테일한 개념을 소개한다. 우리는 종종 함수를 다음과 같은 방식으로 쓰곤 한다.

$$
\begin{align}
f: \mathbb R^D \to \mathbb R \tag{5.1a} \\
\boldsymbol x \mapsto f(\boldsymbol x) \tag{5.1b}
\end{align}
$$

(5.1a)는 $f$가 $\mathbb R^D$에서 $\mathbb R$로 mapping함을 말해주고, (5.1b)는 input $\boldsymbol x$를 함수값 $f(\boldsymbol x)$에 할당함을 말해준다. 함수 $f$는 모든 input $\boldsymbol x$를 정확히 하나의 함수값 $f(\boldsymbol x)$로 할당한다.

본 챕터에서 우리는 함수의 gradient를 계산하는 방법을 배울 것이며, 이는 머신러닝에서 학습을 용이하게 해준다. 그러므로 벡터 미적분은 머신러닝에서 가장 중요한 수학적 도구 중 하나이다.

> 본 포스트는 머신러닝에 필요한 선형대수 및 확률과 같은 수학적 개념을 정리한 포스트이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.

## Differentiation of Univariate Functions

일변수 함수(univariate function)의 미분을 간략하게 다시 살펴보자. 우선 일변수 함수 $y = f(x), x, y \in \mathbb R$의 차분몫(difference quotient)에서 시작해보자. 그리고 미분(derivative)을 정의해보자.

<div class="notice--warning" markdown="1">

**Definition 5.1** (Difference Quotient)

**difference quotient**은 다음과 같이 정의되며, $f$ 그래프 상의 두 점을 지나는 secant line의 기울기를 계산한다.

$$
\frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x} \tag{5.3}
$$

</div>

아래 그림은 $x _0$와 $x _0 + \delta x$사이의 difference quotient를 보여준다.

![image](https://user-images.githubusercontent.com/47516855/119263542-6be0d380-bc1a-11eb-9073-1ca42f62b90f.png){: .align-center}{:width="300"}

difference quotient는 또한 $f$를 linear function으로 가정했을 경우, $x _0$와 $x _0 + \delta x$사이 $f$의 평균 기울기와도 같다. $\delta x \to 0$, 즉 함수 $f$의 점 $0$에서의 극한은 $f$의 $x$에서의 접선(tangent)를 얻게 된다 ($f$가 미분가능일 때만). 접선은 그러면 $f$의 $x$에서의 미분이 된다.

<div class="notice--warning" markdown="1">

**Definition 5.2** (Derivative).

$h > 0$에 대해 $f$의 $x$에서의 **derivative(미분)**은 다음과 같은 극한으로 정의된다.

$$
\frac{\text{d} f}{\text{d} x} := \lim\limits _{h \to 0} \frac{f(x+h) - f(x)}{h} \tag{5.4}
$$

</div>

앞선 그림의 할선은 접선이 된다.

$f$의 미분은 $f$가 가장 가파르게 상승하는 방향을 나타낸다.

### Taylor Series

Taylor Series(테일러 급수)는 어떤 함수 $f$를 항의 무한한 합으로 표현한 것이다. 이러한 항들은 $f$가 $x _0$일 때의 미분값을 이용하여 결정된다.

<div class="notice--warning" markdown="1">

**Definition 5.3** (Taylor Polynomial (테일러 다항식))

$f: \mathbb R \to \mathbb R$의 $x _0$에서의 $n$차 **Taylor polynomial(테일러 다항식)**은 다음과 같이 정의된다.

$$
T _n(x) := \sum^n _{k=0} \frac{f^{(k)}(x _0)}{k!} (x - x _0)^k  \tag{5.7}
$$

</div>

$f^{(k)}(x _0)$는 $f$의 $x _0$에서의 $k$-th derivative이고 (존재할 것이라고 가정), $\frac{f^{(k)}(x _0)}{k!}$는 polynomial의 coefficient이다.

<div class="notice--warning" markdown="1">

**Definition 5.4** (Taylor Series)

Smooth function(매끄러운 함수) $f \in \mathcal C^\infty, f: \mathbb R \to \mathbb R$에 대해 ($f \in \mathcal C^\infty$은 무한번 연속 가능하다는 뜻이다), $f$의 $x _0$에서의 **Taylor series**는 다음과 같이 정의된다.

$$
T _\infty (x) := \sum^\infty _{k=0} \frac{f^{(k)}(x _0)}{k!}(x - x _0)^k \tag{5.8}
$$

</div>

$x _0 = 0$인 경우에 우리는 테일러 급수의 특수한 경우인 **Maclaurin series(맥클로린 급수)**를 얻을 수 있다. $f(x) = T _\infty (x)$인 경우, $f$는 **analytic(해석적, 해석함수)**이라고 부른다.

*Remark.* 일반적으로 $n$차 테일러 다항식은 함수의 근사로, 다항식으로 표현할 필요가 없는 함수이다. 테일러 다항식은 $x _0$ 근처에서 $f$와 비슷하다. 그러나 $n$차 테일러 다항식은 $k \leq n$인 다항식 $f$를 정확히 표현할 수 있는데, 이는 $ i > k$인 모든 미분 $f^{(i)}$가 사라지기 때문이다.
{: .notice}

> $f(x) = x^4$를 생각해보자. $k$가 4보다 크게 되면 그의 미분값이 0이 된다.

<div class="notice" markdown="1">

*Remark.* 테일러 급수는 멱급수의 특수한 경우이다.

$$
f(x) = \sum^{\infty} _{k=0} a _k(x - c)^k
$$

$a _k$는 계수이고 $c$는 상수이다. 이는 Definite 5.4 Taylor series의 특수한 형태이다.

</div>

### Differentiation Rules

이제 간단하게 기본적인 미분의 규칙을 기술해보자. $f$의 미분은 $f'$으로 쓴다.

$$
\begin{align}
& \text{Product rule:} ~~ (f(x)g(x))' = f'(x)g(x) + f(x)g'(x) \tag{5.29} \\
& \text{Quotient rule:} ~~ (\frac{f(x)}{g(x))})' = \frac{f'(x)g(x) - f(x)g'(x)}{(g(x))^2} \tag{5.30} \\
& \text{Sum rule:} ~~ (f(x) + g(x))' = f'(x) + g'(x) \tag{5.31} \\
& \text{Chaine rule:} ~~ (g(f(x)))' = (g \circ f)'(x) = g'(f(x))f'(x) \tag{5.32} \\
\end{align}
$$

## Partial Differentiation and Gradient

앞서 설명한 미분은 스칼라값 $x \in \mathbb R$의 함수 $f$에 적용한다. 이제는 일반적인 경우를 생각해보자. 바로 함수 $f$가 한개 혹은 그 이상의 변수 $\boldsymbol x \in \mathbb R^n$에 대한 것이다 ($f(\boldsymbol x = f(x _1, x_2)$). 여러 변수를 갖는 함수에 대한 미분의 일반적인 형태를 **gradient**라고 부른다.

우리는 함수 $f$의 $\boldsymbol x$에 대한 gradient를 **한 인자를 변수로**하고, 나머지를 상수로 고정시켜 구할 것이다. 이러면 이제 gradient는 **partial derivative(편미분)**의 모임이 된다.

<div class="notice--warning" markdown="1">

**Definition 5.5** (Partial Derivative)

$n$개의 변수 $x _1, \cdots, x _n$을 갖는 함수 $f: \mathbb R^n \to \mathbb R, \boldsymbol x \mapsto f(\boldsymbol x), x \in \mathbb R^n$에 대해 partial derivative를 다음과 같이 정의한다.

$$
\begin{align}
\frac{\partial f}{\partial x _1} =& \lim\limits _{h \to 0} \frac{f(x _1+h, x _2, \cdots, x _n) - f(\boldsymbol x)}{h} \\
&\vdots \\
\frac{\partial f}{\partial x _n} =& \lim\limits _{h \to 0} \frac{f(x _1, x _2, \cdots, x _n + h) - f(\boldsymbol x)}{h} \\ \tag{5.39}
\end{align}
$$

</div>

그리고 이를 행 벡터로 모으게 되면,

$$
\nabla _{\boldsymbol x} f = \text{grad} f = \frac{df}{dx} =
\begin{bmatrix}
\frac{\partial f(\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial f(\boldsymbol x)}{\partial x _m}
\end{bmatrix} \in \mathbb R^{1 \times n} \tag{5.40}
$$

$n$은 변수의 갯수이고, 1은 $f$의 image/range/codomain이 된다. 여기서 column vector는 $\boldsymbol x = [x _1, \cdots, x _n]^\intercal$로 정의한다. (5.40)의 행 벡터는 $f$의 **gradient** 혹은 **Jacobian**이라고 불리며, 앞서 배운 미분의 일반화된 형태이다.

*Remark.* 여기서의 Jacobian의 정의는 vector-valued function(벡터함수)에 대한 Jacobian의 일반적 정의의 특수한 형태로, 편미분의 모임으로 표현되었다. 이는 추후 [Gradients of Vector-Valued Functions](#gradients-of-vector-valued-functions)에서 다시 살펴볼 것이다. 
{: .notice}

*Remark* (Gradient as a Row Vector). 일반적으로 vector를 column vector로 표현함에도 불구하고 문헌에서 gradient vector를 column vector로 표현하는 경우는 드물다. Gradient vector가 row vector로 표현하는 이유는 두 가지가 있다. 우선 함수 $f$를 벡터함수인 $f: \mathbb R^n \to \mathbb R^m$으로 일반화할 경우, gradient가 자연스럽게 matrix로 변화하기 때문이다. 두번째로는 row vector 형태에서는 gradient의 차원에 주의를 기울이지 않고도 multi-variate chain rule을 즉시 적용시킬 수 있기 때문이다. 이에 대해서는 [Gradients of Vector-Valued Functions](#gradients-of-vector-valued-functions)에서 다시 다룰 것이다.
{: .notice}

### Basic Rules of Partial Differentiation

Multivariate의 경우에도 ($\boldsymbol x \in \mathbb R^n$) 기본적인 미분 규칙 (e.g., sum rule, product rule, chain rule) 또한 여전히 적용된다. 그러나 vector $\boldsymbol x \in \mathbb R^n$에 대한 미분을 계산할 때 주의를 기울여야 한다. 이때부터 gradient에는 벡터와 행렬이 포함되고, 행렬곱이 commutative하지 않기 때문이다. 즉, 순서가 중요해진다.

다음은 product rule, sum rule, chain rule이다.

$$
\begin{align}
& \text{Product rule:} ~~ \frac{\partial}{\partial \boldsymbol x}(f(\boldsymbol x)g(\boldsymbol x)) = \frac{\partial f}{\partial \boldsymbol x}g(\boldsymbol x) + f(\boldsymbol x)\frac{\partial g}{\partial \boldsymbol x} \tag{5.46} \\
& \text{Sum rule:} ~~ \frac{\partial}{\partial \boldsymbol x}(f(\boldsymbol x) + g(\boldsymbol x)) = \frac{\partial f}{\partial \boldsymbol x} + \frac{\partial g}{\partial \boldsymbol x} \tag{5.47} \\
& \text{Chaine rule:} ~~ \frac{\partial}{\partial \boldsymbol x}(g \circ f)(x) = \frac{\partial}{\partial \boldsymbol x} (g (f(\boldsymbol x)))= \frac{\partial g}{\partial f} \frac{\partial f}{\partial \boldsymbol x} \tag{5.48}
\end{align}
$$

Chain rule을 좀 더 자세히 들여다보자. (5.48)을 보면 행렬곱을 할 때 이웃한 차원끼리 일치해야 되는 것과 비슷하게 진행된다. 왼쪽에서 오른쪽으로 진행하는걸 보면 chain rule도 이와 비슷한 성질을 보이는 것을 볼 수 있다. $\partial f$가 첫번째 인수의 분모에서 나타나고, 두번째 인수의 분자에서 나타난다. 이 인수들을 서로 곱하면, $\partial f$의 차원이 맞아 떨어지며, $\partial f$가 "취소"되고, $\partial g / \partial \boldsymbol x$가 남게된다.

> 여기서 "약분"이 아니라 "취소"라고 표현했음에 유의하자. derivative는 약분이 되지 않는다.

### Chain Rule

두개의 변수 $x _1, x _2$에 대한 함수 $f: \mathbb R^2 \to \mathbb R$를 생각해보자. 또한, $x _1(t), x _2(t)$는 $t$에 대한 함수이다. $f$의 $t$에 대한 gradient를 계산하기 위해 다변량 함수의 chain rule (5.48)를 다음과 같이 적용할 필요가 있다.

$$
\frac{df}{dt} =
\begin{bmatrix}
  \frac{\partial f}{\partial x _1} & \frac{\partial f}{\partial x _2}  
\end{bmatrix}
\begin{bmatrix}
  \frac{\partial x _1(t)}{\partial t} \\ \frac{\partial x _2(t)}{\partial t}
\end{bmatrix}
= \frac{\partial f}{\partial x _1} \frac{\partial x _1}{\partial t} + \frac{\partial f}{\partial x _2} \frac{\partial x _2}{\partial t} \tag{5.49}
$$

여기서 d는 gradient를, $\partial$은 partial derivative를 나타낸다.

만일 $f(x _1, x _2)$가 $x _1, x _2$에 대한 함수이고, $x _1(s, t), x _2(s, t)$가 두 변수 $s, t$에 대한 함수라면, 체인룰은 다음과 같은 편미분 결과를 낸다.

$$
\begin{align}
\frac{\partial f}{\partial \color{orange}{s}} 
= \frac{\partial f}{\partial \color{blue}{x _1}}\frac{\partial \color{blue}{x _1}}{\partial \color{orange}{s}} + \frac{\partial f}{\partial \color{blue}{x _2}}\frac{\partial \color{blue}{x _2}}{\partial \color{orange}{s}} \tag{5.51} \\
\frac{\partial f}{\partial \color{orange}{t}} 
= \frac{\partial f}{\partial \color{blue}{x _1}}\frac{\partial \color{blue}{x _1}}{\partial \color{orange}{t}} + \frac{\partial f}{\partial \color{blue}{x _2}}\frac{\partial \color{blue}{x _2}}{\partial \color{orange}{t}} \tag{5.52}
\end{align}
$$

그리고 gradient는 행렬곱을 통해 얻어지게 된다.

$$
\frac{df}{d(s, t)} = \frac{\partial f}{\partial \boldsymbol x} \frac{\partial \boldsymbol x}{\partial (s, t)} =
\underbrace{
  \begin{bmatrix}
  \frac{\partial f}{\color{blue}{\partial x _1}} & \frac{\partial f}{\color{orange}{\partial x _1}}
  \end{bmatrix}
} _{\frac{\partial f}{\partial \boldsymbol x}}
\underbrace{
  \begin{bmatrix}
  \color{blue}{\frac{\partial x _1}{\partial s}} & \color{blue}{\frac{\partial x _1}{\partial t}} \\
  \color{orange}{\frac{\partial x _2}{\partial s}} & \color{orange}{\frac{\partial x _2}{\partial t}}
  \end{bmatrix}
} _{\frac{\partial \boldsymbol x}{\partial (s, t)}}
$$

이렇듯 체인룰을 간단하게 행렬곱으로 표현하는 것은 gradient가 row vector로 정의되었을 때만 가능하다. 그렇지 않을 경우 차원을 일치시키기 위해 transpose해야 할 것이다. 이는 gradient가 벡터나 행렬의 형태라면 굉장히 직관적인 방법일 것이다. 그러나 텐서가 된다면 얘기가 달라질 것이다.

## Gradients of Vector-Valued Functions

여태까지는 함수 $f: \mathbb R^n \to \mathbb R$에 대한 편미분과 gradient만을 다뤘다. 여기서는 이를 vector-valued function (벡터함수) $\boldsymbol f: \mathbb R^n \to \mathbb R^m$으로 확장해볼 것이다.

함수 $\boldsymbol f: \mathbb R^n \to \mathbb R^m$와 vector $\boldsymbol x = [x _1, \cdots, x _n]^\intercal \in \mathbb R^n$에 대해 이에 대항하는 함수값 벡터는 다음과 같이 주어진다.

$$
\boldsymbol f(\boldsymbol x) = 
\begin{bmatrix}
  f _1 (\boldsymbol x) \\ \vdots \\ f _m (\boldsymbol x)
\end{bmatrix} \in \mathbb R^m \tag{5.54}
$$

이런 표현법은 벡터함수 $\boldsymbol f: \mathbb R^n \to \mathbb R^m$를 함수로 이루어진 벡터 $\begin{bmatrix} f _1, \cdots, f _m \end{bmatrix}^\intercal, f _i: \mathbb R^n \to \mathbb R$로 표현할 수 있게 해준다. 모든 $f _i$에 대한 미분 규칙이 앞장에서 적용한 것과 정확히 똑같이 적용된다.

따라서 벡터함수 $\boldsymbol f: \mathbb R^n \to \mathbb R^m$의 $x _i \in \mathbb R$에 대한 편미분값은 벡터로 주어진다.

$$
\frac{\partial \boldsymbol f}{\partial x _i} 
=
  \begin{bmatrix}
  \frac{\partial f _1}{\partial x _i} \\ \vdots \\ \frac{\partial f _m}{\partial x _i} 
  \end{bmatrix}
=
  \begin{bmatrix}
  \lim _{h \to 0} \frac{f _1(x _1, \cdots, x _{i-1}, x _i +h, x _{i+1}, \cdots x _n) - f _1(\boldsymbol x)}{h} \\ \vdots \\ \lim _{h \to 0} \frac{f _1(x _1, \cdots, x _{i-1}, x _i +h, x _{i+1}, \cdots x _n) - f _m(\boldsymbol x)}{h}
  \end{bmatrix} \in \mathbb R^m  \tag{5.55}
$$

(5.40)으로부터 $\boldsymbol f$의 벡터에 대한 gradient가 편미분으로 이루어진 row vector임을 알 수 있다. (5.55)에서 모든 편미분 ${\partial \boldsymbol f}{\partial x _i}$는 column vector가 된다. 그러므로, $\boldsymbol f: \mathbb R^n \to \mathbb R^m$의 $\boldsymbol x \in \mathbb R^n$에 대한 gradient는 이러한 편미분값을 모아서 얻을 수 있다.

$$
\begin{align}
\frac{d \boldsymbol f (\boldsymbol x)}{d \boldsymbol x} & =
  \begin{bmatrix}
    \frac{\partial \boldsymbol f (\boldsymbol x)}{\partial x _1} & \cdots & \frac{ \boldsymbol f (\boldsymbol x)}{\partial x _n}
  \end{bmatrix} \tag{5.56a} \\
& = 
  \begin{bmatrix}
    \frac{\partial f _1 (\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial f _1 (\boldsymbol x)}{\partial x _n} \\
    \vdots & & \vdots \\
    \frac{\partial f _m (\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial f _m (\boldsymbol x)}{\partial x _n}
  \end{bmatrix} \tag{5.56b}
\end{align}
$$

<div class="notice--warning" markdown="1">

**Definition 5.6** (Jacobian)

벡터함수 $\boldsymbol f: \mathbb R^n \to \mathbb R^m$의 $\boldsymbol x \in \mathbb R^n$의 first order partial derivative를 **Jacobian (자코비안)**이라고 부른다. Jacobian $\boldsymbol J$는 $m \times n$ 행렬이며, 다음과 같이 정렬할 수 있다.

$$
\begin{align}
\boldsymbol J &= \boldsymbol \nabla _{\boldsymbol x} \boldsymbol f = \frac{d \boldsymbol f (\boldsymbol x)}{d \boldsymbol x} = 
  \begin{bmatrix}
    \frac{\partial \boldsymbol f (\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial \boldsymbol f (\boldsymbol x)}{\partial x _n}
  \end{bmatrix} \tag{5.56} \\
& = 
  \begin{bmatrix}
    \frac{\partial f _1 (\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial f _1 (\boldsymbol x)}{\partial x _n} \\
    \vdots & & \vdots \\
    \frac{\partial f _m (\boldsymbol x)}{\partial x _1} & \cdots & \frac{\partial f _m (\boldsymbol x)}{\partial x _n}
  \end{bmatrix} \tag{5.58} \\
\boldsymbol x &= 
  \begin{bmatrix}
    x _1 \\
    \vdots \\
    x _n \\
  \end{bmatrix}, J(i, j) = \frac{\partial f _i}{\partial x _j} \tag{5.59}
\end{align}
$$

</div>

식 (5.58)의 특수한 케이스로 함수 $f: \mathbb R^n \to \mathbb R^1$는 row vector로 이루어진 Jacobian을 갖는다 ($1 \times n$).

*Remark.* 이 책에서는 **numerator layout (분자중심표현)**을 사용하여 미분을 표현한다. 즉, $\boldsymbol f \in \mathbb R^m$의 $\boldsymbol x \in \mathbb R^n$에 대한 미분 $d \boldsymbol f / d \boldsymbol x$은 $m \times n$ 행렬이 되고, $\boldsymbol f$의 원소들은 행을, $\boldsymbol x$의 원소들은 열을 이뤄 Jacobian을 생성한다. 이의 반대개념인 **denominator layout (분모중심표현)**도 있고, 이는 분자중심표현의 transpose한 것과도 같다.
{: .notice}

추후 probability distribution에서 change-of-variable(변수변환, 치환)을 위해 Jacobian이 어떻게 사용되는지 살펴볼 것이다.

앞서 행렬식은 parallelogram의 영역을 계산하는데 이용할 수 있다고 했다. 만일 두 벡터 $\boldsymbol b _1 = [1, 0]^\intercal, \boldsymbol b _2 = [0, 1]^\intercal$를 unit square (아래 그림의 파란색)라고 한다면, 이의 영역은 1이 될 것이다.

![image](https://user-images.githubusercontent.com/47516855/119687301-0d1d9300-be82-11eb-90f3-2a6c81c3c4c6.png){: .align-center}{:width="400"}

두 벡터 $\boldsymbol c _1 = [-2, 1]^\intercal, \boldsymbol c _2 = [1, 1]^\intercal$를 보면, 이 두 벡터가 주는 영역(아래 그림의 오렌지색)은 행렬식의 절댓값인 3이 될 것이다. 즉, 이는 unit square의 3배가 된다. 이러한 scaling factor는 unit square를 다른 square로 transform하는 mapping을 찾음으로서 알아낼 수 있다. 이러한 mapping을 두 가지 방법을 사용해서 알아보자.

첫번째 방법은 이 변환이 선형임을 이용하는 것이다. $\{\boldsymbol b _1,  \boldsymbol b _2\}$와 $\{\boldsymbol c _1,  \boldsymbol c _2\}$은 모두 $\mathbb R^2$의 기저이다. 여기서 하려는 것은 기저를 변환하는 변환행렬을 찾는 것이다. 이러한 행렬 $\boldsymbol J$는 다음과 같다. 바꾸는 방법은 [Basis Change](/project/mml/Linear-Algebra/#basis-change)를 확인하자.

$$
\boldsymbol J = 
  \begin{bmatrix}
    -2 & 1 \\
    1 & 1
  \end{bmatrix} \tag{5.62}
$$

즉, $\boldsymbol J \boldsymbol b _1 = \boldsymbol c _1 = \boldsymbol J \boldsymbol b _2 = \boldsymbol c _2$가 된다. $\boldsymbol J$의 행렬식은 3이 된다.

그러나 이 방법은 선형변환에 대해서만 가능하므로 비선형 변환(Section 6.7과 연관)을 하기 위해 partial derivative를 이용, 더 일반적인 방법을 이용한다. 

이를 위해 함수가 variable transformation를 수행한다고 하자. 예제에서 $\boldsymbol f$는 $(\boldsymbol b _1,  \boldsymbol b _2)$로 표현된 어떠한 vector $\boldsymbol x$의 좌표표현을 $(\boldsymbol c _1,  \boldsymbol c _2)$로 변환한다. 우리가 하려는 것은 이 변환의 실체를 알아내어 행렬식을 계산하는 것이므로, $\boldsymbol x$를 약간 변화시켰을 때 $\boldsymbol f$가 얼마만큼 변하는지 알아내야 한다. 이 질문은 Jacobian matrix $\frac{d \boldsymbol f}{d \boldsymbol x} \in \mathbb R^{2 \times 2}$을 통해 완벽하게 답변할 수 있다.

$$
\begin{align}
y _1 = -2 x _1 + x _2 \tag{5.63} \\
y _2 = x _1 + x _2 \tag{5.64}
\end{align}
$$

앞선 맵핑은 위와 같이 쓸 수 있으며, $\boldsymbol x$와 $\boldsymbol y$ 사이의 함수 관계를 얻을 수 있다. 이를 편미분하면,

$$
\frac{\partial y _1}{\partial x _1} = -2, \frac{\partial y _1}{\partial x _2} = 1, \frac{\partial y _2}{\partial x _1} = 1, \frac{\partial y _2}{\partial x _2} = 1 \tag{5.65}  
$$

이를 모아보면 다음과 같은 자코비안을 만들 수 있다.

$$
\boldsymbol J = 
\begin{bmatrix}
  \frac{\partial y _1}{\partial x _1} & \frac{\partial y _1}{\partial x _2} \\
  \frac{\partial y _2}{\partial x _1} & \frac{\partial y _2}{\partial x _2}
\end{bmatrix}

\begin{bmatrix}
  -2 & 1 \\
  1 & 1
\end{bmatrix}
$$

이 자코비안은 우리가 찾던 좌표표현을 나타낸다. 만일 좌표 변환이 linear라면 이 표현은 정확히 basis change를 나타내게 되지만, 비선형이라면 선형변환으로 근사한 것에 불과하다.

자코비안 행렬식과 variable transformation은 Section 6.7 Change of Variables/Inverse Transform과 연관이 있다. 이는 확률변수와 확률분포를 변환하는 것이다. 이러한 변환은 뉴럴 네트워크를 **reparametrization trick** 혹은 **infinite perturbation analysis**를 사용하여 학습한다는 맥락에서 머신러닝과 매우 연관이 깊다.

본 챕터에서 우리는 함수의 미분을 알아보았다. 다음 그림은 이러한 미분의 차원이다. 함수의 차원과 벡터의 차원에 따라 어떻게 변화하는지, 그 결과는 row vector인지 column vector인지 잘 살펴보도록 하자.

![image](https://user-images.githubusercontent.com/47516855/119828292-000dac00-bf35-11eb-8316-ba1a057c03de.png){: .align-center}{:width="250"}

> 본 장에서 설명한 내용에 대해 이해가 안된다면 책에 있는 예제를 꼭 풀어보도록 하자.

> 다음은 Linear Model의 Least-Square Loss에 대해 gradient를 구하는 예제이다. 보통은 예제를 번역하진 않지만, 본 예제는 중요하다고 생각하여 특별하게 소개한다.

**Example 5.11 (Gradient of a Least-Squares Loss in a Linear Model)**

다음과 같은 linear model을 생각해보자.

$$
\boldsymbol y = \boldsymbol \Phi \boldsymbol \theta \tag{5.75}
$$

$\boldsymbol \theta \in \mathbb R^D$는 파라미터 벡터, $\boldsymbol \Phi \in \mathbb R^{N \times D}$는 input feature, $\boldsymbol y \in \mathbb R^N$는 레이블이다. 다음과 같이 로스함수와 에러를 함수로 정의해보자.

$$
\begin{align}
L(\boldsymbol e) := \| \boldsymbol e \|^2 \tag{5.76} \\
\boldsymbol e (\boldsymbol \theta) := \boldsymbol y - \boldsymbol \Phi \boldsymbol \theta \tag{5.77}
\end{align}
$$

여기서 얻고자 하는 것은 $\partial L / \partial \boldsymbol \theta$이고, 이를 위해 연쇄법칙을 사용한다. $L$은 **Least squae** loss function이다. 이를 계산하기전에 차원부터 살펴보면,

$$
\frac{\partial L}{\partial \boldsymbol \theta} \in \mathbb R^{1 \times D} \tag{5.78}
$$

연쇄법칙에 의해 계산하면 다음과 같은 식이 나온다.

$$
\frac{\partial L}{\partial \boldsymbol \theta} = \color{blue}{\frac{\partial f}{\partial \boldsymbol e}} \color{orange}{\frac{\partial \boldsymbol e}{\partial \boldsymbol \theta}} \tag{5.79}
$$

이의 $d$번째 원소는 다음과 같이 주어진다.

$$
\frac{\partial L}{\partial \boldsymbol \theta} [1, d] = \sum^N _{n=1} \frac{\partial L}{\partial \boldsymbol e} [n] \frac{\partial \boldsymbol e}{\partial \boldsymbol \theta} [n, d] \tag{5.80}
$$

$\| \boldsymbol e \|^2 = \boldsymbol e^\intercal \boldsymbol e$ 이므로 ([Inner Product](/project/mml/Analytics-Geometry/#inner-product) 참고),

$$
\color{blue}{\frac{\partial L}{\partial \boldsymbol e} = 2 \boldsymbol e^\intercal} \in \mathbb R^{1 \times N} \tag{5.81}
$$

또한,

$$
\color{orange}{\frac{\partial \boldsymbol e}{\partial \boldsymbol \theta} = -\boldsymbol \Phi} \in \mathbb R^{N \times D} \tag{5.82}
$$

따라서 편미분의 결과는 다음과 같다.

$$
\frac{\partial L}{\partial \boldsymbol \theta} = \underbrace{\color{blue}{2 \boldsymbol e^\intercal}} _{1 \times N} ~~ \underbrace{\color{orange}{-\boldsymbol \Phi}} _{N \times D} \in \mathbb R^{1 \times D} \tag{5.83}
$$

<div class="notice" markdown="1">

*Remark.* 이에 대해 체인룰을 사용하지 않고 직접 미분을 하여 똑같은 결과를 얻어낼 수 있다.

$$
L _2(\boldsymbol \theta) = \| \boldsymbol y - \boldsymbol \Phi \boldsymbol \theta\|^2 = (\boldsymbol y - \boldsymbol \Phi \boldsymbol \theta)^\intercal (\boldsymbol y - \boldsymbol \Phi \boldsymbol \theta) \tag{5.84}
$$

이는 $L _2$와 같은 간단한 형태에 대해서는 매우 실용적이지만 함수가 복잡할 경우 사용하기 어렵다.

</div>


> 스칼라, 벡터, 행렬이 나오다보니 슬슬 헷갈리기 시작한다. 아래는 위키피디아에서 가져온 자료로, 함수와 변수의 형태에 따른 미분의 결과값이다.
>
> ![image](https://user-images.githubusercontent.com/47516855/120656228-660eac00-c4be-11eb-8d4b-678a387c010d.png){: .align-center}{:width="800"}


> 아래는 스칼라 벡터, 벡터, 행렬에 대해 미분한 결과를 나타낸 것이다.
>
> ![image](https://user-images.githubusercontent.com/47516855/119852475-491c2b00-bf4a-11eb-9577-c3184d0598ae.png){: .align-center}{:width="550"}
>
> 출처: [다크 프로그래머: 벡터 미분과 행렬 미분](https://darkpgmr.tistory.com/141)

## Gradients of Matrices

이제 행렬 또는 벡터에 대한 행렬의 gradient를 구해보도록 하자. 이 결과로 multidimensional tensor를 얻게 될 것이다. 이러한 텐서는 편미분을 모아놓은 다차원의 배열로 생각해볼 수 있다.

예를 들어 $m \times n $의 행렬 $\boldsymbol A$를  $p \times q$ 행렬 $\boldsymbol B$로 미분하여 gradient를 계산한다면, 그 결과는 Jacobian으로 $(m \times n) \times (p \times q)$, 즉, 4차원짜리 tensor $\boldsymbol J$가 될 것이다. 이의 원소는 $\boldsymbol J _{ijkl} = \partial \boldsymbol A _{ij}/\partial \boldsymbol J _{kl}$로 주어진다.

행렬은 선형변환으로 표현할 수 있기 때문에, $\mathbb R^{m \times n}$과 $\mathbb R^{mn}$사이에 vector-space isomorphism (linear, invertible mapping)이 존재한다는 사실을 이용할 수 있다. 따라서 앞선 행렬을 $mn$, $pq$의 벡터로 다시 표현할 수 있다. 이러한 $mn$ 벡터를 이용한 gradient는 사이즈 $mn \times pq$의 Jacobian을 결과로 얻게 된다. 다음 그림은 본 두개의 접근법을 나타낸다.

![image](https://user-images.githubusercontent.com/47516855/120342931-3b441c80-c333-11eb-8877-435eca33ff83.png){: .align-center}{:width="800"}

## Useful Identities for Computing Gradients

다음을 통해 머신러닝에서 자주 쓰이는, 유용한 gradient의 목록을 보도록 하겠다. 여기서 $\text{tr}(\cdot)$은 trach로, $\text{det}(\cdot)$는 determinant로, $\boldsymbol f(\boldsymbol X)^{-1}$은 $\boldsymbol f(\boldsymbol X)$의 inverse로 사용한다.

![image](https://user-images.githubusercontent.com/47516855/120076122-02136e80-c0df-11eb-9d2f-79520ffe08ed.png){: .align-center}{:width="600"}

## Backpropagation and Automatic Differentiation

머신러닝에서는 gradient를 이용하여 모델의 파라미터를 찾는다 (Section 7.1). 이는 모델 파라미터에 대한 objective function의 gradient를 찾는 것과 같다.

다음과 같은 함수를 생각해보자.

$$
f(x) = \sqrt{x^2 + \text{exp}(x^2)} + \text{cos}(x^2 + \text{exp}(x^2)) \tag{5.109}
$$

Chain rule과 미분이 linear하다는 점을 이용하여 이의 gradient를 구할 수 있다.

$$
\begin{align}
\frac{df}{dx} &= \frac{2x + 2x ~ \text{exp}(x^2)}{2\sqrt{x^2 + \text{exp}(x^2)}} - \text{sin}(x^2 + \text{exp}(x^2))(2x + 2x ~ \text{exp}(x^2)) \\
&= 2x \left(\frac{1}{2 \sqrt{x^2+\text{exp}(x^2)}} - \text{sin}(x^2 + \text{exp}(x^2)) \right ) (1+\text{exp}(x^2)) \tag{5.110}
\end{align}
$$

이러한 방법으로 gradient를 구하는 것은 실용적이지 않은데, 이는 매우 긴 식이 나올수도 있기 때문이다. 이 뜻은 즉 우리가 실제로 사용할 때 주의를 기울이지 않으면 gradient의 구현이 함수를 계산하는 것보다 더욱 expensive하고, 이는 불필요한 overhead를 발생시킨다. deep neural network를 학습할 때 **backpropagation algorithm**을 사용하는데, 이는 error function을 모델 파라미터에 대해 미분하는데 매우 효율적인 방법이다.

### Gradients in a Deep Network

체인룰을 사용하는 딥러닝에서 $\boldsymbol y$를 계산하는데 많은 함수의 합성을 거치게 된다.

$$
\boldsymbol y = (f _K \circ f _{K-1} \circ \cdots \circ f _1)(\boldsymbol x) = f _K(f _{K-1}(\cdots (f _1(\boldsymbol x)))) \tag{5.111}
$$

여기서 $\boldsymbol x$는 input이고, $\boldsymbol y$는 observation, 모든 함수는 파라미터를 가르킨다.

뉴럴 네트워크에서 $i$번째 레이어에는 다음과 같은 함수 $f _i(\boldsymbol x _{i-1}) = \sigma (\boldsymbol A _{i-1} \boldsymbol x _{i-1} + \boldsymbol b _{i-1})$가 존재한다. 여기서 $\boldsymbol x _{i-1}$ $i-1$번째 레이어의 아웃풋이고, $\sigma$는 활성화함수이다. 이러한 모델을 학습하기 위해 모델 파라미터에 대해 loss function $L$의 gradient를 구하게 된다. 이는 모든 레이어에 걸쳐서 일어나게 된다.

$$
\begin{align}
\boldsymbol f _0 & := \boldsymbol x \tag{5.112} \\
\boldsymbol f _i & := \sigma _i (\boldsymbol A _{i-1} \boldsymbol f _{i-1} + \boldsymbol b _{i-1}), i = 1, \cdots, K \tag{5.113}
\end{align}
$$

이를 그림으로 표현하면 다음과 같다. 

![image](https://user-images.githubusercontent.com/47516855/120079622-b0271480-c0ef-11eb-953a-bc34cf558aa0.png){: .align-center}{:height="300"}

여기서 관심있는 것은 파라미터 $\boldsymbol A _h, \boldsymbol b _j$ for $j=0, \cdots, K -1$이고, 이는 squared loss를 최소화한다.

$$
\boldsymbol L (\boldsymbol \theta) = \|\boldsymbol y - \boldsymbol f _K(\boldsymbol \theta, \boldsymbol x) \|^2 \tag{5.114}
$$

여기서 theta는 $\boldsymbol \theta = \\{\boldsymbol A _0, \boldsymbol b _0, \cdots, \boldsymbol A _{K-1}, \boldsymbol b _{K-1} \\}$이다.

Parameter set $\boldsymbol \theta$에 대한 gradient를 얻기 위해 $L$을 파라미터 $\boldsymbol \theta _j = \\{\boldsymbol A _j, \boldsymbol b _j \\}$에 대해 편미분한 것 과 같다. 이를 모든 레이어에 대해 체인룰을 적용할 수 있다.

$$
\begin{align}
\frac{\partial L}{\partial \boldsymbol \theta _{K-1}} &= \frac{\partial L}{\partial \boldsymbol f _{K}} \color{blue}{\frac{\partial \boldsymbol f _{K}}{\partial \boldsymbol \theta _{K-1}}} \tag{5.115} \\
\frac{\partial L}{\partial \boldsymbol \theta _{K-2}} &= \frac{\partial L}{\partial \boldsymbol f _{K}} \boxed{\color{orange}{\frac{\partial \boldsymbol f _{K}}{\partial \boldsymbol f _{K-1}}} \color{blue}{\frac{\partial \boldsymbol f _{K-1}}{\partial \boldsymbol \theta _{K-2}}}} \tag{5.116} \\
\frac{\partial L}{\partial \boldsymbol \theta _{K-3}} &= \frac{\partial L}{\partial \boldsymbol f _{K}} \color{orange}{\frac{\partial \boldsymbol f _{K}}{\partial \boldsymbol f _{K-1}}} \boxed{\color{orange}{\frac{\partial \boldsymbol f _{K-1}}{\partial \boldsymbol f _{K-2}}} \color{blue}{\frac{\partial \boldsymbol f _{K-2}}{\partial \boldsymbol \theta _{K-3}}}} \tag{5.117} \\
\frac{\partial L}{\partial \boldsymbol \theta _{i}} &= \frac{\partial L}{\partial \boldsymbol f _{K}} \color{orange}{\frac{\partial \boldsymbol f _{K}}{\partial \boldsymbol f _{K-1}}} \color{orange}{\cdots} \boxed{\color{orange}{\frac{\partial \boldsymbol f _{i+2}}{\partial \boldsymbol f _{i+1}}} \color{blue}{\frac{\partial \boldsymbol f _{i+1}}{\partial \boldsymbol \theta _{i}}}} \tag{5.118}
\end{align}
$$

오렌지색은 layer의 output에 대해 이의 intput으로 편미분한 결과이고, 파란색은  layer의 output에 대해 이의 parameter로 편미분한 결과이다. 우리가 이미 편미분 $\partial L / \partial \boldsymbol \theta _{i+1}$을 계산했으면, 이는 다음 $\partial L / \partial \boldsymbol \theta _{i}$를 계산하는데 재사용된다. 그러면 우리는 박스로 된 부분만 추가적으로 계산하면 된다.

![image](https://user-images.githubusercontent.com/47516855/120081255-c3d67900-c0f7-11eb-9140-3e125b2eb102.png){: .align-center}{:width="700"}

### Automatic Differentiation

역전파는 nemerical analysis의 특별한 경우로, **automatic differentiation**으로 불린다. automatic differentiation은 수치적으로 컴퓨터가 연산가능한 수준까지 gradient를 계산하는 것이다. 이때 적절하게 체인룰을 사용한다.

![image](https://user-images.githubusercontent.com/47516855/120081277-df418400-c0f7-11eb-93a8-da859199a6e9.png){: .align-center}{:width="500"}

위에 대해 $dy/dx$를 계산해보면,

$$
\frac{dy}{dx} = \frac{dy}{db} \frac{db}{da} \frac{da}{dx} \tag{5.119}
$$

와 같이 된다.

> 번역에 애를 쓸만큼 어려운 부분이 없으므로 번역하지 않는다. 궁금한 사람은 직접 보면 좋을듯하다.

## Higher-Order Derivatives

여태까지 알아본 gradient는 first-order derivative이다. 그러나 종종 더 높은 차수의 미분을 필요로 할 때가 있을 것이다 (e.g. Newton's Method). 앞서 Taylor Series에서 우리는 polynomial을 이용하여 함수를 근사하는 방법을 배웠다. 다변수의 경우에도 이는 그대로 적용될 수 있다.

어떤 함수 $f: \mathbb R^2 \to \mathbb R$을 생각해보자. 이 함수는 두 변수 $x, y$를 필요로 한다. 높은 차수의 편미분과 gradient에 대해 다음 notation을 이용할 것이다.

- $\frac{\partial^2 f}{\partial x^2}$은 $x$에 대한 이계도함수이다.
- $\frac{\partial^n f}{\partial x^n}$은 $x$에 대한 $n$계도함수이다.
- $\frac{\partial^2 f}{\partial x \partial y}$은 $x$에 대한 미분에 $y$에 대한 미분을 구한 것이다.

**Hessian**은 모든 2차 미분의 모음이다.

만약 $f(x, y)$가 두번 미분 가능하다면,

$$
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} \tag{5.146}
$$

이 성립한다. 즉, 미분의 순서는 중요하지 않다. 그리고 이에 대응하는 **Hessian matrix**는,

$$
\boldsymbol H = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial x \partial y} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix} \tag{5.147}
$$

이는 symmetric하다. Hessian은 $\nabla^2 _{x, y}f(x, y)$로 쓴다. 일반적으로 n차원의 벡터와 벡터함수에 대해, Hessian은 $n \times n$ 행렬이 된다. Hessian은 $(x, y)$ 근처에서 함수의 곡률을 나타낸다.

*Remark* (Hessian of a Vector Field). $f: \mathbb R^n \to \mathbb R^m$이 vector field이면, Hessian은 $(m \times n \times n$)의 텐서이다.
{: .notice}

## Linearization and Multivariate Taylor Series

종종 함수 $f$의 $x$에 대한 gradient $\nabla$는 $\boldsymbol x _0$ 근처에서 $f$의 locally approximation을 구하기 위해 사용된다.

$$
f(\boldsymbol x) \approx f(\boldsymbol x _0) + (\boldsymbol \nabla _{\boldsymbol x} \boldsymbol f)(\boldsymbol x _0)(\boldsymbol x - \boldsymbol x _0) \tag{5.148}
$$

여기서 $(\boldsymbol \nabla _{\boldsymbol x} \boldsymbol f)(\boldsymbol x _0)$는 점 $\boldsymbol x _0$에서의 $\boldsymbol x$에 대한 $f$의 gradient이다. 이는 아래의 그림을 통해 확인할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/120343892-0c7a7600-c334-11eb-95bb-7bbd395dc529.png){: .align-center}{:width="600"}


원래의 함수는 직선으로 근사되었다. 근사된 선은 일부분에서만 정확하고 $\boldsymbol x _0$에서 벗어날 수록 값이 멀어지는 걸 확인할 수 있다. 위 식 (5.148)은 $\boldsymbol x _0$에서 $f$의 다변수에 대한 테일러 급수의 특수한 경우로, 오직 앞의 두개의 항만 고려하는 경우이다. 다음을 통해 더욱 일반적인 경우를 살펴보고, 이를 통해 더 나은 근사값을 계산하도록 해보자.

<div class="notice--warning" markdown="1">

**Definition 5.7** (Multivariate Taylor Series)

다음과 같은 함수를 고려해보자.

$$
\begin{align}
f: \mathbb R^D & \to \mathbb R \tag{5.149} \\
\boldsymbol x & \mapsto f(\boldsymbol x), \boldsymbol x \in \mathbb R^D \tag{5.150}
\end{align}
$$

이는 $\boldsymbol x _0$에서 smooth하다. Difference vector $\boldsymbol \delta := \boldsymbol x - \boldsymbol x _0$을 정의할 때, f의 ($\boldsymbol x _0$)에서의 **multivariate Taylor series**는 다음과 같이 정의된다.

$$
f(\boldsymbol x) = \sum^\infty _{k=0} \frac{D^k _{\boldsymbol x} f(\boldsymbol x _0)}{k!} \boldsymbol \delta^k \tag{5.151}
$$

$D^k _{\boldsymbol x} f(\boldsymbol x _0)$는 점 $\boldsymbol x _0$에서 $f$를 $\boldsymbol x$에 대해 $k$번째 (전)미분값이 된다. ($D^k _{\boldsymbol x} f(\boldsymbol x _0)$ is the $k$-th (total) derivative of $f$ with respect to $\boldsymbol x$, eval-uated at $\boldsymbol x _0$.)

</div>

> 앞서 정의한 테일러 급수를 다변수에 대해 다시 정의한 것이라 보면 된다. 따라서 terminology를 다시 재정의하였다.

<div class="notice--warning" markdown="1">

**Definition 5.8** (Taylor Polynomial)

$f$의 $\boldsymbol x _0$에서의 n차 **Taylor Polynomial**은 급수 (5.151)의 $n+1$개의 항으로 이루어져 있고, 이는 다음으로 정의된다.

$$
T _n(\boldsymbol x) = \sum^n _{k=0} \frac{D^k _{\boldsymbol x} f(\boldsymbol x _0)}{k!} \boldsymbol \delta^k \tag{5.152}
$$
</div>

(5.151)과 (5.152)에서 $\boldsymbol \delta^k$라는 약간 조잡한 표기법을 사용했는데, 이는 벡터 $\boldsymbol x \in \mathbb R^D$에 대해 $D >1, k >1$에 대해 아직 정의하지 않았다. 여기서 $D^k _{\boldsymbol x} f$와 $\boldsymbol \delta^k$는 $k$-차원의 tensor이다. $k$차원의 텐서 $\boldsymbol \delta^k \in \mathbb R^{\overbrace{D \times D \times \cdots \times D} _{k\text{ times}}}$는 $k$번 외적을 통해 계산된다. 외적은 $\otimes$로 표현한다.

$$
\begin{align}
\boldsymbol \delta^2 &:= \boldsymbol \delta \otimes \boldsymbol \delta  = \boldsymbol \delta \boldsymbol \delta^\intercal, ~~ \boldsymbol \delta^2[i, j] = \delta[i]\delta[j] \tag{5.153} \\ 
\boldsymbol \delta^3 &:= \boldsymbol \delta \otimes \boldsymbol \delta \otimes \boldsymbol \delta, ~~ \boldsymbol \delta^3[i, j, k] = \delta[i]\delta[j]\delta[k] \tag{5.154}
\end{align}
$$

아래 그림은 이에 대한 도식화이다.

![image](https://user-images.githubusercontent.com/47516855/120209847-1c2d8800-c26a-11eb-9efb-0f5f068357ce.png){: .align-center}{:width="800"}

일반적으로 우리는 테일러 급수를 통해 다음과 같은 식을 얻을 수 있다.

$$
D^k _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^k = \sum^D _{i _1 =1} \cdots \sum^D _{i _k =1} D^k _{\boldsymbol x} f(\boldsymbol x _0) [i _1, \cdots, i _k]\delta[i _1] \cdots \delta[i _k] \tag{5.155}
$$

여기서 $D^k _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^k$는 $k$번째 polynomial을 포함하고 있다.

Vector field에 대해 테일러 급수를 정의했으니, 이제 $k=0, \cdots, 3$과 $\boldsymbol \delta := \boldsymbol x - \boldsymbol x _0$에 대해, 테일러 급수 $D^k _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^k$를 전개했을 때 나오는 첫번째 항을 적어보도록 하자.

$$
\begin{align}
k&=0~ : ~ D^0 _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^0 = f(\boldsymbol x _0) \in \mathbb R \tag{5.156} \\
k&=1~ : ~ D^1 _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^1 = \underbrace{\boldsymbol \nabla _{\boldsymbol x} f (\boldsymbol x _0)} _{1 \times D} \underbrace{\boldsymbol \delta} _{D \times 1} = \sum^D _{i=1} \boldsymbol \nabla _{\boldsymbol x} f (\boldsymbol x _0)[i] \delta[i] \in \mathbb R \tag{5.157} \\
k&=2~ : ~ D^2 _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^2 = \text{tr}(\underbrace{\boldsymbol H(\boldsymbol x _0)} _{D \times D} \underbrace{\boldsymbol \delta} _{D \times 1} \underbrace{\boldsymbol \delta^\intercal} _{1 \times D} = \boldsymbol \delta^\intercal \boldsymbol H(\boldsymbol x _0 \boldsymbol \delta \tag{5.158} \\
&= \sum^D _{i=1} \sum^D _{j=1} \boldsymbol H[i, j] \delta[i] \delta[j] \in \mathbb R \tag{5.159} \\
k&=3~ : ~ D^3 _{\boldsymbol x} f(\boldsymbol x _0) \boldsymbol \delta^3 = \sum^D _{i=1} \sum^D _{j=1} \sum^D _{k=1} \boldsymbol D^3 _{\boldsymbol x} f(\boldsymbol x _0)[i, j, k] \delta[i] \delta[j] \delta[k] \in \mathbb R \tag{5.160}
\end{align}
$$

여기서 $\boldsymbol H(\boldsymbol x _0)$는 $x _0$에서의 $f$의 Hessian이다.

## Further Reading

머신러닝에서 우리는 종종 기댓값을 계산해야할 때가 있다. 즉, 다음과 같은 적분을 풀어야 한다.

$$
\mathbb E _{\boldsymbol x}[f(\boldsymbol x)] = \int f(\boldsymbol x)p(\boldsymbol x) d \boldsymbol x \tag{5.181}
$$

$p(\boldsymbol x)$가 심지어 매우 편리한 형태 (e.g. 가우시안)라도, 이러한 적분형태는 해석적으로 풀기 쉽지 않다. $f$의 테일러 급수는 근사해를 구하는 방법 중 하나이다. $p(\boldsymbol x) = \mathcal N(\boldsymbol \mu, \boldsymbol \Sigma)$라 가정하자. 그러면 first-order Taylor series expansion은 $\boldsymbol \mu$ 근처에서 지역적으로 비선형 함수 $f$를 선형화한다. 선형함수에 대해서는 평균과 공분산을 정확하게 계산할 수 있다 (Section 6.5) 참고. 이러한 성질은 **extended Kalman filter**에서 매우 잘 이용된다. 적분을 근사하는 또 다른 deterministic 방법으로는 **unscented transform**으로, gradient나 **Laplace approximation**을 필요로 하지 않는다.