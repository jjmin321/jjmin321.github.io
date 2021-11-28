---
title:  "머신러닝을 위한 수학 정리: Analytics Geometry"
toc: true
toc_sticky: true
permalink: /project/mml/Analytics-Geometry/
categories:
  - Mathmatics
  - Machine Learning
tags:
  - linear algebra
  - vector
  - matrix
use_math: true
last_modified_at: 2021-05-09
---

> [이전 포스트](/machine-learning/linear-algebra1/)에서는 matrix와 vector에 관한 내용을 살펴보았다. 이번에는 앞서 배운 개념을 기하학적으로 해석하여 직관적으로 이해하여 보자.
> 
> 본 포스트는 머신러닝에 필요한 선형대수 및 확률과 같은 수학적 개념을 정리한 포스트이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.

## Norms

우리가 기하학적인 벡터를 생각할 때 직관적으로 이의 길이는 원점으로부터 이의 "끝"까지로 생각할 것이다. Norm의 개념을 통하여 벡터 길이의 개념에 대해 살펴볼 것이다.

<div class="notice--warning" markdown="1">

**Definition 3.1** (Norm).

벡터공간 $V$에서 **Norm(노름)**은 어떤 함수로 다음을 의미한다.

$$
\begin{align}
\| \cdot \|: & V \rightarrow \mathbb R, \tag{3.1} \\
& \boldsymbol x \mapsto \| \boldsymbol x \| \tag{3.2}
\end{align}
$$

</div>

이는 각 vector $\boldsymbol x$에 대해 그에 해당하는 길이를 도출한다. 그리고 $\forall \lambda \in \mathbb R \text{ and } \boldsymbol x, \boldsymbol y \in V$에 대해 다음을 만족한다.
- Absolutely homogeneous: $ \| \lambda \boldsymbol x \| = \rvert \lambda \lvert \| \boldsymbol x  \|$
  - 길이의 곱은 방향과 관계가 없다. 따라서 상수의 절댓값을 취하게 된다.
- Triangle inequality: $\| \boldsymbol x + \boldsymbol y  \| = \| \boldsymbol x \| + \| \boldsymbol y \| $
  - 삼각형 양변의 합은 다른 변의 길이보다 작거나 같다.
- Positive definite: $ \| \boldsymbol x \| \geq 0 $
  - $ \| \boldsymbol x \| = 0 \Leftrightarrow \boldsymbol x = 0$

대표적으로는 Manhattan norm ($l _1$)과 Euclidean norm ($l _2$)로 구분되고, 이후 구분없이 norm으로 표현하는 것은 euclidean norm을 가르킨다.

![image](https://user-images.githubusercontent.com/47516855/114568958-8d5fae80-9caf-11eb-9020-09891ff8e372.png){: .align-center}{: width="700"}

## Inner Product

Inner product를 통해 두 vector 사이의 길이, 각도, 거리 등과 같은 기하학적인 개념을 직관적으로 이해할 수 있다. 가장 큰 목적은 이를 통해 vector간의 orthogonal 관계를 도출하는 것이다.

### Dot Product (Scalar Product)

아마도 inner product의 일종인 dot product는 어디선가 자주 봤을 것이다. 이는 다음과 같이 정의된다.

$$
\begin{align}\\
\boldsymbol x^ \intercal \boldsymbol y = \sum ^n _{i=1} \boldsymbol x _i \boldsymbol y _i
\end{align}
$$

위와 같은 특정한 inner product를 dot product라 할 것이다. 그러나 inner product는 이보다는 더욱 일반적인 특성으로, 어떤 특정한 성질을 만족한다.

### General Inner Product

이전 장의 [linear mapping](/project/mml/Linear-Algebra/#linear-mapping)을 생각해보자. 그때 mapping을 덧셈과 스칼라곱에 대해 재정리하였다. **Bilinear mapping** $\Omega$는 선형성을 갖는 두 argument를 mapping하는 것으로, linear mapping에서와 같이 스칼라를 더하거나 곱할 수 있다 (linear property). 즉, 모든 $\boldsymbol x, \boldsymbol y, \boldsymbol z \in V, \lambda, \psi \in \mathbb R$에 대해 다음이 성립한다.

$$
\begin{align}
\Omega(\lambda \boldsymbol x + \psi \boldsymbol y, \boldsymbol z) = \lambda \Omega(\boldsymbol x, \boldsymbol z) + \psi \Omega(\boldsymbol y, \boldsymbol z) \tag{3.6}\\
\Omega(\boldsymbol x + \lambda \boldsymbol y, \psi \boldsymbol z) = \lambda \Omega(\boldsymbol x, \boldsymbol y) + \psi \Omega(\boldsymbol x, \boldsymbol z) \tag{3.7}
\end{align}
$$

식 (3.6)은 첫번째 argument에 대한 linear를, (3.7)은 두번째 argument에 대한 linear를 나타낸다. (2.87 참고)

<div class="notice--warning" markdown="1">

**Definition 3.2.**

$V$를 vector space라 하고, $\Omega: V \times V \to \mathbb R$를 두 vector를 하나의 실수로 mapping하는 bilinear mapping이라 하자. 그러면 다음이 성립한다. 
- 모든 $\boldsymbol x, \boldsymbol y \in V$에 대해 $\Omega (\boldsymbol x, \boldsymbol y) = \Omega (\boldsymbol y, \boldsymbol x)$가 성립하면, $\Omega$는 **symmetric**이다.
- $\forall \boldsymbol x \in V \setminus \{ 0 \}: \Omega (\boldsymbol x, \boldsymbol x) > 0, ~~\Omega (\boldsymbol 0, \boldsymbol 0) = 0$이면, $\Omega$는 **positive definite**이다.

</div>

<div class="notice--warning" markdown="1">

**Definition 3.3**  

$V$를 vector space라 하고, $\Omega: V \times V \to \mathbb R$를 두 vector를 하나의 실수로 mapping하는 bilinear mapping이라 하자. 그러면 다음이 성립한다. 
- positive definite symmetric bilinear mapping $\Omega: V \times V \to \mathbb R$를 vector space에 대한 inner product라 하고, 일반적으로 $\langle \boldsymbol x, \boldsymbol y \rangle$로 표현한다.
- $(V, \langle \cdot , \cdot \rangle)$을 **inner product space** 혹은 inner product의 vector space라고 한다.
- Dot product를 사용하는 경우, $(V, \langle \cdot , \cdot \rangle)$을 **Euclidean vector space**라고 한다.

</div>

### Symmetric, Postivie Definite Matrix

Symmetric positive definite matrix는 머신러닝에서 중요한 역할을 하며, 이는 inner product를 통해 정의된다. 추후 Section 4.3 matrix decomposition에서 이를 다시 살펴볼 것이며, Section 12.4 Kernel에서 중요한 역할을 하게 된다.

어떤 n-dimensional vector space $V$의 inner product : $\langle \cdot , \cdot \rangle: V \times V \to \mathbb R$와 $V$에 대한 ordered basis $B = (\boldsymbol b _1, \cdots , \boldsymbol b _n)$을 고려해보자. 어떤 벡터 $\boldsymbol x, \boldsymbol y$는 basis vector의 linear combination으로 표현할 수 있다 ($\boldsymbol x = \sum ^n _{i=1} \psi _i \boldsymbol b _i, \boldsymbol y = \sum ^n _{j=1} \lambda _j \boldsymbol b _j$). Inner product의 bilinearity로 인해, 모든 $\boldsymbol x, \boldsymbol y \in V$에 대해 다음이 성립한다.

$$
\langle \boldsymbol x, \boldsymbol y \rangle 
= \langle 
\sum ^n _{i=1} \psi _i \boldsymbol b _i, 
\sum ^n _{j=1} \lambda _j \boldsymbol b _j 
\rangle
= \sum ^n _{i=1} \sum ^n _{j=1} \psi _i 
\langle \boldsymbol b _i, \boldsymbol b _j \rangle \lambda _j
= \hat {\boldsymbol x}^\intercal A \hat{\boldsymbol y} \tag{3.10}
$$

$A _{ij}:=\langle \cdot, \cdot \rangle$이며, $\hat {\boldsymbol x}, \hat{\boldsymbol y}$는 basis $B$에 대한 $\boldsymbol x, \boldsymbol y$의 좌표이다. 이는 inner product $\langle \cdot, \cdot \rangle$가 $A$를 통해 unique하게 결정된다는 뜻이다. Inner product가 symmetric이란 것은 $A$ 또한 symmetric하다는 것을 의미한다. 더 나아가 inner product의 positive definiteness는 다음을 암시적으로 나타낸다.

$$
\forall \boldsymbol x \in V \setminus \{ 0 \}: \boldsymbol x^\intercal A \boldsymbol y > 0 \tag{3.11}
$$

즉, innder product의 결과가 항상 양수임을 의미한다.

<div class="notice--warning" markdown="1">

**Definition 3.4** (Symmetric, Positive Definite Matrix).

(3.11)을 만족하는 symmetric matrix $\boldsymbol A \in \mathbb R^{n \times n}$은 **symmetric, positive definite** 혹은 단순히 **positive definite**하다고 한다. 만일 (3.11)에서 $\geq$만을 만족하면, **symmetric positive semidefinite**하다고 한다.

</div>

$\boldsymbol A \in \mathbb R^{n \times n}$가 symmetric, positive definite라면, 

$$
\langle \boldsymbol x, \boldsymbol y \rangle = \hat{\boldsymbol x}^\intercal \boldsymbol A \hat{\boldsymbol y} \tag{3.14}
$$

는 ordered basis $B$에 대한 inner product를 정의한다. $\hat{\boldsymbol x}, \hat{\boldsymbol y}$는 $B$에 대한 $\boldsymbol x, \boldsymbol y \in V$의 coordinate representation이다.

> positive semi definite는 이변수함수의 극점과 연관이 있다. 어떤 이변수함수 $f(x, y) = ax^2+2bxy+cy^2$를 고려해보자. 이를 행렬로 나타내면
>
> $$
f(x, y) = ax^2+2bxy+cy^2 =
\begin{bmatrix}
x & y 
\end{bmatrix}
\begin{bmatrix}
a & b \\ c & d
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix}
$$
>
> 이 행렬이 positive semi definite하다는 것은 극솟점이 존재한다는 것이다.
>
> 출처: [[선형대수학] 양의 정부호 행렬(positive definite matrix)이란?.txt](https://bskyvision.com/205)

<div class="notice--info" markdown="1">

**Theorem 3.5.**

real-valued, finite-dimensional vector space $V$와 ordered basis $B$에 대해, $\langle \cdot , \cdot \rangle: V \times V \to \mathbb R$는 inner produdct임과 다음을 만족하는 sysmetric, positive semi definite matrix $A \in \mathbb R^{n \times n}$가 존재함은 동치이다.

$$ 
\langle \boldsymbol x , \boldsymbol y \rangle = \hat{\boldsymbol x}^\intercal \boldsymbol A \hat{\boldsymbol y} \tag{3.15}
$$

만일 $A \in \mathbb R^{n \times n}$가 sysmetric, positive semi definite matrix라면 다음 성질을 만족한다.
- $\forall \boldsymbol x \neq \boldsymbol 0, \boldsymbol x^\intercal A \boldsymbol y > 0$ 이기 때문에 ($\textrm{if }\boldsymbol x \neq \boldsymbol 0, \text{ then } A \boldsymbol x \neq \boldsymbol 0 $), $A$의 null space는 오직 0으로만 구성된다.
- $A$의 diagonal elements $a _{ij} = \boldsymbol e^\intercal _i A \boldsymbol e _i >0 $이기 때문에, positive하다.

</div>



## Length and Distance

앞서 vector의 길이를 계산하기 위해 norm을 사용하였다. 이는 다음과 같이, 

$$
\| \boldsymbol x \| := \sqrt{\langle \boldsymbol x, \boldsymbol y \rangle}
$$

어떠한 **inner product라도 norm을 자연스럽게 도출**한다는 점에서 둘은 밀접한 관계에 있다. 그러나 모든 norm이 inner product를 통해 도출되지는 않는다. 앞서 살펴본 Manhattan norm이 이런 예시가 된다. 이번에는 inner product을 통해 도출되는 norm에 초점을 맞춰 길이와 거리, 각도와 관련된 기하학적 개념에 대해 살펴보자.


<div class="notice" markdown="1">

*Remark* (Cauchy-Schwarz Inequality)

Inner product vector space $(V, \langle \cdot , \cdot \rangle)$에서 도출된 $\| \cdot \|$은 Cauchy-Schwarz inequality를 만족한다.

$$
\rvert \langle \boldsymbol x, \boldsymbol y \rangle \lvert \leq \| \boldsymbol x \| \| \boldsymbol y \|
$$

</div>

<div class="notice--warning" markdown="1">

**Definition 3.6.** (Distance and metric). 어떤 inner product space $(V, \langle \cdot , \cdot \rangle)$를 고려해보자. 그러면,

$$
d(\boldsymbol x, \boldsymbol y) := \| \boldsymbol x - \boldsymbol y \| = \sqrt{\langle \boldsymbol x - \boldsymbol y, \boldsymbol x - \boldsymbol y \rangle}
$$  

는 $\boldsymbol x$와 $\boldsymbol y$ 사이의 **거리**이다. 만일 dot product를 inner product로 사용하면, 이 거리는 **Euclidean distance**라 한다.

다음과 같은 mapping은 **metric** (distance function, 거리함수)이라 부른다.

$$
\begin{align}
d : V \times V \to \mathbb R \tag{3.22} \\ 
(\boldsymbol x, \boldsymbol y) \mapsto d(\boldsymbol x, \boldsymbol y) \tag{3.23}
\end{align}
$$

</div>

vector의 길이와 비슷하게 벡터간의 거리는 inner product를 필요로 하지 않고, norm으로도 충분하다. 만일 inner product를 통해 도출한 norm이 있다면, 거리는 inner product의 선택에 따라 norm이 달라질 수 있다.

Metric $d$는 다음을 만족한다:
1. Postive semi definite
    - 모든 $\boldsymbol x, \boldsymbol y \in V$에 대해 $d(\boldsymbol x, \boldsymbol y) \geq 0$
    - $d(\boldsymbol x, \boldsymbol y) = 0 \iff \boldsymbol x = \boldsymbol y$
2. Symmetric
    - 모든 $\boldsymbol x, \boldsymbol y \in V$에 대해 $d(\boldsymbol x, \boldsymbol y) = (\boldsymbol y, \boldsymbol x)$
3. Triangle inequality
    - $d(\boldsymbol x, \boldsymbol z) \leq d(\boldsymbol x, \boldsymbol y) +d(\boldsymbol y, \boldsymbol z) $ 

*Remark*. 앞서 봤던 inner product의 성질과 metric의 성질은 매우 비슷하다. 그러나 Definition 3.3과 Definition 3.6을 비교해보면 $\langle \boldsymbol x, \boldsymbol y \rangle$과 $d(\boldsymbol x, \boldsymbol y)$는 반대 방향으로 동작하는 것을 알 수 있다. 매우 비슷한 $\boldsymbol x$와 $\boldsymbol y$는 inner product에선 큰 값을, metric에선 작은 값을 내놓을 것이다.
{: .notice}

## Angles and Orthogonality

Inner product를 통해 vector의 길이를 정의하고 distance를 정의할 뿐만 아니라 각도 $\omega$를 구할 수 있다. Cauchy-Schwarz inequality를 두 vector $\boldsymbol x, \boldsymbol y$ 사이의 inner product space내의 각도를 정의할 수 있으며, 이러한 표현은 $\mathbb R^2, \mathbb R^3$에 대한 직관과 일치한다. $\boldsymbol x \neq \boldsymbol 0, \boldsymbol y \neq \boldsymbol 0$라 가정하자. 그러면,

$$
-1 \leq \frac{\langle \boldsymbol x, \boldsymbol y \rangle}{\| \boldsymbol x \|\| \boldsymbol y \|} \geq 1 \tag{3.24}
$$

따라서 Figure 3.4에 묘사된 것과 같은 유일한 $\omega \in [0, \pi]$가 존재하며, 이는 다음과 같다.

$$
\cos{\omega} = \frac{\langle \boldsymbol x, \boldsymbol y \rangle}{\| \boldsymbol x \|\| \boldsymbol y \|}  \tag{3.25}
$$

![image](https://user-images.githubusercontent.com/47516855/114734977-b51e4780-9d7f-11eb-852c-196d917609c1.png){: .align-center}{: width="500"} 

$\omega$는 두 vector $\boldsymbol x, \boldsymbol y$ 사이의 **각도**이다. 직관적으로 이는 이 둘의 orientation이 얼마나 비슷한지 말해준다.

> orientation과 direction의 차이는 [다음](https://www.mathemania.com/lesson/vectors/)을 참고하자.

Inner product의 핵심은 우리에게 orthogonal(직교)하는 vector의 특징을 기술하는데 있다.

<div class="notice--warning" markdown="1">

**Definition 3.7** (Orthogonality).

두 벡터 $\boldsymbol x, \boldsymbol y$가 **orthogonal**함은 $\langle \boldsymbol x, \boldsymbol y \rangle = 0$과 동치이다 (iff). 이는 $ \boldsymbol x \perp \boldsymbol y$로 쓴다. 추가적으로 $\| \boldsymbol x \|=1 = \| \boldsymbol y \|$ 이면 (i.e. unit vector), $\boldsymbol x$와 $\boldsymbol y$는 **orthonomal**하다.

</div>

이는 $\boldsymbol 0$-vector는 모든 vector에 대해 orthogonal함을 의미한다.

*Remark.* Orthogonality는 dot product일 필요 없는 bilinear form에 대해 수직하는 개념을 일반화하는 것이다. 기하학적으로, orthogonal vectors를 특정한 inner product에 대해 직각을 이루는 것으로 생각할 수 있다.

<div class="notice--warning" markdown="1">

**Definition 3.8** (Orthogonal Matrix)

Square matrix $A \in \mathbb R^{n \times n}$가 orthogonal matrix임은 이의 column이 orthonomal함과 동치이다. 따라서,

$$
AA^\intercal = I = A^\intercal A \tag{3.29}
$$

가 성립하고, 이는 다음을 의미한다.

$$
A^{-1} = A^\intercal, \tag{3.30} 
$$

즉, inverse matrix를 단순히 transpose하는 것으로 얻을 수 있다.

</div>

Orthogonal matrix를 이용한 transformation이 특별한 이유는 $\boldsymbol x$의 길이가 변하지 않기 때문이다. Dot product에 대해 다음을 얻을 수 있다.

$$
\| A \boldsymbol x \| 
= (A \boldsymbol x)^\intercal(A \boldsymbol x) 
= \boldsymbol x^\intercal A^\intercal A \boldsymbol x 
= \boldsymbol x^\intercal I \boldsymbol x = \boldsymbol x^\intercal \boldsymbol x 
= \| \boldsymbol x \|^2. \tag{3.31}
$$

또한, inner product에 의해 측정된 두 벡터 $\boldsymbol x, \boldsymbol y$사이의 각도는 orthogonal matrix $A$를 통해 변환하더라도 변하지 않는다. Dot product를 inner product라 가정하면, image $A \boldsymbol x$와 $A \boldsymbol y$의 각도는 다음과 같이 주어진다.

$$
\cos{\omega} = \frac{(A \boldsymbol x)^\intercal(A \boldsymbol y)}{\|A \boldsymbol x \|\|A \boldsymbol y \|} 
= \frac{\boldsymbol x^\intercal A^\intercal A \boldsymbol y}{\sqrt{\boldsymbol x^\intercal A^\intercal A \boldsymbol x \boldsymbol y^\intercal A^\intercal A \boldsymbol y }}
= \frac{\langle \boldsymbol x, \boldsymbol y \rangle}{\| \boldsymbol x \|\| \boldsymbol y \|}   \tag{3.32}
$$

이 뜻은 orthogonal matrix $A$는 각도와 거리 모두 보존한다는 뜻이다. 이는 orthogonal matrix 회전 변환을 정의하기 때문이다 (혹은 flip). 이는 Section 3.9에서 다시 살펴본다.

## Orthonormal Basis

Section 2.6.1 Generating Set and Basis에서 basis vector는 n-dimensional vector space에서 n 개의 basis vector를 필요로 한다는 것을 발견했다. 이는 곧 n개의 vector가 linearly independent하다는 뜻이다. Section 3.3 (Lengths and Distances)와 Section 3.4 (Angles and Orthogonality)에서는 inner product를 활용하여 벡터의 길이와 각도를 계산하였다. 이번에는 basis vector가 서로 orthogonal하며 길이가 1인 특수한 경우를 살펴본다.

이를 좀 더 formal하게 써보자.

<div class="notice--warning" markdown="1">

**Definition 3.9** (Orthonormal Basis). n-dimensional vector space $V$와 basis $\{\boldsymbol b _1, \cdots, \boldsymbol b _n \}$를 고려해보자.

$$
\begin{align}
& \langle \boldsymbol b _i, \boldsymbol b _j \rangle = 0 \text{ for } i \neq j \tag{3.33} \\
& \langle \boldsymbol b _i, \boldsymbol b _i \rangle = 1 \tag{3.34} 
\end{align}
$$

만일 다음이 모든 $i, j$에 대해 성립하는 경우 **orthonomal basis (ONB)**라고 하며, 만일 (3.33)만 성립한다면, **orthogonal basis**라 부른다. (3.34)는 길이/norm이 1임을 내포한다.

</div>

Orthonomal basis의 컨셉은 Chapter 10, 12에서 SVM과 PCA를 살펴볼 때 다시 언급하도록 한다.

## Orthogonal Complement

정의한 orthogonality를 통해 서로 orthogonal한 vector space를 살펴보도록 하자. 이는 Chapter 10에서 차원 축소를 행할 때 기하학적인 관점을 제공해준다.

D-dimensional vector space $V$와 M-dimensional subspace $U \subseteq V$를 생각해보자. 그러면 이의 **orthogonal complement (직교여공간)** $U^\perp$는 $V$의 $(D-M)$-dimensional subspace가 되고, $V$내에 있는 모든 벡터는 $U$에 있는 모든 벡터와 orthogonal하다. 더불어 $U \cup U^\perp={\boldsymbol 0}$으로 어떠한 벡터 $\boldsymbol v \in V$는 다음으로 유일하게 분해된다.

$$
\boldsymbol x = \sum^M _{m=1} \lambda _m \boldsymbol b _m + \sum^{D-M} _{j=1} \psi _j \boldsymbol b^\perp _j, \lambda _m, \psi _j \in \mathbb R, \tag{3.36}
$$

여기서 $\{\boldsymbol b _1, \cdots, \boldsymbol b _M \}$은 $U$의 basis vector이며 $\{\boldsymbol b^\perp _1, \cdots, \boldsymbol b^\perp _{D_M} \}$은 $U^\perp$의 basis이다.

따라서 orthogonal complement는 3-dimensional vector space 내에 있는 plane $U$ (2-dimensional space)를 설명하는데 사용할 수 있다. 더욱 구체적으로, 길이가 1이고 plane $U$와 orthogonal한 $\boldsymbol w$는 $U^\perp$의 basis vector이다. 다음 Figure 3.7은 이러한 모습을 보여주고 있다.

![image](https://user-images.githubusercontent.com/47516855/114734744-78525080-9d7f-11eb-883e-1284fcb1e36b.png){: .align-center}{: width="700"}

일반적으로 orthogonal complement는 n 차원의 vector/affine space를 기술하는데 이용할 수 있다.

## Inner Product of Functions

지금까지 inner product의 성질을 이용하여 길이와 각도, 거리를 계산하였다. 우리는 유한 차원의 벡터공간에서의 내적만을 살펴보았는데, 이번엔 함수에 대한 내적을 살펴보자.  
지금까지 설명한 내적은 유한한 차원의 벡터에만 한정지었다. $\boldsymbol x \in \mathbb R$은 n개의 함수값을 갖는 함수로 생각할 수 있다. 내적의 개념은 무한 차원의 벡터 (countably infinite), continuous-valued function (uncountably infinite)에도 일반화할 수 있다. 그러면 vector의 개별 원소 (Equation 3.5 참고)의 덧셈은 적분으로 바뀌게 된다.  
두 함수의 내적 $u: \mathbb R \rightarrow \mathbb R$과 $v: \mathbb R \rightarrow \mathbb R$은 다음과 같은 definite integral (정적분)이 된다.

$$
\langle u, v \rangle := \int^b _a u(x)v(x)dx \tag{3.37}
$$

이를 통해 norm과 orthogonality를 정의할 수 있다. 식 (3.37)이 0이 되면 $u$와 $v$는 orthogonal하다.

Section 6.4.6에서 linear product의 unconventional한 타입인 확률변수의 내적을 알아보도록하자.

## Orthogonal Projections

linear transformation에서 회전과 반사와 함께 project은 중요한 개념이며, 그래픽, 코딩이론, 머신러닝에서 중요한 역할을 한다. 머신러닝에서 우리는 종종 높은 차원을 다루게 되는데, 이는 분석하거나 시각화하기 까다롭다. 그러나 높은 차원의 데이터는 일반적으로 일부 차원에 대부분의 정보가 몰려있고 대부분의 다른 차원은 데이터를 묘사하는데 핵심적인 요소가 아니다. 

높은 차원의 데이터를 압축하거나 시각화하면 정보를 손실하게 되는데, 이를 최소화히기 위한 이상적인 방법은 데이터 내에 가장 정보가 많은 차원을 찾아내는 것이다. 

구체적으로 설명하면 우리는 높은 차원의 데이터를 낮은 차원의 feature space로 project한 다음, 이 공간에서 작업하여 데이터셋에 대해 더 자세하게 배우도록 하고 연관된 패턴을 추출하도록 한다. 예를 들어 PCA, neural network (auto encoder)는 이러한 차원축소의 개념을 극대화한다. 이 장에서는 orthogonal projection을 살펴볼 것이며, 이는 Chapter 10의 inear dimensionality reduction과 12의 classification에서 활용된다. 심지어 linear regression (Chapter 9)에서도 이를 이용하여 해석할 수 있다. 주어진 낮은 차원의 subspace에 대해, 높은 차원의 데이터를 orthogonal projection은 가능한 최대로 많은 정보를 유지하며 투영된 차원과 원본 데이터의 차이를 최소화한다.

![image](https://user-images.githubusercontent.com/47516855/114744809-0bdc4f00-9d89-11eb-865a-6d546f3bff93.png){: .align-center}{: width="600"}

<div class="notice--warning" markdown="1">


**Definition 3.10** (Projection). 

$V$를 vector space, $U \in V$를 V의 subspace라 하자. Linear mapping $\pi: V \rightarrow U$가 있을 때, $\pi^2 = \pi \cdot \pi = \pi$라면 이를 **projection**이라 한다.

</div>

Linear mapping은 transformation matrix로 표현할 수 있기 때문에, 위의 정의는 특수한 transformation matrix인 **projection matrix** $P _\pi $에도 똑같이 적용된다. 이는 $P^2 _\pi = P _\pi $라는 특징을 갖는다.  
이제 내적공간 $(\mathbb R^n, \langle \cdot, \cdot \rangle)$내 벡터의 orthogonal projection을 subspace로 유도할 것이다. 1차원의 subspace인 선에서 시작하자. 특별히 언급한게 아닌이상, dot product를 inner product로 간주한다.

### Projection onto One-Dimensional Subspaces (Lines)

원점을 통과하고 Basis vector $\boldsymbol b \in \mathbb R^n$을 갖는 선분이 있다고 하자. 이 선분은 1차원의 subspace $U \in \mathbb R^n$으로 $\boldsymbol b$에 의해 span한다. $\boldsymbol x \in \mathbb R^n$을 $U$에 project하면, $\boldsymbol x$와 가까운 $\pi _U (\boldsymbol x) \in U$를 찾을 수 있다. 기하학적 논증 (geometric arguments)을 이용하여 projection $\pi _U (\boldsymbol x)$의 특징을 살펴보자 (Figure 3.10(a)).

![image](https://user-images.githubusercontent.com/47516855/114748419-e2252700-9d8c-11eb-888d-bbd57a4d0981.png){: .align-center}{: width="600"}

- Projection $\pi _U (\boldsymbol x)$는 $\boldsymbol x$와 가장 가깝다. 여기서 가장 가깝다는 것은 거리 $ \| \color{red}{\boldsymbol x - \pi _U (\boldsymbol x)} \|$가 가장 작다는 뜻을 내포한다. 따라서 $\color{red}{\pi _U (\boldsymbol x) - \boldsymbol x}$는 $U$와 orthogonal하고, basis vector와도 orthogonal하다. Orthogonality 조건에 의해 $\langle \color{red}{\pi _U (\boldsymbol x) - \boldsymbol x}, \color{orange}{\boldsymbol b} \rangle=0$이 되는데, 이 둘 사이의 각도는 내적을 통해 정의되기 때문이다.
- $\boldsymbol x$에서 $U$로의 projection $\pi _U (\boldsymbol x)$는 반드시 $U$의 element여야 한다. 그러므로, basis $\boldsymbol b$의 배수가 $U$를 span한다. 따라서 $\pi _U (\boldsymbol x) = \lambda \boldsymbol b$이다.

*Remark.* Chapter 4를 다 보게되면 $\pi _U (\boldsymbol x)$가 $P _\pi$의 eigenvector가 됨을 알 수 있고, 이에 해당하는 eigenvalue는 1이 되는 것을 확인할 수 있다.
{: .notice}

### Projection onto General Subspaces

앞으로 $\boldsymbol x \in \mathbb R^n$의 $\text{dim}(U)=m \geq 1$인 lower dimensional subspace $U \in \mathbb R^n$로의 projection을 살펴볼 것이다. 이는 Figure 3.11에 나타나있다.

![image](https://user-images.githubusercontent.com/47516855/114751659-6331ed80-9d90-11eb-81c5-fdfd33c947f5.png){: .align-center}{: width="600"}

$(\boldsymbol b _1, \cdots, \boldsymbol b _m)$을 $U$의 ordered basis라 하자. $U$로의 어떤 projection $\pi _U (\boldsymbol x)$는 반드시 $U$의 element여야 한다. 따라서 이는 basis의 linear combination으로 표현할 수 있다.  
Projection $\pi _U (\boldsymbol x)$와 projection matrix $P _\pi$를 찾기 위해 3단계를 거친다.
1. Projectiond의 coordinate $\lambda _1, \cdots, \lambda _2$ 찾기

    우선 $\pi _U (\boldsymbol x)$를 다음과 같이 표현한다.

    $$
    \begin{align}
    & \pi _U (\boldsymbol x) = \sum^m _{i=1}\lambda _i \boldsymbol b _i = B \Lambda \\
    & B = [b _1, \cdots, b _m] \in \mathbb R^{n \times m}, \Lambda = [\lambda _1, \cdots, \lambda _m]^\intercal \in \mathbb R^m
    \end{align}
    $$

    앞서 봤던 예와 같이 가장 가까운 space를 찾아야 한다. $\boldsymbol x - \pi _U (\boldsymbol x)$는 $U$의 basis에 orthogonal하므로, m개의 simultaneous condition을 얻는다.

    Dot product를 inner product로 가정하면, 

    $$
    \begin{align} 
    \langle b _1, \boldsymbol x - \pi _U (\boldsymbol x) \rangle &= b^\intercal _1 (\boldsymbol x - \pi _U (\boldsymbol x)) = 0 \tag{3.51} \\
    &\vdots \\
    \langle b _m, \boldsymbol x - \pi _U (\boldsymbol x) \rangle &= b^\intercal _m (\boldsymbol x - \pi _U (\boldsymbol x)) = 0 \tag{3.52}
    \end{align}
    $$

    이고, $\pi _U (\boldsymbol x)=B \Lambda$를 대입하면,

    $$
    \begin{align} 
    b^\intercal _1 (\boldsymbol x &- B \boldsymbol \lambda) = 0 \tag{3.53} \\
    &\vdots \\
    b^\intercal _m (\boldsymbol x &- B \boldsymbol \lambda) = 0 \tag{3.54}
    \end{align}
    $$

    이 된다. 이를 정리하면 다음과 같은 homogeneous linear equation system을 얻을 수 있다.

    $$
    \begin{align} 
    \begin{bmatrix}
    b^\intercal _1 \\ \cdots \\ b^\intercal _m
    \end{bmatrix}
    \begin{bmatrix}
    \boldsymbol x &- B \boldsymbol \lambda
    \end{bmatrix} 
    = 0 & \iff B^\intercal (\boldsymbol x - B \boldsymbol \lambda) = 0 \tag{3.55} \\
    & \iff B^\intercal B \boldsymbol \lambda = B^\intercal \boldsymbol x \tag{3.56}
    \end{align}
    $$

    위의 마지막 식은 **normal equation**이다. $B$의 column vector는 $U$의 basis이기 때문에, linear independent하므로, $B^\intercal B \in \mathbb R^{m \times m}$은 regular하며, invertible하다. 따라서 위 식으로부터 아래와 같이 $\Lambda$를 구할 수 있다.

    $$
    \boldsymbol \lambda = (B^\intercal B)^{-1}B^\intercal \boldsymbol x \tag{3.57}
    $$

    $(B^\intercal B)^{-1}B^\intercal$는 $B$의 **pseudo-inverse**라고 부른다. $(B^\intercal B)$가 positive definite, 즉, full-rank를 갖으면 pseudo-inverse를 사용할 수 있다.

2. Projection $\pi _U (\boldsymbol x) \in U$ 찾기

    $\pi _U (\boldsymbol x) = B \Lambda$임을 알고 있으므로, 앞서 구한 $\Lambda = (B^\intercal B)^{-1}B^\intercal \boldsymbol x$를 대입하여 $\pi _U (\boldsymbol x)$를 구할 수 있다. 

    $$
    \pi _U (\boldsymbol x) = B \Lambda = B(B^\intercal B)^{-1}B^\intercal \boldsymbol x \tag{3.58}
    $$

3. Projection matrix $P _\pi$ 구하기

    $P _\pi \boldsymbol x = \pi _U (\boldsymbol x)$이기 때문에, 바로 구할 수 있다.

    $$
    P _\pi = B(B^\intercal B)^{-1}B^\intercal \tag{3.59}
    $$

Projection은 $A\boldsymbol x = \boldsymbol b$가 해가 없는 linear system을 다룰 수 있게한다. 이는 $\boldsymbol b$가 $A$의 span에 없다는 뜻이다. 해를 찾을 수 없는 상황에서, **approximate solution**을 찾을 수 있다. 즉, $A$에 $\boldsymbol b$가 없지만, 이에 최대한 가까운 해를 찾는다는 것이다. 이에 대한 해를 **least square**라고 부른다. 이는 Section 9.4 (Maximum Likelihood as Orthogonal Projection)에서 다시 다룰 것이다.