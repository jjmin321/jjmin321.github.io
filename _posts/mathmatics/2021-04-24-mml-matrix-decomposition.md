---
title:  "머신러닝을 위한 수학 정리: Matrix Decomposition"
toc: true
toc_sticky: true
permalink: /project/mml/Matrix-Decomposition/
categories:
  - Mathmatics
  - Machine Learning
tags:
  - linear algebra
  - vector
  - matrix
use_math: true
last_modified_at: 2021-05-20
---

> 본 포스트는 머신러닝에 필요한 선형대수 및 확률과 같은 수학적 개념을 정리한 포스트이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.


## Determinant and Trace

Determinant (행렬식)은 수학적 객체로, 해석과 선형 시스템의 해에서 사용된다. 이는 square matrix에서만 정의된다. 본 책에서는 $\text{det}(\boldsymbol A)$ 혹은 $\rvert \boldsymbol A \lvert$로 쓴다. $\boldsymbol A$의 **determinant**는 어떤 함수로,  $\boldsymbol A$를 어떤 실수로 mapping한다.

**Theorem 4.1.**  
어떠한 square matrix $\boldsymbol A \in \mathbb R^{n \times n}$든지 $\boldsymbol A$가 invertible하다는 뜻은 $\text{det}(\boldsymbol A) \neq 0$임과 동치이다.
{: .notice--info}

만일 어떤 square matrix $\boldsymbol T$가 $T _{i, j} = 0$ for $ i > j $이면 **upper-triangular matrix**라 부른다 (즉 대각행렬 아래로는 모두 0). 이 반대는 **lower-triangular matrix**라 부른다. triangular matrix $\boldsymbol T \in \mathbb R^{n \times n}$에 대해 행렬식은 대각성분의 곱과 같다.

$$
\text{det}(\boldsymbol T) = \prod^n _{i=1} T _{ii} \tag{4.8}
$$

또한 행렬식의 개념은 이를 $\mathbb R^n$의 어떤 객체를 spanning하는 n개의 벡터 집합으로부터의 mapping으로 생각하는게 자연스럽다. 곧 밝혀지겠지만 $\text{det}(\boldsymbol A)$는 행렬 $\boldsymbol A$의 columns가 형성하는 n차원의 평행육면체(parallelepiped)의 부호가 있는 부피 (signed volumn)이다.

![image](https://user-images.githubusercontent.com/47516855/115955691-14741880-a533-11eb-8933-2a2da5607850.png){: .align-center}{:width="300"}

만일 두 벡터가 이루는 각도가 작아진다면, 이에 따라 평행육면체 (이 경우 평행사변형)의 넓이는 줄어든다.

> 행렬식이 음수를 갖는 것과 행렬식이 부피를 나타낸다는 것이 동시에 이해가 되지 않을 것이다. 이는 orientation과 연관이 있다. 행렬식이 음수를 갖는 것은 plane이 뒤집히는 것으로 이해할 수 있다. 즉, 부피는 행렬식의 절댓값만큼 증가하고, xy plane이 yx plane으로 뒤집힌다.
>
> ![image](https://user-images.githubusercontent.com/47516855/116278946-ef490980-a7c1-11eb-9102-81be4b1abf25.png){: .align-center}{:width="500"}

$\boldsymbol A \in \mathbb R^{n \times n}$에 대해 행렬식은 다음의 성질을 만족한다.
- $\text{det}(\boldsymbol A \boldsymbol B) = \text{det}(\boldsymbol A) \text{det}(\boldsymbol B)$
- $\text{det}(\boldsymbol A) = \text{det}(\boldsymbol A ^\intercal)$
- $\boldsymbol A$가 regular(invertible)하면, $\text{det}(\boldsymbol A^{-1}) = \frac{1}{\text{det}(\boldsymbol A)}$
- 두 행렬이 닮음(similarity)이라면, 행렬식도 같다. 따라서, linear mapping
$\Phi: V \to V $에 대해 모든 transformation matrix $\boldsymbol A _{\Phi}$는 같은 행렬식을 갖는다. 그러므로 행렬식은 linear mapping의 basis에 invariant하다.
  - Recall: $\boldsymbol{\tilde A} = S^{-1}\boldsymbol{A}S$인 regular matrix $S \in \mathbb R^{n \times n}$가 존재하면, 두 matrix $\boldsymbol{A}, \boldsymbol{\tilde A} \in \mathbb R^{n \times n}$은 서로 **similar**하다고 한다.
- 행/열을 여러개 추가하여도 행렬식은 변하지 않는다.
- $\text{det}(\lambda \boldsymbol A) = \lambda^n \text{det}(\boldsymbol A)$
- 두 개의 행/열을 바꾸면 행렬식의 부호가 바뀐다.

마지막 3개의 성질로 인해, 가우스 소거법을 사용하여 행렬식을 구할 수 있다.

**Theorem 4.3.**  
정방행렬 $\boldsymbol A \in \mathbb R^{n \times n}$의 행렬식이 0이 아닌 것은 $\text{rk}(\boldsymbol A)=n$임과 동치이다. 즉, $\boldsymbol A$의 역행렬이 존재하는 것과 full rank임은 동치이다 (iff).
{: .notice--info}

행렬식은 characteristic polynomial(특성방정식)을 통해 eigenvalue(고유값)와 고유벡터(eigenvector)를 배울 때 유용하다.

<div class="notice--warning" markdown="1">

**Definition 4.4.**

정사각 행렬 $\boldsymbol A \in \mathbb R^{n \times n}$의 **Trace(대각합)**은 다음과 같이 정의된다.

$$
\text{tr}(\boldsymbol A) := \sum^n _{i=1} a _{ii} \tag{4.18}
$$

</div>

trace는 다음과 같은 성질을 만족한다.
- $\text{tr}(\boldsymbol A \boldsymbol B) = \text{tr}(\boldsymbol A) + \text{tr}(\boldsymbol B)$ for $\boldsymbol A, \boldsymbol B \in \mathbb R^{n \times n}$
- $\text{tr}(\alpha \boldsymbol A) = \alpha \text{tr}(\boldsymbol A), \alpha \in \mathbb R$ for $\boldsymbol A \in \mathbb R^{n \times n}$
- $\text{tr}(\boldsymbol I _n) = n$
- $\text{tr}(\boldsymbol A \boldsymbol B) = \text{tr}(\boldsymbol B \boldsymbol A)$ for $\boldsymbol A \in \mathbb R^{n \times k}, \boldsymbol B \in \mathbb R^{k \times n}$
- $\text{tr}(\boldsymbol K \boldsymbol L \boldsymbol A) = \text{tr}(\boldsymbol A \boldsymbol K \boldsymbol L)$ for $\boldsymbol A \in \mathbb R^{a \times k}, \boldsymbol K \in \mathbb R^{k \times l}, \boldsymbol L \in \mathbb R^{l \times a}$
- $ \text{tr}(\boldsymbol x \boldsymbol y^\intercal) = \text{tr}(\boldsymbol y^\intercal \boldsymbol x) = \boldsymbol y^\intercal \boldsymbol x \in \mathbb R$

선형사상 $\Phi: V \to V $에 대해, 이 사상에 대한 대각합을 이의 행렬 표현의 대각합을 이용하여 정의할 수 있다. 주어진 $V$의 기저에 대해, $\Phi$를 transformation matrix $\boldsymbol A$를 통해 설명할 수 있다. 이러면 $\Phi$의 대각합은 $\boldsymbol A$의 대각합이 된다. 또 다른 기저에 대해, 적절한 $\boldsymbol S$의 basis change $\boldsymbol S^{-1} \boldsymbol A \boldsymbol S$의 형태를 통해 $\Phi$의 transformation matrix $\boldsymbol B$를 구할 수 있다 ([Basis Change](/project/mml/Linear-Algebra/#basis-change) 참고). 이는 즉 다음이 성립한다는 의미이다.

$$
\text{tr}(\boldsymbol B) = \text{tr}(\boldsymbol S^{-1} \boldsymbol A \boldsymbol S)  \stackrel{4.19}{=} \text{tr}(\boldsymbol A \boldsymbol S \boldsymbol S^{-1}) = \text{tr}(\boldsymbol A) \tag{4.21}
$$

이를 통해 선형사상의 행렬표현은 기저에 의존적인 반면, 선형사상 $\Phi$의 대각합은 기저에 독립적이다.

여기서 행렬식과 대각합을 정사각 행렬을 기술하는 함수로서 생각할 것이다. 대각합과 행렬식에 대한 이해를 종합하여 이제 다항식 측면에서 행렬 $\boldsymbol A$를 설명하는 중요한 방정식을 정의할 수 있다.

<div class="notice--warning" markdown="1">

**Definition 4.5 (Characteristic Polynomial (특성다항식))**

$\lambda \in \mathbb R$과 정사각 행렬 $\boldsymbol A \in \mathbb R^{n \times n}$에 대해 다음을 $\boldsymbol A$의 **Characteristic Polynomial**이라 한다.

$$
\begin{align}
p _{\boldsymbol A} & := \text{det}(\boldsymbol A - \lambda \boldsymbol I) \tag{4.22a} \\
&= c _0 + c _1 \lambda + c _2 \lambda^2 + \cdots + c _{n-1} \lambda^{n-1} +  -1^n \lambda^n \tag{4.22b}
\end{align}
$$

$c _0 + \cdots + c _{n-1} \in \mathbb R$이다. 위의 값들은 아래와 같이 정의된다.

$$
\begin{align}
c _0 = \text{det}(\boldsymbol A), \tag{4.23} \\
c _{n-1} = -1^{n-1} \text{tr}(\boldsymbol A). \tag{4.24}
\end{align}
$$

</div>

Characteristic Polynomial (4.22a)을 통해 eigenvalue와 eigenvector를 계산할 수 있게된다.

> 즉, 특성다항식의 해는 eigenvalue이다.

## Eigenvalues and Eigenvectors

[Matrix Representation of Linear Mappings](/project/mml/Linear-Algebra/#matrix-representation-of-linear-mappings)에서 모든 선형사상은 ordered basis에 대해 고유한 transformation matrix가 있다. "Eigen" 분석을 수행하여 선형사상과 이와 관련된 transformation matrix를 해석할 수 있다. 추후 알아보겠지만, 선형 사상의 eigenvalue가 특별한 벡터의 집합이 (eigenvector) 선형사상을 통해 어떻게 변환되는지를 알려준다.

<div class="notice--warning" markdown="1">

**Definition 4.6.**

$\boldsymbol A \in \mathbb R^{n \times n}$를 정사각 행렬이라 하자. 아래의 식이 성립할 때, 

$$
\boldsymbol A \boldsymbol x = \lambda \boldsymbol x \tag{4.25}
$$

$\lambda \in \mathbb R$을 $\boldsymbol A$의 **eigenvalue**, $\boldsymbol x \in \mathbb R^n \setminus \{0\}$을 이에 대응하는  $\boldsymbol A$의 **eigenvector**라 한다. (4.25)는 **eigenvalue equation**이라고 부른다.

</div>

다음의 명제들은 동치이다.
- $\lambda$는 $\boldsymbol A \in \mathbb R^{n \times n}$의 eigenvalue이다.
- $\boldsymbol A \boldsymbol x = \lambda \boldsymbol x$ 혹은 $(\boldsymbol A - \lambda \boldsymbol I)\boldsymbol x=0$이 non-trivial solution을 갖으면, $x \in \mathbb R^n \setminus \{ 0 \}$가 존재한다.
- $\text{rk}(\boldsymbol A - \lambda \boldsymbol I) < n $
- $\text{det}(\boldsymbol A - \lambda \boldsymbol I) = 0$

> Eigenvalue와 eigenvector가 의미하는 것은 결국 어떤 벡터($\boldsymbol x$)를 선형변환($\boldsymbol A$)했을 때, 크기만 변할 뿐 원래벡터($\boldsymbol x$)와 평행한 벡터가 존재하느냐이다. 여기서 eigenvalue는 변한 크기(scaling factor)가 된다.

> 식 (4.25)를 다시쓰면 $(\boldsymbol A -  \lambda) \boldsymbol x = \boldsymbol 0$가 된다. 이를 성립하기 위해서는 non-trivial solution을 갖아야 하고, 즉, 이의 행렬식은 0이 되어야 유의미한 $\lambda$를 찾을 수 있다 (행렬식이 0이 아닐 경우 어떠한 $\lambda$에 대해서도 $\boldsymbol x=0$에 대해서 위가 성립하게 된다.)

<div class="notice--warning" markdown="1">

**Definition 4.7(Collinearity and Codirection).**

서로 같은 방향을 가르키고 있는두 벡터는 **codirect**됐다고 한다. 두 벡터가 서로 같은/반대 방향을 가리키고 있다면, 이를 두 벡터가 **collinear**하다고 말한다.

</div>

<div class="notice" markdown="1">

*Remark (Non-uniqueness  of  eigenvectors)*. 

만일 $\boldsymbol A$의 eigenvector가 $\boldsymbol x$라면, 어떠한 $c \in \mathbb R \setminus \{0\}$에 대해 $c \boldsymbol x$도 똑같은 eigenvalue을 갖는 eigenvector이다. 이는 다음에 의해 만족한다.

$$
\boldsymbol A (c \boldsymbol x) = c \boldsymbol A (\boldsymbol x) = c \lambda \boldsymbol x = \lambda c \boldsymbol x \tag{4.26}
$$

</div>

따라서, $\boldsymbol x$ collinear인 모든 벡터 또한 $\boldsymbol A$의 eigenvector가 된다.

**Theorem 4.8.** $\lambda \in \mathbb R$가 $\boldsymbol A \in \mathbb R^{n \times n}$의 eigenvalue임과, $\lambda$가 $\boldsymbol A$의 특성다항식 $p _{\boldsymbol A}$의 root인 것은 동치이다.
{: .notice--info}

<div class="notice--warning" markdown="1">

**Definition 4.9.** 

정사각 행렬 $\boldsymbol A$가 eigenvalue $\lambda _i$를 갖는다 하자. $\lambda _i$의 **Algebraic multiplicity(기하적 중복도)**는 특성다항식에서 근의 갯수이다.

</div>

> 행렬의 고윳값은 특성방정식의 절대값을 0으로 만드는 $\lambda$로 정의된다. 특성방정식은 $\lambda$에 대한 m차 방정식으로 나타낼 수 있다. 
>
> $$
> \text{det}(\boldsymbol A -  \lambda \boldsymbol I)= c _0 + c _1 \lambda + c _2 \lambda^2 + \cdots + c _{n-1} \lambda^{n-1} +  -1^n \lambda^n
> $$ 
>
> 대수학의 기본정리에 의해, 특성방정식은 복소수를 포함하여 m개의 근을 갖는다. 여기서 근이 중근일 경우, 고윳값은 중복을 포함하여 구해지게 된다. 앞서 식을 다시 중근을 포함하여 나타내게 되면,
> 
> $$
> \begin{align}
> \text{det}(\boldsymbol A -  \lambda \boldsymbol I)= c (\lambda - \lambda _1)^{a _1} (\lambda - \lambda _2)^{a _2} \cdots (\lambda - \lambda _k)^{a _k} \\
> k \geq n
> \sum^k _{i=1} a _i = n
> \end{align}
> $$
>
> 위와 같이 표현했을 때 행렬 $\boldsymbol A$는 서로 다른 $k$개의 고윳값을 가지며, $\lambda _i$는 $a _i$개만큼 중복된다. 이를 고윳값 $\lambda _i$가 대수적 중복도 $a _i$를 갖는다고 정의한다.
>
> [출처: 고유값의 대수적 중복도와 기하적 중복도](https://freshrimpsushi.github.io/posts/multiplicity-of-eigen-value/)

<div class="notice--warning" markdown="1">

**Definition 4.10 (Eigenspace and Eigenspectrum).**

$\boldsymbol A \in \mathbb R^{n \times n}$에 대해, eigenvalue $\lambda$와 연관된 모든 eigenvector의 집합은 n차원의 subspace를 span한다. 이는 eigenvalue $\lambda$에 대한 $\boldsymbol A$의 **eigenspace(고유공간)**이라고 하고 $E _{\lambda}$로 표현한다. $\boldsymbol A$의 고윳값 집합은 **eigenspectrum(고유스펙트럼)** 혹은 그냥 스펙트럼이라고 한다.

</div>

$\lambda$가 $\boldsymbol A \in \mathbb R^{n \times n}$의 고윳값이면, 이에 대응하는 고유공간 $E _{\lambda}$는 동차시스템 $(\boldsymbol A - \lambda \boldsymbol I) \boldsymbol x = \boldsymbol 0 $의 해공간이 된다. 기하학적으로 이는 선형사상 non-zero eigenvalue에 의해 늘려지는(stretched) 방향을 가르키게 된다. 고윳값은 이 늘려짐의 정도를 결정하게 된다. 고윳값이 음수라면, 늘려지는 방향은 뒤집히게 된다.

> 이는 아래 그림을 확인하면 빠르다.
>
> ![image](https://user-images.githubusercontent.com/47516855/118388890-0319ba00-b662-11eb-82b4-0675843305a2.png){: .align-center}{: width="300"}

고윳값과 고유벡터에 대한 유용한 특성을 살펴보자.
- 행렬 $\boldsymbol A$과 이의 transpose는 같은 eigenvalue를 갖지만, 꼭 같은 eigenvector를 갖을 필요는 없다.
- Eigenspace $E _\lambda$는 $\boldsymbol A - \lambda \boldsymbol I$의 null space이다.

  $$
  \begin{align}
  \boldsymbol A \boldsymbol x= \lambda \boldsymbol x & \iff \boldsymbol A \boldsymbol x - \lambda \boldsymbol x = \boldsymbol 0 \tag{4.27a} \\
  & \iff (\boldsymbol A - \lambda \boldsymbol I) \boldsymbol x \iff \boldsymbol x \in \text{ker}(\boldsymbol A - \lambda \boldsymbol I) \tag{4.27b}
  \end{align}
  $$

- Similar matrix는 같은 고윳값을 같는다. 따라서 선형 변환 $\Phi$는 이의 transformation matrix의 기저에 상관없이 eigenvalue를 갖는다. 이러한 특성은 eigenvalue를 determinant, trace와 함께 linear mapping의 특성을 나타내는 key parameter로 만들어준다. (basis change에 invariant)
- symmetric, positive definite matrix는 항상 양의 실수인 eigenvalue를 갖는다.

> 마지막 성질을 통해 우리는 symmetric, positive definite matrix의 새로운 판별법을 알 수 있다. 이는 또한 다음과 동치이다.
>
> $\boldsymbol Q^{\intercal} = \boldsymbol Q$이면, SPD $\boldsymbol A$에 대해 $\boldsymbol S = \boldsymbol Q \boldsymbol A \boldsymbol Q^{\intercal}$도 SPD이다. 
> 
> 이는 이 둘은 닮음이기 때문에 성립한다.

<div class="notice--warning" markdown="1">

**Definition 4.11.**

$\lambda _i$를 정사각 행렬 $\boldsymbol A$의 eigenvalue라 하자. 그러면 eigenvalue의 **geometric multiplicity(기하적 중복도)**는 eigenvalue에 대응하는 linearly independent eigenector의 갯수가 된다. 즉, 고윳값과 대응하는 eigenvector가 span하는 eigenspace의 차원과도 같다.

</div>

*Remark.* 특정한 고윳값의 geometric multiplicity는 반드시 1이상이 된다. 이는 모든 고윳값에는 최소 하나의 eigenvector가 대응하기 때문이다. 또한, algebraic muliplicity를 초과할 수 없고 이보다 낮다.
{: .notice}

> 행렬 $\boldsymbol A$ 의 고유값 $\lambda _i$ 에 대해 $\boldsymbol x _1, \boldsymbol x _2 \in \mathbb C^m$ 이 행렬방정식 $\boldsymbol A \boldsymbol x= \lambda _i \boldsymbol x$ 의 해가 된다고 두자. 그러면 두 벡터 $\boldsymbol x _1, \boldsymbol x _2$ 는 같은 고유값 $\lambda _i$에 해당하는 고유벡터가 될 것이다. 물론 한 고유값에 대해서 고유벡터는 무한히 존재하긴 한다. 기하학적으로 설명하자면 고유벡터 $\boldsymbol x$ 의 크기를 늘이고 줄인 $\alpha \boldsymbol x$가 존재하기 때문이다.
>
> 만일 $\boldsymbol x _1, \boldsymbol x _2$가 수직이라면, 같은 고윳값을 공유하지만 선형독립이기 때문에 서로를 표현할 수가 없다. 다음과 같은 집합을 특정 고윳값 $\lambda _i$에 대응하는 고유벡터의 집합이라 하자. 이의 차원을 구하게 되면 이 차원은 고윳값을 공유하되 서로 수직이 되는 고유벡터의 종류가 된다. 이를 기하적 중복도라 표현한다.
>
> $$
> S _{\lambda _i} = \{\boldsymbol A \boldsymbol x = \lambda _i \boldsymbol x \}
> $$
>
> 
> [출처: 고유값의 대수적 중복도와 기하적 중복도](https://freshrimpsushi.github.io/posts/multiplicity-of-eigen-value/)

### *Graphical Intuition in Two dimensions*

서로다른 선형변환을 이용하는 행렬식, 고윳값, 고유벡터에 대한 직관적 이해를 해보자. 아래 그림들은 5가지 변환행렬 $\boldsymbol A _1, \cdots, \boldsymbol A _5$와 원점을 중심으로 하는 사각형 모양의 데이터에 대한 영향을 표현한 것이다.

$\boldsymbol{A _1} = \begin{bmatrix} \frac{1}{2} & 0 \\\\ 0 & 2 \end{bmatrix}$. 두 고유벡터의 방향은 canonical basis와 대응한다. 수직축은 $\lambda _1 =2$에 의해 확장되고, 수평축은 $\lambda _2 = 1/2$에 의해 축소된다. 이 변환은 공간을 그대로 유지한다 ($\text{det}(\boldsymbol A _1) = 1 = 2 \cdot 1/2$)

![image](https://user-images.githubusercontent.com/47516855/118501137-62abbe80-b763-11eb-8cd8-06346202f708.png){: .align-center}{: width="500"}

$\boldsymbol{A _2} = \begin{bmatrix} 1 & \frac{1}{2} \\\\ 0 & 1 \end{bmatrix}$는 shearing mapping에 해당한다. 이 변환은 말 그대로 1, 2 사분면은 오른쪽으로 늘리고, 3, 4분면은 반대쪽으로 늘린다. 이 변환 역시 공간을 유지한다. 고유값은 중근으로 1이 된다. 따라서 고유벡터는 collinear하다. 두 고유벡터의 방향은 canonical basis와 대응한다 (아래 그림은 서로 다른 방향을 강조하기 위해 다음과 같이 고유벡터를 표현하였다). colinear가 의미하는 것은 이 변환이 한 방향으로만 작용한다는 것이다.

![image](https://user-images.githubusercontent.com/47516855/118508181-e9fc3080-b769-11eb-85e0-95aa10fb1979.png){: .align-center}{: width="500"}

$\boldsymbol{A _3} = \begin{bmatrix} cos(\pi/6) & -sin(\pi/6) \\\\ sin(\pi/6) & cos(\pi/6) \end{bmatrix}$는 점을 반시계 방향으로 30도 회전시키는 rotation matrix이다. 이는 복수수의 고윳값을 갖으며, 이는 이 변환이 단순한 회전이라는 것을 의미한다 (따라서 고유벡터가 없다). 회전변환이기 때문에 넓이를 유지한다.

![image](https://user-images.githubusercontent.com/47516855/118508699-6262f180-b76a-11eb-9849-45b015e03e0b.png){: .align-center}{: width="500"}

$\boldsymbol{A _4} = \begin{bmatrix} 1 & -1 \\\\ -1 & 1 \end{bmatrix}$는 2차원 1차원으로 붕괴시키는 변환이다. 하나의 고윳값이 0이므로 이에 대응하는 고유벡터도 0이된다. 이에 직각인 고유벡터는 $\lambda _2 = 2$만큼 차원을 늘리게된다. 변환 후의 공간은 이 고유벡터가 span하게 되고, 넓이는 0이된다.

![image](https://user-images.githubusercontent.com/47516855/118512386-b7543700-b76d-11eb-9d12-b3b8feab05d0.png){: .align-center}{: width="500"}

$\boldsymbol{A _5} = \begin{bmatrix} 1 & 1/2 \\\\ 1/2 & 1 \end{bmatrix}$는 늘림과 축소를 동시에 진행한다. 행렬식은 $3/4$이기 때문에 넓이가 축소되고, 빨간색 고유벡터 방향으로 $\lambda _2$인 2만큼 늘리고, 이에 직각인 파랑방향으로 0.5만큼 줄인다.

![image](https://user-images.githubusercontent.com/47516855/118514507-8ffe6980-b76f-11eb-91f9-4a5661edbe24.png){: .align-center}{: width="500"}


**Theorem 4.12.** $\boldsymbol A \in \mathbb R^{n \times n}$의 서로 다른 n개의 고유벡터 $\boldsymbol x _1, \cdots, \boldsymbol x _n$는 서로 linearly independent하다.
{: .notice--info}

> 대부분은 $n$개의 선형독립인 고유벡터를 갖는다. 이때는 고윳값의 중복이 없다. 반대로 고윳값이 중복되는 경우가 있다. 그렇다고 이 둘이 반드시 linearly dependent하다는 뜻은 아니다.

본 정리가 의미하는 것은 어떤 행렬의 서로다른 n개의 고유벡터는 $\mathbb R^n$차원을 형성한다는 것이다.

<div class="notice--warning" markdown="1">

**Definition 4.13.** 

정사각 행렬 $\boldsymbol A \in \mathbb R^{n \times n}$이 자신의 차원 $n$보다 작은 linearly independent eigenvector를 갖는다면, 이를 **defective**하다고 말한다.

</div>

non-defective matrix $\boldsymbol A \in \mathbb R^{n \times n}$는 반드시 n개의 다른 eigenvalue를 갖을 필요는 없지만, 이의 eigenvector들이 $\mathbb R^n$의 basis를 형성해야 한다. defective matrix의 eigenspace를 살펴보면, eigenspace의 차원의 합이 n보다 작게 된다. 특히, 최소 하나의 대수적 중복도가 $m > 1$이고 기하적 중복도가 $m$보다 작은 eigenvalue $\lambda _i$를 갖는다.

*Remark*. Defective matrix는 n개의 서로 다른 고윳값을 갖을 수 없다. 서로 다른 eigenvalue는 linearly independent eigenvector를 갖기 때문이다 (Theorem 4.12)
{: .notice}

<div class="notice--info" markdown="1">

**Theorem 4.14.**

$\boldsymbol A \in \mathbb R^{m \times n}$에 대해 항상 sysmetric, positive semidefinite matrix $\boldsymbol S \in \mathbb R^{n \times n}$를 다음과 같이 얻을 수 있다.

$$
\boldsymbol S := \boldsymbol A^{\intercal} \boldsymbol A \tag{4.36}
$$

</div>

*Remark*. $\text{rk}(\boldsymbol A) = n$이 성립하면, $\boldsymbol S := \boldsymbol A^{\intercal} \boldsymbol A$는 symmetric positive definite하다.
{: .notice}

정리 4.14가 어째서 성립하는지 이해하는 것은 우리가 어떻게 symmetrized matrix를 이용할 수 있는지에 대한 통찰력이 필요하다. 대칭이라는 것은 $\boldsymbol S^{\intercal} = \boldsymbol S$가 만족한다는 뜻이고, 식 (4.36)를 통해 $\boldsymbol S = \boldsymbol A^{\intercal} \boldsymbol A = \boldsymbol A^{\intercal} (\boldsymbol A^{\intercal})^{\intercal} = (\boldsymbol A^{\intercal} \boldsymbol A)^{\intercal} = \boldsymbol S^{\intercal}$를 얻을 수 있다. 또한, positive semidefiniteness가 만족하려면 $\boldsymbol x^\intercal \boldsymbol A \boldsymbol x \geq 0$을 만족해야한다. 이를 (4.36)과 연결시키면, $\boldsymbol x^\intercal \boldsymbol S \boldsymbol x = \boldsymbol x^\intercal \boldsymbol A^{\intercal} \boldsymbol A \boldsymbol x = (\boldsymbol x^\intercal \boldsymbol A^{\intercal}) (\boldsymbol A \boldsymbol x)=(\boldsymbol A \boldsymbol x)^\intercal (\boldsymbol A \boldsymbol x) \geq 0$를 얻을 수 있다. 이는 dot product가 sum of square를 계산하기 때문에 non-negative가 되기 때문이다.


<div class="notice--info" markdown="1">

**Theorem 4.15** (Spectral Theorem).

$\boldsymbol A$가 symmetric하면 $\boldsymbol A$의 eigenvector가 이루는 vector space $V$는 orthonormal basis를 갖고, eigenvalue는 실숫값을 갖는다.

</div>

Spectral theorem이 의미하는 것은 symmetric matrix의 eigendecomposition이 존재하고 (eigenvalue는 실수), $\boldsymbol A = \boldsymbol P \boldsymbol D \boldsymbol P^{\intercal} $이 되는 eigenvector의 ONB를 찾을 수 있다는 것이다. 여기서 $\boldsymbol D$는 diagonal matrix이고, $\boldsymbol P$의 column은 eigenvector를 포함한다.

교윳값과 고유벡터에 관한 내용을 마무리하기전에, 행렬식과 대각합에 대한 개념을 통해 이러한 matrix characteristic을 하나로 연결하고자 한다.

<div class="notice--info" markdown="1">

**Theorem 4.16**

$\boldsymbol A \in \mathbb R^{n \times n}$의 행렬식은 이의 고윳값의 곱과도 같다. 즉,

$$
\text{det}(\boldsymbol A) = \prod^n _{i=1} \lambda _i \tag{4.42}
$$

$\lambda _i \in \mathbb C$인 eigenvalue이다.

</div>

<div class="notice--info" markdown="1">

**Theorem 4.17**

$\boldsymbol A \in \mathbb R^{n \times n}$의 대각합은 이의 고윳값의 합과도 같다. 즉,

$$
\text{det}(\boldsymbol A) = \sum^n _{i=1} \lambda _i \tag{4.43}
$$

$\lambda _i \in \mathbb C$인 eigenvalue이다.

</div>

이 두 정리에 대해 기하학적인 직관을 더해보자. 어떤 행렬 $\boldsymbol A \in \mathbb R^{2 \times 2}$가 두 개의 linearly independent eigenvector $\boldsymbol x _1, \boldsymbol x _2$를 갖는다고 가정하자. $(\boldsymbol x _1, \boldsymbol x _2)$는 2차원의 ONB라고 가정하여 서로 orthogonal하며 이의 넓이는 1이 된다.

![image](https://user-images.githubusercontent.com/47516855/118786594-ee8d2a00-b8cc-11eb-9edd-62244981c938.png){: .align-center}{: width="600"}

앞서 행렬식은 transformation $\boldsymbol A$에 의해 변화하는 넓이라고 언급하였다. 이 예제에서 이 변화하는 넓이를 명시적으로 구할 수 있다.

$\boldsymbol A$를 이용하여 eigenvector를 맵핑하면 다음과 같은 벡터를 얻게 된다: $\boldsymbol v _1 = \boldsymbol A \boldsymbol x _1 = \lambda \boldsymbol x _1 $, $\boldsymbol v _2 = \boldsymbol A \boldsymbol x _2 = \lambda \boldsymbol x _2 $. 이는 새롭게 얻은 vector $\boldsymbol v _i$가 eigenvector $\boldsymbol x _i$를 스케일링 하는 것과 동일하다. $\boldsymbol v _1, \boldsymbol v _2$는 여전히 orthogonal하며, 이들이 span하는 직사각형의 넓이는 $\lambda _1 \lambda _2$가 된다.

$\boldsymbol x _1, \boldsymbol x _2$가 orthonormal하면 이의 둘레를 $2(1+1)$로 계산할 수 있다. $\boldsymbol A$를 이용하여 eigenvector를 맵핑하여 생기는 직사각형의 둘레는 $2(\lvert \lambda _1 \rvert + \lvert \lambda _2 \rvert$)가 된다. 따라서, 고윳값의 절댓값을 더하면 transformation matrix $\boldsymbol A$에 의해 변화하는 둘레가 무엇인지 알 수 있게된다.

## Cholesky Decomposition

머신러닝에서 종종 마주치는 특수한 종류의 행렬을 분해하는 방법은 여러 종류가 있다. Symmetric, positive definite matrix에 대해 우리는 제곱근을 이용하여 분해할 수 있다. 이는 **Cholesky Decomposition/Cholesky Factorization**이라 부른다.

<div class="notice--info" markdown="1">

**Theorem 4.18** (Cholesky Decomposition)

Symmetric, positive definite matrix $\boldsymbol A$는 두 행렬의 곱으로 표현할 수 있다. $\boldsymbol A = \boldsymbol L \boldsymbol L^{\intercal}$. $\boldsymbol L$은 lower triangular matrix로, 이의 대각성분은 양수값이 된다.

$$
\begin{bmatrix}
a _{11} & \cdots & a _{1n} \\ \vdots & \ddots & \vdots \\ a _{n1} & \cdots & a _{nn}
\end{bmatrix}
=
\begin{bmatrix}
l _{11} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ l _{n1} & \cdots & l _{nn}
\end{bmatrix}
=
\begin{bmatrix}
l _{11} & \cdots & l _{n1} \\ \vdots & \ddots & \vdots \\ 0 & \cdots & l _{nn}
\end{bmatrix}
\tag{4.44}
$$

$\boldsymbol L$은 $\boldsymbol A$의 **Cholesky factor**라 부르고, $\boldsymbol L$은 유일하다.

</div>

Cholesky Decomposition은 머신러닝 기저에서 numerical computation에 대해 중요한 역할을 한다. 여기서 symmetric positive definite matrix는 빈번하게 사용되는데, multivaraite Gaussian variable에 대한 covaraince matrix (Section 6.5)를 구하는 것이 그 예이다. Cholesky decompostion을 covariance matrix에 적용하면 Gaussian 분포로부터 샘플을 생성할 수 있게된다. 이는 우리로 하여금 확률변수의 linear transformation을 가능케한다. 이는 deep stochastic model에서 gradient를 계산하는데 매우 자주 활용되며, 대표적으로는 variational auto-encoder가 있다. 

Cholesky decompostion는 또한 행렬식을 매우 효율적으로 계산하게 해준다. Cholesky decompostion $\boldsymbol A = \boldsymbol L \boldsymbol L^{\intercal}$로부터 $\text{det}(\boldsymbol A) = \text{det}(\boldsymbol L) \text{det}(\boldsymbol L^{\intercal})$임을 알 수 있다. $\boldsymbol L$은 triangular matrix이므로, 이의 행렬식은 단순하게 대각성분을 곱하는 것으로 구할 수 있다. 따라서$\text{det}(\boldsymbol A)=\prod _i l^2 _{ii}$가 된다. 많은 수치연산 소프트웨어 패키지들은 이러한 Cholesky Decomposition를 활용하여 연산을 더욱 효율적으로 진행한다.

## Eigendecomposition and Diagonalization

**Diagonal matrix (대각행렬)**은 대각성분을 제외한 나머지 성분이 모두 0인 행렬을 의미한다. 이를 통하여 행렬식, 제곱, 역행렬 등을 빠르게 계산할 수 있다. 행렬식은 대각성분의 곱을 통해 진행되고, 행렬의 제곱은 각 대각성분을 제곱하는 것으로 얻을 수 있고, 역행렬은 대각성분의 역수를 통해 만들 수 있다.

이 장에서 행렬을 대각 형태로 바꾸는 법을 배우게 될텐데, 이는 basis change와 eigenvalue에 대해 중요한 역할을 한다.

앞서 $\boldsymbol P$의 역행렬이 존재하여 $\boldsymbol D = \boldsymbol P^{-1} \boldsymbol A \boldsymbol P$가 성립하면 $\boldsymbol A, \boldsymbol D$는 닮음 행렬이라고 하였다. 더욱 구체적으로 $\boldsymbol A$가 대각 행렬 $\boldsymbol D$와 닮았음은 $\boldsymbol D$에 대각 성분에 $\boldsymbol A$의 eigenvalue를 포함하고 있음을 살펴보게 될 것이다.

> 좀 더 자세히 살펴보자. if $\boldsymbol A \boldsymbol x = \lambda \boldsymbol x$ then, $(\boldsymbol B \boldsymbol A \boldsymbol B^{-1}) (\boldsymbol B \boldsymbol x) = \boldsymbol B \boldsymbol A \boldsymbol x = \boldsymbol B \lambda \boldsymbol x = \lambda (\boldsymbol B \boldsymbol x)$. 즉, $\boldsymbol A$와 $\boldsymbol B \boldsymbol A \boldsymbol B^{-1}$는 similar이다. 이 관계를 통해 $\boldsymbol B \boldsymbol A \boldsymbol B^{-1}$와 $\boldsymbol A$의 고윳값이 같음을 알 수 있다.
>
> 이는 $\boldsymbol A$가 너무커서 특성다항식을 구하기가 어려울 때 쉬운 $\boldsymbol B$를 찾는 것으로 대신 구할 수 있다. 이 때, $\boldsymbol B \boldsymbol A \boldsymbol B^{-1}$는 가능한한 triangular matrix로 만든다. 이의 고윳값은 대각성분을 통해 구할 수 있기 때문이다.

<div class="notice--warning" markdown="1">

**Definition 4.19** (Diagonalizable).

행렬 $\boldsymbol A \in \mathbb R^{n \times n}$가 대각행렬과 닮으면 **diagonalizable**이라고 한다. 즉, $\boldsymbol D = \boldsymbol P^{-1} \boldsymbol A \boldsymbol P$가 성립하는 $\boldsymbol P \in \mathbb R^{n \times n}$가 존재한다.

</div>

다음을 통해 대각행렬 $\boldsymbol A \in \mathbb R^{n \times n}$가 똑같은 선형사상에 대해 다른 basis로 표현하는 방법이라는 것을 살펴볼 것이다. 또한 이는 $\boldsymbol A$의 고유벡터로 이루어진 basis이다.

$\boldsymbol A \in \mathbb R^{n \times n}$, $\lambda _1, \cdots, \lambda _n$을 스칼라 집합, $\boldsymbol p _1 , \cdots, \boldsymbol p _n$을 $\mathbb R^n$의 벡터라고 가정하자. 우리는 $\boldsymbol P := [\boldsymbol p _1 , \cdots, \boldsymbol p _n]$이라고 정의하고, $\boldsymbol D \in \mathbb R^{n \times n}$를 대각성분 $\lambda _1, \cdots, \lambda _n$를 갖는 대각행렬이라고 하자. 그러면, 우리는 다음 (4.50)과 $\boldsymbol A$의 고윳값과 고유행렬이 각각 $\lambda _1, \cdots, \lambda _n$, $\boldsymbol p _1 , \cdots, \boldsymbol p _n$임은 동치이다.

$$
\boldsymbol A \boldsymbol P = \boldsymbol P \boldsymbol D \tag{4.50}
$$

이는 다음을 통해 참임을 확인할 수 있다.

$$
\begin{align}
& \boldsymbol A \boldsymbol P = \boldsymbol A [\boldsymbol p _1 , \cdots, \boldsymbol p _n] = [\boldsymbol A \boldsymbol p _1 , \cdots, \boldsymbol A \boldsymbol p _n] \tag{4.51} \\
& \boldsymbol P \boldsymbol D = [\boldsymbol p _1 , \cdots, \boldsymbol p _n] 
\begin{bmatrix}
\lambda _1 & & 0 \\ & \ddots & \\ 0 & & \lambda _n
\end{bmatrix}
= [\lambda _1 \boldsymbol p _1 , \cdots, \lambda _n \boldsymbol p _n] \tag{4.52}
\end{align}
$$

게다가 (4.50)은 다음을 암시한다.

$$
\begin{align}
\boldsymbol A \boldsymbol p _1 &= \lambda _1 \boldsymbol p _1 \tag{4.53} \\
& \vdots \\
\boldsymbol A \boldsymbol p _n &= \lambda _n \boldsymbol p _n \tag{4.54}
\end{align}
$$

따라서 $\boldsymbol P$의 column은 $\boldsymbol A$의 고유벡터가 된다.

대각화에 대한 정의는 $\boldsymbol P \in \mathbb R^{n \times n}$의 역행렬이 존재하는 것을 전제로 한다. 즉, $\boldsymbol P$는 full rank를 갖는다 (Theorem 4.3). 이는 우리로 하여금 $n$개의 선형 독립인 eigenvector $\boldsymbol p _1 , \cdots, \boldsymbol p _n$를 필요로 한다. 즉, $\boldsymbol p _i$는 $\mathbb R^n$의 basis를 이룬다.

<div class="notice--info" markdown="1">

**Theorem 4.20** (Eigendecomposition).

$\boldsymbol A$의 고유벡터가 $\mathbb R^n$의 basis를 이루는 것과 정사각행렬 $\boldsymbol A \in \mathbb R^{n \times n}$이 다음과 같이 분해되는 것은 동치이다.

$$
\boldsymbol A = \boldsymbol P \boldsymbol D \boldsymbol P^{-1} \tag{4.55}
$$

$\boldsymbol P \in \mathbb R^{n \times n}$이고, $\boldsymbol D$는 대각행렬로, 그 성분이 $\boldsymbol A$의 고유값이 된다.

</div>

위 정리는 오직 non-defective matrix만이 대각화가 가능함과, $\boldsymbol P$의 column이 $\boldsymbol A$의 $n$개의 고유벡터임을 내포한다. 대칭행렬에서는 더욱 유용한 결론을 낼 수 있다.

**Theorem 4.21**. 대칭행렬 $\boldsymbol S \in \mathbb R^{n \times n}$은 항상 대각화가 가능하다.
{: .notice--info}

위 정리는 spectral theorem 4.15로부터 바로 도출된다. 또한 spectral theorem은 $\mathbb R^n$의 eigenvector로 이루어진 ONB를 찾을 수 있음을 말해준다. 이를 통해 $\boldsymbol P$는 orthogonal matrix로 $\boldsymbol P \boldsymbol A \boldsymbol P^{\intercal}$를 만족함을 보일 수 있다.

*Remark*. 행렬의 Jordan normal form은 defective matrix에 대한 decomposition을 제공하나, 본 책에서 다루는 범위를 벗어나므로 생략한다.
{: .notice}

> 본 책에서 다루지 않는 성질을 이야기해보자. 앞서 배운 기하적 중복도와 대수적 중복도가 같으면 대각화가 가능하다.
>
> 1. Eigenvectors (geometric): There are non-zero solutions to $\boldsymbol A \boldsymbol x = \lambda \boldsymbol x$
> 2. Eigenvalues (algebraic): The determinant of $\boldsymbol A - \lambda \boldsymbol I$ is zero
>
> $$
> \text{algebraic multiplicity} \geq \text{geometric multiplicity}
> $$
> 
> 등호는 diagonalizable일 때 성립한다.

### Geometric Intuition for the Eigendecomposition

이전과 마찬가지로 eigendecomposition 또한 다음과 같이 기하학적으로 해석할 수 있다. $\boldsymbol A$를 선형변환의 transformation이라 하자. $\boldsymbol P^{-1}$은 표준기저에서 eigenbasis로 기저 변환을 수행한다. 그러면 대각행렬 $\boldsymbol D$는 이 기저 방향으로 eigenvalue만큼 벡터를 스케일링한다. 마지막으로 $\boldsymbol P$는 스케일링한 벡터를 다시 표준기저로 되돌린다.

![image](https://user-images.githubusercontent.com/47516855/118828837-94a25980-b8f8-11eb-8532-2ab54d4d2f84.png){: .align-center}{: width="600"}

- 대각행렬 $\boldsymbol D$의 제곱은 쉽게 계산할 수 있다 (대각행렬의 제곱). 그러므로 $\boldsymbol A$의 제곱은 eigenvaluedecomposition을 통해 수행할 수 있다.

    $$
    \boldsymbol A^k = (\boldsymbol P \boldsymbol D \boldsymbol P^{-1})^k = \boldsymbol P \boldsymbol D^k \boldsymbol P^{-1} \tag{4.62}
    $$
- eigendecomposition $\boldsymbol A = \boldsymbol P \boldsymbol D \boldsymbol P^{-1}$이 존재한다고 가정해보자. 그러면, 다음과 같이 효율적인 계산이 가능해진다.

    $$
    \begin{align}
    \text{det}(\boldsymbol A) &= \text{det}(\boldsymbol P \boldsymbol D \boldsymbol P^{-1}) = \text{det}(\boldsymbol P) \text{det}(\boldsymbol D) \text{det}(\boldsymbol P^{-1}) \tag{4.63a} \\ 
    &= \text{det}(\boldsymbol D) = \prod _i d _{ii} \tag{4.63b}
    \end{align}
    $$

고윳값분해는 정방행렬을 필요로한다. 만일 다양한 행렬에 대해 분해를 적용할 수 있다면 이는 매우 유용할 것이다. 다음에 배울 것이 바로 일반적인 행렬의 분해방법이며, 이는 sigular value decompostion이라 부른다.

## Singular Value Decomposition

행렬의 singular value decomposition (SVD)는 선형대수의 핵심적인 분해 방법이다. 이는 "선형대수의 중요 정리"라는 이름으로 불리기도 했는데, 정사각행렬뿐만 아니라 모든 행렬에 대해 적용가능하며, 항상 존재하기 때문이다. 앞으로 더 살펴보겠지만 추가적으로 $\boldsymbol A$의 SVD는 linear mapping $\Phi: V \to W$를 나타내는데, 이는 두 벡터공간 사이의 기저에 깔린 기하학적인 변화를 측정하는 방법이기도 하다.

<div class="notice--info" markdown="1">

**Theorem 4.22** (SVD Theorem (특이값분해))

$\boldsymbol A \in \mathbb R^{m \times n}$이 직사각 행렬이라 하고, 이의 랭크가 $r \in [0, min(m, n)]$이라 하자. $\boldsymbol A$의 SVD는 다음과 같은 형태의 분해가 된다.

![image](https://user-images.githubusercontent.com/47516855/118835864-55770700-b8fe-11eb-8669-5bdd8e9c8770.png){: .align-center}{: width="300"}

$\boldsymbol U \in \mathbb R^{m \times m}$, $\boldsymbol V \in \mathbb R^{n \times n}$가 orthogonal matrix이다. $\boldsymbol \Sigma \in \mathbb R^{m \times n}$은 $\Sigma _{ii} = \sigma _i \geq 0$이고, $\Sigma _{ij} = 0$이 된다.

</div>

대각성분 $\sigma _{i}$는 **singular value(특이값)**으로, $\boldsymbol U$의 column vector $\boldsymbol u _i$는 left-singular vector, $\boldsymbol V$의 column vector $\boldsymbol v _j$는 right singular vector라 부른다. 관습적으로 singular value들은 순서가 정해져있다. ($\sigma _1 \geq \sigma _2 \geq \sigma _r \geq 0$)

Singular value matrix $\boldsymbol \Sigma$는 유일하고 $\boldsymbol A$와 동일한 크기를 갖는다. 따라서 $\boldsymbol A$가 직사각형일 경우 zero padding이 필요하다.

$$
\begin{align}
\boldsymbol \Sigma =
  \begin{bmatrix}
    \sigma _1 & 0 & 0 \\
    0 & \cdots & 0 \\
    0 & \cdots & \sigma _n \\
    0 & \cdots & 0 \\
    \vdots &  & \vdots \\
    0 & \cdots & 0 \\
  \end{bmatrix} \tag{4.65} \\
\boldsymbol \Sigma =
  \begin{bmatrix}
    \sigma _1 & 0 & 0 & 0 & \cdots & 0 \\
    0 & \ddots & 0 & \vdots &  & \vdots \\
    0 & 0 & \sigma _m & 0 & \cdots & 0 \\
  \end{bmatrix} \tag{4.66} 
\end{align}
$$

### Geometric Intuitions for the SVD

SVD는 기하학적인 직관을 제공하여 변환행렬에 특성을 알아볼 수 있게 해준다. SVD를 sequential linear transformation의 형태로 알아볼 것이다.

SVD는 이에 해당하는 선형변환를 세 가지 요소로 분해하는 것으로 해석할 수 있다. 넓게 말하면 SVD는 $\boldsymbol V^{\intercal}$을 통한 basis change를 수행하는 것이며, 그후 스케일링과 augmentation을 수행하는 것이다.

![image](https://user-images.githubusercontent.com/47516855/118839477-79881780-b901-11eb-8f8e-1f6de2bdc379.png){: .align-center}{: width="600"}

표준기저 $B, C$에 대한 선형변환 $\Phi: \mathbb R^n \to \mathbb R^m$의 변환행렬을 생각해보자. 또한, 또 다른 기저 $\tilde B$ of $\mathbb R^n$, $\tilde C$ of $\mathbb R^m$도 있다고하자. 그러면,

1. 행렬 $\boldsymbol V$는 정의역 $\mathbb R^n$에서 $\tilde B$로부터 $B$까지의 기저 변환을 수행한다 (좌상단 그림의 빨간색/오렌지색 벡터 $\boldsymbol v _1, \boldsymbol v _2$). $\boldsymbol V^{\intercal} = \boldsymbol V^{-1}$는 $B$에서 $\tilde B$로의 기저 변환을 수행한다. 빨간색과 오랜지색 벡터가 이제는 기저벡터와 같은 방향으로 정렬된 것을 확인할 수 있다.
2. $\tilde B$로 바뀐 좌표계에서, $\boldsymbol \Sigma$는 새로운 좌표계를 singular value $\sigma _i$를 통해 스케일한다 (그리고 차원을 더하거나 삭제함). 즉, $\boldsymbol \Sigma$는 $\tilde B$와 $\tilde C$에 대한 transformation matrix $\Phi$이다. 이는 $e _1 - e _2$평면에서 늘려진 빨간색/오랜지색 벡터이고, 3차원에 임베딩된 것을 확인할 수 있다.
3. $\boldsymbol U$는 공역 $\mathbb R^m$에서, $\tilde C$로부터 $\mathbb R^m$의 표준기저로의 basis change를 수행한다. 이는 빨간색/오랜지색 벡터의 회전으로 표현되어 있다.

SVD는 기저변환을 정의역과 공역 둘 다에서 표현한다. 이는 고유값분해가 같은 벡터공간에서 같은 기저변환을 적용하고 되돌리는 것과는 대조적이다. SVD를 특별하게 만드는 것은 이러한 두 개의 기저가 특이값 행렬 $\boldsymbol \Sigma$에 읳해 동시에 연결되기 때문이다.

아래 그림은 다음 행렬을 사용한 예제이다.

$$
\begin{align}
\boldsymbol A & =
  \begin{bmatrix}
    1 & -0.8 \\
    0 & 1 \\
    1 & 0
  \end{bmatrix}
= \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal} \tag{4.67a} \\
& =
  \begin{bmatrix}
    - 0.79 & 0 & -0.62 \\
    0.38 & -0.78 & -0.49 \\
    -0.48 & -0.62 & 0.62 \\
  \end{bmatrix}
  \begin{bmatrix}
    1.62 & 0 \\
    0 & 1.0 \\
    0 & 0 \\
  \end{bmatrix} 
  \begin{bmatrix}
    -0.78 & 0.62 \\
    -0.62 & -0.78 \\
  \end{bmatrix}   
\tag{4.67b} 
\end{align}
$$

![image](https://user-images.githubusercontent.com/47516855/118847375-a3910800-b908-11eb-8913-9387666713cc.png){: .align-center}{: width="600"}

행렬 연산은 뒤에서부터 시작하므로 $\boldsymbol V^{\intercal}$를 먼저 보자. 우선 $\boldsymbol V^{\intercal} \in \mathbb R^{2 \times 2}$가 격자 모양의 데이터를 회전시킨다. 그 후 해당 벡터들은 특이값 행렬 $\boldsymbol \Sigma$를 통해 아래 오른쪽 그림과 같이 3차원 위로 옮겨진다. 이때 벡터들은 3차원으로 변환되었지만, plane이 그대로 유지되는 것을 확인할 수 있다. 이후 마지막으로 $\boldsymbol U$가 해당 벡터들을 오른쪽 위 그림과 같이 3차원 공간안에 존재하도록 변환시켜준다.

### Construction of the SVD

이제 SVD가 왜 존재하고 이를 어떻게 계산하는지 살펴보자. 일반적인 행렬의 SVD는 정방행렬의 고윳값 분해와 유사한 측면이 있다.

<div class="notice" markdown="1">

*Remark.* SPD (Symmetric, Positive Definite) 행렬의 고윳값 분해를 비교해보자.

$$
\boldsymbol S = \boldsymbol S^{\intercal} = \boldsymbol P \boldsymbol D \boldsymbol P^{\intercal} \tag{4.68}
$$

이에 대응하는 SVD는

$$
\boldsymbol S = \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal} \tag{4.69}
$$

만일 다음과 같이 세팅한다면,

$$
\boldsymbol U = \boldsymbol P =\boldsymbol V, \boldsymbol D=\boldsymbol \Sigma \tag{4.70}
$$

SPD 행렬의 SVD는 이의 고윳값분해와 같아지는 것을 확인할 수가 있다.

</div>

이제 Theorem 4.22 (SVD Theorem)이 왜 성립하고, SVD가 어떻게 구성되는지 살펴보자. SVD를 계산하는 것은 두 개의 orthonomal basis를 찾는 것과 동일하다. 이러한 ordered basis를 통해 우리는 ONB를 찾을 수 있다.

먼저 right singular vectors $\boldsymbol v_1, \cdots, \boldsymbol v_n \in \mathbb R^m$의 orthonomal set을 구성하고, left singular vectors $\boldsymbol u_1, \cdots, \boldsymbol u_n \in \mathbb R^n$의 orthonomal set을 구성해보자. 그후에 이 둘을 연결하고, 변환 $\boldsymbol A$ 이후에도 $\boldsymbol v _i$의 orthogonality가 유지됨을 보일 것이다. 이는 상 $\boldsymbol A \boldsymbol v _i$가 orthogonal vector의 집합을 형성한다는 점에서 중요하다. 그리고 이 상을 scalar factor, 즉, singular value를 통해 normalize할 것이다.

Spectral theorem에 의해 대칭행렬의 고윳값은 ONB가 된다. 이를 우리는 대각화할 수 있다. 또한, theorem 4.14를 통해 우리는 항상 symmetric, positive semidefinte 행렬 $\boldsymbol A^{\intercal} \boldsymbol A \in \mathbb R^{n \times n}$을 만들 수 있다. 따라서 우리는 언제나 $\boldsymbol A^{\intercal} \boldsymbol A$를 대각화 할 수 있고 다음을 얻을 수 있다.

$$
\boldsymbol A^{\intercal} \boldsymbol A = \boldsymbol P \boldsymbol D \boldsymbol P^{\intercal} = \boldsymbol P 
  \begin{bmatrix}
    \lambda _1 & \cdots & 0 \\
    \vdots & \ddots & \vdots \\
    0 & \cdots & \lambda _n \\
  \end{bmatrix}
\boldsymbol P^{\intercal} \tag{4.71}
$$

$\boldsymbol P$는 orthogonal matrix이고, 이는 orthonormal engenbasis로 이루어져있다. $\lambda _i \geq 0$은 $\boldsymbol A^{\intercal} \boldsymbol A$의 고윳값이 된다. 이제 $\boldsymbol A$의 SVD가 존재하고, (4.64)를 (4.71)에 넣어보자.

$$
\boldsymbol A^{\intercal} \boldsymbol A 
= (\boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal})^{\intercal} (\boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal}) 
= \boldsymbol V \boldsymbol{\Sigma^{\intercal}} \boldsymbol{U^{\intercal}} \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal} \tag{4.72}
$$

여기서 $\boldsymbol U, \boldsymbol V^{\intercal}$는 orthogonal matrix가 된다. 그러므로, $\boldsymbol U \boldsymbol U^{\intercal} =\boldsymbol I $를 이용,

$$
\boldsymbol A^{\intercal} \boldsymbol A  
= \boldsymbol V \boldsymbol{\Sigma^{\intercal}} \boldsymbol \Sigma \boldsymbol V^{\intercal} 
= \boldsymbol V \boldsymbol V^{\intercal} \tag{4.73}
$$

(4.71)과 (4.73)을 비교하면 다음을 얻을 수 있다.

$$
\begin{align}
\boldsymbol{V{\intercal}} = \boldsymbol{P{\intercal}} \tag{4.74} \\
\sigma^2 _{i} = \lambda _i \tag{4.75}
\end{align}
$$

따라서 $\boldsymbol P$를 이루는 $\boldsymbol A^{\intercal} \boldsymbol A$의 고유벡터는 $\boldsymbol A$의 right-singular vectors $\boldsymbol V$가 된다 (4.74 참고). $\boldsymbol A^{\intercal} \boldsymbol A$의 고윳값은 $\boldsymbol{\Sigma}$의 2제곱이 된다 (4.75참고).

left singular vector $\boldsymbol U$를 얻기 위해, 이와 비슷한 과정을 진행한다. 이번엔 $\boldsymbol A \boldsymbol A^{\intercal} \in \mathbb R^{m \times m}$에 대해 SVD를 계산한다.

$$
\begin{align*}
\boldsymbol A \boldsymbol A^{\intercal} 
& = (\boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal}) (\boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal})^{\intercal}
= \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal} \boldsymbol V \boldsymbol{\Sigma^{\intercal}} \boldsymbol{U^{\intercal}} \tag{4.76a} \\
& = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{\Sigma^{\intercal}} \boldsymbol{U^{\intercal}} \tag{4.76b}
\end{align*}
$$

스펙트럼 정리는 $\boldsymbol A \boldsymbol A^{\intercal} = \boldsymbol S \boldsymbol D \boldsymbol S^{\intercal}$가 대각화가 가능하며, 따라서 $\boldsymbol A \boldsymbol A^{\intercal}$에 대한 고유벡터의 ONB가 있음을 말해준다. 이는 $\boldsymbol S$를 통해 찾을 수 있다. $\boldsymbol A \boldsymbol A^{\intercal}$의 orthonormal eigenvector는 left-singular vector $\boldsymbol U$이며, SVD 공역의 orthonormal basis를 형성한다.

이는 우리에게 $\boldsymbol \Sigma$의 구조에 대한 질문을 남긴다. $\boldsymbol A \boldsymbol A^{\intercal}$와 $\boldsymbol A^{\intercal} \boldsymbol A$는 똑같은 non-zero eigenvalue를 갖고, $\boldsymbol \Sigma$의 0이 아닌 성분은 $\boldsymbol A \boldsymbol A^{\intercal}$와 $\boldsymbol A^{\intercal} \boldsymbol A$에서 똑같기 때문이다.

본 과정들을 통해 $\boldsymbol V$의 right-singular vectors에 대한 orthonormal set을 얻었다. SVD를 끝내기 위해 이를 orthonormal vector $\boldsymbol U$와 연결하자. 이를 달성하기 위해 우리는 $\boldsymbol A$에 대한 $\boldsymbol v _i$의 상은 orthonormal해야된다는 점을 이용하자. 이는 앞서 Angles and Orthogonality에서 배운 것을 활용하면 된다. 우리는 $\boldsymbol A \boldsymbol v _i$와 $\boldsymbol A \boldsymbol v _j$의 inner product를 계산하여 $i \neq j$인 경우에 0을 얻어야만 한다.

$$
(\boldsymbol A \boldsymbol v _i)^{\intercal} (\boldsymbol A \boldsymbol v _j)
= \boldsymbol v _i^{\intercal}(\boldsymbol A^{\intercal} \boldsymbol A)\boldsymbol v _j
= \boldsymbol v _i^{\intercal}(\lambda _j \boldsymbol v _j) = \lambda _j \boldsymbol v _i^{\intercal} \boldsymbol v _j \tag{4.77}
$$

$m \geq r$인 경우에 $\{\boldsymbol A \boldsymbol v _1, \cdots, \boldsymbol A \boldsymbol v _r \}$가 r차원의 basis라는게 성립한다.

이제 left-singluar vector가 orthonormal임을 보이자. Right-singular vector의 상을 normalize하면,

$$
\boldsymbol u _i 
:= \frac{\boldsymbol A \boldsymbol v _i}{\| \boldsymbol A \boldsymbol v _i \|}
= \frac{1}{\sqrt{\lambda _i}} \boldsymbol A \boldsymbol v _i
= \frac{1}{\sqrt{\sigma _i}} \boldsymbol A \boldsymbol v _i, \tag{4.78}
$$

마지막 등식은 (4.75)와 (4.76b)으로부터 얻어진다.

그러므로, $\boldsymbol A^{\intercal} \boldsymbol A$의 eigenvector는 right singular vector $\boldsymbol v _i$가 되고, $\boldsymbol A$에 의한 normalized image인 left singular vector $\boldsymbol u _i$는 특이값 행렬을 통해 연결되는 두 개의 self-consistent ONB를 형성하게 된다.

(4.78)을 재정리하면 우리는 **singular value equation**을 얻을 수 있다.

$$
\boldsymbol A \boldsymbol v _i = \sigma _i \boldsymbol u _i, ~ i = 1, ..., r. \tag{4.79}
$$

이 식은 eigenvalue equation (4.25)와 매우 유사하지만, 좌변의 벡터와 우변의 벡터가 서로 다름에 주의하자.

$n < m$인 경우 (4.79)는 $i \leq n$에 대해서 성립하고, $i > n$인 $\boldsymbol u _i$에 대해서는 어떻게 되는지 알 수 없지만 구조적으로 orthonormal인 것을 알 수 있다. 이와 반대로, $m < n$인 경우 (4.79)는 $i \leq m$인 경우에만 성립한다. $i > m$인 경우 우리는 $\boldsymbol A \boldsymbol v _i = \boldsymbol 0$을 얻게 되고, 여전히 $\boldsymbol v _i$가 orthonormal set을 이룸을 알 수 있다. 이는 $N(\boldsymbol A)$의 orthonormal basis를 SVD가 포함한다는 뜻이다.

$\boldsymbol v _i$를 $\boldsymbol V$의 column으로, $\boldsymbol u _i$를 $\boldsymbol U$의 column으로 concatenation하면 다음을 얻을 수 있다.

$$
\boldsymbol A \boldsymbol V = \boldsymbol U \boldsymbol \Sigma , \tag{4.80}
$$

$\boldsymbol \Sigma$는 $\boldsymbol A$와 같은 차원을 갖고 있고, 행 $1, ..., r$에 대해 대각 성분을 갖고 있다. 그러므로 $\boldsymbol V^{\intercal}$을 오른쪽에 곱하면 $\boldsymbol A = \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal}$를 얻게되고, 이는 $\boldsymbol A$의 SVD가 된다.

### Eigenvalue Decomposition vs. Singular Value Decomposition

고윳값 분해 $\boldsymbol A = \boldsymbol P \boldsymbol D \boldsymbol P^{\intercal}$와 SVD $\boldsymbol A = \boldsymbol U \boldsymbol \Sigma \boldsymbol V^{\intercal}$를 비교해보고, 이들의 핵심 개념을 다시 살펴보자.

- SVD는 어떠한 행렬에 대해서도 존재한다. 고윳값 분해는 정방행렬에 대해서만 정의되며, $\mathbb R^n$의 고유벡터의 기저가 존재할 경우에만 진행할 수 있다.
- 고윳값 분해 행렬 $\boldsymbol P$의 벡터는 직교할 필요는 없다. 즉, 기저를 변환시켜도 단순한 회전과 스케일링으로 표현하기가 어렵다. 반면 SVD에서의 $\boldsymbol U$와 $\boldsymbol V$의벡터는 orthonormal하며, 따라서 해당 벡터의 변화로 회전과 스케일링을 나타낼 수 있다.
- 고윳값 분해와 SVD 모두 다음과 같은 세가지 선형변환으로 구성된다.
  - 정의역의 기저를 바꾸는 행렬
  - 각각의 새로운 기저 벡터를 스케일링하고, 정의역에서 공역으로 맵핑하는 행렬
  - 기저를 공역으로 전환하는 행렬
- SVD와 고윳값 분해의 가장 큰 차이는 SVD에서는 정의역과 공역이 서로 다른 차원의 벡터공간이라는 것이다.
- SVD에서 left/right-singular vector matrix는 일반적으로 서로의 역행렬이 아니다 (차원이 다름). 하지만 고윳값 분해는 이 둘이 역행렬관계이다.
- SVD에서 특이값은 음이 아닌 실수이지만, 고윳값 분해는 그렇지 않다.
- SVD와 고윳값 분해는 사영을 내릴 때 많은 연관성이 있다.
  - $\boldsymbol A$의 left-singular vector는 $\boldsymbol A \boldsymbol A^{\intercal}$의 고윳값이다.
  - $\boldsymbol A$의 right-singular vector는 $\boldsymbol A^{\intercal} \boldsymbol A$의 고윳값이다.
  - $\boldsymbol A$의 non-zero singular value는 $\boldsymbol A \boldsymbol A^{\intercal}$의 non-zero eigenvalue의 제곱근이고, $\boldsymbol A^{\intercal} \boldsymbol A$의 non-zero eigenvalue의 제곱근이다.
- 대칭행렬에서의 고윳값분해와 SVD는 spectral theorem에 의해 서로 같다.

## Matrix Approximation

앞서 SVD가 세개의 행렬로 분해됨을 확인하였다. 우리는 SVD를 온전히 이용한 분해 대신, $\boldsymbol A$를 간단한 (low-rank) 행렬 $\boldsymbol A _i$의 합으로 표현할 것이다.

rank-1 행렬 $\boldsymbol A _i \in \mathbb R^{m \times n}$을 다음과 같이 구성한다.

$$
\boldsymbol A _i := \boldsymbol u _i \boldsymbol{v _i}^{\intercal}, \tag{4.90}
$$

이는 $\boldsymbol U$와 $\boldsymbol V$의 $i$-th orthogonal column vector의 외적이 된다. 다음 그림은 스톤헨지에 대해 rank-1 행렬을 표현한 것이다.

![image](https://user-images.githubusercontent.com/47516855/118863422-d394d700-b919-11eb-9ebf-b4caef12a7b8.png){: .align-center}{: width="600"}

rank $r$을 갖는 행렬 $\boldsymbol A \in \mathbb R^{m \times n}$은 rank-1 행렬 $\boldsymbol A _i$의 합으로 쓸 수 있다.

$$
\boldsymbol A = \sum^r _{i=1} \sigma _i \boldsymbol u _i \boldsymbol{v _i}^{\intercal} =  \sum^r _{i=1} \sigma _i \boldsymbol A _i \tag{4.91}
$$

이는 outer product 행렬 $\boldsymbol A _i$가 $i$-th singular value $\sigma _i$를 통해 weight 된 형태이다. (4.91)이 성립하는 이유는 대각성분의 곱이 이에 해당되는 left/right singular vector $ \boldsymbol u _i \boldsymbol{v _i}^{\intercal}$에만 적용되고, 이를 스케일링하기 때문이다. 모든 $i \neq j$에 대한 $\sum _{ij} \sigma _i \boldsymbol u _i \boldsymbol{v _i}^{\intercal}$ term은 사라지게 되고, $i > r$에 대해서도 이에 해당하는 특이값이 0이기 때문에 사라지게 된다.

앞서 우리는 rank-1 행렬 $\boldsymbol A _i$를 살펴보았는데, 이를 $r$개의 rank-1 행렬을 더함으로써 rank-$r$ 행렬을 얻어보자. 이는 $k < r$까지만 적용하여 **rank-k approximation**을 얻을 수 있다.

$$
\hat{\boldsymbol{A}}(k) := \sum^k _{i=1} \sigma _i \boldsymbol u _i \boldsymbol{v _i}^{\intercal} =  \sum^k _{i=1} \sigma _i \boldsymbol A _i \tag{4.92}
$$

이의 랭크는 $k$가 된다. 다음 그림은 앞선 그림에 대해서 low-rank approximation $\hat{\boldsymbol{A}}(k)$를 적용한 모습이다.

![image](https://user-images.githubusercontent.com/47516855/118864558-10150280-b91b-11eb-929e-1f92d0c4dc32.png){: .align-center}{: width="600"}

$\boldsymbol A$와 이의 rank-$k$ approximation $\hat{\boldsymbol{A}}(k)$ 사이의 차이 (error)를 측정하기 위해 우리는 norm의 개념이 필요하다.

<div class="notice--warning" markdown="1">

**Definition 4.23** (Spectral Norm of a Matrix)

어떤 $\boldsymbol x \in \mathbb R^n \setminus \\{ \boldsymbol 0 \\} $에 대해, $\boldsymbol A \in \mathbb R^{m \times n}$의 **spectral norm**은 다음과 같이 정의된다.

$$
\| \boldsymbol A \| _2 := \text{max} _{\boldsymbol x} \frac{\| \boldsymbol A \boldsymbol x \| _2 }{\| \boldsymbol x \| _2 }. \tag{4.93}
$$

</div>

matrix norm 내에 밑첨자에 대한 notation을 처음으로 도입하였는데 (좌변) 이는 벡터의 유클리디안 노름과 비슷하다. Spectral norm (4.93)은 어떤 벡터 $\boldsymbol x$가 $\boldsymbol A$를 곱했을 경우 갖을 수 있는 최대한의 거리를 의미한다.

**Theorem 4.24**. Spectral norm은 그 행렬의 가장 큰 특이값 $\sigma _i$이다.
{: .notice--info}

<div class="notice--info" markdown="1">

**Theorem 4.25** (Eckart-Young Theorem (Eckart and Young, 1936)).

$\boldsymbol A \in \mathbb R^{m \times n}$이 rank $r$을 갖고, $\boldsymbol B \in \mathbb R^{m \times n}$가 rank $k$를 갖는다고 하자. 어떤 $k \leq r$인 $\sum^k _{i=1} \sigma _i \boldsymbol u _i \boldsymbol{v _i}^{\intercal}$에 대해, 다음이 성립한다.

$$
\begin{align}
\hat{\boldsymbol{A}}(k) := \text{argmin} _{\text{rk}(\boldsymbol B)=k} \| \boldsymbol{A} - \boldsymbol{B} \| _2, \tag{4.94} \\
\| \boldsymbol{A} - \hat{\boldsymbol{A}}(k) \| _2 = \sigma _{k+1} \tag{4.95}
\end{align}
$$

</div>

Eckart-Young theorem은 rank-$k$ approximation을 통해 행렬 $\boldsymbol{A}$를 근사한 값의 오차를 설명한다. SVD를 통해 얻어진 rank-$k$ approximation은 full-rank matrix $\boldsymbol{A}$를 낮은차원(rank-at-most-$k$)으로 사영하는 것으로 생각할 수 있다. 가능한 모든 사영에 대해, SVD는 rank-$k$ approximation을 기존 행렬과의 오차를 최소화한다.

(4.95)가 성립하는 이유를 생각해보자. $\boldsymbol{A} - \hat{\boldsymbol{A}}(k)$는 남은 rank-1 행렬의 합으로 이루어져있음을 알 수 있다.

$$
\boldsymbol{A} - \hat{\boldsymbol{A}}(k) = \sum^r _{i=k+1} \sigma _i \boldsymbol u _i \boldsymbol{v _i}^\intercal  \tag{4.96}
$$

Theorem 4.24에 의해 difference matrix의 spectral norm이 $\sigma _{k+1}$임을 즉시 알 수 있다. (4.94)에 대해 더욱 자세히 들여다보자. 만약 다른 행렬 $\boldsymbol{B}$의 차원이 $k$보다 작거나 같다고 해보자.

$$
\| \boldsymbol{A} - \boldsymbol{B} \| _2 < \| \boldsymbol{A} - \hat{\boldsymbol{A}}(k) \| _2 \tag{4.97}
$$

그러면 최소한 $(n-k)$ 차원의 null space $Z \subseteq \mathbb R^n$이 존재하며, $\boldsymbol x \in Z$는 $\boldsymbol B \boldsymbol x = \boldsymbol 0$임을 의미한다. 이는 다음을 만족하며,

$$
\| \boldsymbol{A} - \boldsymbol{x} \| _2 = \| (\boldsymbol{A} - \boldsymbol{B}) \boldsymbol{x} \| _2 \tag{4.98}
$$

코시-슈바르츠 부등식 (3.17)을 통해 아래와 같음을 도출할 수 있다.

$$
\| \boldsymbol{A} - \boldsymbol{x} \| _2 \leq \| (\boldsymbol{A} - \boldsymbol{B}) \| _2 \| \boldsymbol{x} \| _2 < \sigma _{k+1} \|\boldsymbol x\| _2 \tag{4.99}
$$

그러나 $\| \boldsymbol{A} - \boldsymbol{x} \| _2 \geq \sigma _{k+1} \|\boldsymbol x\| _2$인 $(k+1)$의 부분공간이 존재하면, 이는 행렬 $ \boldsymbol{A}$의 right-singular vector $ \boldsymbol v _j,~ j \leq k+1$로 span한다. 이러한 두 공간의 차원을 합치면 $n$보다 크게 되는데, 이는 두 공간에 non-zero vector가 있음을 의미한다. 이는 앞서 살펴본 rank-nullity theorem과 모순된다.

The  Eckart-Young  theorem는 SVD를 이용하여 rank-$r$ 행렬의 차원을 rank-$k$차원의 행렬로 줄일 수 있음을 말해준다. 행렬의 rank-$k$ approximation은 손실압축으로 생각할 수 있다. 이는 머신러닝 분야의 이미지 처리, 노이즈 필터링, ill-posed probability의 정규화 등에서 확인할 수 있다. 또한 이는 Chapter 10의 PCA에서 중요한 역할을 하게된다.

## Matrix Phylogeny

![image](https://user-images.githubusercontent.com/47516855/118869228-1659ad80-b920-11eb-9272-53f7968bb5ac.png){: .align-center}{: width="600"}

위 그림은 여태까지 보았던 선형대수의 개념이다. 본 장에서 우리는 행렬과 선형변환에 대한 핵심적인 성질을 살펴보았다. 위 그림은 계통나무로, 서로 다른 형태의 행렬들의 관계를 표현한다.

검은색은 subset을 의미하며, 파랑색은 이를 수행하기 위한 개념을 의미한다.

우리는 모든 real matrix $\boldsymbol A \in \mathbb R^{n \times m}$을 다룬다. 직사각행렬에 대해서는 **SVD가 존재**한다. 

정사각 행렬 $\boldsymbol A \in \mathbb R^{n \times n}$을 살펴보면, **행렬식**은 **역행렬이 존재**하지 않는지를 알려준다고 했다. 즉, regular, invertible matrix의 클래스에 속하는지를 알 수 있다.

만일 행렬이 **non-defective**하다면, 고윳값분해가 가능하다 (Theorem 4.12). 중복된 고윳값(중근)은 defective matrix가 되고, 대각화가 불가능하다.

Non-singular, non-defective matrix는 같은 개념이 아니다. 회전행렬은 역행렬이 존재하지만 (행렬식이 0이 아님), 실수에서는 대각화가 불가능하다 (고윳값이 실수임을 보장할 수 없다).

$\boldsymbol A \boldsymbol A^{\intercal} = \boldsymbol A^{\intercal} \boldsymbol A $이 성립하면 $\boldsymbol A$는 **normal**이다. 또한, $\boldsymbol A \boldsymbol A^{\intercal} = \boldsymbol A^{\intercal} \boldsymbol A = \boldsymbol I $가 성립하면 $\boldsymbol A$는 **직교한다** (Definition 3.8). Orthogonal matrix의 집합은 regular matrix의 부분집합이고 $\boldsymbol A^{\intercal} = \boldsymbol A^{-1}$을 만족한다.

Normal matrix는 자주 접하는 부분집합이 있는데, 바로 대칭행렬 $\boldsymbol S \in \mathbb R^{n \times n}$이다. 대칭행렬은 $\boldsymbol S = \boldsymbol S^{\intercal}$을 만족한다. 대칭행렬은 오직 양의 고유값만을 갖는다. 대칭행렬의 부분집합은 positive definite matrix $\boldsymbol P$로, 모든 non-zero vector에 대해 $\boldsymbol x^{\intercal} \boldsymbol P \boldsymbol x$ > 0를 만족한다. 이 경우 유일한 Cholesky decomposition이 존재한다 (Theorem 4.18). Positive definite matrix는 오직 양의 고윳값만을 갖으며, 항상 역행렬이 존재한다 (즉, 행렬식이 0이 아니다).

Positive definite의 다른 부분집합으로는 **대각행렬**이 있다. 대각행렬은 곱셈과 덧셈에 대해 닫혀있지만 group을 형성할 필요는 없다 (역행렬이 가능한 경우에만 성립).