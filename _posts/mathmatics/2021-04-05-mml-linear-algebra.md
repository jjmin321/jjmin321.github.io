---
title:  "머신러닝을 위한 수학 정리: Vector, Matrix"
toc: true
toc_sticky: true
permalink: /project/mml/Linear-Algebra/
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

산업공학도로서 부끄러운 이야기지만 딥러닝, 머신러닝에서 사용하는 수학적인 개념에 대해 헷갈리는 경우가 꽤나 자주있다. 지방대다 보니 학부과정에서 그냥 넘어가는 경우도 있고, 공부를 잘하는 편이 아니기 때문에 내가 까먹은 것도 있지만, 어쨌든간에 석사학위를 보유하고 있는 엔지니어로서 참으로 부끄러운 일이라고 생각한다. 본 포스트는 이러한 문제점을 해결하고 나 스스로 레퍼런스로 삼을만한 것을 만들기 위해 기획하였다. 기본적인 선형대수학부터 통계, 정보이론, 최적화 등 딥러닝과 머신러닝 전반적으로 다루는 수학적인 개념에 대해 다룰 것이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.


## Background

![image](https://user-images.githubusercontent.com/47516855/113588310-f2415600-966a-11eb-900a-5fb63d56e293.png){: .align-center}{: width='500'}

머신러닝의 이론과 관련하여 위 그림과 같은 것들을 배우게 될 것이다.
- 선형대수: 데이터를 벡터와 매트릭스로 표현해야 함.
- 해석기하학: 두 개의 대상을 표현하는 두 개의 벡터에 대해 유사도를 정의해야 함. 이는 두 벡터가 비슷하다면, 머신러닝에 의해 비슷한 결과를 내기 때문임. 이러한 유사도를 표현하기 위해 두 개의 인풋을 받고, 이들의 유사도를 반환하는 operation이 필요함.
- 행렬분해: 이를 통해 데이터의 직관적인 해석과 효율적인 학습을 가능하게 함
- probability theory: 간혹 데이터를 true underlying signal에 대한 noisy observation으로 간주하는 경우가 있음. 머신러닝을 이용하여 이러한 노이즈는 제거하고 진짜 신호를 찾고 싶음. 이를 위해서는 노이즈를 정량화할 언어가 필요함. 또한, 종종 predictor가 일종의 불확실성을 표현하게 만들 필요가 있음 

대수와 선형대수가 다루는 것은 다음과 같다.
- 대수: 수학적 기호들과 이런 기호를 다루는 규칙에 대한 것
    - ax^2+bx+c=0
- 선형대수: 선형 방정식 (Ax=b)를 다루는 수학의 한 갈래

이 책에서 다루는 vector는 다음과 같다.
- 기하학적 vector $\boldsymbol v $: 스칼라곱(scaling),
- 다항식: 역시 addtion과 scaling을 만족. 기하학적 벡터보다는 좀 더 일반화된 형태
- Audio signal
- Elements of $\mathbb R^n$: n차원 실수차원에 포함되는 n쌍의 성분으로, 다항식보다도 더 일반화된 형태의 벡터이며, 본 과정에서 주로 다룰 예정

## Systems of Linear Equations

Systems of Linear Equations (선형 연립방정식)은 선형대수의 핵심개념이다. 다양한 문제들이 이를 통해 표현될 수 있으며, 선형대수는 이를 해결하는 도구를 제시한다.

선형 방정식의 real-valued system에 대해 일반적으로는 해가 없거나, 정확히 하나거나, 무수히 많은 결과를 얻게 된다. Chapter 9의 linear regression은 해가 없는 경우 적용할 수 있다.

## Matrix

선형방정식을 표현하는 용도와 section 2.7에서 다루게 될 선형함수 (linear functions, linear mapping)을 표현하는 용도로 사용된다.

## Solving Systems of Linear Equations

$\boldsymbol{A} \boldsymbol x = \boldsymbol b $ 형태의 linear equation을 풀어보며, solution의 종류, 다양한 matrix의 형태, variable의 종류 등을 살펴보자

### Particular and General Solution

다음은 linear equation의 solution의 종류이다

**Particular solution (Special solution)**  
$\boldsymbol{A} \boldsymbol x = \boldsymbol b $의 형태 (inhomogenous equation)에 대한 solution을 말하는 것으로 특정한 $\boldsymbol x$값을 갖는다.

$$
\boldsymbol x = 
\begin{bmatrix} 
x _1  \\ x _2 \\ x _3 \\ x _4 \\ x _5
\end{bmatrix}
= \begin{bmatrix} 
2  \\ 0 \\ -1 \\ 1 \\ 0
\end{bmatrix}
$$

$\boldsymbol{A} \boldsymbol x = \boldsymbol b $의 해는 존재할 수도, 아닐 수도 있다. 이는 가우스 소거법 등을 통해 구할 수 있다.

그러나 이는 본 선형시스템의 유일한 해가 아니다. 다른 해를 모두 찾으려면 $\boldsymbol 0$을 non-trivial하게 생성해내야 한다. 또한, $\boldsymbol 0$을 special solution에 더하는 것은 아무런 영향을 끼치지 않는다. 

> $\boldsymbol x _p$를 special solution이라 하고, $\boldsymbol x _n$을 $\boldsymbol{A} \boldsymbol x = \boldsymbol b $의 해라 하자. 그러면,
> 
> $$
\begin{align}
\boldsymbol A \boldsymbol x = \boldsymbol b \\ 
\boldsymbol A (\boldsymbol x _p + \boldsymbol x _n) = \boldsymbol b \\ 
\boldsymbol A \boldsymbol x _p + \boldsymbol A \boldsymbol x _n = \boldsymbol b \\ 
\boldsymbol A \boldsymbol x _p  = \boldsymbol b
\end{align}
$$
>
> 이를 통해 우리의 해에 $\boldsymbol x _n$를 더하는 것은 아무런 영향을 끼치지 않음을 확인하였다. 따라서 이러한 $\boldsymbol x _n$을 우리의 해에 더해주게 될 것이다.

**Homogeneous Solution (Null Space Solution)**  
$\boldsymbol{A} \boldsymbol x = 0 $의 형태에 대한 solution으로, kernel 혹은 null space 상의 basis of solution이라 부른다. 이는 Vector space의 subspace이다.

$$
\lambda _1 \begin{bmatrix} 
2  \\ 1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
, \lambda _2 \begin{bmatrix} 
2  \\ 0 \\ -1 \\ 2 \\ 1
\end{bmatrix}, \textrm{where }\lambda _1, \lambda _2 \in \mathbb R
$$

**General Solution (Complete Solution)**  
Particular solution과 homogeneous solution을 합친 것으로, $\boldsymbol{A} \boldsymbol x = \boldsymbol b $에 대한 vector space 상의 모든 해를 의미한다.

$$
\boldsymbol x = 
\begin{bmatrix} 
2  \\ 0 \\ -1 \\ 1 \\ 0
\end{bmatrix}
+
\lambda _1 \begin{bmatrix} 
2  \\ 1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
+ 
\lambda _2 \begin{bmatrix} 
2  \\ 0 \\ -1 \\ 2 \\ 1
\end{bmatrix}, \textrm{where }\lambda _1, \lambda _2 \in \mathbb R
$$

>
> 선형시스템 $\boldsymbol A \boldsymbol x = \boldsymbol b $의 해를 구하는 방법은 $\boldsymbol A$에 따라 달라진다. $\textrm{rk}(A) = r$, $m$를 row의 갯수, $n$을 column의 갯수라 할 때,
> 
> ### Full Rank ($r= m = n $)
> 가장 익숙한 경우인데, $\boldsymbol A $가 정사각 행렬인 경우로, 정의에 의해 모든 column vector가 linearly independent한 경우를 의미한다. 이 때 fundamental space를 그려보면 다음과 같은 모양이 그려진다. 
>
> ![image](https://user-images.githubusercontent.com/47516855/115241805-2f721180-a15c-11eb-8203-200e473ea64e.png){: .align-center}{: width="700"}
> 
> 그림의 null space/left null space는 0벡터가 된다 (the column vectors are linearly independent each other). $\boldsymbol b$는 항상 $\boldsymbol A$의 column space 내에 있고, 이들은 단 한 개의 해를 갖는다.
> 
> ### Full Column Rank ($r = n < m $)
> $\boldsymbol A $가 상하로 길게 늘어선 모양의 행렬로, rank가 column vector의 갯수와 같은 경우를 의미한다. 이런 경우를 full column rank라 한다. 이때 fundamental space를 그려보면 다음과 같다.
> 
> ![image](https://user-images.githubusercontent.com/47516855/115242213-9e4f6a80-a15c-11eb-82d9-5931376de16b.png){: .align-center}{: width="700"}
> 
> 이때 nullspace는 오직 0 벡터만을 포함한다. column vector가 부족하기 때문에 $\boldsymbol b$가 column space에 없을 수도 있다. 이럴 때는 least square를 사용하여 최적해를 구하게 된다.
>
> ### Full Row Rank ($r = m < n $)
> 이 경우는 좌우로 긴 직사각 행렬이면서 행의 갯수와 랭크가 일치하는 경우이다. 
> 
>![image](https://user-images.githubusercontent.com/47516855/115409545-cdcda800-a22c-11eb-932b-9f15d34a62ea.png){: .align-center}{: width="700"}
>
> 이때 left-nullspace는 오직 영벡터만을 포함한다. 열벡터가 매우 많기 때문에 $\boldsymbol b$는 항상 열공간 위에 존재한다. 그렇기 때문에 $\boldsymbol A \boldsymbol x = \boldsymbol b $의 해가 존재한다. 그러나 셀 수 없이 많은 해가 존재한다. 이는 complete solution이 particular solution과 Homogeneous Solution의 합으로 이루어지기 때문이다. (앞선 케이스에서는 null space가 영벡터인 경우)
>
> ### $r = m < n $
> 가장 일반적인 경우이다. 이 경우 $C(A)$에 $\boldsymbol b$가 존재할 경우 infinite many solution을, 없을 경우 해를 갖지 않는다.
>
>![image](https://user-images.githubusercontent.com/47516855/116277446-6f6e6f80-a7c0-11eb-8f2a-84e0728aac0a.png){: .align-center}{: width="700"}
>
> 출처: [[선형대수학] 선형연립방정식(Ax = b)의 해 구하기.txt](https://bskyvision.com/271?category=619292)


## Vectors

벡터는 어떤 객체로, 서로 더할 수 있거나 스칼라를 곱할 수 있으며, 그 결과는 여전히 벡터인 것을 의미한다. 이를 자세하게 살펴보자.

## Vector space

<div class="notice--warning" markdown="1">

**Definition 2.9** (Vector Space)

Real-valued **vector space** $V = (\mathcal V, +, \cdot)$는 다음 두 개의 연산과 집합 $\mathcal V$이다.

$$
\begin{align}
+: \mathcal V + \mathcal V \tag{2.62} \\
\cdot : \mathbb R \times \mathcal V \\
\end{align}
$$

where
1. $(\mathcal V, +)$는 Abelian group이다.
2. Distributivity:
    - $\forall \lambda \in \mathbb R , \lambda ( \boldsymbol x + \boldsymbol y) = \lambda \boldsymbol x + \lambda \boldsymbol y $
    - $\forall \lambda, \psi \in \mathbb R , (\lambda + \psi) \boldsymbol x = \lambda \boldsymbol x + \psi \boldsymbol x $
3. Associativity (outer operation):
    - $\forall \lambda \in \mathbb R, \lambda (\psi \boldsymbol x) = (\lambda \psi) \boldsymbol x $
4. 곱셈에 대한 항등원이 존재
    - $ 1 \cdot \boldsymbol x = \boldsymbol x$

</div>

여기서 원소 $\boldsymbol x \in V$를 **vector**라고 한다.

### Vector Subspace

Vector subspace는 vector space의 부분 집합으로, 머신러닝에서 중요한 개념이다. 이후에 배울 dimensionality reduction에서 중요한 역할을 한다.

<div class="notice--warning" markdown="1">

**Definition 2.10** (Vector Subspace)

$V=(V, +, \cdot )$을 vector space라 할 때, subspace는 $ U \subseteq V, U \neq 0 $인, vector space의 부분 집합이라 볼 수 있다. 따라서, vector space의 성질을 그대로 따른다 ($U=(U, +, \cdot )$). 이는 "닫혀 있다"고 말한다. 

</div>



## Linear independence

Vector space의 기준이 되는 vector의 집합을 basis라 하고, 이에 vector를 더하거나, scaling하여 vector space 전부를 표현할 수 있다.

<div class="notice--warning" markdown="1">

**Definition 2.11** (Linear Combination)

어떤 벡터공간 $V$와 유한차원의 벡터 $\boldsymbol x _1, \cdots, \boldsymbol x _k \in V $를 생각해보자. 그러면 모든 $\boldsymbol v \in V$와 $\lambda _1, \cdots, \lambda _k \in \mathbb R, k \in \mathbb N$에 대해 다음을 벡터 $\boldsymbol x _1, \cdots, \boldsymbol x _k$의 **Linear combination**이라고 부른다.

$$
\boldsymbol v = \lambda _1 \boldsymbol x _1 + \cdots + \lambda _k \boldsymbol x _k = \sum ^k _{i=1} \lambda _i \boldsymbol x _i \in V \tag{2.65}
$$

</div>

0 vector는 $ 0 = \sum ^k _{i=1} 0 x _i$가 성립하기 때문에 항상 참이다 (trivial linear combination). 따라서 우리는 non-trivial한 케이스에 관심을 갖도록 한다.

<div class="notice--warning" markdown="1">

**Definition 2.12** (Linear (In)dependence).

어떤 벡터공간 $V$와 $k \in \mathbb N$, $\boldsymbol x _1, \cdots, \boldsymbol x _k \in V $를 고려해보자. 만일 non-trivial linear combination을 통해 $ 0 = \sum ^k _{i=1} \lambda _i x _i$를 표현할 수 있다면, $\boldsymbol x _1, \cdots, \boldsymbol x _k$는 **linearly independent**하다

</div>

이는 선형독립인 벡터들은 중복이 없다는 것과 마찬가지이며, 만약 하나를 제거한다면 우리는 무엇인가를 잃게 된다.

<div class="notice" markdown="1">
*Remark.* 다음 성질들은 벡터가 linearly independent한지 파악하는데 도움이 된다. 

- k 개의 벡터는 무조건 선형독립이거나, 선형독립이 아니다. 세번째 옵션은 없다.
- vector $\boldsymbol x _1, \cdots, \boldsymbol x _k$중 하나라도 0벡터라면, linearly dependent하다. 두 벡터가 같은 경우에도 이는 성립한다.
- 오직 trivial solution ($\lambda _i = 0$)만 있다면, vector $ x _1, \cdots x _k$는 linearly independent하다.
- Linearly dependent한 경우 span하는데에 있어 불필요한 vector가 있다는 것을 의미한다.
</div>



## Basis and Rank

### Generating Set and Basis

벡터공간 $V$에 대해, 우리가 특별하게 관심을 갖는 것은 어떤 특정한 벡터의 집합 $\mathcal A$로, 이의 선형조합을 통해 어떠한 $\boldsymbol v \in V$를 얻을 수 있다.

<div class="notice--warning" markdown="1">

**Definition 2.13** (Generating set and span(생성원))

어떤 벡터공간 $V = ( \mathcal V, +, \cdot)$과 집합 $\mathcal A = \{\boldsymbol x _1, \cdots,  \boldsymbol x _k \} \subseteq \mathcal V$을 고려해보자. 만일 $\boldsymbol v \in \mathcal V$가 $\{\boldsymbol x _1, \cdots,  \boldsymbol x _k \}$의 선형조합으로 표현될 경우, $\boldsymbol{A}$를 $V$에 대한 **generating set**이라 한다. $\mathcal A$의 linear combination으로 만들어지는 집합을 **span** of $\mathcal A$라 하고, $\mathcal A$가 vector space $V$로 span한 것을 $V=\text{span}(\mathcal A)$ 혹은 $V=\text{span}[\boldsymbol x _1, \cdots,  \boldsymbol x _k]$ 로 표현한다.

</div>

Generating set은 벡터공간을 span하는 벡터의 집합이다. 즉, 해당 vector들의 linear combination은 generating set의 모든 vector를 표현할 수 있다. 이제 더욱 구체적으로 가장 작은 generating set에 대해 살펴보자.

<div class="notice--warning" markdown="1">

**Definition 2.14** (Basis(기저))

Vector space $V=(\mathcal V, +, \cdot )$, set of vectors $\mathcal{A} =  \subseteq \mathcal V$를 고려해보자. $\mathcal V$의 generating set $\mathcal{A}$는 $V$를 span하는 더 작은 집합 $\tilde{\mathcal A} \subseteq \mathcal A \subseteq \mathcal V$이 존재하지 않으면 **minimal**하다고 한다. 모든 선형독립인 generating set은 minimal하고, 이를 $V$의 **basis(기저)**라 한다.

</div>

$V=(V, +, \cdot)$, set of vectors $\mathcal B \subseteq V, \mathcal B \neq \emptyset$ 일 때 아래의 명제들은 동치이다.
- $\mathcal B$는 $V$의 기저이다.
- $\mathcal B$는 minimal generating set이다.
- $\mathcal B$는 $V$의 선형독립인 벡터의 집합이다. 즉, 어떠한 벡터를 더하면, 이 집합은 더 이상 선형독립이 아니다.
- 모든 vector $\boldsymbol x \in V$는 $\mathcal B$의 linear combination이고, 각각의 linear combination은 유일하다.

*Remark*. basis는 유일하지 않다. 즉, 수 많은 basis가 있고, 각 각의 basis는 서로 다르다.
{: .notice}

여기서는 오직 유한한 차원만을 다루며, 이 경우 vector space의 차원은 basis vector의 수이며, 이를 $\textrm{dim}(V)$라 쓴다. $V$의 subspace $U$에 대해 $\textrm{dim}(U) \leq \textrm{dim}(V)$이고, $U = V$ 일때, $\textrm{dim}(U) = \textrm{dim}(V)$이 성립한다. 직관적으로 vector space의 차원은 vector space 상의 독립적인 방향의 수이다.

*Remarks*. Vector space의 차원은 반드시 vector element의 수와 동일한 것은 아니다.
{: .notice}

### Rank

$\boldsymbol{A} \in \mathbb R^{m \times n}$의 linearly independent column의 수와 linearly independent row의 수는 같고, 이를 **rank**라 하며, $\textrm{rk}(\boldsymbol{A})$로 표현한다.

행렬의 랭크는 중요한 성질이 있다.  
- $\textrm{rk}(\boldsymbol{A})$ = $\textrm{rk}(\boldsymbol{A}^\intercal)$. 즉, column rank와 row rank는 같다.
- $\boldsymbol{A} \in \mathbb R^{m \times n}$의 column은 subspace $U \in \mathbb R^m$을 span하고, $\textrm{dim}(U)$는 $\textrm{rk} (\boldsymbol{A})$와 같다.
    - 이 subspace는 추후에 **image** 혹은 **range(치역)**로 부를 것이다.
    - $\boldsymbol{A}$의 basis는 gaussian elimination을 통해 알 수 있다.
- $\boldsymbol{A}^\intercal \in \mathbb R^{m \times n}$의 row는 subspace $W \in \mathbb R^n$을 span하고, $\textrm{dim}(W)$는 $\textrm{rk} (\boldsymbol{A})$와 같다.
    - 이는 $\boldsymbol{A}^\intercal$의 gaussian elimination을 통해 알 수 있다.
- $\boldsymbol{A} \in \mathbb R^{n \times n}$의 rank가 n이면 \boldsymbol{A}는 **regular matrix** 혹은 **invertible matrix**라 부른다
- $\boldsymbol{A} \in \mathbb R^{m \times n}$이고, $\boldsymbol b \in \mathbb R^m$일 때, $\textrm{rk}(\boldsymbol{A}) = \textrm{rk}(\boldsymbol{A} \rvert \boldsymbol b)$ (augmented form)이면, linear equation으로 해를 도출할 수 있다.
- $\boldsymbol{A} \in \mathbb R^{m \times n}$에 대해 $\boldsymbol{A} \boldsymbol x = \boldsymbol 0 $에 대한 subspace의 차원은 dimension n - rk(\boldsymbol{A})이며, 이를 **kernel (핵)** 혹은 **null space**라 표현한다.
- Matrix $\boldsymbol{A} \in \mathbb R^{m \times n}$가 **full rank**를 갖으면, rank는 matrix의 가장 큰 차원과 같다. 즉, min(m, n)이 된다.
    - Full rank를 갖지 않으면 **rank deficient**하다고 표현한다.

## Linear mapping

여기서는 vector space의 구조를 보존하는 mapping에 대해 살펴보자. 이로서 우리는 좌표에 대한 개념을 정의할 수 있다. 본 챕터의 초창기에서 벡터는 더하거나 스칼라곱을 해도 여전히 벡터인 어떤 객체를 나타낸다고 했다. 따라서 어떤 mapping을 적용했을 때 이러한 성질을 만족해야 한다. 두 개의 real vector space $V, W$를 고려해보자. Mapping $\Phi: V \to W$는 다음을 만족하면 벡터공간의 성질을 보존한다.

$$
\begin{align}
\Phi(\boldsymbol x + \boldsymbol y) = \Phi(\boldsymbol x) + \Phi(\boldsymbol y)  \tag{2.85} \\
\Phi(\lambda \boldsymbol x) = \lambda \Phi(\boldsymbol x) \tag{2.86}
\end{align}
$$

이는 또한 linearity의 정의이기도 하다. 이를 요약하면 다음과 같다.

<div class="notice--warning" markdown="1">

**Definition 2.15** (Linear Mapping)

Vector space $V, W$에 대해, 다음을 만족하는 mapping $\Phi: V \to W$를 **linear mapping (선형 사상) (or Linear transformation (선형변환), vector space homomorphism (준동형사상))**이라 한다.

$$
\forall \boldsymbol x, \boldsymbol y \in V, \forall \lambda, \psi \in \mathbb R: \Phi(\lambda \boldsymbol x + \psi \boldsymbol y) = \lambda \Phi(\boldsymbol x) + \psi \Phi(\boldsymbol y) \tag{2.87}
$$

</div>

머지않아 linear mapping을 matrix로 표현하는 방법을 배울 것이다. matrix로 표현하는 경우, 이를 **matrix representation (행렬 표현)**이라 한다.

<div class="notice--warning" markdown="1">

**Definition 2.16** (Injective(단사), Surjective(전사), Bijective(전단사))

임의의 집합 $\mathcal V, \mathcal W$에 대한 mapping $\Phi: \mathcal V \to \mathcal W$을 고려해보자. 그러면 $\Phi$는,
- **Injective(one-to-one, 단사)**: $\Phi(\boldsymbol x) = \Phi(\boldsymbol y) $이면, $\boldsymbol x = \boldsymbol y$. 즉, $\text{Im}(f) =Y$이다.
    - 만일 $\Phi$가 subjective하면, $\Phi$를 이용해 $\mathcal V$를 이동시켜 $\mathcal W$의 모든 요소를 나타낼 수 있다. 
- **Subjective(onto, 전사)**: $\Phi(\mathcal V) = \mathcal W $
- **Bijective(one-to-one correspondence, 전단사)**: Injective and Subjective
    - Bijective $\Phi$는 되돌려질 수 있다 (undone). 즉, $\Phi(\boldsymbol x)$를 다시 $\boldsymbol x$로 보내는 linear mapping $\Psi: \mathcal W \rightarrow \mathcal V$가 존재한다. (i.e. $\Psi \circ \Phi (\boldsymbol x) = \boldsymbol x$) 이러한 mapping $\Psi$는 $\Phi$의 inverse라 하고, $\Phi^{-1}$로 표현한다.

</div>

> 아래 그림은 injective, surjective, bijective에 대한 그림이다.
> 
> ![image](https://user-images.githubusercontent.com/47516855/113878484-99052e00-97f4-11eb-878a-dbec54402dd8.png){: .align-center}{: width='600'}

이와 같은 정의를 이용하여, vector space $V, W$ 사이 linear mapping에 대한 특수한 경우를 살펴보자
- **Isomorphism(동형사상)** : $\Phi: V \rightarrow W$ linear and bijective
    - 따라서 $V$에서의 vector addition과 scalar multiplication은 $W$의 연산으로 완전히 전환되어 계산할 수 있다.
    - $T(c \boldsymbol v _1 + \boldsymbol v _2) = cT(\boldsymbol v _1) + T(\boldsymbol v _2)$
- **Endomorphism(자기사상)** : $\Phi: V \rightarrow V$ linear
    - 즉, 시작과 끝이 같다 (정의역=공역)
- **Automorphism(자기동형사상)** : $\Phi: V \rightarrow V$ linear and bijective
- $\textrm{id}_V: V \rightarrow V, \boldsymbol x \rightarrow \boldsymbol v$를 $V$의 **identity mapping (항등 사상, identity automorphism)**이라 한다.

**Theorem 2.17**
유한한 차원의 vector space $V, W$에 대해 $\textrm{dim} (V) = \textrm{dim} (W)$과 isomorphic (동형)는 동치이다 (iff).
{: .notice--info}

Theorem 2.17 차원이 같은 vector space 내에 linear, bijective mapping이 존재한다는 것을 이야기한다. 즉, 차원이 같은 vector space는 같은 것이며, 적절한 변환을 통해 손실없이 변환된다는 뜻이다. 이때 이러한 mapping을 $V$와 $W$사이의 isomorphism이라고 부른다.  

이는 또한 $\mathbb R^{m \times n}$과 $\mathbb R^{mn}$을 같은 차원으로 취급하게 해주며, 이에도 역시 linear, bijective mapping이 존재한다

<div class="notice" markdown="1">
*Remark.* vector space $V, W, X$를 고려해보자. 그러면,
- linear mapping $\Phi: V \rightarrow W$과 $\Psi: W \rightarrow  X$에 대해 mapping $\Psi \circ \Phi: V \rightarrow  X$ 또한 linear하다.
- $\Phi: V \rightarrow W$이 isomorphic하면, $\Phi^{-1}: W \rightarrow V$ 또한 isomorphic하다.
- $\Phi: V \rightarrow W$과 $\Psi: W \rightarrow  X$이 linear하면, $\Phi + \Psi$, $\lambda \Phi$ 또한 linear하다.
</div>

### Matrix Representation of Linear Mappings

어떠한 n차원 vector space는 $\mathbb R^n$과 isomorphic하다 (Theorem 2.17). n차원 vector space $V$ 내에 있는 basis $\{ \boldsymbol b _1, \cdots, \boldsymbol b _n \}$를 고려해보자. 이 경우 basis의 순서가 중요해진다. 따라서 앞으로 $V$의 basis가 $\{\boldsymbol b _1, \boldsymbol b _2, \cdots, \boldsymbol b _n \}$ 이라고 할 때, 이를 정렬하여 n-tuple로 나타낸 것을 **ordered basis** of V라 하고, $B=(\boldsymbol b _1, \boldsymbol b _2, \cdots, \boldsymbol b _n)$로 정리한다.

<div class="notice--warning" markdown="1">

**Definition 2.18** (Coordinate)

Vector space V와 그에 해당하는 ordered basis B가 있을 때, V에 포함하는 모든 벡터 $\boldsymbol x$는 linear combination으로 unique하게 표현가능하다.

$$
\boldsymbol x = \alpha _1 \boldsymbol b _1 + \cdots + \alpha _n \boldsymbol b _n \tag{2.90}
$$

</div>

그러면 $\boldsymbol x$의 $B$에 대한 **coordinate(좌표)**는 $\alpha _1, \cdots, \alpha _n$이 된다. 다음 vector는 $\boldsymbol x$의 ordered basis $B$에 대한 **coordinate vector/coordinate representation**가 된다.

$$
\boldsymbol \alpha =
\begin{bmatrix} 
 \alpha _1  \\ \vdots \\ \alpha _5
\end{bmatrix}
\in \mathbb R^n
$$

N차원 vector space $V$, $V$의 ordered basis $B$, $\Phi: \mathbb R^n \rightarrow V$인 mapping $\Phi$가 있을 때, n차원 실수 공간의 단위 벡터 $(e _1, \cdots, e _n)$은 $\Phi$에 의해 $(b _1, b _2, \cdots, b _n)$로 mapping된다.

> 예를 들어 $(-3, 5)$는 $-3\boldsymbol e _1 + 5\boldsymbol e _2$로 표현이 가능하다. 그러나 이를 ordered basis로 표현한다면 기저는 무시하고 그 앞의 스칼라로만 표현할 수 있다.

*Remark.* 어떤 n차원 vector space와 ordered basis $B$에 대해 mapping $\Phi: \mathbb R^n \to V, \Phi(\boldsymbol e _i) = \boldsymbol b _i, i = 1, \cdots, n$,는 linear하고, 또한 Theorem 2.17 때문에 isomorphic하다.
{: .notice}

그러면 이제 우리는 finite-dimensional vector spaces간의 선형사상들과 행렬들을 명시적으로 연결할 준비가 되었다.

<div class="notice--warning" markdown="1">

**Definition 2.19** (Transformation Matrix(변환행렬))

Vector space $V, W$가 있고, 그에 대한 ordered basis $B, C$가 각각 있을 때, $\Phi : V \rightarrow W$인 $\Phi$가 있으면, $C$로 $\Phi (b _j)$를 다음과 같이 unique하게 표현할 수 있다.

$$
\Phi (b _j) = \alpha _{1j} \boldsymbol c _j + \cdots + \alpha _{mj}\boldsymbol c _j = \sum ^m _{i=1} \alpha _{ij}\boldsymbol c _i \tag{2.92}
$$
</div>

이때 모든 j에 대해 $\alpha$를 모으면 **transformation matrix** $\boldsymbol{A} _\Phi (i, j) = \alpha _{ij}$를 만들 수 있다.


### Basis change

이제 $V, W$의 basis가 변화할 때 $\Phi: V \rightarrow W$의 transformation matrix가 어떻게 변화하는지 자세히 살펴보자. $V, W$에 대해 각기 두개의 ordered basis를 살펴보자:

$$
\begin{align}
B=(\boldsymbol b _1, \boldsymbol b _2, \cdots, \boldsymbol b _n), \tilde{B}=(\tilde{\boldsymbol b _1}, \tilde{\boldsymbol b _2}, \cdots, \tilde{\boldsymbol b _n}) \tag{2.98} \\
C=(\boldsymbol c _1, \boldsymbol c _2, \cdots, \boldsymbol c _n), \tilde{C}=(\tilde{\boldsymbol c _1}, \tilde{\boldsymbol c _2}, \cdots, \tilde{\boldsymbol c _n}) \tag{2.99}
\end{align}
$$

또한, $\boldsymbol{A} _\Phi \in \mathbb R^{m \times n}$를 기저 $B, C$에 대한 사상의 변환 행렬이라하자. 마찬가지로 $\tilde{\boldsymbol{A}} _\Phi \in \mathbb R^{m \times n}$를 기저 $\tilde{B}, \tilde{C}$에 대한 사상의 변환 행렬이라하자. 다음을 통해 어떻게 $A와 \tilde A$가 연결되어 있는지, 즉, $B, C$로부터 $\tilde{B}, \tilde{C}$로의 basis change를 수행하여 어떻게 $\boldsymbol{A}$를 $\tilde{\boldsymbol{A}}$로 변환할 수 있는지 살펴보도록 한다.

*Remarks.* identity mapping $\text{id}_v$에 대해 서로 다른 coordinate representation을 생각해보자. 이는 즉, $(\boldsymbol{e _1}, \boldsymbol{e _2})$에 대한 좌표를 $(\boldsymbol{b _1}, \boldsymbol{b _2})$로 mapping하는 것으로 이해할 수 있다. 기저와 이에 대응하는 벡터의 표현의 변화를 통해, 이런 새로운 기저를 통해 변환 행렬을 더욱 간략하게 만들어 직관적인 계산을 하게 만들 수 있다.
{: .notice}

이 다음에 어떤 기저에 대한 좌표 벡터를 다른 기저에 대한 좌표 벡터로 변환하는 사상에 대해 살펴볼 것이다. 우선 주요 결론을 먼저 기술하고, 이후 이에 대한 설명을 이어가도록 하겠다.

<div class="notice--info" markdown="1">

**Theorem 2.20** (Basis Change)  
앞선 예제와 마찬가지로 vector space $V, W$와 각각의 ordered basis $B, C$와 linear mapping $\Phi: V \rightarrow W$를 생각해보자.

$$
\begin{align}
B=(\boldsymbol b _1, \boldsymbol b _2, \cdots, \boldsymbol b _n), \tilde{B}=(\tilde{\boldsymbol b _1}, \tilde{\boldsymbol b _2}, \cdots, \tilde{\boldsymbol b _n}) \tag{2.103} \\
C=(\boldsymbol c _1, \boldsymbol c _2, \cdots, \boldsymbol c _n), \tilde{C}=(\tilde{\boldsymbol c _1}, \tilde{\boldsymbol c _2}, \cdots, \tilde{\boldsymbol c _n}) \tag{2.104}
\end{align}
$$

$B, C$에 대한 $\Phi$의 변환행렬 $\boldsymbol A _\Phi$, 그리고 이에 대응하는 $\tilde B, \tilde C$에 대한 $\tilde{\boldsymbol{A} _\Phi}$이 다음과 같이 주어졌을 때,

$$
\tilde{\boldsymbol{A _\Phi}} = \boldsymbol{T^{-1}} \boldsymbol{A _\Phi} \boldsymbol{S} \tag{2.105}
$$  

$\boldsymbol{S}$는 $\text{id} _V$의 변환행렬로, $\tilde B$에 대한 좌표를 $B$로 맵핑한다. 또한, $\boldsymbol{T}$는 $\text{id} _W$의 변환행렬로, $\tilde C$에 대한 좌표를 $C$로 맵핑한다.

</div>

![image](https://user-images.githubusercontent.com/47516855/117107051-ef499a80-adbb-11eb-92b0-87a70e0826a8.png){: .align-center}{: width="500"}

위 그림은 다음을 설명한다. 어떤 homomorphism $\Phi: V \to W$와 $V$의 ordered basis $B, \tilde{B}$, $W$의 ordered basis $C, \tilde{C}$를 생각해보자. Mapping $\Phi _{CB}$는 $B$의 기저벡터를 $C$의 기저벡터의 선형조합으로 map한다. 우리가 ordered basis $B, C$에 대한 $\Phi _{CB}$의 변환행렬 $\boldsymbol{A _{\Phi _{CB}}}$를 안다고 가정하자. $V$의 $B$를 $\tilde{B}$로, $W$의 $C$를 $\tilde{C}$로 기저변환을 수행하면, 이에 대응하는 변환행렬 $\tilde{\boldsymbol{A}} _{\Phi _{CB}}$를 다음과 같이 결정할 수 있다. 우선 선형변환 $\Psi _{B\tilde{B}}: V \to V$의 행렬표현을 찾을 수 있다. 이는 새로운 기저 $\tilde{B}$에 대한 좌표를 **이전** 기저 $B$ (V)에 대한 (유일한) 좌표로 map할 수 있다. 그러면 $\Phi _{CB}$의 변환행렬 $\boldsymbol{A} _{\Phi}: V \to W$를 이용하여 V의 이전 기저에 대한 좌표를 $W$ 내의 $C$에 대한 좌표로 변환할 수 있다. 마지막으로, 선형사상 $\Xi _{\tilde{C}C}: W \to W$를 이용하여 $C$에 대한 좌표를 $\tilde{C}$로 map하게 된다. 그러므로 선형사상 $\Phi _{\tilde{C}\tilde{B}}$를 **이전** 기저를 포함하는 선형사상의 조합으로 표현할 수 있다.

$$
\Phi _{\tilde{C}\tilde{B}} = \Xi _{\tilde{C}C} \circ \Phi _{CB} \circ \Psi _{B\tilde{B}} = \Xi ^{-1} _{C\tilde{C}} \circ \Phi _{CB} \circ \Psi _{B\tilde{B}} \tag{2.114}
$$

구체적으로, $\Psi _{B\tilde{B}}=\text{id} _V$ $\Xi _{C\tilde{C}} = \text{id} _W$를 사용한다. 즉, 벡터들을 자기 자신으로 맵핑하지만, basis가 달라지게 된다.

<div class="notice--warning" markdown="1">

**Definition 2.21** (Equivalence)

$\tilde{\boldsymbol{A}} = \boldsymbol{T^{-1}} \boldsymbol{A} \boldsymbol{S}$인 regular matrix $S \in \mathbb R^{n \times n}$와 $T \in \mathbb R^{m \times m}$이 존재하면 두 matrix $\boldsymbol{A}, \tilde{\boldsymbol{A}} \in \mathbb R^{m \times m}$은 서로 **equivalent**하다고 한다.
</div>

<div class="notice--warning" markdown="1">

**Definition 2.22** (Similarity)

$\tilde{\boldsymbol{A}} = S^{-1}\boldsymbol{A}S$인 regular matrix $S \in \mathbb R^{n \times n}$가 존재하면, 두 matrix $\boldsymbol{A}, \tilde{\boldsymbol{A}} \in \mathbb R^{n \times n}$은 서로 **similar**하다고 한다.
</div>

*Remarks.* Similar matrix는 항상 equivalent하지만, equivalent matrix는 항상 similar하지 않다 (similar $\subset$ equivalent)
{: .notice}

> 뭔가 말이 어렵다. 간단하게 설명하자면 각각의 vector space마다 basis가 다른 경우를 생각해보자. 어떤 벡터를 여러 공간에서 표현할 때, 서로의 기저가 다르므로 단순하게 벡터의 값만으론 같은 벡터를 표현하기 어려울 것이다 (선형변환을 통해 기저벡터가 변화하므로). 이러한 동일성을 유지하는 것이 이러한 basis change의 핵심 개념이라고 생각하면 된다. 

## Image and Kernel

선형사상의 Image와 kernel은 subspace로, 특정 중요한 성질을 갖고있다. 다음을 통해 이를 살펴보자.

<div class="notice--warning" markdown="1">

**Definition 2.23** (Image and Kernel).

$\Phi: V \to W$에 대해, Kernel (핵)/null sapce(영공간)은 다음과 같이 정의된다.

$$
\textrm{ker} (\Phi) := \Phi^{-1}(\boldsymbol 0 _W) = \{\boldsymbol v \in V: \Phi(\boldsymbol v) = \boldsymbol 0 _W\} \tag{2.122}
$$

Image(상)/range(치역)은 다음과 같다.  

$$
\textrm{Im} (\Phi) := \Phi (V) = \{\boldsymbol w \in W \rvert \exists \boldsymbol v \in V: \Phi(\boldsymbol v) = \boldsymbol w\}
$$

</div>

직관적으로 kernel은 Vector space $V$ 안에 있는 subset이 vector space W로 mapping될 때, $0 _w$가 되는 vector space다. Image는 vector space $V$의 벡터가 vector space W의 어떤 subset으로 mapping되는 경우이다. 이에 대한 그림이 아래에 있다.

![image](https://user-images.githubusercontent.com/47516855/114015685-ef31aa00-98a4-11eb-9368-beb7bb4e8df3.png){: .align-center}{: width="500"}

<div class="notice" markdown="1">
*Remarks.* 선형사상 $\Phi: V \to W$를 생각해보자.
- 언제나 $\Phi(\boldsymbol 0 _V)=\boldsymbol 0 _W$가 성립한다. 그러므로 $\boldsymbol 0 _V \in \textrm{ker} (\Phi)$가 성립한다. 특히 null space는 절대로 빈 공간이 될 수 없다.
- $\textrm{Im} (\Phi) \subseteq W$는 $W$의 subspace이다.
- $\Phi$는 $\textrm{ker} (\Phi) = \{\boldsymbol 0 \}$인 경우에만 injective하다.
</div>

<div class="notice" markdown="1">
*Remark*. (Null Space and Column Space). $\boldsymbol{A} \in \mathbb R^{m \times n}$과 linear mapping $\Phi : \mathbb R^n \to \mathbb R^m, \boldsymbol x \mapsto \boldsymbol A \boldsymbol x$이 있을 때 다음이 성립한다.
- $\boldsymbol{A}$를 $\boldsymbol{A}$의 column vector로 표현해 $\boldsymbol{A} = [a _1, \cdots, a _n]$으로 나타내면 다음 식이 성립한다.

    $$
    \begin{align}
    \textrm{Im} (\Phi) &=\{ \boldsymbol{A} \boldsymbol x: \boldsymbol x \in \mathbb R^n = \{ \sum^n _{i=1} x _i a _i : x _1, \cdots , x _n \in \mathbb R\} \tag{2.124a} \\
    &= \textrm{span}[a _1, \cdots, a _n] \subseteq \mathbb R^m \tag{2.124b}
    \end{align}
    $$
- 즉, image는 $\boldsymbol{A}$의 column vector의 span이다. 이는 **column space**라 부른다. 따라서 column space (image)는 $\mathbb R^m$의 subspace이다 ($m$은 matrix의 "높이")
- $\textrm{rk}(\boldsymbol{A})=\textrm{dim}(\textrm{Im}(\Phi))$이다
- Kernel/null space인 $\textrm{ker} (\Phi)$는 $\boldsymbol{A} \boldsymbol x = 0 $과 같은 homogeneous system of linear equation의 **일반해**이다.
- kernel은 $\mathbb R^n$의 subspace이다. ($n$은 matrix의 "가로"이다.)
- Kernel은 column들 간의 관계에 집중한다. 이를 이용하여 한 column을 다른 column의 선형조합으로 나타낼 수 있는지, 어떻게 나타내는지 결정할 수 있다.
</div>


**Theorem 2.24** (Rank-Nullity Theorem(차원정리))  
Vector space $V, W$와 $\Phi: V \rightarrow W$에 대해 다음이 성립한다.  
$$
\textrm{dim}(\textrm{ker}(\Phi)) + \textrm{dim}(\textrm{Im}(\Phi)) = \textrm{dim}(V) \tag{2.129}
$$
{: .notice--info}

rank-nullity theorem은 또한 **linear mapping의 주요 이론**이라고도 부른다. 다음은 rank-nullity theorem와 직접적인 연관이 있다.
- $\textrm{dim}(\textrm{Im}(\Phi)) < \textrm{dim}(V)$이면, $\textrm{ker}(\Phi)$가 non-trivial하다. 즉, kernel이 0 이외의 값을 갖고, $\textrm{dim}(\textrm{ker}(\Phi))$가 1이상이다.
- $\textrm{dim}(V) = \textrm{dim}(W)$이면, $\Phi$는 injective, surjective, bijective하다. 이는 $\textrm{Im}(\Phi) \subseteq W$이기 때문이다.

> 앞서 선형시스템의 해는 special solution과 homogeneous solution의 합으로 이루어져 있다고 하였다. 행렬은 선형사상과 isomorphism관계에 있기 때문에 이는 선형사상에서도 똑같이 적용된다. Rank-Nullity theorem은 이에 대한 관계를 나타내는 것이다.


## Affine space

이제 원점에서 떨어진 공간들을 살펴보자. 즉, 이는 더이상 부분공간이 아니게 된다. 더 나아가 이러한 affine space 사이 사상의 성질을 간략하게 논의해보자.

### Affine subspace

<div class="notice--warning" markdown="1">

**Definition 2.25** (Affine Subspace).

Vector space $V$와 $x _0 \in V$와 V의 subspace $U \subseteq V$가 있을 때, 다음 부분집합 $L$을 $V$의 **affine space** 혹은 $V$의 **linear manifold**라한다.

$$
\begin{align}
L &= \boldsymbol x _0 + U := {\boldsymbol x _0 + \boldsymbol u : u \in U} \tag{2.130a}\\
&= \{\boldsymbol v \in V \rvert \exists \boldsymbol u \in U: \boldsymbol v = \boldsymbol x _0 + \boldsymbol u\} \subseteq V \tag{2.130b}
\end{align}
$$

</div>

$U$는 **direction/direction space**라 하고, $x _0$는 **support point**라고 한다. 이후 SVM에서는 이러한 subspace를 hyperplane이라고 부를 것이다.  

이때 $x _0 \notin U$이면 affine space에 0이 포함되지 않았으므로 affine space는 linear subspace (vector space)가 아니게되고, 따라서 affine subspace를 V의 subspace라고 할 수 없다.  

Affine space는 주로 parameter를 이용해 많이 표현된다. k-dimensional affine space인 $L = \boldsymbol x _0 + U$가 있을 때, $(\boldsymbol b _1, \boldsymbol b _2, \cdots, \boldsymbol b _n)$를 $U$의 ordered basis라 하면, affine space 내의 원소 $\boldsymbol x$는 다음과 같이 directional vector $\boldsymbol b$와 parameter $\lambda$를 이용해 표현할 수 있다.

$$
\boldsymbol x = x _0 + \lambda _1 b _1 + \cdots + \lambda _k b _k
$$

### Affine mapping

Linear mapping과 마찬가지로 affine mapping은 두 affine space 사이의 변환을 나타낸다.

<div class="notice--warning" markdown="1">

**Definition 2.26** (Affine Mapping)

Vector space $V, W$와 $\Phi: V \rightarrow W, \alpha \in W$가 있을 때, $V$에서 $W$로의 **affine mapping**은 다음과 같이 나타낼 수 있다.

$$
\begin{align}
\varphi: V \rightarrow W \tag{2.132} \\ 
\boldsymbol x \mapsto \boldsymbol a + \Phi (x) \tag{2.133}
\end{align}
$$

여기서 벡터 $\boldsymbol a$를 $\varphi$의 **translation vector** 라고 부른다.

</div>

- 모든 affine mapping  $\varphi: V \rightarrow W$는 linear mapping $\Phi: V \rightarrow W$와 translation $\tau: W \rightarrow W$로 이루어진다. $\varphi$에 대해 $\Phi, \tau$는 unique하게 정해지고, $\tau \circ \Phi$로 표현할 수 있다.
- 두 affine mapping은 서로 결합해도 여전히 affine mapping이다.
- affine mapping은 변환 전후의 dimension, parallism 같은 기하학적인 구조를 그대로 유지시킨다.
