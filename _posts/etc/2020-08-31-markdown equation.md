---
title:  "Mathjax 수식 정리"
excerpt: "자주쓰이는 Mathjax 수식 정리"
toc: true
toc_sticky: true

categories:
  - IT

use_math: true
last_modified_at: 2020-08-31
---

Markdown을 통해 Github pages를 작성하다 보니 수식을 입력하는게 생각보다 까다롭다. Colab에선 이러지 않았는데... 따라서 직접 사용하고 정리해보는 Markdown 수식을 작성해보았다. 본 post는 기본적으로 markdown에 대한 이해가 어느정도 있는 상태에서 사용할 것을 권장한다. 기본적인 자료는 구글링하면 쉽게 구할 수 있으니 한 번 찾아보자.

## 정렬

`\begin{align}`과 `\end{align}`을 사용하면 **정렬이 오른쪽**으로 된다. 따라서 `&`를 통해 어디서부터 시작할지 정해주면 잘 작동하게 된다. 다음은 예시이다.

**오른쪽 정렬**
$$
\begin{align}
R _k = \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

```latex
\begin{align}
R _k = \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
```

---

**왼쪽 정렬**

$$
\begin{align}
R _k &= \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$

```latex
$$
\begin{align}
R _k &= \{ x^LM _k, \overrightarrow {h^{LM} _{k, j}}, \overleftarrow {h^{LM} _{k, j}} \lvert j=1, ..., L\} \\
&= \{ h^{LM} _{k, j} \lvert j=0, ..., L \},
\end{align}
$$
```

## Numbering

`\tag`를 사용하면 식에 넘버링을 할 수 있다.

$$
x := f(x) \tag{1}
$$

```latex
$$
x := f(x) \tag{1}
$$
```

보다시피 $R _k$옆에 &을 넣어줘서 다음부터는 $=$부터 정렬이 되게끔 만들었다.

## 문자열

### Greek Letter

그리스 문자는 escape sequence에 알파벳을 써주면 작성할 수 있다. 대문자의 경우는 첫 글자를 Capital로 시작한다.

| Greek Letter | Markdown |
| :---: | :---: |
| $\alpha$ | \alpha | 
| $\theta$ | \theta |
| $\Theta$ | \Theta |

### 폰트

| 폰트 | Markdown | 사용법 |
| :---: | :---: | :---: |
| $\mathcal X$ | \mathcal X | 집합 표현 |
| $\mathbb R$ | \mathbb R | 차원 표현 |
| $\mathbf x $ | \mathbf x |  굵은 표시 | 
| $\pmb x $ | \pmb x | 굵은 표시 (이탤릭) |

# 선형대수

vector와 matrix는 다음과 같이 사용한다

$$
\begin{bmatrix} 0 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 \end{bmatrix}
$$

```latex
\begin{bmatrix} 
0 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 0 
\end{bmatrix}
```

한 row에서 다음 row로 넘어갈 때는 \\\를, column을 이동할 때는 &를 사용한다.


| 폰트 | Markdown | 사용법 |
| :---: | :---: | :---: |
| $\vec x$ | \vec x| 벡터 (화살표) | 
| $\overrightarrow x$, $\overleftarrow x$ | \overrightarrow x, \overrleftarrow x | 벡터 (화살표) |
| $\pmb x ^\intercal$ | \pbm x^\intercal | transpose |
| $\boldsymbol{A}$ | \boldsymbol{A} | matrix |
| $\textbf{A}$ | \textbf{A} | matrix |
| $\pmb{A}$ | \pmb{A} | matrix |


# Sums and Intergrals

| Symbol | 	Script |
| :---: | :---: |
|$\sum _{i=1}^{10} t_i$ |	\sum_{i=1}^{10} t_i|
|$\int _0 ^\infty $ | \mathrm{e}^{-x},\mathrm{d}x$ 	\int_0^\infty \mathrm{e}^{-x},\mathrm{d}x |
| $\sum$ |	\sum |
| $\prod$ |	\prod |
| $\coprod$ |	\coprod |
|$\bigoplus$ | 	\bigoplus |
|$\bigotimes$| 	\bigotimes|
|$\bigodot$ |	\bigodot|
|$\bigcup$ 	|\bigcup|
|$\bigcap$ 	|\bigcap|
|$\biguplus$ |	\biguplus|
|$\bigsqcup$ 	|\bigsqcup|
|$\bigvee$ 	|\bigvee|
|$\bigwedge$ |	\bigwedge|
|$\int$ |	\int|
|$\oint$ |	\oint|
|$\iint$ 	|\iint|
|$\iiint$ |	\iiint|
|$\idotsint$ | 	\idotsint |
|$\sum_{\substack{0<i<m \\ 0<j<n}} P(i, j)$  |	\sum_{\substack{0<i<m \\ 0<j<n}} P(i, j) |
|$\int \limits_a^b$ |	\int \limits_a^b |

# Reference

https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols

https://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/

[https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

