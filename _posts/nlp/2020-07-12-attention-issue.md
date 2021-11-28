---
title:  "PyTorch Attention 구현 issue 정리"
excerpt: "포트폴리오 용으로 Attention을 구현하던 중 헷갈리는 개념을 정리하였다."
toc: true
toc_sticky: true
permalink: /project/nlp/attention-issue/
categories:
  - NLP
tags:
  - Attention
  - PyTorch
use_math: true
last_modified_at: 2020-07-12
---

포트폴리오 용으로 Attention을 구현하던 중 헷갈리는 개념을 정리하였다. 원본은 [https://stackoverflow.com/questions/62444430/implementing-attention/62445044#62445044](https://stackoverflow.com/questions/62444430/implementing-attention/62445044#62445044%5D) 을 참고하면 된다. 구현 repo는 [다음](https://github.com/InhyeokYoo/NLP/tree/master/papers/1.Attention)을 참고.

## 1. Decoder의 초기 hidden state $s_0$는 어떤 값인가??

이를 물어본 이유는 내가 참고한 implementation에서 $s_0$를 구하는 과정이 각기 다 달랐기 때문이다. 아래는 PyTorch 재단에서 제공하는 NLP 튜토리얼인데, 이 부분에서 $s_0$는 context vector로 정의하고 있다.  
[https://colab.research.google.com/drive/1UQM0wbmeetka4Sm8jE-hdfdgBkkKU0DZ?authuser=1#scrollTo=GiEY4xb75GPt](https://colab.research.google.com/drive/1UQM0wbmeetka4Sm8jE-hdfdgBkkKU0DZ?authuser=1#scrollTo=GiEY4xb75GPt)

알고보니 원래는 zero vector가 맞았는데, 나중에 encoder의 final hidden state를 사용하거나, output($h\_j$의 모음)의 평균을 사용하는 것을 추세가 바뀌었다고 한다. 마지막 hidden state가 아닌, 평균을 쓸 경우, propagation과정에서 gradient가 좀 더 직접적으로 encoder로 흘러 들어올 수 있지만, 그닥 효과는 없는 듯...

## 2\. 원래 논문에선 maxout layer를 사용하였는데, dropout으로 대체될 수 있는가?

사실 maxout layer에 대한 지식이 없어서 drop out과 비슷한 건줄 알았다.  
maxout은 non-linear layer의 일종으로, linear layer 두개를 합친 후, max값을 취해 linearity를 제거한다. 그러나, 두 개의 linear layer를 계산해야 하므로 비효율적이다. 수식으로 표현하면 다음과 같다.  

$$  
f(x) = \max(w_1^Tx + b_1, w_2^Tx + b_2)  
$$

## 3\. Encoder와 decoder의 dropout이 다르던데 왜 그러는건가?

잘 모르겠다. 아무래도 empirical하게 얻는 결과인듯.

## 4\. Alignment model 중 concat해서 사용하는 경우, bias term은 있어야 하는가?

즉, 다음과 같은 alignment model을 가정한다.

$$  
f(s_{i-1}, h_j) = Ws_{i-1} + U h_j = W[s_{i-1}; h_j]  
$$

이 때, $U, W$를 계산하기 위해 bias term이 필요하냐는 질문이었다. 이 또한 implement에 따라 bias term 구현이 달랐기 때문에 질문하였다.

답변은 '대부분은 사용한다'이고, 아무래도 같은 layer에서 같은 initiallization을 공유하기 때문에 update 또한 비슷하게 될 것이다.

## 5\. 원문에서 Encoder의 경우 Bi-directional 한 hidden state를 concat하여 dimension이 2h가 되는데, decoder의 경우 dimension이 h이다. 이 경우 encoder에 linear layer를 추가하여 차원을 맞춰주나??

마찬가지로 원문에서 알 수 없는 정보였다. 근데 생각해보니, $s_0$를 zero vector로 초기화하면 굳이 차원을 맞춰줄 필요가 없었다. 따라서 이 질문은 $s_0=f([\overleftarrow h_T; \overrightarrow h_T])$ 인 경우에만 생각하면 되겠다.

만약 위와 같은 경우라면, 차원을 바꿔주면 된다. 다만, 경우에 따라 tanh와 같은 activation function을 넣어주는 경우도 있는 듯 하다.

**추가**  
논문을 안 읽고 구현했더니 이런 참사가 일어났다. 논문의 appendix에 이에 대한 설명이 있으므로 살펴보면 되겠다.  
![](http://cfile8.uf.tistory.com/image/998A893F5EF22EC820B7FD)