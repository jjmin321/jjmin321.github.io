---
title:  "The Quest for Machine Learning (1) - Feature Engineering"
excerpt: "데이터 사이언스를 위한 인터뷰 문답집 (1) Feature Engineering편"
toc: true
toc_sticky: true
permalink: /project/ds-interview/feature-engineering/
categories:
  - DS-Interview
tags:
use_math: true
last_modified_at: 2020-07-22
---

본 페이지는 [데이터 과학자와 데이터 엔지니어를 위한 인터뷰 문답집](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791190665230&orderClick=LOA&Kc=) **1장 Feature Engineering**을 읽고 정리한 내용을 개인적인 공부 기록을 위해 남겨 놓는 페이지입니다. 별은 난이도를 뜻합니다.

> 제가 *개인적으로 아는 것들*은 이탤릭체와 block quote를 이용하여 표현하였습니다.

# Introduction

본 장에서는 두 가지 유형의 데이터에 대해 살펴본다.
1. 정형 데이터: RDB의 테이블과 같은 것으로, 각 열은 매우 명확하게 정의되어있으며, 수치/카테고리 형 데이터를 포함함.
2. 비정형: 텍스트/이미지/음성/비디오 등으로, 하나의 수치로는 표현하기 힘듬

# 1. Feature Normalization

- feature 사이 dimension의 영향을 제거
- 이러면 서로 다른 지표가 비교 가능
    - 예를 들어 키(m), 몸무게(kg)의 사람의 데이터를 분석한다면,
    - m는 1.6 - 1.8의 수치 범위에 있고, 체중은 50kg - 100kg까지의 범위에 있음
    - 따라서 분석 결과 수치 범위가 비교적 넓은 체중이라는 **feature에 bias가 생김**

## ★ 수치형 데이터에 대한 feature normalization이 중요한 이유는 무엇인가?

- 수치형 데이터에 대해 feature normalization을 진행할 경우, 모든 feautre가 대략 비슷한 수치 구간 내로 이동.
- **min-max scaling/Z-score normalization**이 대표적임.
    - **min-max scaling**의 경우, outlier의 영향을 줄임.
- *Outlier을 적당한 percentile에 따라 clip하거나 winsorization하는 방법도 있음*
- *Rank transformation도 있는데, outlier에는 다른 scaler보다 효과적임*
- Linear model (SVM, Logistic regression, linear regression), ANN에서 주로 사용함
- Tree 계열에서는 할 필요가 없음


# 2. Categorical Feature

## ★★ 데이터 정제 작업을 진행할 때 categorical feature는 어떻게 처리해야 하는가?

1. Ordinal encoding
    - 클래스 사이에 **대소관계가 있는** 경우 사용 가능
2. One-hot encoding
    - **대소 관계 없을 때** 사용
    - sparse vector로 표현
    - 다만 차원이 너무 클수도 있음 -> Overfitting 발생
    - *0 - 1 scaling이 가능*
3. Binary encoding
    - 순번 인덱스를 활용하여 각 클래스에 ID를 부여하고, 각 클래스 ID를 통해 이진법 코드로 나타냄.
        - e.g. 혈액형
        - |혈액형|클래스 ID|Binary encoding| one-hot encdoing |
            |:----:|:----:|:----:|:----:|
            |A|1|0 0 1|1 0 0 0|
            |B|2|0 1 0|0 1 0 0|
            |AB|3|0 1 1 |0 0 1 0|
            |O|4|1 0 0|0 0 0 1|
    - 일종의 hashmapping이라 생각하자
    - 최종적으로 0/1의 고유벡터를 얻고, 차원 수는 원-핫 인코딩보다 작다
    - *linear regression의 경우 [dummy trap](https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a)문제가 발생하기 때문에 이 방법이 좋을 수 있다*
    - 이 외에도 Helmert Contrast, Sum, Contrast, Polynomial Contrast, Backward Difference Contrast 등의 방법이 있다.

# 3. interaction feature

## ★★ interaction feature란 무엇이고, 이는 어떤 방식으로 feature engineering해야 하는가?

- 복잡한 데이터 관계를 보다 잘 적합하게 하기 위해 일차원의 discreter feature를 쌍으로 조합시켜 고차원의 iteraction feature로 만드는 작업을 진행한다.
- feature $x_i$와 $x_j$가 결합하면, 둘의 차원만큼의 데이터가 추가로 생긴다.
- 다만 ID형태의 데이터가 들어오면 무지막지 늘어나기 때문에 주의해야 한다.
    - 이를 해결하는 한 가지 방법은 저차원으로 mapping하는 방법이 있음.

## ★★ 효율적인 interaction feature는 어떻게 찾는가?

- 다양한 고차원의 feature가 있는데, 무작위로 조합할 수도 없고, overfitting 이슈도 있음.
- 따라서 적절하게 찾을 수 있어야 함.
- facebook의 연구 [He, X., et al. (2014, August)](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)을 따라 decision tree를 통해 feature 조합을 찾을 수 있다.
- 초기 데이터가 정해진 상태에서 효과적으로 의사결정 트리를 구성하려면 Gradient Boosting Tree를 쓰는 것이 제일 쉽고 간단하다.

# 4. Text 표현 모델

## ★★ 텍스트 표현 모델과 장단점
![word representation](https://user-images.githubusercontent.com/47516855/88257173-21d84a00-ccf8-11ea-9bc8-49c164966e13.png){: width="900"}{: .align-center}
*임베딩 기법의 역사와 종류 [^4]*
{: .text-center}
- BOW: 각 단어를 **순서에 상관없이 고유한 인덱스로 맵핑한 뒤, 빈도수를 추출**하여 embedding.
- TF-IDF: BOW 단어에 대해 빈도수가 아닌 **TF-IDF값을 통해 표현**하는 방식이다. **TF**는 각 단어가 갖는 **빈도수/가중치**이고, **IDF**는 각 단어가 **문서에 등장하는 횟수에 대해 역수를 취해 penalizing**하는 값이다.
- N-gram: **language model**로, **n개의 단어**들을 모아서 bag of words처럼 표현한다.
- LSA: SVD[^1] 를 사용하여 Term-Document matrix를 분해한 후, 특정 k개의 siguar value를 통해 원본 데이터를 근사한다. 이를 통해 latent한 단어 관계를 표현할 수 있다. [^2] [^3]
- LDA: LSA와 비슷한 토픽 모델링 방법이다. 이는 추후에 다루도록 한다.
- Word embedding: **단어를 continous한 vector로 표현**하는 것이다.  
*word2vec, glove, fasttext 등이 있다. 그러나 요새 대세는 Pre-trained Language Model 인 것 같다.*


[^4]: 오픈카톡방 '자연어처리와 딥러닝'의 '자연'
[^1]: 공돌이의 수학노트: [특이값 분해(SVD)](https://angeloyeo.github.io/2019/08/01/SVD.html)
[^2]: 내 맘을 알아주는 검색: [Latent Semantic Analysis(LSA)](https://sragent.tistory.com/entry/Latent-Semantic-AnalysisLSA)
[^3]: 로츠카츠의 AI 머신러닝: [[선형대수] 특이값 분해(Singular Value Decomposition)의 의미](https://losskatsu.github.io/linear-algebra/svd/)

## ★★★ word2vec은 무엇이고 LDA와 무슨 차이가 있는가?

- *솔직히 LDA에 관련된 질문은 면접에서 나올 것 같진 않다. Generative model 물어보기엔 좀...*
- word2vec은 간단하므로 간략하게 설명
    - word2vec은 **비슷한 맥락에 등장하는 단어들은 유사**한 의미를 지니는 경향이 있다는 **distributional hypothesis**를 기반으로 만들어진 모델이다.
    - CBOW: 특정 단어를 이 단어의 context window $c$만큼의 주변 단어들을 통해 분류하는 모델을 만들어 word embedding을 수행한다.  
    즉, $c=2$에 대해, $p(w_t|w_{t-1}, w_{t-2}, w_{t+1}, w_{t+2})$를 최대화 시키는 것이다.
    - Skip-gram: CBOW와 반대라고 보면 된다. 즉, 중심 단어를 가지고 context words를 예측한다.  
    즉, $c=2$에 대해, $p(w_{t-1}, w_{t-2}, w_{t+1}, w_{t+2}|w_t)$를 최대화 한다.
    - 그 외에 중요한 포인트로는 **negative sampling**이 있다.

# 5. 이미지 데이터가 부족할 때는 어떻게 처리하는가?

## ★★ 이미지 분류 문제에서 훈련 데이터가 부족하면 어떻게 되고, 이를 해결하려면 어떻게 하는가?

- 과적합 발생하기 쉬움
    - 모델을 쉽게 만들거나, normalizing (L1, L2) 추가, 앙상블, dropout 등이 있음
    - Transfer learning
    - Data augmentation
    - up-sampling
    - **GAN**