---
title:  "[미완성] 한국어 전처리의 이해"
excerpt: "검색해도 안 나오는 한국어 전처리에 대해 공부해보자."
toc: true
toc_sticky: true

categories:
  - NLP
tags:
  - Preprocessing
  - TODO
use_math: true
last_modified_at: 2020-11-04
---

## 들어가며

석사과정에서 연구할 때에는 text mining 위주로 진행했기 때문에, lemmatization이라던가, stemming, stop word removing 등의 작업을 당연히 거쳤었다. 그러나 졸업 후 딥러닝을 공부하며 이러한 작업을 전혀 보지 못해 의아해 하고 있었는데, 최근에 회사 과제로 한국어 전처리에 대해 조사를 하며 이번 기회에 오개념이나 새롭게 알게된 추가 지식을 정리해볼까 한다.

## 한국어 전처리의 순서

**김기현의 자연어처리 딥러닝 캠프** 책을 보면 다음과 같이 쓰여져 있다.
> 한국어 전처리 과정은 목적에 따라 약간씩 다르지만 대체로 다음과 같습니다.  
> 1. 코퍼스 수집
> 2. 정제
> 3. 문장 단위 분절
> 4. 분절
> 5. (병렬 코퍼스 정렬)
> 6. 서브워드 분절

여기서 **4번 분절**과 **6번 서브워드 분절**의 의미에 혼동이 생긴다. 4번의 경우 책에서 설명하기를, **형태소 분석/단순한 분절, 띄어쓰기 교정 등을 통해 정규화를 수행한다**고 되어 있고, 6번 서브워드 분절의 경우 **BPE, WPM, Sentencepiece 등을 통해 단어를 더 작은 단어의 모음으로 쪼갠다**라고 하는데, 결국 형태소 분석도 단어를 더 작은 단어의 모음으로 분해하는 과정이 아닌가하는 의문이 생겼다.

**한국어 임베딩**을 살펴봐도 이와 비슷한 맥락으로 설명한다.
> 웹 문서나 json 파일 같은 형태의 데이터를 순수 텍스트 파일로 바꾸고 여기에 형태소 분석을 실시하는 방법을 설명한다. 형태소 분석 방법에는 국어학 전문가들이 태깅한 데이터로 학습된 모델로 분석하는 지도 학습 기법과 우리가 가진 말뭉치의 패턴을 학습한 모델을 적용하는 비지도 학습 기법 등이 있다.


## 날짜, 돈, 사람이름 등은 어떻게 전처리를 해야하나?

Entity recognition이랑 비슷하지만, 이를 전처리 단계에서 신경써주어야 하는 의문이 든다.

## 형태소 분석과 subword segmentation

우선 형태소란 무엇인가? 다음은 이기창의 **한국어 임베딩**에서 발췌한 내용이다.

> 형태소(morpheme)란 의미를 가지는 최소 단위를 말하는 것으로, 더 쪼개면 뜻을 잃어버리게 된다. 이때 의미는 **어휘**뿐만 아니라 **문법**적인 것도 포함된다. 그러나 언어학자들이 형태소를 분석하는 방법은 조금 다른데, 대표적으로는 계열 관계(paradigmatic relation)가 있다. 이는 해당 형태소 자리에 다른 형태소가 대치돼 쓰일 수 있는가를 따지는 것이다. 즉, distributional hypothesis와 밀접한 관계를 갖고 있다.

즉, tokenization의 일종으로 보인다.

[HanBert의 발표자료](https://www.slideshare.net/YoungHCHO/hanbert-korquad-20-by-twoblock-ai)에 따르면, 형태소 분석을 먼저 실시하여 tokenizing을 하여 corpus를 만들고, 이 중 일부를 이용해 Vocab을 구성한다고 되어 있다. 이 때, BPE를 사용할지, 한글 음절을 기준으로 사용할지를 고민하는 것을 보면 이 둘은 다른 프로세스로 보인다. **김기현의 자연어처리**에서도 마찬가지로 형태소 분석과 subword 분절을 별개의 것으로 취급하는 것으로 보인다.

그러나 [핑퐁 블로그](https://blog.pingpong.us/dialog-bert-tokenizer/)를 보면, 공백 기반/형태소 분석기 기반/ BPE 기반 tokenizer로 구분하는 것을 볼 수 있다. 또한, [당근 마켓 블로그](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d)를 보면 형태소 분석과 subword 분절을 같은 level의 작업으로 보고 있는 것 같다. 

> 사실 BERT 논문에서 wordpiece 모델을 사용했던 이유 중에 하나가 다국어에 대해서 동작하게 만들고 싶었기 때문입니다. 하지만 당근마켓은 한국에서만 서비스를 하고 있어서 다국어를 고려할 필요가 없습니다. 이럴 때는 wordpiece보다는 mecab과 같이 한국어에 맞게 만들어진 형태소 분석기를 사용하는 것이 더 좋은 경우가 많습니다. mecab은 다음 링크를 통해 볼 수 있습니다. 설치하는 것이 살짝 까다로운데 가이드를 잘 따라가면 설치가 됩니다(mecab-ko-dic도 설치해야 합니다). Python으로 사용해야 하므로 mecab-python도 설치해야 합니다.  
...  
두 가지의 tokenizer를 모두 사용해서 BERT를 학습시키고 결과를 비교했습니다. 논문에 나온 것처럼 두 개의 tokenizer 모두 30000개의 token으로 사전을 구축했습니다. 결과적으로는 mecab을 tokenizer로 사용해서 학습했을 때 결과가 더 좋게 나와서 현재는 mecab을 사용하고 있습니다. 추후에 mecab 사전에 단어들을 추가하는 작업을 할 생각입니다.

그러나 [DEVIEW 2019](https://deview.kr/2019/schedule/285)를 보면 Mecab + SentencePiece를 사용하여 성능을 높혔다고 한다. 이는 experiment로 측정해서 사용해야 할듯하다.

다음은 한국어 tokenization을 정리한 표이다.

| 종류 | 모델 | 설명
| --- | :---: | :---: |
| Supervised | [Khaiii](https://github.com/kakao/khaiii) | 카카오에서 개발한 형태소 분석기로, CNN 사용 |
| Supervised | [KoNLP](https://konlpy-ko.readthedocs.io/ko/v0.4.3/) | 은전한닢, 꼬꼬마, 한나눔, Okt, 코모란 등 5개 오픈소스 형태소 분석기를 파이썬 환경에서 사용할 수 있도록 인터페이스를 통일한 한국어 자연어 처리 패키지 |
| Unsupervised | [SoyNLP](https://github.com/lovit/soynlp) |  형태소 분석, 품사 판별 등을 지원하는 파이썬 기반 한국어 자연어 처리 패키. 하나의 문장 혹은 문서에서보다는 어느 정도 규모가 있으면서 동질적인 문서 집합(Homogeneous Documents)에서 잘 작동함 |
| Unsupervised | [SentencePiece](https://github.com/google/sentencepiece) | 구글에서 분석한 subword segmentation으로, BPE, Unigram, WordPiece 등을 지원 |

SentencePiece, soynlp는 띄어쓰기에 영향을 많이 받으므로 이를 미리 고쳐주는 작업이 필요하다.

## Lemmatization, stemming

**김기현의 자연어처리** 8.3 절을 보면 다음과 같은 내용이 나온다.
> 필자가 종종 받는 질문 중 하나는, lemmatization 또는 stemming을 수행하여 접사 등을 제거한 이후에 텍스트 분류를 적용해야 하는지에 관한 것입니다.  
...  
따라서 코퍼스가 부족한 상황에서는 이처럼 lemmatization이나 stemming가 같은 문장에 대해 같은 샘플로 취급하여 희소성 문제에서 어느정도 타협을 볼 수 있습니다.  
...   
하지만 딥러닝 시대에 접어들어 성공적으로 차원 축소를 수행할 수 있게 되면서, 희소성 관련 문제는 더 이상 큰 장애물이 되지 않습니다. **따라서 lemmatization 및 stemming이 반드시 정석이라 하기는 어렵습니다.**  
...  
따라서 처음부터 lemmatization 또는 stemming을 한 후에 텍스트 분류 문제에 접근하는 것보다는, **일단은 하지 않은 상태에서, 이후 설명할 신경망 모델을 사용하여 텍스트 분류 문제 해결을 시도하여 베이스라인 성능을 확보**함이 바람직합니다. 이후에 성능 향상을 위한 차원에서 여러 가지 튜닝 및 시도를 할 때 **코퍼스 양의 부족이 성능 저하의 원인이라는 가정이 성립**되면, 그때 lemmatization 또는 stemming을 추가로 실험해보는 편이 낫습니다.

아래는 한국어 문법 및 처리에 대한 reference이다. 

[말뭉치를 이용한 한국어 용언 분석기 (Korean Lemmatizer) ](https://lovit.github.io/nlp/2019/01/22/trained_kor_lemmatizer/)

[한국어 용언의 활용 함수 (Korean conjugation)](https://lovit.github.io/nlp/2018/06/11/conjugator/)

[어간 추출(Stemming) and 표제어 추출(Lemmatization)](https://settlelib.tistory.com/57)

밑의 논문은 한국어 subword representation에 대한 연구이다.

[Subword-level Word Vector Representations for korean](https://catsirup.github.io/ai/2020/03/12/subword-level-word-vector-representations-for-korean.html)

[KR-BERT: A Small-Scale Korean-Specific Language Model](https://www.semanticscholar.org/paper/KR-BERT%3A-A-Small-Scale-Korean-Specific-Language-Lee-Jang/c0ba595b2bef54f2552ec4716bb187901f52f4a3)

[Advanced Subword Segmentation and Interdependent Regularization Mechanisms for Korean Language Understanding](https://ieeexplore.ieee.org/document/8903977)

[국민은행의 KB-Albert](https://github.com/KB-Bank-AI/KB-ALBERT-KO#3-subtoken-based-model%EC%9D%98-%EA%B2%BD%EC%9A%B0-bpe%EC%9D%B4%EC%A0%84-%EB%8B%A8%EA%B3%84%EC%97%90-%EC%96%B4%EA%B7%BC%EC%96%B4%EB%AF%B8-%EB%B6%84%EB%A6%AC%EB%A5%BC-%EC%A7%84%ED%96%89%ED%95%98%EB%8F%84%EB%A1%9D-%EC%B6%94%EA%B0%80)에서는 다음과 같이 밝히고 있다.
>- 한글에 BPE를 바로 적용하는 것 보다 형태소 분석 후에 적용하는 것이 일반적으로 성능이 더 좋음.
> - 하지만 보통 형태소 분석기의 경우 50여개의 형태소 태그를 분류하기 위한 연산의 오버로드와 형태소의 원형을 복원하기 위한 오버로드가 상당한데, 실제로 BPE의 입력에는 해당 부분이 사용 되지 않음.
> - 그래서 형태소 분석 말뭉치를 변형하여 간단히 어근과 어미만 분리(실제로는 명사와 동사 어근)하기 위한 말뭉치를 생성 해냈으며, 해당 말뭉치와 CRF를 이용하여 간단한 전처리기(어근 분리기)를 구현 ⇒ 세종 말뭉치에 대해서 99.1%의 정확도 보임.


## PLM vs. TF-IDF/Word embedding

PLM을 학습시킬 여력이 안되는 경우엔 울며 겨자먹기로 TF-IDF 혹은 word2vec 등을 써야되는 경우가 생길텐데, 이들의 성능 차이가 과연 얼마나 나는지 궁금했다. 또한, 자연어처리 오픈카톡방에서 누군가가 

![image](https://user-images.githubusercontent.com/47516855/93410635-46a20580-f8d4-11ea-9cfc-c096b04ce569.png){: .align-center}

라고 얘기를 해서 비교를 해보고 싶어졌다.

다음은 [이곳](https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794)에서 찾은 결과인데, tf-idf, word2vec, BERT의 성능을 비교하여 정리한 것이다.

---


**TF-IDF**

![TF-IDF performance](https://miro.medium.com/max/764/1*iPL_8iJOuTJ_mrLvftwUEw.png){: .align-center}{: width="600"}

---

**Word2vec**

![word2vec performance](https://miro.medium.com/max/764/1*a39MMTNXnDaFOKFur2Z7xQ.png){: .align-center}{: width="600"}

---

**PLM (BERT)**

![PLM performance](https://miro.medium.com/max/764/1*NsiKi7b0JGlCQPLpeVkftA.png){: .align-center}{: width="600"}


엄청나게 성능이 향상하는 것은 아니지만, 어쨋든 fine-tuning의 문제도 있고, 성능이 올라간 것은 사실이기 때문에 PLM을 쓰는게 좋다는 생각이 들긴 한다. 한번 쓰고 버릴 것도 아니니까...


# Task에 따라 전처리과정이 달라지나?

NLP에는 정말 다양한 task가 있다. 예를들어 relevant한 문서를 검색하기 위해 BM25를 사용하거나 (IR) LDA와 같은 topic modeling을 이용하여 clustering을 이용하고, BERT같은 LM에 올리는 작업 등이 있다.
그렇다면 **각 task에 따라 전처리 파이프라인을 다르게 구성해야 하나?**

**IR**

https://arxiv.org/abs/1905.09217v1

![stop-words attention](https://user-images.githubusercontent.com/40360823/72411227-d45dac00-37ad-11ea-9732-eabe3027eb33.png)

위 그림에서 보면 stop word인 *in*에 attention이 강하게 생기는 것을 확인. 기존에는 보기 힘들던 구조.


If you are working with basic NLP techniques like BOW, Count Vectorizer or TF-IDF(Term Frequency and Inverse Document Frequency) then removing stopwords is a good idea because stopwords act like noise for these methods. If you working with LSTM’s or other models which capture the semantic meaning and the meaning of a word depends on the context of the previous text, then it becomes important not to remove stopwords.