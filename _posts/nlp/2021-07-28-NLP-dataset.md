---
title:  "[미완성] NLP benchmark dataset에 대해 알아보자"
toc: true
toc_sticky: true
permalink: /project/nlp/NLP-benchmark/
categories:
  - NLP
tags:
  - benchmark
  - TODO
use_math: true
last_modified_at: 2021-07-15
---

## 들어가며

현재 작성 중

## SuperGLUE

## 1. Language Model

## LAMBADA

LAMBADA는 (Paperno et al., 2016) 텍스트 내 긴 문장에 대한 의존성을 테스트하는 데이터셋으로, 문장의 마지막 단어를 예측하는 것을 목표로 한다. 

![](https://paperswithcode.com/media/datasets/LAMBADA-0000002422-52650e4e_B4dJstl.jpg){: .align-center}{: width='700'}

[LAMBADA 설명 (papers with code)](https://paperswithcode.com/dataset/lambada)

## Natural Language Inference (NLI)

premise가 뭔지 설명

Natural Language Inference(NLI)는 두 문장 사이의 관계를 이해하는 능력을 측정한다. 실질적으로 이러한 작업은 두 개 혹은 세 개의 클래스를 분류하는 문제로 구성되며, 모델은 두번째 문장이 논리적으로 첫번째 문장 뒤에 올 수 있는지, 이에 반대되는지, 아니면 참인지 (자연스러운지) 분류한다.

### Recognizing Textual Entailment (RTE)

Textual Entailment Recognition has been proposed recently as a generic task that captures major semantic inference needs across many NLP applications, such as Question Answering, Information Retrieval, Information Extraction, and Text Summarization. This task requires to recognize, given two text fragments, whether the meaning of one text is entailed (can be inferred) from the other text.

## Adversarial Natural Language Inference (ANLI)

- [Paper: Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599)
- [Repo:  facebookresearch/anli](https://github.com/facebookresearch/anli)

ANLI는 FAIR에서 개발한 데이터셋 및 데이터셋 구축 절차로, 기존의 NLI가 모델에 의해 쉽게 정복되는 현상을 방지하고, **오래 버티도록** 설계된 데이터셋이다. AI가 사람 수준까지 정복되는데 MNIST는 15년, ImageNet은 7년 정도의 시간이 걸린 반면, NLU에서는 모델의 발전에 따라 쉽게 무너지고 있다. 특히 BERT의 발전 이후 GLUE같은 데이터셋 너무나 쉽게 무너져서 SuperGLUE 데이터셋의 필요성을 야기하였다.  

ANLI는 벤치마크 데이터셋의 수명과 견고성 문제를 해결하는 NLU 데이터셋 수집을 위한, 반복적이고, 적대적인 human-and-model-in-the-loop solution을 제공한다. 즉, 다음과 같은 것을 목표로 한다.

> The primary aim of this work is to create a new large-scale NLI benchmark on which current state- of-the-art models fail.

주의할 점은 본 ANLI는 데이터셋을 제공하는 것 뿐만 아니라 좋은 데이터셋 수집을 위한 절차를 제공한다는 것이다. 아래는 ANLI 논문에서 밝히고 있는 contribution이다.

> 1) We introduce a novel human-and-model-in-the-loop dataset, consisting of three rounds that progressively increase in difficulty and complexity, that includes annotator-provided explanations. 
> 2) We show that training models on this new dataset leads to state-of-the-art performance on a variety of
popular NLI benchmarks. 
> 3) We provide a detailed analysis of the collected data that sheds light on the shortcomings of current models, categorizes the data by inference type to examine weaknesses, and demonstrates good performance on NLI stress tests.

첫 단계에서는 human annotator가 현재 최고의 모델이 정답을 맞추지 않게끔하는 example을 고안해낸다. 이를 통해 모델의 취약점을 포함하는 hard example을 생성하게 되고, 이를 training 셋에 포함하여 학습한 후 더 좋은 모델을 만들어낸다. 그후 강화된 모델을 대상으로 같은 절차를 진행하고, 몇개의 라운드를 통해 약점을 수집한다. 각 라운드가 끝날 때마다 새로운 모델을 학습시키고, 따로 test set을 마련한다. Never-ending learning (Mitchell et al., 2018) 세팅처럼 계속해서 반복적으로 이 절차를 진행하고, test 셋은 매 라운드가 지날 때마다 점점 어려워진다. 따라서 데이터셋은 현존하는 벤치마크보다 어려울뿐만 아니라, 정적인 벤치마크가 언젠가는 정복되는 것과는 다르게 "앞으로 전진하는", NLU 시스템에 대한 동적인 목표가 되는 것이다.

![image](https://user-images.githubusercontent.com/47516855/128608942-b4f04400-5b17-4cbd-adaa-b7b10d48e170.png)

## Generation task

 abstractive question answering and summarization



Reading comprehension

Most current question answering datasets frame the task as reading comprehension where the question is about a paragraph or document and the answer often is a span in the document. The Machine Reading group at UCL also provides an overview of reading comprehension tasks.

http://nlpprogress.com/english/question_answering.html#reading-comprehension



데이터셋 구축 과정
{: .text-center}
