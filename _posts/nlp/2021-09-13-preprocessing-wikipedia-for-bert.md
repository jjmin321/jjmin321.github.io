---
title:  "BERT 학습을 위한 wikipedia 전처리"
toc: true
toc_sticky: true
categories:
  - NLP
tags:
  - preprocessing
use_math: true
last_modified_at: 2021-09-13
---

## 들어가며

이번 포스트에서는 BERT 학습을 위한 wikipedia 전처리 방법을 알아보자. 이를 위해서는 우선, 

1. 한국어 위키 데이터 수집
2. 위키 데이터 전처리
3. BERT 학습 용 데이터 전처리

의 세가지 과정을 거쳐야 한다.

## 데이터 수집

[dump 파일 링크](https://dumps.wikimedia.org/kowiki/20210901/)

위키데이터는 직접 수집하지 않고 dump파일을 이용하여 받도록 한다. 현재 기준으로 가장 최신의 위키피디아 데이터는 2021년 09월 01일에 수집된 것이다.

```bash
!wget https://dumps.wikimedia.org/kowiki/20210901/kowiki-20210901-pages-articles.xml.bz2
```

## 마크다운 전처리

위키피디아 데이터는 위키엔진의 마크업 등으로 인해 전처리가 필요한 상태이다. 위키피디아를 전처리 해주는 방법은 크게 두 가지가 있다.

1. [wikiextractor](https://github.com/attardi/wikiextractor)
2. [gensim](https://radimrehurek.com/gensim/corpora/wikicorpus.html)


### 1.wikiextractor

wikiextractor의 경우 bz2 파일채로 전처리를 진행하며, 매우 빠른 속도를 보여준다. 다음 코드를 활용하면 쉽게 전처리를 진행할 수 있다.

```bash
# reference: https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67
#!/bin/sh
set -e

WIKI_DUMP_FILE_IN=$1
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# clone the WikiExtractor repository
git clone https://github.com/attardi/wikiextractor.git

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
python3 -m wikiextractor.WikiExtractor  $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"
```

다만 이렇게 진행할 경우 문서 제목 이후에 바로 이어서 본문이 시작된다. 이게 싫다면 argparse로 인자를 넣어 데이터를 추출한다.

```bash
python -m wikiextractor.WikiExtractor <Wikipedia dump file> -o <output file>
```

그러나 이렇게 진행하더라도 html 코드가 그대로 남아있다던가 파일, 링크 등의 마크업문서가 지워지지 않은채로 그대로 있게 된다.

### 2. gensim

`gensim.corpora.Wikicorpus`도 유용한 전처리 기능을 제공한다. `Wikicorpus`의 경우 위키피디아 문서 하나하나를 읽어와 이를 tokenize하는 기능을 제공한다.

```python
from gensim.corpora import WikiCorpus, Dictionary

def tokenize(content, token_min_len=2, token_max_len=100, lower=True):
    # do something
    return content

wiki = WikiCorpus("/home/vaiv/citypulse/sci-bert/datasets/pre-train/kowiki-20210901-pages-articles.xml.bz2", tokenizer_func=tokenize, dictionary=Dictionary())

for text in wiki.get_texts():
    print(text)
```

뿐만 아니라 `gensim.corpora.wikicorpus.filter_wiki`과 `gensim.corpora.wikicorpus.remove_markup`을 이용하면 정규표현식을 이용하여 마크업 문서를 전처리해준다.

그러나 이 또한 문제가 있는데, `wikiextractor`는 마크업 문서를 잘 처리하는 반면 `gensim`은 약간의 잔여물을 남겨놓고 전처리가 된다는 점이다. 테이블을 말끔하게 지우지 않고 픽셀 정보를 남겨놓고 삭제된다던가, 헤더는 남겨놓는다던가, 아래와 같이 마크업 문서가 깨진채로 있게 된다.

```
파일:20150228세종대학교에서 바라본 광진구 전경42.jpg|섬네일|세종대학교에서 바라본 광진구 주변의 모습. 주위에 아파트가 많이 모여 있는 것이 특이하다.
분류:서울특별시의 구
```

차라리 마크업이 남으면 해결할 수 있을테지만, 이러면 어디까지가 파일에 대한 정보인지 알기가 힘들다.

따라서 아래와 같이 둘의 기능을 적절하게 섞어서 전처리기능을 만들어주었다. 마크업 문서 외에도 빈 괄호기호를 삭제해주는 등의 기능을 추가하였다.

```python
def tokenize(text):
    FILE_COMPILE = re.compile("\[\[파일:.*?.*?\]\]")
    CAT_COMPILE = re.compile("\[\[분류:.*?.*?\]\]")
    LINK_COMPILE = re.compile("\[\[(?!파일|분류)(.*?)\|*(.*?)\]\]")
    EMPTY_PARENTHESIS = re.compile(r"\([\s,]*\)")
    MULTI_EMPTY_PARENTHESIS = re.compile("(,\s)+")
    MISSING_LEAD_PARENTHESIS = re.compile("\(,\s+")
    MISSING_TAIL_PARENTHESIS = re.compile(",\s+\)")
    
    DOC_START_PATTERN = re.compile("<doc .*?>")
    EMPTY_SPACES = re.compile("[\t]")
    
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    text = DOC_START_PATTERN.sub("", text)
    text = MISSING_LEAD_PARENTHESIS.sub("(", text)
    text = MISSING_TAIL_PARENTHESIS.sub(")", text)
    text = MULTI_EMPTY_PARENTHESIS.sub("", text)
    text = EMPTY_PARENTHESIS.sub("", text)
    text = FILE_COMPILE.sub("", text)
    text = CAT_COMPILE.sub("", text)
    text = EMPTY_SPACES.sub("", text)

    for match in re.finditer(LINK_COMPILE, text):
        m = match.group(0)
        caption = m[:-2].split("|")[-1]
        text = text.replace(m, caption, 1)
    
    return text
```

## BERT 학습용 데이터 생성

구글에서 제공하는 [BERT](https://github.com/google-research/bert) 레포를 통해 손쉽게 pre-training 할 수 있다. 단, 문서와 문서 사이는 빈 공백으로 만들어야 하며, 각 line이 한 문장이 되어야 한다. 이는 `kss`를 통해 해결해주도록 한다. MS에서 제공하는 blingfire도 사용해봤지만, 구분하지 못하는 문장이 많았다.

이러한 과정을 전부 거친 코드는 아래와 같다. 문서가 너무 작거나 빈 공백일 경우 전처리를 진행하지 않도록 만들어주었다.

```python
import argparse
import logging
import re
from pathlib import Path

import kss
from gensim import utils


def tokenize(text):
    FILE_COMPILE = re.compile("\[\[파일:.*?.*?\]\]")
    CAT_COMPILE = re.compile("\[\[분류:.*?.*?\]\]")
    LINK_COMPILE = re.compile("\[\[(?!파일|분류)(.*?)\|*(.*?)\]\]")
    EMPTY_PARENTHESIS = re.compile(r"\([\s,]*\)")
    MULTI_EMPTY_PARENTHESIS = re.compile("(,\s)+")
    MISSING_LEAD_PARENTHESIS = re.compile("\(,\s+")
    MISSING_TAIL_PARENTHESIS = re.compile(",\s+\)")
    
    DOC_START_PATTERN = re.compile("<doc .*?>")
    EMPTY_SPACES = re.compile("[\t]")
    
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    text = DOC_START_PATTERN.sub("", text)
    text = MISSING_LEAD_PARENTHESIS.sub("(", text)
    text = MISSING_TAIL_PARENTHESIS.sub(")", text)
    text = MULTI_EMPTY_PARENTHESIS.sub("", text)
    text = EMPTY_PARENTHESIS.sub("", text)
    text = FILE_COMPILE.sub("", text)
    text = CAT_COMPILE.sub("", text)
    text = EMPTY_SPACES.sub("", text)

    for match in re.finditer(LINK_COMPILE, text):
        m = match.group(0)
        caption = m[:-2].split("|")[-1]
        text = text.replace(m, caption, 1)
    
    return text

def make_corpus(out_f):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # empty doc handler
    handler = logging.FileHandler("/creat_dataset.log")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # read text file
    exceptions = []
    with open(out_f, "a") as write_file:
        for file in Path("/kowiki-20210901-pages-articles/").glob("**/wiki_*"):
            logger.info(file)
            with open(file, encoding="utf-8") as reader:
                text_file = reader.read()
                for i, doc in enumerate(text_file.split("</doc>")):
                    try:
                        sentences = tokenize(doc)
                        if len(sentences) == 0:
                            logger.info("Empty: "+doc+"\n")
                            continue
                        if len(sentences) < 2:
                            logger.info("Less doc: "+doc+"\n")
                            continue
                        sentences = "\n".join(kss.split_sentences(sentences))
                        write_file.write(sentences + "\n")
                        if i % 10000 == 0:
                            print(i)
                    except:
                        exceptions.append(doc)
        print("Processing complete!")

    with open("/exception_wiki.txt", "a") as write_file:
        for line in exceptions:
            write_file.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Location of output files")
    args = parser.parse_args()
    make_corpus(args.output_path)
```

상황에 맞게 logger와 예외처리를 추가하면 될 것 같다.