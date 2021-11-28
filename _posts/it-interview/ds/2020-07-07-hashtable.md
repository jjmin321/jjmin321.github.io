---
title:  "CS 비전공자의 IT 기업 면접 뽀개기 - (1) Data Structure - HashTable"
excerpt: "HashTable 자료구조를 알아보자"
toc: true
toc_sticky: true
permalink: /project/IT-interview/DS/hashtable/
categories:
  - IT-Interview
  - Data Structure
tags:
  - CS
  - HashTable
last_modified_at: 2020-07-07
---

자료구조(Data structure)는 데이터들의 모임, 관계, 함수, 명령 등의 집합을 의미한다. 즉, 처리하고자 하는 데이터들이 모여 있는 형태, 혹은, 처리하고자 하는 데이터들 사이의 관계 (수직, 상하, 일방적, 상호 등)를 정의한 것, 혹은, 데이터들을 사용하기 용이하게 저장해 놓은 형태라고 볼 수 있다.

자료구조를 잘 선택하면 사용하는 메모리와 시간, 공간적 효율성을 확보할 수 있다.

# HashTable

![](https://media.vlpt.us/post-images/cyranocoding/8d25f580-b225-11e9-a4ce-730fc6b3757a/1iHTnDFd3sR5FqjHD1FDu9A.png)

연관배열(associative array)<sup>1</sup> 구조를 이용하여 key-value를 저장하는 자료구조로, 위 그림과 같이 key, hash function, hash, value, bucket(slot)으로 구성되어 있다.

-   **key**: 고유한 값으로, hash function의 input이 된다. 이 상태로 bucket에 저장된다면 다양한 길이 만큼의 저장소를 구성해 두어야 하기 때문에 해시 함수로 값을 바꿔야 한다.
    
-   **Hash function**: 다양한 길이의 key를 hash로 변경하는 역할을 한다. 서로 다른 key가 같은 hash를 갖을 경우 해시 충돌이 일어나므로, 조심해야 한다.
    
-   **hash**: hash function의 결과물로, bucket에서 value와 매칭되어 저장된다.
    
-   **value**: bucket에 최종적으로 저장되는 값으로, key와 매칭되어 저장, 삭제, 검색, 접근이 가능해야 한다.

Hashtable의 특징은 다음과 같다.
    
-   순서가 있는 배열에는 어울리지 않는다.
    
-   공간 효율성이 떨어진다.
    
-   Hash function의 의존도가 높다.
    

# Hash function의 종류

## 1. Division method

나눗셈 법은 입력 값을 테이블의 크기로 나누고, 나머지를 테이블의 주소로 사용한다.

-   어떤 값이든 테이블의 크기로 나누면, 그 나머지는 절대로 테이블의 크기를 넘지 않는다.
-   테이블의 크기를 n이라 하면, `[0, n-1]`의 주소를 반환함을 보장한다.
-   테이블의 크기는 소수로 정하는 것이 좋다고 알려져 있다.

## 2. Digit folding

숫자의 각 자릿수를 더해 해시 값을 만든다.

-   문자열에 잘 어울린다. 문자열의 각 요소를 ASCII 코드 번호로 바꾸고, 이 값들을 다 더하면 hash table 내의 주소로 변환된다.  
    ![](https://t1.daumcdn.net/cfile/tistory/161C99415027FF6E18)

# Hash 충돌을 피하는 방법(Collision resolution)

## 1. Separate chaining

![](https://media.vlpt.us/post-images/cyranocoding/329e7e60-b226-11e9-a4ce-730fc6b3757a/16eBeaqTti8MxWPsw4xBgw.png)

위 그림을 보면 1번 John의 값에 Sandra가 중복되어 충돌이 발생했다. 이러한 경우 chaining은 bucket에서 충돌이 일어난 기존 값과 해당 값을 연결한다. 이 때, 이전에 배웠던 linkedlist를 사용하여 이 둘을 연결한다. 이 때 tail에 넣으면, 길이 전체를 탐색해야 하므로, 맨 앞에다가 넣는다. Chaining의 장단점은 아래와 같다.

장점:

-   한정된 bucket을 효율적으로 사용한다.
-   Hash function을 선택하는 중요성이 상대적으로 작다
-   상대적으로 적은 메모리를 사용한다.
-   미리 공간을 잡을 필요가 없다.

단점:

-   Cluster<sup>2</sup> 발생 시 검색 효율이 낮아진다.
-   외부 저장 공간을 사용한다. 즉, hash table의 저장 공간 외에 새로운 linkedlist를 사용한다.
-   linkedlist의 단점을 갖는다.

삽입 시, linkedlist의 삽입과 같으므로, O(1)의 시간복잡도를 갖는다.

삭제나 탐색 시, O(키의 개수/ bucket의 길이)만큼의 시간복잡도를 갖는다. 이는 평균적으로 bucket에서 1개의 hash당 들어있는 키의 개수이다.

## 2. Open Addressing

![](https://media.vlpt.us/post-images/cyranocoding/7c9f8040-b226-11e9-89af-8fc0a61dbc3e/19O8Eyd9wEhZKhwrXzKJaw.png)

위와는 다르게 비어있는 hash를 찾아 저장하는 기법이다. 따라서 1:1 관계가 유지된다. 이 때, 비어있는 hash를 찾는 과정은 동일해야 한다. 이러한 hash를 찾는 기법은 다음과 같이 구분할 수 있다.

1.  선형 탐색(linear probing): 다음 hash(+1)나 n개(+n)를 건너뛰어 비어있는 hash에 저장한다. 이 경우 cluster 현상이 매우 잘 발생한다.
2.  제곱 탐색(Quadratic Probing): 충돌이 일어난 해시의 제곱에 데이터를 저장한다. 그러나 같은 hash를 갖는 경우 2차 cluster가 발생한다.
3.  이중 해시(Double Hashing): cluster 방지를 위해 다른 해시 함수를 한 번 더 사용한다.

장점과 단점은 다음과 같다

장점:

-   hash table 내의 저장공간에서 해결이 가능하다.

단점:

-   Hash function의 성능에 의해 좌지우지된다.
-   데이터의 길이가 늘어나면 그에 해당하는 저장소를 마련해야 한다.

삽입, 삭제, 탐색 모두 Hash 과정에 따라 계산된다. Hash가 비어있는 경우 O(1)이지만, 최악의 경우 O(N)이 된다.

# Remark

1.  연관배열(associative array): key 1개와 value 1개가 1:1로 연관되어 있는 자료구조를 의미한다.
2.  Cluster: 일부 지역의 주소들을 집중적으로 반환 하는 결과로 데이터들이 한 곳에 모이는 문제를 뜻한다.

# Reference

[https://velog.io/@cyranocoding/Hash-Hashing-Hash-Table%ED%95%B4%EC%8B%9C-%ED%95%B4%EC%8B%B1-%ED%95%B4%EC%8B%9C%ED%85%8C%EC%9D%B4%EB%B8%94-%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0%EC%9D%98-%EC%9D%B4%ED%95%B4-6ijyonph6o](https://velog.io/@cyranocoding/Hash-Hashing-Hash-Table%ED%95%B4%EC%8B%9C-%ED%95%B4%EC%8B%B1-%ED%95%B4%EC%8B%9C%ED%85%8C%EC%9D%B4%EB%B8%94-%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0%EC%9D%98-%EC%9D%B4%ED%95%B4-6ijyonph6o)

[https://luyin.tistory.com/191](https://luyin.tistory.com/191)