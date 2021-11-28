---
title:  "CS 비전공자의 IT 기업 면접 뽀개기 - (1) Data Structure - Array/LinkedList"
excerpt: "Array/LinkedList를 알아보자"
toc: true
toc_sticky: true
permalink: /project/IT-interview/DS/array-linkedList/
categories:
  - IT-Interview
  - Data Structure
tags:
  - Array
  - List
last_modified_at: 2020-07-07
---

자료구조(Data structure)는 데이터들의 모임, 관계, 함수, 명령 등의 집합을 의미한다. 즉, 처리하고자 하는 데이터들이 모여 있는 형태, 혹은, 처리하고자 하는 데이터들 사이의 관계 (수직, 상하, 일방적, 상호 등)를 정의한 것, 혹은, 데이터들을 사용하기 용이하게 저장해 놓은 형태라고 볼 수 있다.

자료구조를 잘 선택하면 사용하는 메모리와 시간, 공간적 효율성을 확보할 수 있다.

# 배열(Array)

-   같은 종류의 데이터(int, string, float, etc.)를 하나의 이름으로 관리한다.
-   배열 인덱스는 값에 대한 유일무이한 식별자이다.
-   크기가 정해져있고, 변경할 수 없다.
-   array를 assign한 변수에는 배열의 첫 원소의 메모리 주소를 갖고 있어 index를 사용하여 빠르게 접근할 수 있다.
-   데이터가 삭제되어도 삭제된 element의 공간이 그대로 남는다. 따라서 element의 index는 변하지 않는다.
-   순차탐색(Sequential Search)에서도 배열은 연속된 메모리 공간에 할당하므로, 연결리스트(Linkedlist)보다도 빠르다.
-   운영체제의 캐시 지역성(cache locality)<sup>1</sup>을 활용할 수 있다. 즉, cache hit<sup>2</sup>의 가능성이 커져서 성능에 도움이 된다.

Java에서는 array의 사이즈를 미리 정의한 후 데이터를 저장하거나 삭제하고, 데이터의 type 또한 미리 정의한 후 사용했던 것 같다. 그러나 python에서는 이런 일이 일어나지 않는데, 이는 기본 자료형으로 array가 아닌 list를 제공하기 때문이다. List는 array와 마찬가지로 자료구조의 일종이다. 아래는 list의 특징이다.

-   list는 index가 아닌 element의 순서가 중요하다. 따라서 list를 다른 말로는 sequence라고도 부른다.
-   list에서 index는 몇 번째 데이터인가 정도의 의미를 가진다. 반면, array에서의 index는 element에 대한 유일무이한 식별자가 된다.
-   빈 엘리먼트는 허용하지 않는다. 즉, 중간에 삭제가 된다면, array처럼 빈 공간이 생기는 것이 아닌, index가 하나씩 당겨진다.
-   순차성을 보장하지 못하기 때문에 spatial locality<sup>3</sup>가 보장이 되지 않아서 cash hit가 어렵다.
-   데이터의 수가 fix되어 있고, 자주 사용된다면 array가 더 효율적이다.
-   python list는 크기가 가변적이고 어떤 원소 타입이던지 쉽게 저장할 수 있다. 그러나 메모리가 더 많이 필요하다.

Python에서의 array는 `array.array()`를 통해 사용할 수 있다. 다음은 python에서의 arary와 list의 속도 차이이다.

```python
import array # array
import resource

startMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

mylist = []
for i in range(1,100000):
    mylist.append(i)

listMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

myarray = array.array('i')
for i in range(1,100000):
    myarray.append(i)

arrayMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print("list를 만드는 데는 :", listMem-startMem)
print("array를 만드는 데는 : ", arrayMem-listMem)
```

실행결과:

```
list를 만드는 데는 : 3768320
array를 만드는 데는 :  286720
```

Python에서 list에 관한 연산량은 다음과 같다

![python-O(N)](https://user-images.githubusercontent.com/47516855/95674634-72bb5880-0bec-11eb-8c86-54a57657648c.png)


## Remark

1.  locality: 운영체제에선 물리적으로 근접한 위치의 데이터가 주로 활용되기 때문에 미리 캐쉬에 넣어둠으로써 CPU의 성능을 향상시킨다. 배열은 물리적으로 연속된 공간에 데이터를 저장하기 때문에 이러한 locality를 잘 활용할 수 있다.
2.  cache hit: 1과 같이 지역성을 활용하기 위해 캐쉬에 저장해놓은 메모리에 CPU가 참조하고자 하는 메모리가 있다면 cahce hit, 캐쉬 적중이라고 한다. 반대의 개념은 cache miss.
3.  spatial locality: 1에서 설명한 지역성은 시간 지역성(Temporal locality)과 공간 지역성(Spatial Locality)으로 나뉜다. 시간 지역성(Temporal locality)이란 가장 최근에 읽어온 data는 다시 읽어올 때도 빠르게 access할 수 있다는 뜻이다. for나 while 같은 반복문에 사용하는 조건 변수처럼 한번 참조된 데이터는 잠시 후에 또 참조될 가능성이 높다. **공간 지역성**이란 `A[0]`, `A[1]`과 같은 데이터 배열에 연속으로 접근할 때 참조된 데이터 근처에 있는 데이터가 잠시 후에 사용될 가능성이 높다는 것이다.

## Reference

[https://wayhome25.github.io/cs/2017/04/17/cs-18-1/](https://wayhome25.github.io/cs/2017/04/17/cs-18-1/)  
[https://ko.wikipedia.org/wiki/%EB%B0%B0%EC%97%B4](https://ko.wikipedia.org/wiki/%EB%B0%B0%EC%97%B4)  
[https://hashcode.co.kr/questions/1093/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90-list%EA%B0%80-%EC%9E%88%EB%8A%94%EB%8D%B0-arrayarray%EB%8A%94-%EC%99%9C-%EC%93%B0%EB%8A%94-%EA%B1%B4%EA%B0%80%EC%9A%94](https://hashcode.co.kr/questions/1093/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90-list%EA%B0%80-%EC%9E%88%EB%8A%94%EB%8D%B0-arrayarray%EB%8A%94-%EC%99%9C-%EC%93%B0%EB%8A%94-%EA%B1%B4%EA%B0%80%EC%9A%94)  
[https://daimhada.tistory.com/106](https://daimhada.tistory.com/106)

# 연결리스트(LinkedList)

-   아래 그림과 같이 element 간의 연결을 이용하여 list를 구성한 것이다.
    
    ![](https://wayhome25.github.io/assets/post-img/cs/linked-list.png)
-   python에서는 list 기본 자료형에 포함되어 있다.
    
-   길이를 동적으로 조절 가능하다.
    
-   데이터의 삽입과 삭제가 쉽다.
    
-   임의의 노드에 바로 접근할 수가 없다.
    
-   다음 노드의 위치를 저장하기 위해서는 추가로 공간이 필요하다.
    
-   cache locality를 활용하여 물리적으로 근접한 데이터를 사전에 cache에 저장하기가 어렵다.
    
-   linkedlist를 거꾸로 탐색하기가 어렵다.
    

## 이중연결리스트(Doubly-linked-list)

![](https://wayhome25.github.io/assets/post-img/cs/doubly-linked-list.png)

-   linkedlist에서 각 노드에 next에 대한 포인터 외에 prev에 대한 포인터가 있는 자료 구조이다.
-   linkedlist에서 할 수 없는 거꾸로 탐색하는 기능이나, 특정 노드의 이전 노드에 삽입하거나 이전 노드를 삭제하는 것이 가능하다.

## Reference

[https://daimhada.tistory.com/72?category=820522](https://daimhada.tistory.com/72?category=820522)  
[https://wayhome25.github.io/cs/2017/04/17/cs-19/](https://wayhome25.github.io/cs/2017/04/17/cs-19/)