---
title:  "백준에서 파이썬으로 입력받기"
excerpt: "sys.stdin을 사용해보자"
toc: true
toc_sticky: true

categories:
  - Algorithm
tags:
  - BOJ
  - Python
  - sys.stdin
last_modified_at: 2020-07-08
---

백준에서 알고리즘을 풀려면 데이터를 입력 받아야 한다. 여태까지는 `input()`을 썼지만, 이러면 속도가 느리기 때문에 `sys.stdin`에 있는 메소드를 사용하는 것이 좋다.

## 한 줄에 여러 데이터가 입력되는 경우

`sys.stdin.readline()`을 이용한다. `sys.stdin.readline()`는 띄어쓰기도 입력 받으니 적절히 처리해주어야 한다. 이 때, `map`을 이용하여 casting하면 더 빠르다.

```python
import sys 

a = map(int, sys.stdin.readline().split())

print(list(a))
```

결과

```
1 2 3
[1, 2, 3]
```

map은 iterator를 반환한다는 사실에 유의하자.




## 여러 줄에 걸쳐 데이터가 입력되는 경우

두 가지 방법이 있다. 첫 번째는 반복되는 수만큼 `sys.stdin.readline()`를 사용하는 것이다.


```python
import sys

num = int(sys.stdin.readline())
li = list()

for i in range(num):
    li.append(sys.stdin.readline().strip())

print(li)
```

결과
```
3
1
2
3
['1', '2', '3']
```


혹은 다음과 같은 방법을 이용할 수 있다. 아래의 경우가 위의 경우보다 더 빠르다고 한다 ([출처](https://choisblog.tistory.com/25)).


```python
import sys

num = int(sys.stdin.readline())
a = [sys.stdin.readline().strip() for i in range(num)]
print(a)
```

결과

```
3
1
2
3
['1', '2', '3']
```