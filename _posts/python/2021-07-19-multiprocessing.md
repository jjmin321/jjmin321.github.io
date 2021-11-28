---
title:  "[작성 중] Concurrency와 Parallelism"
toc: true
toc_sticky: true

categories:
  - Python
tags:
  - TODO
use_math: true
last_modified_at: 2021-07-19
---

## 들어가며

최근 실무에서 일을 처리하며 병렬 처리의 필요성을 느끼게 되었다. 예를들어 현재 처리하고 있는 프로그램의 구성을 보면 BERT를 통해 분석한 후, 형태소 분석이나 키워드 추출 등의 과정을 거치게 되는데, 사실 이 둘의 프로세스는 따로 구성이 가능하므로 직렬로 연결하지 않고 병렬로 연결하여 구성하는 것이 더욱 time-efficient하다. 또한, 병렬처리는 딥러닝에서도 필수적이라 볼 수 있는데, 큰 모델의 경우 병렬 GPU를 활용하게 되기 때문이다. 따라서 병렬화에 대한 이해도를 높힐 필요가 있겠다.

이번 포스트에서는 병렬처리의 개념과 파이썬에서의 활용법에 대해 소개해보고자 한다. 아래는 이번 시간에 다룰 목록이다.

- 동시성/병렬성 개념 소개
  - 동시성은 threading이고 병렬성은 multiprocessing?
- `multiprocessing` vs. `threading`
- `multiprocessing.Process` vs. `multiprocessing.Pool`
- future, async, 
- 용어정리
  - process, thread, 동기, 비동기, 

언제나 그렇듯, 공부를 위한 기록을 남겨놓는 것이니 그 외 자세한 내용은 직접 찾아보길 바란다.

우선은 앞으로 다루게 될 용어에 대해 간략하게 살펴보자.

## Process와 Thread, task

![image](https://user-images.githubusercontent.com/47516855/127119914-9ae704bc-61b8-48a6-800c-6664f2f9eee4.png)

**프로그램(Program)**은 **어떤 작업을 위해 실행할 수 있는 파일**이다. 우리가 작성한 코드라던가 데이터로 이루어져있다. 

**프로세스(Process)**란 **실행되고 있는 컴퓨터 프로그램**으로, 메모리(RAM)에 올라와 실행되고 있는 프로그램의 인스턴스를 의미한다 (당연하지만 메모리 뿐만 아니라 많은 리소스가 할당된다).

**스레드(Thread)**란 **프로세스 내에서 실행되는 여러 흐름의 단위**로, CPU에 의해 스케줄링된다. 

**태스크(task)**란 **작업의 최소 단위**로, process, thread가 된다.

## Concurrency vs. Parallelism

동시성과 병렬성도 꽤나 헷갈리는 개념 중 하나이다. 우선, 동시성(Concurrency)의 정의는 다음과 같다.

> Concurrent computing is a form of computing in which several computations are executed **concurrently**.

병렬성(Parallelism)의 정의는 다음과 같다.

> Parallel computing is a type of computation where many calculations or the execution of processes are carried out **simultaneously**.

concurrently와 simultaneously는 사전적으로 '동시에'라는 뜻을 갖는다. 그렇다면 프로그래밍에서는 어떻게 다른것일까?

![image](https://user-images.githubusercontent.com/47516855/127114301-d6111712-6407-4492-82e4-d9e47dbc8792.png){: .align-center}{: width="400"}

위 그림을 보면 task A와 task B가 '동시에' 진행되고 있다. 현재로서는 **concurrent**하게 진행되는 것은 맞지만 아직 **simultaneously**한지는 알 수 없다.

![image](https://user-images.githubusercontent.com/47516855/127114500-bca4ab7c-fa8b-4bfd-b3ba-d22d4453ce86.png){: .align-center}{: width="400"}

Case 1은 single core에서 multi-thread를 다루는 것으로 CPU가 번갈아가며 두 작업을 진행한다.

Case 2는 multi core에서 multi-thread를 다루는 것으로 두 개의 작업이 동시에 진행된다.

즉, Case 1은 **concurrently**하지만 simultaneously는 아니다.  
Case 2는 **concurrently**하며 **simultaneously**하다.

프로그래밍에서의 정의는 다음과 같다.
- concurrently : 2개 이상의 ask (= 코드, 알고리즘 등) 를 수행할 때, 각 task 는 다른 task 의 수행시점에 상관없이 수행이 가능하다는 의미 (sequencial 의 반의어). 즉, N 개의 task 의 실행 시간이 타임라인 상에서 겹칠 수 있음.
- simultaneously : 일반적으로 사용하는 '동시에'

다음의 예시를 살펴보자.

> 우선 동시성(Concurrency)과 병렬성(Parallelism)의 차이에 대해 이야기 해보자. 철수가 소파에 앉아있고, 앞에 3개의 TV 가 서로 적당한 간격을 두고 있다고 가정하자. 철수는 자신이 가장 좋아하는 예능 프로그램인 무한도전, 1박2일, 아는형님을 각각의 TV 에 틀어놓았다. 3개의 예능 프로그램을 너무나도 좋아한 나머지 동시에 보고싶었기 때문이다. 철수는 우선 무한도전을 틀어놓은 TV를 10초 보다가, 1박2일을 틀어놓은 TV를 3초 보다가, 아는형님을 틀어놓은 TV를 5초간 보았다.
여기서 알 수 있는 사실은, **철수는 3개의 예능 프로를 동시에(Concurrent) 시청하고 있지만 한 번에 하나의 TV 만 볼 수 있다**. 즉, 눈이 3쌍이 아니기 때문에 3개의 TV 를 한 번에(Parallel)볼 수는 없는 것이다. 물론 한국어와 영어 단어 사이에 1대1 대응이 어려워 두 행위 모두 '동시에'라고 얘기할 수 있기 때문에 헷갈릴 수도 있다. 하지만 철수가 한 번에 하나의 TV 만 볼 수 있기 때문에 병렬성은 없는 것이며, 동시에 여러개의 예능 프로를 본다는 것은 동시성은 있는 것이다.
> 
> 컴퓨터의 세계에서는 철수를 컴퓨터로, 철수의 눈 한 쌍이 CPU 의 코어 하나라고 볼 수 있고, 각 예능 프로그램이 스레드라고 볼 수 있다. 여러개의 스레드가 실행중이지만 싱글 코어 CPU 이기 때문에 한 번에 하나의 작업만 수행할 수 있는 것이다. 물론 여러 스레드를 아주 빠른 속도로 번갈아 수행하면 마치 병렬적으로 수행되는것처럼 보일 수도 있다. 반면 멀티 코어 CPU 에서 여러개의 스레드가 동시에 여러개의 코어에서 실행될 수 있다. 어떤 특정 시점에 두 개 이상의 스레드가 코어에서 실행 중이라면 그것은 병렬성이 있다고 하는것이다.
>
> [파이썬과 동시성 프로그래밍](https://sgc109.github.io/2020/11/25/python-and-concurrency/)

즉, 정리해보면, 병렬성은 동시성에 포함되는 개념이라고 볼 수 있다.

그렇다면 앞서 살펴보면 스레드와 프로세스는 동시성과 병렬성에 어떻게 연관이 될까? 병렬성과 동시성은 **모두 스레드와 프로세스를 사용**한다. 즉, 스레드와 프로세스는 어느쪽에서든 **도구로 이용**된다는 뜻이다. 따라서 그 자체로 구분의 지표가 될 수 없다. 스레드는 동시성을 달성하기 위한 수단으로 사용되지만, 유일한 방법은 아니다. 병렬성은 하나의 같은 작업에 대해한 여러개의 하드웨어 리소스(single CPU에서의 유휴자원을 끌어다가 쓰는 것 이상으로)를 의미한다. 아래는 "파이썬 병렬 프로그래밍"에서 발췌한 내용이다.

> 동시성 프로그래밍 내부에서 프로그램이 여러 작업자에게 할당한 후, 이 작업자들이 태스크를 실행하기 위해 CPU를 사용하려고 서로 경쟁하는 시나리오가 있다. 경쟁이 발생하는 단계에서는 특정 시점에 자원을 사용하기에 적합한 작업자를 정의하는 기능을 갖춘 CPU 스케줄러가 제어한다. 대부분의 경우에서는 CPU 스케줄러가 프로세스를 훑는 태스크를 순식간에 실행하기 때문에 의사병렬을 느낄 수도 있다. 그런 이유로 동시성 프로그래밍은 병렬 프로그래밍의 추상화다.
>
> *동시성 시스템은 태스크를 실행하기 위해 동일한 CPU를 두고 경쟁한다*
>
> 작업자들이 CPU에 동시 접근할 필요가 없는 멀티코어 환경에서는 프로그램 데이터가 작업자를 생성해 특정 태스크를 동시에 실행하는 방식으로 병렬 프로그래밍을 정의할 수 있다.
>
> *병렬 시스템은 태스크를 동시에 실행한다.*

## python에서의 예시: `threading`과 `multiprocessing`

Python에선 동시성 프로그래밍을 위한 두 가지 모듈을 제공한다. 바로 `threading`과 `multiprocessing`이다.


### `threading`

Python에서의 `threading.Thread`은 진정한 의미에서의 multi-threading이라 할 수 없다. 이는 Python의 Global Interpreter Lock (GIL) 때문이다.

GIL로 인해 파이썬에서는 하나의 process 내에서 여러개의 thread가 병렬적으로 실행될 수 없다. 즉, 멀티 코어 CPU에서 동작한다고 하더라도 하나의 프로세스는 동시에 여러개의 코어를 사용할 수 없다는 뜻이다. 그렇기 때문에 만약 수행하고자 하는 작업이 CPU bound job이고 multi-core CPU 환경인 경우에는 멀티프로세싱을 사용하는 것이 유리하다. 왜냐하면 하나의 프로세스 내에서 아무리 여러개의 스레드를 만들어봐야, 하나의 스레드에서 순차적으로 수행하는것과 비교하여 딱히 성능이 좋아지지 않기 때문이다. context switching을 생각하면 멀티스레딩 쪽이 오히려 더 느릴 수도 있다. 게다가 여러개의 스레드를 사용하면 메모리 사용량도 많아진다.

하지만 만약 수행하고자 하는 작업이 I/O bound job 이라면 이야기가 달라진다. 어떤 스레드가 I/O 를 수행하기 위해 block 되면 GIL을 반환하게 되고, 그 동안 다른 스레드가 실행될 수 있기 때문이다. 물론 복수의 스레드가 복수의 코어에서 병렬적으로 실행될 수 없다는 사실은 변함이 없지만, 하나의 스레드만 사용하여 여러 작업을 동시에 수행하고자 하는 경우에는 이 스레드가 block 이 되면 아무런 일도 하지 않게 되기 때문에 이런 경우에는 멀티스레딩을 사용할 가치가 충분히 있는것이다. 하지만 스레드는 직접 사용하기가 까다롭다. 경쟁상태(race condition)도 발생할 수 있고, 메모리 사용량과 context switching 측면에서도 비용이 비싸다.

`threading.Thread`를 사용하는 방법에는 두 가지가 있다. 하나는 클래스로 상속하는 것이고, 다른 하나는 `Thread` 객체에 함수를 집어넣어 활용하는 것이다.

```python
# 1: threading.Thread를 상속하여 진행
import threading
import time

class Worker(threading.Thread):
    def __init__(self, name, time):
        super().__init__()
        self.name = name    # thread 이름 지정
        self.time = time

    def run(self):
      # Override
        print("sub thread start ", threading.currentThread().getName())
        time.sleep(self.time)
        print("sub thread end ", threading.currentThread().getName())


print("main thread start")

t1 = Worker("1", 3)   # sub thread 생성
t1.start()    # sub thread의 run 메서드를 호출

t2 = Worker("2", 5)   # sub thread 생성
t2.start()    # sub thread의 run 메서드를 호출

t1.join()   # thread가 작업을 완료할 때 까지 응답대기
t2.join()   # thread가 작업을 완료할 때 까지 응답대기

print("main thread post job")
print("main thread end")
```

```python
# 2: Thread에 함수를 넣어 실행
import threading


def run(sec):
    print("sub thread start ", threading.currentThread().getName())
    time.sleep(sec)
    print("sub thread end ", threading.currentThread().getName())
    
print("main thread start")

t1 = Thread(target=run, args=(3, ))   # sub thread 생성
t1.start()    # sub thread의 run 메서드를 호출

t2 = Thread(target=run, args=(5, ))   # sub thread 생성
t2.start()    # sub thread의 run 메서드를 호출

t1.join()   # thread가 작업을 완료할 때 까지 응답대기
t2.join()   # thread가 작업을 완료할 때 까지 응답대기

print("main thread post job")
print("main thread end")
```

예제 코드에서 thread를 만든 후, `start`를 통해 실행시켜주었다. 그리고 `join`을 통해 각 스레드가 종료되기 전까지 대기하도록 하였다.

이 결과 `t1`, `t2` thread가 작업을 완료하면 main thread가 동작한다. 대기시간이 3초로 더 짧은 `t1`에만 `join`을 걸어놓으면 `t2`의 완료여부에 상관없이 main thread는 종료된다.

### `multiprocessing`

Below information might help you understanding the difference between Pool and Process in Python multiprocessing class:

Pool:    
When you have junk of data, you can use Pool class. Only the process under execution are kept in the memory.
I/O operation: It waits till the I/O operation is completed & does not schedule another process. This might increase the execution time.
Uses FIFO scheduler.

Process:    
When you have a small data or functions and less repetitive tasks to do.
It puts all the process in the memory. Hence in the larger task, it might cause to loss of memory.
I/O operation: The process class suspends the process executing I/O operations and schedule another process parallel.
Uses FIFO scheduler.

#### `Process`

이제 Process 에 대해서 살펴봅시다
Process 는 Pool 과는 다르게 인수를 퐁당퐁당 건너서 제공하는 것은 아니고 그저 하나의 프로세스를 하나의 함수에 적당한 인자값을 할당 해주고(없어도 됩니다) 더이상 신경을 안씁니다.  예제를 보겠습니다.

각의 프로세스사이에서 서로 값의 communication 이 필요할때 multiprocessing 은 두가지 방법을 제공합니다. Queue 와 Pipe 입니다. 

> ‘Process’ halts the process which is currently under execution and at the same time schedules another process. ‘Pool’ on the other hand waits till the current execution in complete and doesn’t schedule another process until the former is complete which in turn takes up more time for execution. ‘Process’ allocates all the tasks in the memory whereas ‘Pool’ allocates the memory to only for the executing process. You would rather end up using Pool when there are relatively less number of tasks to be executed in parallel and each task has to be executed only once.

---

Its used when function based parallelism is required, where I could define different functionality with parameters that they receive and run those different functions in parallel which are doing totally various kind of computations.



#### `Pool`

> "A prime example of this is the Pool object which offers"
> "a convenient means of parallelizing the execution of a function across"
> "multiple input values, distributing the input data across processes" 
> "(data parallelism)"

위는 Python document 에 있는 Pool 에 대한 설명입니다.
" 입력값을 process들을 건너건너 분배하여 함수실행의 병렬화 하는 편리한 수단을 제공한다. "
정확한 표현인듯 합니다. 이를 설명하는 예제를 살펴보겠습니다.

---

It offers a convenient means of parallelizing the execution of a function across multiple input values, distributing the input data across processes i.e. data based parallelism. The following example demonstrates the common practice of defining such functions in a module so that child processes can successfully import that module.

## `concurrent.futures`

추상화된 버전.





https://jinwoo1990.github.io/dev-wiki/python-concept-4/

http://pertinency.blogspot.com/2019/10/join.html

https://www.ellicium.com/python-multiprocessing-pool-process/

https://zzaebok.github.io/python/python-multiprocessing/

https://github.com/remzi-arpacidusseau/ostep-translations/blob/master/korean/README.md

https://tutorialedge.net/python/python-multiprocessing-tutorial/

https://soooprmx.com/concurrent-futures/

https://data-newbie.tistory.com/231

https://velog.io/@chan33344/%EB%8F%99%EC%8B%9C%EC%84%B1-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EB%B9%84%EB%8F%99%EA%B8%B0-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D

https://yganalyst.github.io/data_handling/memo_17_parallel/

https://velog.io/@wltjs10645/Python-thread2

https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python

https://www.notion.so/Python-adv-Thread-a3a151a2507c42cfab85d8573f1ca513

https://velog.io/@wltjs10645/Python-thread2

https://velog.io/@otzslayer/Ray%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%B4-Python-%EB%B3%91%EB%A0%AC-%EC%B2%98%EB%A6%AC-%EC%89%BD%EA%B2%8C-%ED%95%98%EA%B8%B0



Celery 는 Python 동시성 프로그래밍에서 가장 많이 사용하는 방법 중 하나이며, 분산 메시지 전달을 기반으로 동작하는 비동기 작업 큐(Asynchronous Task/Job Queue)이다.
이는 Python Framework 라고도 하지만 보통 Worker라고 불린다.
Worker는 웹 서비스에서 Back단의 작업을 처리하는 별도의 프레임이며, 사용자에게 즉각적인 반응을 보여줄 필요가 없는 작업들로 인해 사용자가 느끼는 Delay를 최소하 화기 위해 사용 된다.

예를 들어, 웹 서비스에서 응답 시간은 서비스의 생명과 직결되므로 비동기로 작업을 처리하게 넘기고 바로 응답을 하기 위해 사용 된다.

Celery는 메시지를 전달하는 역할(Publisher)과 메시지를 Message Broker에서 가져와 작업을 수행하는 Worker의 역할을 담당하게 된다.

프로세스는 '실행 중인 프로그램'입니다.

즉, 프로그램이 실행 중이라는 것은 디스크에 있던 프로그램을 메모리에 적재하여 운영체제의 제어를 받는 상태가 되었다는 것입니다.

이는 프로세서를 할당받고, 자신만의 메모리 영역이 있음을 의미하고,
프로그램이 프로세스가 되려면 프로세서 점유 시간, 메모리 그리고 현재의 활동 상태를 나타내는 PC(Program Counter), SR(Status Register) 등이 포함됩니다.

따라서 프로그램은 단지 정적 데이터만을 포함하는 정적인 개체,
프로세스는 현재 활동 상태를 포함하고 있어 동적인 개체라고 볼 수 있습니다.
[출처] 프로세스, 스레드, 프로그램의 차이|작성자 예비개발자


참고자료:
- https://vagabond95.me/posts/concurrency_vs_parallelism/
- https://gmlwjd9405.github.io/2018/09/14/process-vs-thread.html