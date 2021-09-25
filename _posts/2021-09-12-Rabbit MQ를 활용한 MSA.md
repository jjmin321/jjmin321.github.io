---
title: "Rabbit MQ를 활용한 MSA 2021-09-12"
date: 2021-09-12 00:01:28 -0400
categories: Development
---

마이크로서비스 아키텍처는 이벤트 브로커와 메시지 브로커를 기반으로 효율적으로 동작합니다. 이번 글에서는 메시지 브로커의 한 종류인 Rabbit MQ를 직접 사용해보며 배워봅니다.
<hr>

## Docker로 띄워보기
아래 명령어를 통해 이미지를 다운받고, 15672 포트로 실행해보겠습니다. (로컬의 15672번 포트를 도커의 15672번 포트, 즉 Rabbit Mq에 포워딩시키고 컨테이너 이름을 blog라고 지었습니다)
```
docker pull rabbitmq:3-management
docker run -p 15672:15672 --name blog rabbitmq:3-management
```
![image](https://user-images.githubusercontent.com/52072077/134161866-69247607-97d5-42d0-b7ef-9a001127fa8f.png)
이렇게 실행 후 15672 포트로 접속했을 때 화면이 정상적으로 출력하면 됩니다. (안 될 경우 도커의 15672번 포트 외 다른 포트로 접속을 시도해보세요)
<br>

## Queues의 정의
일단 기본 id와 pw는 guest, guest입니다. 로그인을 하면 이 화면이 출력됩니다. 
![image](https://user-images.githubusercontent.com/52072077/134162620-c737d197-92c0-487e-9e9f-43bc51287238.png)
이 화면에 보이는 Queues 가 바로 Rabbit Mq의 핵심적인 역할을 하는 것입니다. 메시지 브로커에서 메시지가 추가되고 처리될 수 있게 해줍니다.
<br>
좀 더 쉽게 말하자면 하나의 Rabbit Mq 인스턴스에 여러 개의 Queue를 만들어 처리할 수 있는데, A에 관련된 메세지는 A Queue에, B 이벤트에 관련된 메세지는 B Queue에 각각 만들어 사용할 수 있다는 뜻입니다. 

## Queue 만들기
Prefix로 TEST를 사용하고, Name으로 GO_TO_JAVA를 사용하여 Queue를 만들어 보겠습니다.
![image](https://user-images.githubusercontent.com/52072077/134671502-47eeb3b2-ca86-43b6-ac23-ae4db46f2b08.png)
Queue를 만들었으니 만들어진 Queue에 들어가서 메세지를 한 번 넣어보겠습니다. 
![image](https://user-images.githubusercontent.com/52072077/134671863-fa8b31d3-2884-47ea-bdb8-a3bf3621f8cd.png)
메세지를 넣게 되면 아래 그래프에 메세지가 추가되는 것을 볼 수 있으며, 이를 통해 실시간으로 메세지에 대해서 모니터링할 수 있습니다.
![image](https://user-images.githubusercontent.com/52072077/134671983-34e64c32-2814-44ea-ba15-ff112c821bc1.png)


