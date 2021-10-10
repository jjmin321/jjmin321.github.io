---
title: "Rabbit MQ를 활용한 MSA 2021-09-12"
date: 2021-09-12 00:01:28 -0400
categories: Development
---

마이크로서비스 아키텍처는 이벤트 브로커와 메시지 브로커를 기반으로 효율적으로 동작합니다. 이번 글에서는 메시지 브로커의 한 종류인 Rabbit MQ를 직접 사용해보며 배워봅니다.
<hr>

## Docker로 띄워보기
아래 명령어를 통해 이미지를 다운받고, 15672 포트로 실행해보겠습니다. (5672번은 rabbitmq의 기본 포트이고, 15672번은 모니터링 기본 포트입니다)
```
docker pull rabbitmq:3-management
docker run -p 5672:5672 -p 15672:15672 --name rabbitmq rabbitmq:3-management
```
![image](https://user-images.githubusercontent.com/52072077/134161866-69247607-97d5-42d0-b7ef-9a001127fa8f.png)
이렇게 실행 후 15672 포트로 접속했을 때 화면이 정상적으로 출력하면 됩니다.
<br>

## Queues의 정의
일단 기본 id와 pw는 guest, guest입니다. 로그인을 하면 이 화면이 출력됩니다. 
![image](https://user-images.githubusercontent.com/52072077/134162620-c737d197-92c0-487e-9e9f-43bc51287238.png)
이 화면에 보이는 Queues 가 바로 Rabbit Mq의 핵심적인 역할을 하는 것입니다. 메시지 브로커에서 메시지가 추가되고 처리될 수 있게 해줍니다.
<br>
좀 더 쉽게 말하자면 하나의 Rabbit Mq 인스턴스에 여러 개의 Queue를 만들어 처리할 수 있는데, A에 관련된 메세지는 A Queue에, B 이벤트에 관련된 메세지는 B Queue에 각각 만들어 사용할 수 있다는 뜻입니다. 

## Queue 만들기
모니터링 페이지에서도 자유롭게 Queue를 만들고, 삭제하고 또는 메세지를 보내고 빼내는 작업을 할 수 있습니다. 하지만 실제로 처리할 때는 코드를 거치는 작업이 필요하므로 GO를 사용해서 테스트해보겠습니다. (공식 문서에서 각 언어별로 모두 example 코드를 지원합니다)
<br>
```go
package main

import (
	"log"

	"github.com/streadway/amqp"
)

func main() {
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	defer conn.Close()

	ch, err := conn.Channel()
	defer ch.Close()

	q, err := ch.QueueDeclare(
		"hello", // name
		false,   // durable
		false,   // delete when unused
		false,   // exclusive
		false,   // no-wait
		nil,     // arguments
	)
}
```
<br>
이렇게 Queue를 생성할 수 있습니다.

![image](https://user-images.githubusercontent.com/52072077/136680964-b83cc0bf-94a3-4761-91e7-0980d9fe2936.png)

## 메세지 생성하기
Queue를 만들었으니 해당 Queue를 사용하여 메세지를 퍼블리싱 해보겠습니다.
<br>

```go
package main

import (
	"log"

	"github.com/streadway/amqp"
)

func main() {
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	defer conn.Close()

	ch, err := conn.Channel()
	defer ch.Close()

	body := "Hello World!"
	err = ch.Publish(
		"",     // exchange
		q.Name, // routing key
		false,  // mandatory
		false,  // immediate
		amqp.Publishing{
			ContentType: "text/plain",
			Body:        []byte(body),
		})
}
```
<br>
이렇게 메세지를 퍼블리싱할 수 있습니다. 코드를 실행해 보면 아래와 같이 메세지가 생성되는 걸 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/52072077/136682291-7ec1d1ea-c71f-4fcd-87e7-0d612cb78853.png)
