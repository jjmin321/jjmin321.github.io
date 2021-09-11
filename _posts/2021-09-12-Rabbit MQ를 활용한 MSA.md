---
title: "Rabbit MQ를 활용한 MSA 2021-09-12"
date: 2021-09-12 00:01:28 -0400
categories: Development
---

마이크로서비스 아키텍처는 메시지 브로커를 기반으로 효율적으로 동작합니다. 메시지 브로커의 한 종류인 Rabbit MQ를 직접 사용해보며 배워봅니다.
<hr>

## Docker로 띄워봅시다
기본 포트는 15692 입니다. (왜인지 모름, 공식사이트에선 5672라는데 어찌저찌 해보니 15692로 됐고, https://www.rabbitmq.com/networking.html 사이트를 참조해보면 됨.)