---
title: "Spring Data JPA를 사용한 CRUD 처리 2021-04-04"
date: 2021-04-04 00:01:28 -0400
categories: Development
---

스프링에서 JPA를 편리하게 사용할 수 있도록 지원하는 Spring Data JPA를 프로젝트에 추가하여 CRUD 처리를 해봅니다
<hr/>

## [ Spring Data JPA ]
JPA를 쓰기 편하게 만들어 놓은 모듈로 쉽고 편하게 사용할 수 있게 도와준다<br>
JPA를 한 단계 추상화 시킨 Repository라는 인터페이스를 제공한다<br>
Repository 인터페이스에 정해진 규칙대로 구현체 없는 메소드를 생성하면, Spring이 알아서 쿼리를 날리는 구현체를 만들어 Bean으로 등록해준다

![img](https://user-images.githubusercontent.com/52072077/113500898-a86a4a00-955c-11eb-9872-eefba124581f.png)