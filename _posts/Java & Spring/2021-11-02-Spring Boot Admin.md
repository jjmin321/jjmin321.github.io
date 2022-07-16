---
title: "Spring Boot Admin"
excerpt: "Spring Boot Admin에 대해 알아보자"
toc: true
toc_sticky: true
date: 2021-11-02 00:01:28 -0400
categories: 
    - Java & Spring
---

서버의 메모리 사용 비율 및 쓰레드 상태를 확인하기 위해 사용할 서버 모니터링 방식 중 하나인 Spring Boot Admin을 간략히 사용해본다.
<hr/>

# Spring Boot Admin

spring boot admin 은 spring boot 어플리케이션들을 모니터링하고 관리하기 위한 웹 어플리케이션이다. 각각의 어플리케이션은 client 로 간주되고 admin server 에 등록된다. spring boot actuator endpoints 만 열어두면 알아서 설정이 된다.

정리해볼 내용은 아래와 같다.

1. spring boot admin client 1개 구현 (with spring boot actuator)

2. spring boot admin server 1개 구현

인증도 추가하여 로그인을 해야만 모니터링을 할 수 있게끔 설계할수도 있지만, 필요 없을듯 하다.

## 1. 의존성 추가 (maven 기준 작성)

모니터링할 클라이언트로 만들기 위해서 필수 dependency 는 actuator 와 admin-client 이다.

```
<dependency>
   <groupId>de.codecentric</groupId>
   <artifactId>spring-boot-admin-starter-client</artifactId>
</dependency>
<dependency>
	<groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

그리고 서버 역할을 하기 위해서는 admin-server도 추가한다.

```yaml
<dependency>
   <groupId>de.codecentric</groupId>
   <artifactId>spring-boot-admin-starter-server</artifactId>
</dependency>
```

## 2. Spring Boot Admin Server 역할로 만들기

`@EnableAdminServer` 을 메인 어플리케이션에 추가한다.

```
@EnableAdminServer
@SpringBootApplication
public class MainApplication {

   public static void main(String[] args) {
      SpringApplication.run(MainApplication.class, args);
   }
}

```

그 후 어플리케이션 설정 파일에 아래와 같은 설정값들을 추가한다.
```
spring.application.name: test
server.port: 8888
spring.boot.admin.context-path: /admin

# spring boot admin 에 hostname 대신 ip 등록을 강제 하기 위해
spring.boot.admin.client.instance.prefer-ip: true

# spring boot admin 에 노출될 앱 정보
info.app.name: 테스트 모듈
info.app.description: 스프링 부트 어드민 테스트로 구동해본 클라이언트 모듈입니다.
info.app.version: 1.0.0

# monitoring metric 수집 및 동작을 위해 actuator 허용
management.endpoints.web.exposure.include: '*'
```

그 후 어플리케이션 구동 후 모니터링 페이지가 잘 올라갔는지 확인한다.
![image](https://user-images.githubusercontent.com/52072077/178140752-636a8bac-1e1f-48c0-88b7-c4805d2839c9.png)


## 3. Spring Boot Admin Client로 설정하기

클라이언트로 등록하기 위한 설정값은 단순히 spring.boot.admin.client 만 있으면 된다. 
서버의 url을 넣어주면 되고, 같은 장비에서 클라이언트와 서버를 올리는 경우에는 본인 서버의 ip로 등록하면 된다.

```
# spring boot admin 접속 설정
spring.boot.admin.client.url: http://127.0.0.1:8888	
``` 

그 후 모니터링 페이지에서 제대로 클라이언트 모듈이 올라갔는지 확인한다.
![image](https://user-images.githubusercontent.com/52072077/178141011-aa7e968c-e6f2-4ce6-8818-4044743006ff.png)


## 4. Spring Boot Admin 모니터링

스프링 부트 어드민을 통해 모니터링할 수 있는 대표적인 예시들이다.

### 4-1. 모듈의 Health, Thread, Process 검사
![image](https://user-images.githubusercontent.com/52072077/178141083-d8a00f7d-a23b-4549-9911-9a8f21a1cecf.png)

### 4-2. 모듈에서 사용중인 properties
![image](https://user-images.githubusercontent.com/52072077/178141147-db08e2ad-f2db-4f7e-ad70-f38fe9df4a49.png)

### 4-3. 모듈에서 사용중인 beans
![image](https://user-images.githubusercontent.com/52072077/178141155-bd102c64-f348-43f4-b72a-1a0deca7ea24.png)

### 4-4. JVM의 threads 및 heap dump
![image](https://user-images.githubusercontent.com/52072077/178141177-5325e1c0-0f85-4621-964b-91bb79f92eb3.png)

![image](https://user-images.githubusercontent.com/52072077/178141184-c4baa9ee-7823-4a89-8b7b-893b5029644b.png)

## 5. 총 정리
위에서 알아봤듯이 아주 짧은 시간을 들여서 Spring Boot Admin을 결합시킬 수 있다. 

설정하는 데에 소요되는 시간에 비해 제공받을 수 있는 유용한 정보들이 꽤나 많다.

스프링 부트 용이라는 게 조금 아쉽지만, 스프링 부트를 사용할 때 간단한 설정들만으로 UI를 구성해서 관리할 수 있는 좋은 모니터링 도구라고 생각된다.