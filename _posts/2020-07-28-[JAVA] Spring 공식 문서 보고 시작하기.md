---
title: "[JAVA] Spring 공식 문서 보고 시작하기 2020-07-28"
date: 2020-07-28 11:15:28 -0400
categories: Development
---

# 🏄‍♂️ Spring Tutorial 7/28/2020 ~ 8/4/2020 🏄‍♂️
🔖 https://spring.io/projects/spring-boot라는 스프링 공식 사이트의 문서를 참조하였습니다. 🔖

## 👨‍💻스프링 웹 개발 👨‍💻
예전에는 JSP, PHP같은 정적 코드를 동적으로 꾸며주는 웹 서버가 주로 개발되었지만 
최근에는 NodeJS, Spring, Django를 이용하여 API 방식을 통해 필요한 데이터만 처리해준다. 
우리나라의 대부분의 기업들이 Spring을 서버 개발자의 기본 사항으로 여기며 배우기에는 어렵지만 배워놓으면 매우 좋은 프레임워크라서 시작하게 되었다.

- 작업 환경 
    - Mac Os Mojave 10.14.6
    - Java 8 
    - InteliJ 2020.1.4
    
- 작업 설정 (https://start.spring.io)
    - Spring Boot 2.3.2
    - Maven Project
    - Thymeleaf Template engine
    - Spring Web Service

## 스프링의 구조 
- MVC 패턴이 고정되어있으며 아래와 같은 폴더 구조를 보인다. 
    - Java/Application은 SpringApplication을 실행하는 메인 클래스를 포함
    - Java/Controller는 클라이언트가 요청한 방식, 데이터를 받아서 로직을 통해 처리함
    - Resources/Static에는 정적파일들을 넣어 사용할 수 있다.
    - Resources/Templates는 정적파일들을 넣지만 컨트롤러와 연동하여 동적으로 사용할 수 있다.

```
src/main    
│
└───Java
│   └─── hello.hellospring
│         └───   Controller 
│         │  Application
│   
└───Resources
    └─── Static 

    └─── Templates
    
    │ application.properties
```

## 스프링에서 정적 파일을 띄우는 법 
Resources/Static 에 정적 파일을 만들고 톰캣으로 정적 파일을 띄울 수 있다. 
<br>
<img width="350" alt="image" src="https://user-images.githubusercontent.com/52072077/88655957-81749200-d10a-11ea-9b41-cee51f361c08.png">
<br>
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/52072077/88656665-a3224900-d10b-11ea-8a21-d9b185cfc235.png">


## 스프링으로 정적 파일을 동적으로 만드는 법
- 쿼리 파라미터로 값 받아서 그에 맞는 다른 값을 서버에서 반환해주는 예시
    - hello.hellospring/Controller에서 hello 메서드를 만들어준다. 

    ```java
    @Controller
    public class HelloController {
        @GetMapping("hello")
        public String hello(@RequestParam("name") String name, Model model) {
            model.addAttribute("name", name);
            return "hello";
        }
    }
    ```
    - return "hello"를 한 메서드의 오브젝트 변수를 스프링이 Resources/Template/hello.html 파일을 찾아 사용할 수 있게 해준다.

    ```html
    <!DOCTYPE html>
    <html xmlns:th="http://www.thymeleaf.org">
    <head>
        <title>Spring Welcome Page</title>
    </head>
    <body>
    <p th:text="'안녕하세요' + ${name}" >안녕하세요. NULL</p>
    </body>
    </html>
    ```
    <br>
    - localhost:8080?name=제정민 을 입력하면 내가 원한대로 동작이 된다.
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/52072077/88768162-a3771e80-d1b5-11ea-9a01-ffac020adb48.png">

## 스프링 빌드 방법
빌드를 통해 보통 리눅스를 사용하는 서버 컴퓨터에서 자바 코드를 모두 작성할 필요 없이 빌드파일로 실행이 가능하다.

1. 프로젝트 폴더 터미널을 통해 ./gradlew build 명령어를 입력한다.
<img width="1376" alt="image" src="https://user-images.githubusercontent.com/52072077/88643291-7adf1e00-d0fc-11ea-9829-3e705cd65301.png">

2. 명령어를 입력한 후 ls명렁어를 통해 하위 폴더를 확인해 보면 build 폴더가 생긴 것을 알 수 있다.
<img width="1375" alt="image" src="https://user-images.githubusercontent.com/52072077/88643325-8599b300-d0fc-11ea-9e11-59bd30cb4aa6.png">


3. cd명령어로 build/libs 경로로 이동을 하면 jar 빌드 파일이 생긴 것을 확인할 수 있다.
<img width="1365" alt="image" src="https://user-images.githubusercontent.com/52072077/88643355-90ecde80-d0fc-11ea-8edc-cf7b083fc567.png">

4. java -jar (파일) 명령어로 SPRING을 빌드할 수 있다.
<img width="1383" alt="image" src="https://user-images.githubusercontent.com/52072077/88643189-5e42e600-d0fc-11ea-99db-50bf7bd6adfc.png">

## MVC 패턴 및 템플릿 엔진 
