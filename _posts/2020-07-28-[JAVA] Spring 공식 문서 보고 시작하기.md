---
title: "[JAVA] Spring 공식 문서 보고 시작하기 2020-07-28"
date: 2020-07-28 11:15:28 -0400
categories: Development
---

# 🏄‍♂️ Spring Tutorial 7/28/2020 ~ 8/4/2020 🏄‍♂️
🔖<a href="https://spring.io/projects/spring-boot">스프링 공식 사이트</a>의 문서를 참조하였습니다. 🔖

# 👨‍💻스프링 웹 개발 👨‍💻
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

# ⚗️ 스프링의 구조 ⚗️
MVC 패턴이 고정되어있으며 아래와 같은 폴더 구조를 보인다. 
- Java/Application은 SpringApplication을 실행하는 메인 클래스를 포함
- Java/Controller는 클라이언트가 요청한 방식, 데이터를 받아서 로직을 통해 처리함
- Resources/Static에는 정적파일들을 넣어 사용할 수 있다.
- Resources/Templates는 정적파일들을 넣지만 컨트롤러와 연동하여 동적으로 사용할 수 있다.
- <a href="https://github.com/jjmin321/Spring_studybook" target="_blank">프로젝트 구조</a>

```
src/main    
│
└───Java
│   └─── hello.hellospring
│         └───   Controller 
|         └───   Domain
|         └───   Repository
|         └───   Service
│         │  Application
│   
└───Resources
    └─── Static 

    └─── Templates
    
    │ application.properties
```

# 🐼 스프링 패턴 🐼
Controller : 외부 요청을 받음
Service : 로직을 만듦
Repository : 데이터 저장을 함
Controller <-> Service <-> Repository 


## 🧸 스프링 빈 🧸
스프링 빈이란 스프링이 관리하는 자바 객체인데 일반적인 객체가 아닌 ApplicationContext가 만들어서 그 안에 담고 있는 객체를 의미한다.
- Component scanning
    - hello.hellospring 하위 폴더들은 스프링에서 자동으로 잡아준다.
- 빈 설정파일에 직접 등록
    - 그 외에는 직접 등록하여야 한다.

스프링에서 사용하는 클래스나 메서드 위에 @가 붙은 어노테이션은 Component scanning을 통해 빈으로 등록되어 스프링이 관리해준다. 
- 어노테이션을 사용할려면 @Component를 붙히면 되지만 많이 쓰는 어노테이션들은 자동으로 Component에서 인식을 해준다. 
- 컨트롤러로 사용할려는 클래스에 @Controller을 붙힌다.
- 테스트할 메서드에 @Test를 붙힌다.
- 스프링의 메인 메서드가 있는 클래스에 @SpringBootApplication을 붙힌다.

# 🚇 스프링 기본 메서드 -> ViewResolver으로 처리 🚇
#### ViewResolver : Resources폴더 하위 파일을 찾는다.
Resources/Static 에 정적 파일을 만들고 톰캣으로 정적 파일을 띄울 수 있다. 
<br>
<img width="350" alt="image" src="https://user-images.githubusercontent.com/52072077/88655957-81749200-d10a-11ea-9b41-cee51f361c08.png">

파라미터로 값을 받아 그에 따른 페이지 출력
- hello.hellospring/Controller에서 hello 메서드 생성
- <a href="https://github.com/jjmin321/Spring_studybook/blob/master/src/main/java/hello/hellospring/controller/HelloController.java" target="_blank">HelloController.java</a>

    ```java
    // src/main/java/hello.hellospring.controller/HelloController.java
    @Controller
    public class HelloController {
        @GetMapping("hello")
        public String hello(@RequestParam("name") String name, Model model) {
            model.addAttribute("name", name);
            return "hello";
        }
    }
    ```
- return "hello"를 하면 Resources/Template 폴더 아래의 hello라는 정적파일과 연동이 된다.
- <a href="https://github.com/jjmin321/Spring_studybook/blob/master/src/main/resources/templates/hello.html" target="_blank">hello.html</a>
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

# 🏗️ 스프링 @ResponseBody -> HttpMessageConverter로 처리 🏗️
### HttpMessageConverter : 파일을 찾지 않고 즉시 응답

문자열을 반환하면 html코드로 변환하여 반환된다.
- <a href="https://github.com/jjmin321/Spring_studybook/blob/master/src/main/java/hello/hellospring/controller/HelloController.java" target="_blank">HelloController.java</a>

```java
// src/main/java/hello.hellospring.controller/HelloController.java
@GetMapping("hello-spring")
    @ResponseBody
    public String helloString(@RequestParam(value = "name", required = false) String name){
        return "hello" + name;
    }
```

객체를 반환하면 JSON코드로 변환하여 반환된다. 
- <a href="https://github.com/jjmin321/Spring_studybook/blob/master/src/main/java/hello/hellospring/controller/HelloController.java" target="_blank">HelloController.java</a>

```java
// src/main/java/hello.hellospring.controller/HelloController.java
@GetMapping("hello-api")
    @ResponseBody
    public Hello helloApi(@RequestParam("name") String name) {
        Hello hello = new Hello();
        hello.setName(name);
        return hello;
    }
```

# 🍾 코드 테스트 🍾
개발한 기능을 실행해서 테스트를 해봐야할 때 main메서드 또는 컨트롤러를 통해서 실행한다. 이러한 방법은 준비를 해야하고, 반복 실행을 하기 귀찮으며 여러 테스트를 한 번에 하기 어렵다는 단점이 있다. JUnit이라는 프레임워크로 테스트를 실행해서 이러한 문제를 해결할 수 있다.

@Test를 붙히면 JUnit이라는 프레임워크가 동작하여 main메서드가 아니더라도 코드를 테스트해볼 수 있다.
@BeforeEach, @AfterEach 로 테스트 코드 실행 전, 실행 후 어떤 로직을 수행할 지 설정 가능하다.
- 장점 : 컴파일 에러 뿐만아니라 실행 에러도 잡을 수 있다.
- 장점2 : 언제든지 어떤 메서드든 테스트할 수 있다. 
- 단점 : 적응할 시간이 많이 필요하다.
- 단점2 : 메서드에서 사용한 객체를 끝나면 지워줘야 한다.
- <a href="https://github.com/jjmin321/Spring_studybook/blob/master/src/test/java/hello/hellospring/repository/MemoryMemberRepositoryTest.java" target="_blank">HelloSpringApplicationsTests.java</a>
    
```java
// src/test/java/hello.hellospring.repository/HelloSpringApplicationsTests.java
    @Test
    public void save() {
        Member member = new Member();
        member.setName("spring");
        repository.save(member);
        Member result = repository.findById(member.getId()).get();
        Assertions.assertEquals(result, member);
    }
```

# 🏛️ 빌드 🏛️
빌드를 통해 보통 리눅스를 사용하는 서버 컴퓨터에서 자바 코드를 모두 작성할 필요 없이 빌드파일로 실행이 가능하다.

1. 프로젝트 폴더 터미널을 통해 ./gradlew build 명령어를 입력한다.
<img width="1376" alt="image" src="https://user-images.githubusercontent.com/52072077/88643291-7adf1e00-d0fc-11ea-9829-3e705cd65301.png">

2. 명령어를 입력한 후 ls명렁어를 통해 하위 폴더를 확인해 보면 build 폴더가 생긴 것을 알 수 있다.
<img width="1375" alt="image" src="https://user-images.githubusercontent.com/52072077/88643325-8599b300-d0fc-11ea-9e11-59bd30cb4aa6.png">


3. cd명령어로 build/libs 경로로 이동을 하면 jar 빌드 파일이 생긴 것을 확인할 수 있다.
<img width="1365" alt="image" src="https://user-images.githubusercontent.com/52072077/88643355-90ecde80-d0fc-11ea-8edc-cf7b083fc567.png">

4. java -jar (파일) 명령어로 SPRING을 빌드할 수 있다.
<img width="1383" alt="image" src="https://user-images.githubusercontent.com/52072077/88643189-5e42e600-d0fc-11ea-99db-50bf7bd6adfc.png">

# MVC 패턴 및 템플릿 엔진 
