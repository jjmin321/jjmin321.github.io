---
title: "Spring IOC & DI & AOP"
excerpt: "Spring의 IOC, DI, AOP에 대해 알아보자"
toc: true
toc_sticky: true
date: 2021-04-10 00:01:28 -0400
categories: 
    - Java & Spring
---

Spring의 핵심 3대 요소인 IOC & DI & AOP에 대해 배워봅니다.
<hr/>

## [ IoC (Inversion of Control, 제어의 역전) ]
IoC란 객체의 제어권을 프레임워크에게 맡기는 것을 컨테이너라 하는데, 이를 부르는 개념이다<br>
객체에 대한 제어권이 개발자가 아닌 컨테이너에게 넘어가면서 객체의 생성과 관리의 모든 것을 컨테이너가 맡아서 하게 된다<br>
이를 두고 제어권이 역전되었다고 하여 IoC(Inversion of Control, 제어의 역전)라고 한다<br>
아래 예시 코드를 보면 이해하기 쉽다

```java
// 개발자가 직접 객체를 생성하는 코드 

public Class Controller {

    private Service service = new Service();

}
```
위 코드에서 Service 클래스를 직접 인스턴스화하여 개발자가 객체를 제어하고 있다<br>
하지만 아래 코드에서는 컨테이너가 생성한 객체를 인스턴스화하지 않고 사용만 한다

```java
// 컨테이너에 의해 생성된 객체를 사용만 하는 코드 

public Class Controller {

    @Autowired // 객체지향 원칙을 지키기 위해서는 생성자 주입을 사용해야 함.
    private Service service;

}
```

스프링에서 자주 볼 수 있는 표현이다<br>
Service라는 객체를 @Autowired를 통해 스프링 컨테이너에게서 생성된 객체를 주입받아 사용할 수 있다<br>
(단, 스프링 컨테이너가 관리할 객체에는 Bean이 등록되어 있어야 한다)<br>
IoC를 통해 개발자는 객체 관리에 신경쓰지 않고 비즈니스 로직만 생각할 수 있다<br>

## [ DI (Dependency Injection, 의존성 주입) ]
의존성 주입이란 IoC의 형태로 오브젝트의 인스턴스 변수를 직접 인스턴스화하지 않고 DI 컨테이너가 애플리케이션 실행 시점에 인스턴스화해둔 인스턴스를 사용하는 것이다<br>
의존성이란 서로 다른 객체간에 의존 관계가 되어 있다는 말이다, 코드를 보자 
```java
public class Controller {

    @Autowired
    private Service service;

    public void getServiceInfo() {
        service.getInfo();
    }

}
```
위 코드는 Service 객체에 변경사항이 생기면 Controller 객체가 영향을 받으므로, Controller -> Service 간에 의존성(의존 관계)이 있다고 표현한다<br>

그렇다면 주입은 무슨 말일까?<br>
주입이란 외부로부터 사용할 객체의 주소값을 전달받아 사용하는 방식이다<br>

클래스에서 새로운 객체를 생성하지 않고 DI 컨테이너가 건네주는 인스턴스를 인터페이스로 받음으로써 인터페이스 기반의 컴퍼넌트화를 구현할 수 있다<br>

DI 컨테이너의 구상 클래스 인스턴스화는 1회만 실행되고 Singleton처럼 하나의 객체로 필요한 곳에서 사용된다<br>

## [ AOP (Aspect Oriented Programming, 관점 지향 프로그래밍) ]

AOP란 여러 오브젝트에 나타나는 Advice(공통 비즈니스 로직 코드)를 분리하여 재사용하는 기법이다<br>

먼저 @AuthorizationCheck(사용자 정의) 어노테이션을 등록한다 <br>
```java
@AuthorizationCheck // Weaving
@GetMapping("/getInfo")
public Response getInfo(~~~) {
    // ~~~
    // return Response ~~~
}
```

Aspect를 핵심 로직 코드에 적용할 수 있게 Weaving(엮는) 역할을 하는 어노테이션을 만든다
```java
@Target(ElementType.METHOD) // Target
@Retention(RetentionPolicy.RUNTIME) // 유지될 시간 설정
public @interface AuthorizationCheck {
}
```

그 후 공통으로 적용될 공통 관심 사항인 Aspect 클래스를 만든다<br>
Weaving 과정이 시작되는 시점을 JoinPoint라 하는데, 미리 만들어둔 어노테이션을 Pointcut으로 지정하여 Target 메서드에 @AuthorizationCheck를 붙힘으로써 JoinPoint를 발생시킬 수 있다
```java
@Aspect
public class AuthorizationAspect {

    @Pointcut("@annotation(매핑할 어노테이션 위치)")
    public void authorizationCheck() {}

    @Before("authorizationCheck() && args(request)")
    public boolean isExist(HttpServletRequest request) {
        if (request.getAttribute("user") == null) {
            throw new AuthorizationException("유저 정보가 없음");
        }
        return true;
    }

}
```

Advice의 종류로는 아래 5가지가 있다<br>
```
@Before - 메서드가 실행되기 전에
@AfterReturning - 메서드가 정상적으로 사용되었을 때
@AfterThrowing - 메서드가 예외를 발생시켰을 때 
@After - 메서드가 끝났을 때
@Around - 비즈니스 로직 전후 모두에 실행되어야 할 때
```

## AOP 용어 추가 설명 
Aspect - 여러 객체에 계층으로 적용될 공통 관심사항 <br>
Target - Aspect가 적용되는 대상 객체<br>
Advice - 이 기능을 적용할 시점<br>
Weaving - Advice가 비즈니스 로직에 적용되는 것<br>
JoinPoint - Weaving 작업이 일어나는 시점 <br>
PointCut - Aspect를 적용할 Target을 선별하는 것<br>
