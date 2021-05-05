---
title: "Spring AOP를 활용한 자동 로그 시스템 만들기 2021-04-29"
date: 2021-04-29 00:01:28 -0400
categories: Development
---

Spring AOP와 Aspect(Around Advice)를 활용하여 자동 로그 시스템을 구축해봅니다.
<hr/>

# 시스템 구축 단계   
> 1. 특정 API의 요청이 들어왔을 때 API의 종류를 로그를 출력한다.<br/>
> 2. 특정 API의 요청에 따른 서버의 Response 값을 추가로 출력한다.<br/>
> 3. 만약 처리과정에서 Exception이 일어난다면, 그 Exception의 정보를 로그로 출력한다.<br/>


## 1. 특정 API의 요청이 들어왔을 때 API의 종류를 출력하기

먼저 @AutoLogging이라는 어노테이션을 추가한다.
```java
@AutoLogging
@GetMapping("/getItem")
public Response getItem(@RequestParam String name) {
   Item item = itemService.getItem(name);
   return new ResponseData<Item>(HttpStatus.OK, "물품 정보 반환 성공", item);
}
```

그 후, AutoLogging 어노테이션을 정의한다 
```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface AutoLogging {
}	
```

그리고 PointCut을 AutoLogging 어노테이션으로 설정하는 메서드를 만든다
```java
@Component
@Aspect
@Slf4j
public class LoggingAspect {
    @Pointcut("@annotation(jejeongmin.MakeAnything.common.annotation.AutoLogging)")
    public void logging() {}
}
```

그 후 실제로 동작하게 될 Around Advice를 만든다
```java
@Around("logging()")
public Response methodLogging(ProceedingJoinPoint joinPoint) {
    Object response = joinPoint.proceed();
    Map<String, Object> params = new HashMap<>();
    params.put("Controller", joinPoint.getSignature().getDeclaringType().getSimpleName());
    params.put("Method", "/"+joinPoint.getSignature().getName());    
    params.put("Time", new Date());
    return response;
}
```

이까지 했으면 자동으로 API 요청이 올 시 시간과, 메서드, 컨트롤러가 출력이 된다.
<img width="812" alt="image" src="https://user-images.githubusercontent.com/52072077/116552245-c8aae000-a933-11eb-9a57-fae918574719.png">


## 특정 API의 요청에 따른 서버의 Response 값을 추가로 출력하기
joinPoint.proceed()는 Aspect를 사용하는 메서드의 실행결과를 반환하는데, 이를 이용해서 메서드의 response 값을 사용할 수 있다.
```java
//methodLogging 메서드에 일부 코드를 추가하자
ResponseData<Object> response = (ResponseData<Object>) joinPoint.proceed();
params.put("Status", response.getStatus());
params.put("Message", response.getMessage());
params.put("Data", response);
```

이제 자동으로 API 요청이 올 시, 추가로 서버의 Response 정보를 확인할 수 있다

<img width="1330" alt="image" src="https://user-images.githubusercontent.com/52072077/116557066-1118cc80-a939-11eb-8704-96b03e2ba777.png">


## 만약 처리과정에서 Exception이 일어난다면, 그 Exception의 정보를 로그로 출력하기

PointCut을 Exception을 핸들링하는 모든 클래스로 지정해준다.
```java
@Pointcut("execution(* jejeongmin.MakeAnything.common.handler.*.*(..))")
public void loggingOnlyException() {}
```

마지막으로 그에 맞는 Around Advice를 만든다
```java
@Around("loggingOnlyException()")
public Response methodLoggingOnlyException(ProceedingJoinPoint joinPoint) throws Throwable {
    ResponseError responseError = (ResponseError) joinPoint.proceed();
    Map<String, Object> params = new HashMap<>();
    params.put("Status", Integer.toString(responseError.getStatus()));
    params.put("Message", responseError.getMessage());
    params.put("Exception", responseError.getError());
    log.info("@AutoLogging {}", params);
    return responseError;
}
```

이까지 모두 했다면 API의 요청에 Exception이 발생하더라도, 그에 따른 서버의 반환값을 로그로 출력해볼 수 있다.

<img width="1312" alt="image" src="https://user-images.githubusercontent.com/52072077/116558077-07dc2f80-a93a-11eb-9269-096aac2d2844.png">




