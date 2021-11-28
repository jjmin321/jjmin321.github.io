---
title: "JPA와 Entity"
excerpt: "자바에서 제공하는 JPA를 알아보자"
toc: true
toc_sticky: true
date: 2021-03-30 00:01:28 -0400
categories: 
    - Java & Spring
---

자바에서 제공하는 인터페이스인 JPA를 알아보고 Entity를 설계해봅니다.
<hr/>

## [ JPA ]
자바 ORM 기술에 대한 표준 명세, 스프링에서 제공하는 것이 아니다<br>
특정 기능을 하는 라이브러리가 아닌 RDBMS를 사용하는 방식을 정의한 인터페이스이다<br>
ORM이기 때문에 자바 클래스와 DB테이블을 매핑할 수 있다
```
ORM이란? 

객체를 통해 개발자가 간접적으로 디비 데이터를 다루는 기술 
작성한 메서드에 맞게 ORM이 SQL을 자동으로 생성해준다
```

## [ JPA 동작 과정 ]
JPA는 애플리케이션과 JDBC 사이에서 동작한다 <br>
개발자가 JPA를 사용하면, 내부에서 JDBC API를 사용하여 SQL를 호출하여 DB와 통신한다
![JPA동작과정](https://user-images.githubusercontent.com/52072077/112945845-c48e7580-916f-11eb-81e3-d4cccecd229d.png)

## [ JPA 특징 ]
비즈니스 로직에 집중하며 객체지향 개발이 가능함<br>
자바 객체와 DB 테이블 사이의 매핑 설정을 통해 SQL을 생성함<br>
지연 로딩, 즉시 로딩과 같은 기법을 개발자가 선택할 수 있게 제공함<br>
```
지연 로딩, 즉시 로딩이란?

지연 로딩: 객체가 실제로 사용될 때 로딩하는 전략
즉시 로딩: JOIN으로 한 번에 연관된 객체까지 미리 조회하는 전략

웬만하면 지연 로딩을 지향하고 필요할 때만 즉시 로딩을 쓰자
잘 활용하면 SQL을 직접 사용하는 것과 유사한 성능을 얻을 수 있다
```
만약 Member와 Team이 @ManyToOne으로 매핑되어 있는 상태에서<br>
지연 로딩이라면 Member.getTeam()으로 Team 객체를 건드릴 때 Team에 대한 쿼리를 날리지만 즉시 로딩이라면 Member 객체를 가져올 때 Join을 통해 한 번에 Team 객체까지 가져옴


## [ JPA 사용 이유 ]
SQL 중심 개발에서 벗어나 객체 중심적인 개발이 가능하다<br>
Spring-data-jpa를 사용한다면 생산성이 더욱 증가한다<br>
```
Spring-data-jpa란?

JPA를 쉽게 사용하기 위해 스프링에서 제공하고 있는 프레임워크 
JpaRepository로 똑똑하게 데이터 검색을 할 수 있다
```

## [ JPA 엔티티 매핑 방법 ]
클래스에 @Entity를 붙히면 JPA가 관리하게 되며, DB 테이블과 매핑된다
```
Entity란?

엔티티는 데이터베이스, SQL 상에 존재하지 않는 개체이다
데이터베이스에 표현하려고 하는 개념이나 정보의 단위다
속성으로 이루어지기 때문에 자바에서 필드로 구성할 수 있다
```

## [ JPA 엔티티 매핑 예시 ]
```java
@Entity(name = "item")
public class Item {
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer idx;

    @OneToOne(targetEntity = User.class, fetch = FetchType.LAZY)
    @JoinColumn(columnDefinition = "user_idx")
    private User user;

    private String name;

    @CreatedDate
    @Column(columnDefinition="TIMESTAMP DEFAULT CURRENT_TIMESTAMP", nullable = false)
    private Date createdAt = new Date();

    @UpdateTimestamp()
    @Column(columnDefinition="TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    private Date updatedAt;
}
```
Entity를 설계할 때 많이쓰는 매핑정보에 관련된 어노테이션들이다<br>

@Entity: 엔티티 이름 지정 
<br>
@Table: 엔티티와 매핑할 테이블 지정 (없으면 Entity의 값 적용)
<br>
@Id: 기본 키 직접 설정 
<br>
@GeneratedValue(strategy = GenerationType.[타입]): 기본 키 간접 설정
<br> 
@Column: 속성(컬럼)에 대한 세부 설정 (없으면 자바 객체 명으로 적용)
<br>
@OneToOne: 테이블 간 관계를 맺을 때 사용 (OneToMany 등 상황에 맞게 사용)
<br>
@JoinColumn: 외래 키를 매핑할 때 사용
<br>
@CreatedDate: 생성 시간을 자동화할때 사용
<br>
@UpdateTimeStamp: 수정 시간을 자동화할때 사용

## [ JPA 엔티티로 테이블 생성 ]
스프링 부트의 application 파일에 spring.jpa.hibernate.ddl-auto 옵션을 추가한다<br>
옵션을 추가하게 되면 Hibernate에서 자동으로 DDL을 생성하여 데이터베이스의 테이블 설정을 자동으로 수행해준다<br>
```
application.yml(properties)란?

스프링부트가 서버를 구동할 때 자동으로 로딩하는 파일이다 
key-value 형식으로 값을 정의하여 사용할 수 있다
환경변수처럼 값을 참조할 수도 있다 (@Value 어노테이션)

DDL이란?

데이터 정의 언어 
데이터의 전체의 골격을 결정하는 역할을 하는 언어
CREATE, ALTER, DROP, TRUNCATE 등이 있다
```
