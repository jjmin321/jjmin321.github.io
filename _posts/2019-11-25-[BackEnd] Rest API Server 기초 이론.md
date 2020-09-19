---
title: "[BackEnd] Rest API Server 기초 이론 2019-11-25"
date: 2019-11-25 00:01:28 -0400
categories: Development
---

# REST란?
1. REST의 개념 
![rest](https://user-images.githubusercontent.com/52072077/69491134-24b63d00-0ed4-11ea-802c-c58f189f48ac.png)


2. REST란? 
- HTTP URI(Uniform Resource Identifier)를 통해 자원(Resource)을 명시하고, HTTP Method를 통해 해당 자원에 대한 CRUD Operation을 적용하는 것을 의미한다.
- 즉, REST는 자원 기반의 구조(ROA, Resource Oriented Architecture) 설계의 중심에 Resource가 있고 HTTP Method를 통해 Resource를 처리하도록 설계된 아키텍쳐를 의미한다.
- 웹 사이트의 이미지, 텍스트, DB 내용 등의 모든 자원에 고유한 ID인 HTTP URI를 부여한다.
- CRUD Operation
    - Create : 생성(POST)
    - Read : 조회(GET)
    - Update : 수정(PUT)
    - Delete : 삭제(DELETE)

3. REST의 정의 
- "Representational State Transfer"의 약자
- 자원을 이름으로 구분하여 해당 자원의 정보를 주고 받는 모든 것 
    - 즉, 자원의 표현에 의한 상태 전달
    - DB의 학생 정보가 자원일 때, 'students'를 자원의 표현이라고 정의함
    - 데이터가 요청되는 시점에서 자원의 상태 전달(JSON 혹은 XML을 통해 주고받는 것이 일반적)

# HTTP Method (GET, POST, PUT, PATCH, DELETE)
내용에서 resource는 데이터나 파일을 뜻함.
1. GET 
    - 데이터베이스의 resource 정보를 요청함.

2. POST 
    - 데이터베이스에 resource를 생성함. 

3. PUT
    - 데이터베이스의 resource를 업데이트함, 없다면 생성함.

4. PATCH
    - 데이터베이스의 resource를 업데이트함.

5. DELETE
    - 데이터베이스의 resource를 삭제하라고 요청
