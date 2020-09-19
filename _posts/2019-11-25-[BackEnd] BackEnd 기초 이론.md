---
title: "[BackEnd] BackEnd 기초 이론 2019-11-25"
date: 2019-11-25 00:01:28 -0400
categories: Development
---

# 클라이언트/서버 (Client/Server)
클라이언트는 서버에게 Request를 보내고 서버는 클라이언트에게 Response를 준다. 브라우저는 이를 번역해서 띄운다.

# REST란?
1. REST의 개념 
![rest](https://user-images.githubusercontent.com/52072077/69491134-24b63d00-0ed4-11ea-802c-c58f189f48ac.png)


2. REST란? 
- HTTP URI(Uniform Resource Identifier)를 통해 자원(Resource)을 명시하고, HTTP Method(POST, GET, PUT, DELETE)를 통해 해당 자원에 대한 CRUD Operation을 적용하는 것을 의미한다.
- 즉, REST는 자원 기반의 구조(ROA, Resource Oriented Architecture) 설계의 중심에 Resource가 있고 HTTP Method를 통해 Resource를 처리하도록 설계된 아키텍쳐를 의미한다.
- 웹 사이트의 이미지, 텍스트, DB 내용 등의 모든 자원에 고유한 ID인 HTTP URI를 부여한다.
- CRUD Operation
    - Create : 생성(POST)
    - Read : 조회(GET)
    - Update : 수정(PUT)
    - Delete : 삭제(DELETE)
    - HEAD: header 정보 조회(HEAD)

3. REST의 정의 
- "Representational State Transfer"의 약자
- 자원을 이름으로 구분하여 해당 자원의 정보를 주고 받는 모든 것 
    - 즉, 자원의 표현에 의한 상태 전달
    - DB의 학생 정보가 자원일 때, 'students'를 자원의 표현이라고 정의함
    - 데이터가 요청되는 시점에서 자원의 상태 전달(JSON 혹은 XML을 통해 주고받는 것이 일반적)

# HTTP Method (GET, POST, PUT, PATCH, DELETE)
내용에서 resource는 웹페이지(html), binary data(그림파일, 소리파일), db data(json/xml/html로 render된 data)를 뜻함.
1. GET 
    - 서버에게 resource를 보내라고 요청
    - 웹 브라우저에 ~ 를 입력하면 서버가 해당 route에 표시되어야 하는 페이지를 찾아 보여줌.
    - 서버(혹은 DB)의 resource는 클라이언트로 전달만 될 뿐 변경되지 않는다.
    - 웹 브라우저 주소창에 주소를 입력하는 신호는 모두 GET 요청 방식

2. POST 
    - 서버에게 resource를 생성해서 보내라고 요청
    - 회원가입을 하면 DB에 새로운 회원정보가 등록된다.
    - 사진을 업로드하면 그 사진이 웹사이트에 등록된다.

3. PUT
    - 서버에게 resource를 업데이트하라고 요청
    - 서버에 resource가 없다면 새로운 resource를 생성해 달라고 요청.
    - 회원 정보 수정 등에 사용된다.
    - PATCH와 달리 전체 데이터를 교체한다.
    - user.id , user.age , user.name 중 id만 업데이트 하더라도 모든 필드값을 가져와서 모든 값을 항상 새로운 값으로 교체한다.

4. PATCH
    - 서버에게 resource를 업데이트하라고 요청
    - 회원 정보 수정 등에 사용된다.
    - PUT과 달리 일부 데이터를 교체한다.
    - user.id , user.age , user.name 중 id만 업데이트하면 id만 받아와서 해당 부분을 업데이트한다.

5. DELETE
    - 서버에게 resource를 삭제하라고 요청
