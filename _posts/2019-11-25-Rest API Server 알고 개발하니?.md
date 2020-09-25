---
title: "REST API Server 알고 개발하니? 2019-11-25"
date: 2019-11-25 00:01:28 -0400
categories: Development
---

# Rest API란?
<br>
1. REST의 개념 
![rest](https://user-images.githubusercontent.com/52072077/69491134-24b63d00-0ed4-11ea-802c-c58f189f48ac.png)


2. REST란? 
- 어플리케이션의 기능들이 생성, 조회, 수정, 식제 4가지인 아키텍쳐
- 클라이언트 / 서버로 구조를 분리하여 독립적으로 개선되어야함
- URI를 통해 자원을 명시하고, HTTP Method를 통해 해당 자원에 대한 명령을 하는 것을 의미
- 즉, REST는 설계의 중심에 자원이 있고 HTTP Method를 통해 자원을 처리하도록 설계된 아키텍쳐를 의미한다
- 클라이언트가 요청을 하면 서버는 자원의 상태를 전달해야함 (JSON 혹은 XML을 통해 주고받는 것이 일반적)
    

# HTTP Method (GET, POST, PUT, PATCH, DELETE)
내용에서 resource는 데이터나 파일을 뜻함
1. GET 
    - 데이터베이스의 resource 정보를 요청함

2. POST 
    - 데이터베이스에 resource를 생성함 

3. PUT
    - 데이터베이스의 resource를 업데이트함, 없다면 생성함

4. PATCH
    - 데이터베이스의 resource를 업데이트함

5. DELETE
    - 데이터베이스의 resource를 삭제하라고 요청

# 어떤 HTTP Method를 사용하면 좋을까?
1. 요청이 오면 사용자의 정보를 데이터베이스에서 읽어와 클라이언트에게 반환해준다 - GET
2. 요청이 오면 사용자의 정보를 데이터베이스에 추가한다 - POST
3. 요청이 오면 사용자의 정보를 데이터베이스에서 수정한다, 값이 없다면 추가한다 - PUT
4. 요청이 오면 사용자의 정보를 데이터베이스에서 수정한다 - PATCH
5. 요청이 오면 사용자의 정보를 데이터베이스에서 삭제한다 - DELETE

# 예시로는?
1. 로그인, 사용자의 정보 보기, 게시물 읽기 - GET
2. 회원가입, 댓글 쓰기 - POST
3. 프로필 이미지 추가, 평점 등록 - PUT
4. 내 정보 수정, 댓글 수정 - PATCH 
5. 회원 탈퇴, 댓글 삭제 - DELETE
