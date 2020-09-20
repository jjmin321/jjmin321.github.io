---
title: "[Database] Database 상식(1) 키란? 2019-11-26"
date: 2019-11-26 08:30:28 -0400
categories: Development
---

# 시작하기에 앞서
데이터베이스에서 중요한 용어들을 정리했다.

1. 데이터베이스 : 특정 조직의 업무를 수행하는 데 필요한 데이터들의 모임

2. 스키마 : 데이터베이스와 90프로 같은 의미이다. 테이블이 하나의 파일이라면 스키마는 그 파일들을 가지고 있는 폴더이다. 

3. 테이블 : 세로줄과 가로줄의 모델을 이용하여 정렬된 데이터 집합의 모임
![table](https://user-images.githubusercontent.com/52072077/93692178-bd384080-fb2a-11ea-9192-429e892a164b.png)
4. 릴레이션 : 테이블과 99프로 같은 의미이지만, 이 사진과 같은 릴레이션의 특성을 위반한 테이블은 릴레이션이라 부르지 못한다.
![relation](https://user-images.githubusercontent.com/52072077/93692167-98dc6400-fb2a-11ea-9364-c204b4e6a10c.png)
5. 릴레이션 스키마 : 속성들의 집합이다.
![relation-scheme](https://user-images.githubusercontent.com/52072077/93692189-eb1d8500-fb2a-11ea-9476-4caf16e2ad57.png)
6. 릴레이션 인스턴스 : 튜플들의 집합이다.
![relation-instance](https://user-images.githubusercontent.com/52072077/93692191-ee187580-fb2a-11ea-8a3a-d4c6034eacaa.png)
7. 속성 : 릴레이션을 스키마를 구성하는 각각의 열
![attribute](https://user-images.githubusercontent.com/52072077/93692219-49e2fe80-fb2b-11ea-8cd2-f3f23ec9dc3e.png)
8. 튜플 : 릴레이션 인스턴스를 구성하는 각각의 행
![tuple](https://user-images.githubusercontent.com/52072077/93692028-bf999b00-fb28-11ea-8f94-da71874ab5b9.png)
# 키 
<mark>키(key)</mark>는 데이터베이스에서 조건에 만족하는 튜플을 
