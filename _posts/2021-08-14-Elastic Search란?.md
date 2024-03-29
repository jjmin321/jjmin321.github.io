---
title: "Elastic Search란? 2021-08-14"
date: 2021-08-14 00:01:28 -0400
categories: Development
---

Elastic Search에 대해 간단히 공부해봅니다.
<hr>

## Elastic Search의 정의
ES는 <mark>시간이 갈수록 증가하는 문제</mark>를 처리하는 분산형 RESTful 검색 및 분석 엔진이며, ES는 손쉽게 확장되는 빠른 검색과 정교하게 조정된 정확도 그리고 강력한 분석을 위해 데이터를 저장합니다.
<br>
로그, 웹 애플리케이션 등 다양한 소스로부터 원시 데이터가 ES로 가게 되고 데이터 수집은 원시 데이터가 ES에 색인 되기 전에 구문 분석, 정규화 등으로 강화되는 프로세스 입니다.
<br>
ES에 색인되면 사용자는 이 데이터에 대해 복잡한 쿼리를 실행하고 집계를 사용하여 데이터의 복잡한 요약을 검색할 수 있습니다.

## Elastic Search의 사용 사례 
- 애플리케이션 내 검색
- 웹사이트 검색
- 로깅과 로그 분석 
- 애플리케이션 성능 모니터링 
- 데이터 분석 및 시각화 

## Elastic Search 인덱스란?
ES 인덱스는 서로 관련되어 있는 문서들의 모임입니다. JSON 문서 형태로 데이터를 저장하며, 각 문서는 일련의 키(필드나 속성의 이름)와 그에 해당하는 값을 연결합니다.
<br>
역 인덱스(문서의 모든 고유한 단어의 목록을 만들고, 각 단어가 있는 모든 문서를 찾음)라고 하는 데이터 구조를 사용하며, 이는 아주 빠른 풀텍스트 검색을 위해 설계되었습니다. 

## Elastic Search 인덱스의 동작 방식
1. ES는 문서를 저장하고 역 인덱스를 구축하여 거의 실시간으로 문서를 검색 가능한 데이터로 만듭니다.
2. 인덱스 API를 사용하여 색인이 시작되고, 이를 통해 사용자가 특정 인덱스의 JSON 문서를 추가, 수정할 수 있습니다.

## Elastic Search를 사용하는 이유 
1. ES는 Lucene(자바로 이루어진 정보 검색 라이브러리)를 기반으로 구축되어, 풀텍스트 검색에 뛰어납니다. 실시간 검색 플랫폼이며, 색인에서 검색 가능 시간까지가 1초 내외입니다. 즉 ES는 보안 분석, 인프라 모니터링 같은 시간이 중요한 사레에 많이 쓰이며 빠릅니다.
2. Beats와 Logstash의 통합은 ES로 색인하기 전 데이터를 훨씬 쉽게 처리하게 해줍니다. Kibana(ES를 위한 시각화 도구)는 ES 데이터의 실시간 시각화를 제공해 UI를 통해 모니터링, 로그, 결과를 신속하게 볼 수 있습니다.

## Elastic Search의 REST API
먼저 ES가 지원하는 언어로는 Java, Javascript, Go, C#, Python 등이 있으며 제공하는 API는 아래와 같습니다.
- 클러스터 상태 확인
- 인덱스에 대한 CRUD
- 검색 작업 실행
- 필터링 및 집계 등의 강력하고 다양한 REST API

## ES와 RDBMS의 데이터 저장 구조

![image](https://user-images.githubusercontent.com/52072077/129449055-6d9888ec-897c-4212-ad53-a451fd927b4c.png)
