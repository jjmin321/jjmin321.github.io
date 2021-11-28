---
title:  "Docker 명령어 정리"
excerpt: "Docker 명령어 정리"
toc: true
toc_sticky: true

categories:
  - IT
tags:
  - docker
use_math: true
last_modified_at: 2020-07-12
---

## 실행 중인 컨테이너 확인

```
docker ps
docker ps -a
```

밑은 중지 중인 컨테이너까지 모두 포함

## 실행 중인 container에 접속

```
docker attach <컨테이너 이름 혹은 아이디>
```

## 컨테이너 이름 변경

```
docker rename <옛날 이름> <새 이름>
```

## 파일 복사

```
docker cp <컨테이너 이름>:<파일> <옮길 주소>
```

## Container 실행
```
docker start <컨테이너 이름>
```

## 실행 중인 container에서 bash 실행
```
docker exec -t -i <컨테이너 이름> /bin/bash
docker exec -ti <컨테이너 이름> /bin/bash
docker exec -ti <컨테이너 이름> sh
```

## volumn 연결

우선 실행 중인 컨테이너를 commit한 후 volumn 설정을 하면 된다

```
docker commit <컨테이너 이름> <새 이미지 이름>

docker run -ti -v <디렉토리1>:<디렉토리2> <이미지 이름> bash
```