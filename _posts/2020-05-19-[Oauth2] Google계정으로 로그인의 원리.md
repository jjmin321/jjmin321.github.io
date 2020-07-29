---
title: "[Oauth2] Google계정으로 로그인의 원리 2020-05-19"
date: 2020-05-19 11:15:28 -0400
categories: Development
---

1. Resource server에게 Client ID, Client Secret을 제공받음 

2. Client에서 Resource owner에게 auth login하는 url를 보내줌 

3. Resource owner이 oauth로 로그인 권한 허락을 하면 자동으로 Resource server에게 Resource owner의 정보가 전송

4. Resource server가 받은 Resource owner 정보를 server에게 code로 전송 

5. server가 1번에서 받아 뒀던 ClientID, ClientSecret값과 code값을 모두 모아 Resource server에게 재전송

6. Resource server가 3개의 값이 모두 유효하는지 확인후 유효하다면 해당 Resource owner의 access token을 server에게 전송

7. server는 받은 access token을 저장 

8. Resource owner가 구글 게정으로 로그인을 한다면 server에서 해당 Resource owner의 access token을 조회 후 있다면 Resource server에게 전송 , 없다면 Client에서 Resource owner에게 auth login하는 url를 전송

9. Resource server가 access token을 분석 후 값이 일치하다면 server에게 Resource owner의 정보를 전송

10. Server는 받은 값을 토대로 서버 기능을 제공하면 된다.