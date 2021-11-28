---
title:  "Docker로 딥러닝 환경 세팅하기 (순한 맛)"
excerpt: "한 번 (대충) 딥러닝 환경을 조성해보았다."
toc: true
toc_sticky: true

categories:
  - IT
  - Deep Learning
  - PyTorch
tags:
  - docker
  - VScode
last_modified_at: 2020-07-05
---


기존에는 colab으로 학습을 진행하고 있었는데, 로컬에 세팅을 하자니 업데이트 할 것도 많고, 버전 꼬일 것도 걱정되고 해서 docker를 사용하기로 결정했다. 컴공 출신이 아니라 linux도 모르고, 일단 들이박느라 개고생을 했지만... 한번 천천히 따라가보자. 그냥 citation덕지덕지 달터이니 review 논문 읽는 셈 치면 될거같다.

## 1\. Docker 설치하기

Docker를 설치하는 것은 간단하다. 구글링을 하면 다양한 환경에서 설치하는 방법이 나온다. 나는 다음 블로그를 참고했다.  
[https://www.quantumdl.com/entry/PyTorchTensorflow%EB%A5%BC-%EC%9C%84%ED%95%9C-Docker-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0](https://www.quantumdl.com/entry/PyTorchTensorflow%EB%A5%BC-%EC%9C%84%ED%95%9C-Docker-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)

참고로 Window 10 Home버전은 안된다고 하는데, 2004 업데이트를 할 경우 가능하다.

NLP용으로 세팅하려면 다음 글을 참고해도 좋을 것 같다.  
[https://beomi.github.io/2019/12/20/DockerImage\_for\_KoreanNLP/](https://beomi.github.io/2019/12/20/DockerImage_for_KoreanNLP/)

만약 GPU를 사용하고 싶다면 추가적인 작업을 해야될거 같다. 나는 랩탑환경이므로, 딱히 필요가 없어서 구축하진 않았다.

~나의 경우는 jupyter notebook이 없는 이미지를 받게 되었는데, notebook을 쓸 때는 colab을 쓰면 되겠지란 생각을 했다. 혹시나 컨테이너에 pip를 통해 추가적으로 환경을 하고, 저장을 하고 싶다면, 다음을 참고하면 된다.  
[https://stophyun.tistory.com/162](https://stophyun.tistory.com/162)~

그냥 깔고, 컨테이너만 유지하자.

## 2\. VS code 설치

결국 하려는건 로컬 개발 환경 세팅이다. 따라서 IDE와 컨테이너를 연결할 수 있어야 했다. 원래는 IDE로 Pycharm을 사용 중이었는데, professional 버전만 도커 컨테이너를 interpreter로 설정할 수 있었다. 따라서 무료인 VS code를 사용하기로 결정했다.  
VS code는 다음을 통해 설치할 수 있다.  
[https://code.visualstudio.com/](https://code.visualstudio.com/)

## 3\. 컨테이너 연결하기

이제는 interpreter에 컨테이너를 연결하면 된다. 다음을 보고 따라하자.  
[https://curioso365.tistory.com/100](https://curioso365.tistory.com/100)

Remote Development는 다음 링크에서 다운 받을 수 있다. Install 시 자동으로 VS code에서 다운을 받는다.  
[https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

나의 경우는 아래와 같이 debug configuration을 설정해야 했다. 첨엔 뭔 소린지 몰라서 당황했는데, 그냥 python file로 하고, 필요하다면 container에 설치하면 된다.  


![위와 같이 뜬다](https://user-images.githubusercontent.com/47516855/86518789-056e8d80-be6f-11ea-9725-e044a93bbbbd.png){: .align-center}



그 후에는 interpreter가 없다고 나오는데, interpreter를 찾아서 설정해주면 된다. 만약 없을 경우 docker container에서 python을 실행시킨 뒤 다음과 같은 코드를 실행하자.

```python
import sys
print(sys.executable)
```

그러면 interpreter path가 나올 것이다. 이를 토대로 세팅하면 된다.

## 4\. local과 container 연결하기

나도 아직 해결하지는 못했는데, 일단 VScode에서 같은 container에 연결하면 지난 기록이 남아있다. 나는 아래와 같은 방법을 통해 하나씩 옮겨버렸다.  
[https://itholic.github.io/docker-copy/](https://itholic.github.io/docker-copy/)

아마 디렉토리채로 옮기거나, volumne을 이용해서 local에 연결하는 방법도 있을터이니 찾아보면 될거같다.

## P.S. VS code 사용하기

일단 세팅해야 되니 다운은 받았는데, IDE 사용법을 몰라 고생했다. Python VS code 사용법은 다음 튜토리얼을 통해서 공부해보자.  
[https://code.visualstudio.com/docs/python/python-tutorial](https://code.visualstudio.com/docs/python/python-tutorial)