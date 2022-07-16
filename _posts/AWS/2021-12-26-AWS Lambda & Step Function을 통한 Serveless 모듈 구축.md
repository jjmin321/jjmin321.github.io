---
title: "AWS Lambda & Step Function을 통한 Serveless 모듈 구축"
excerpt: "AWS Lambda & Step Function 알아보기"
toc: true
toc_sticky: true
date: 2021-12-26 00:01:28 -0400
categories: 
    - AWS
---

AWS Lambda & Step Function에 대해 간단히 알아봅니다.
<hr/>

# 1. AWS Lambda란?

제공한 리소스 내에서 애플리케이션 코드를 실행할 수 있는 경우에 생길 수 있는 애플리케이션 시나리오를 위한 모듈이다.

쉽게 말하면, 보통 클라우드 서비스를 통해 서버를 배포하게 되는데 이 때, 그 서버가 가지는 중요한 메서드가 몇 개 존재하지 않는 경우가 있다. 그럴 때 그 메서드만을 서버를 관리할 필요없이 호출할 수 있게 하여 관리 비용을 절감하자는 취지다. 

Lambda가 이러한 리소스를 관리하므로 컴퓨팅 인스턴스에 로그인하거나 런타임에 사용자가 운영 체제의 환경을 수정할 수 없다. 

따라서 Lambda는 사용자를 대신하여 용량을 관리하고, 모니터링을 도와주며 함수 로깅을 비롯한 운영 및 관리 활동을 대신 수행해주는 Serverless 모듈이라는 것을 알 수 있다.

AWS Lambda가 실제로 어떻게 활용될 수 있는지 예시를 들어보자면 아래와 같다.

## 1.1 사진 처리
Lambda를 호출하면 AWS S3에 새로운 사진이 업로드되었는지 확인하고, 그 사진의 프록시 버전을 제작하여 추가로 저장한다.

## 1.2 웹사이트 크롤링
Lambda에 크롤링을 하는 메서드를 작성해놓으면, 실행이 될 때마다 크롤링을 해오고 결괏값을 받아올 수 있다.

## 1.3 채팅 서비스 관리 
API Gateway의 WebSocket API를 사용해서 사용자가 연결됐을때, 메시지를 전송했을때, 채팅방에서 퇴장했을 때 각각 다른 Lambda를 사용해 메시지를 전송하고, 채팅방에 입장시키거나 퇴장시키는 기능을 하게끔 구현할 수 있다.

# 2. AWS Lambda 사용해보기

## 2.1 Lambda Function 생성

AWS 웹 콘솔에서 Lambda > 함수 > 함수 생성 사이트로 이동해 함수 이름과 런타임 시 실행할 언어를 선택한다.

지원하는 언어로는 C#, Java, Go, Javascript, Python, Ruby가 있고 예제로는 Javascript를 사용할 것이다.

![aws1](https://user-images.githubusercontent.com/52072077/179355598-ff65a02e-b21c-420e-8daf-e6c5be15e98c.png)

## 2.2 Lambda Function 작성
예시로 진격의 거인에 나오는 캐릭터 13명 중, 3명씩 1조로 총 4조를 랜덤으로 뽑고 남은 1명은 제외하는 코드를 작성했다.
![aws2](https://user-images.githubusercontent.com/52072077/179355602-c6f39371-7e20-4e2d-8019-8383da4dadc0.png)

## 2.3 Lambda Function 실행
Deploy를 진행한 후, Test를 눌러보면 아래와 같이 결괏값을 확인할 수 있다.
![aws3](https://user-images.githubusercontent.com/52072077/179355605-8f500eed-26ec-4c7d-b466-9f6d844d80dc.png)

## 2.4 Lambda Function 설정
함수 수정 화면에서 구성을 누르면 환경 변수, 권한 등 설정을 변경 및 확인할 수 있다.
![aws4](https://user-images.githubusercontent.com/52072077/179355609-4c674708-8749-4782-bfed-a1e647211556.png)

# 3. AWS Step Function이란?
AWS Step Functions은 AWS Lambda 및 Amazon ECS의 워크플로우를 연결하여 Serverless 서비스의 설계를 명확하고 유연하게 만들어주는 모듈이다.

AWS Lambda 모듈을 사용하다보면 서비스 플로우에 대한 개선 여지가 생길 수 있는데, 이 때 개선점 중 하나가 될 수 있는 모듈이며 아래와 같은 장점을 가진다.

- Serverless의 가시성이 확보되고 서비스 빌드 및 업데이트를 빠르게

- Lambda Function들을 유기적으로 연결하여 오류와 예외처리를 쉽게 

- 각 단위 Function들의 결합으로 코드를 분리해서 간결하게

위 장점들을 통해 더욱 마이크로서비스 아키텍처로 진화할 수 있지만 장점들을 적절히 이용하기 어렵고 역으로 유지보수가 어려워질 수도 있는 듯 하다.

특정 Lambda 함수에서 다른 Lambda 함수를 호출할 수도 있지만, 그러면 애플리케이션이 복잡해질수록 이러한 모든 연결을 관리하기가 어려워지는 것이 문제이므로 이 때 개선점으로 삼는다.

AWS Step Function을 통한 서비스 플로우 개선 예시는 아래와 같다.

## 3.1 사진 처리 개선
AWS S3에 새로운 사진이 업로드되었는지 확인하고, 그 사진의 프록시 버전을 제작하는 Lambda 모듈이 있을 때, 성능 상 1개의 Lambda 모듈로는 Invoke API를 통해 비동기로 실행하더라도 한계가 있을 것이다.  

그럴 때, 사진이 업로드되었는지 확인하는 Lambda 모듈과, 프록시 버전을 제작하는 Lambda 모듈 총 2개를 제작하여 Step Function을 적용할 수 있다.

## 3.2 웹사이트 크롤링 개선
위 사례와 다르게 크롤링을 하는 Lambda 모듈이 있을 때, 크롤링에 성공할 수도 있을 것이고 실패할 수도 있을 것이다.

만약 성공한다면 성공했을 때 처리하고 싶은 Lambda 모듈과, 실패한다면 실패했을 때 처리하고 싶은 Lambda 모듈을 만들어 두고 Step Function을 통한 분기 처리를 쉽게 가져갈 수 있다.

# 4. AWS Step Function 사용해보기
먼저 AWS Step Function을 사용하여 Lambda 모듈을 제어하기 위해 IAM Role을 생성해야 한다.

## 4.1 AWS IAM Role 생성
IAM > 역할 만들기 > Step Functions > 다음 > 생성
![aws5](https://user-images.githubusercontent.com/52072077/179355610-12483358-b097-447b-8178-7db39cb6ee6e.png)

## 4.2 AWS Step Function 생성
Step Function > 상태 머신 > 기존 역할 선택(위에서 생성한 역할) > 생성
![aws6](https://user-images.githubusercontent.com/52072077/179355638-41f8ab4c-74a9-4504-bea9-9132773fd615.png)

## 4.3 AWS Step Function 작성
축구 유소년 자동 훈련 시스템이라는 이름으로 Step Function을 만들었다.

이 Step Function이 동작하는 방식은 아래와 같다. (각 훈련은 Lambda 모듈로 존재한다고 가정)

TraningType이라는 key로 값을 받는다.
    
1. 값이 공격수라면, 공격 훈련을 진행하고 볼 경합 훈련을 진행하고 종료한다.

2. 값이 미드필더라면, 패스 훈련을 진행하고 프리킥 훈련을 진행하고 종료한다.

3. 값이 수비수라면, 수비 훈련을 진행하고 볼 경합 훈련을 진행하고 종료한다.
![aws7](https://user-images.githubusercontent.com/52072077/179355647-f6cda94b-c670-4a9b-a675-0de350367ea1.png)

## 4.4 Step Function 코드

```json
{
    "Comment": "축구 유소년 자동 훈련 시스템",
    "StartAt": "훈련",
    "States": {
      "훈련": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.TraningType",
            "StringEquals": "공격수",
            "Next": "공격 훈련"
          },
          {
            "Variable": "$.TraningType",
            "StringEquals": "미드필더",
            "Next": "패스 훈련"
          },
          {
            "Variable": "$.TraningType",
            "StringEquals": "수비수",
            "Next": "수비 훈련"
          }
        ]
      },
      "공격 훈련": {
        "Type": "Task",
        "Resource": "공격 훈련 람다 모듈 arn",
        "Next": "볼 경합 훈련"
      },
      "패스 훈련": {
        "Type": "Task",
        "Resource": "패스 훈련 람다 모듈 arn",
        "Next": "프리킥 훈련"
      },
      "수비 훈련": {
        "Type": "Task",
        "Resource": "실행할 람다 모듈 arn",
        "Next": "볼 경합 훈련"
      },
      "프리킥 훈련": {
        "Type": "Task",
        "Resource": "실행할 람다 모듈 arn",
        "End": true
      },
      "볼 경합 훈련": {
        "Type": "Task",
        "Resource": "실행할 람다 모듈 arn",
        "End": true
      }
    }
  }
```
