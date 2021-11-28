---
title:  "백준 DP 풀이 모음집"
excerpt: "Python으로 풀어보는 백준 DP 문제"
toc: true
toc_sticky: true
permalink: /project/algorithm/DP/
categories:
  - Algorithm
tags:
  - DP
  - BOJ
  - Python
use_math: true
last_modified_at: 2020-07-22
---

백준에서 DP 풀이 만을 모아놓았다. 예전에 푼 것은 수정하기가 귀찮아서 그냥 올렸는데 앞으로 푸는 것은 풀이 과정도 정리해서 올릴 예정이다. 
사용언어는 Python이다. 
TOC를 통해 바로가기를 해보자.

# 2698 인접한 비트의 갯수

- [문제보기](https://www.acmicpc.net/problem/2698)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/DP/2698.py)
- 풀이과정:
    - **왜 DP로 풀어야 하는가?**  
    $n$개의 배열에서 $k$개 만큼 인접한 경우의 수는 $n-1$의 배열에서 $k$개 만큼 인접한 경우의 수 $+ 0$ 이거나,
    $n-1$의 배열에서 $k-1$개 만큼 인접한 경우의 수 $+ 1$ 이므로 DP
    - **점화식은 어떻게 세울 수 있는가?**  
    n과 k, 그리고 마지막 숫자를 통해 점화식을 만들 수 있다.  
    $$
    dp[n][k][0] = dp[n-1][k][1] + dp[n-1][k][0]
    $$  
    $$
    dp[n][k][1] = dp[n-1][k-1][1] + dp[n-1][k][0]
    $$

# 9507 Generations of Tribbles
- [문제보기](https://www.acmicpc.net/problem/9507)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/DP/9507.py)

# 2163 초콜릿 자르기
- [문제보기](https://www.acmicpc.net/problem/2163)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/DP/2163.py)