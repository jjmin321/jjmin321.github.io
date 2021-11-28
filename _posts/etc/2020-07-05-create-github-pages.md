---
title:  "쉽게 따라하는 Github pages"
excerpt: "비전공자도 쉽게 따라하는 깃허브 블로그 만들기"
toc: true
toc_sticky: true

categories:
  - Github Pages
tags:
  - jekyll
  - minimal mistake
last_modified_at: 2020-07-05
---

CS 비전공자인 내가 github pages를 만들면서 겪었던 난관을 공유해보고자 한다. Github pages를 만들고 설치하는 과정은 [이곳](https://devinlife.com/howto/)을 따라가도록 한다.

# Ruby 설치

위 블로그는 window 기준이 아니기 때문에 window 사용자는 처음부터 난관이다. [루비 홈페이지](https://rubyinstaller.org/)에서 32/64비트에 맞게 설치해주면 된다. 내 경우에는 Ruby+Devkit 2.5.X(x64)를 설치했다.

만약 WSL을 사용 중이라면 WSL에 ubuntu를 설치한 후 진행하는 방법도 있을 것으로 보인다. 이 경우 사용 불가능한 bundle을 사용할 수 있다는 장점이 있다.

# Jekyll 템플릿 선택

내 경우 선택한 template은 [simple green techblog](http://jekyllthemes.org/themes/SIMPLE-GREEN-TECH/)였는데, Git bash를 이용하여 필요한 bundle을 설치하는데 문제가 발생했다. 이는 내 ruby환경과 달랐기 때문으로 추측한다. 따라서 위 가이드에서 소개한 template를 그대로 따라가는 수 밖에 없었다. Ruby에 능통한 분들은 적절하게 변경해서 시도하거나, ~~local에서 jekyll을 설치하지 않고 진행하면 된다.~~ 설치하지 못하면 할 수가 없다. 포기하거나, 어떻게든 해결하자.

# VScode + github desktop

본 파트는 이번 포스트의 하이라이트로, 사실 이 항목을 설명하기 위해 본 포스트의 작성을 결정했다.

설치 및 셋업을 무사히 마쳤다면, 대체 무엇으로 post를 올려야 하나 고민 될 것이다. 적절한 markdown 편집기를 이용하면 되지만, 나의 경우는 vscode를 사용한다. 이는 github desktop을 이용하는데도 매우 편리하며, git bash 명령어를 몰라도 사용할 수 있기 때문이다. 그러면 github desktop을 설치해보자. 

## Github desktop

다음 그림과 같이 Repo를 clone할 때, 주소로 clone할지, Github desktop을 사용할지 결정할 수 있다. 이 때 Github desktop으로 clone하면 설치할 수 있게 된다.

![image](https://user-images.githubusercontent.com/47516855/86532272-9e4ee880-bf03-11ea-9a53-f14ffbc66aa9.png)

설치를 하면 github desktop을 사용할 수 있는데, 먼저 만들어놨던 `github.io`를 pull한다. 그러면 왼쪽 상단에서 repo를 선택할 수 있을 것이다.

github pages를 pull 한 후, 다음과 같은 목록이 뜨는데, 여기서 VScode와 연동하여 코드를 작성하거나 수정할 수 있게 된다.

![image](https://user-images.githubusercontent.com/47516855/86532359-4369c100-bf04-11ea-9e0a-d7704acbc6a8.png)

## VScode

VScode의 설치과정은 쉬우니 따로 작성하지 않겠다. 설치가 완료된 이후 앞서 보았던 'Open in Visual Studio Code'를 통해 본 repo를 열면 내 github pages repo항목이 보이게 된다. 여기서는 `yml` 및 `md`, `css` 등의 편집이 가능하므로 적극적으로 활용하면 되겠다.

그러나 markdown을 적극적으로 활용하기 위해서는 preview기능이 필수적인데, 아쉽게도 기본 기능에선 제공하지 않는다. 따라서 markdown extension을 추가로 설치한다. ctrl + shift + x를 눌러 markdown extension을 설치하자. 나의 경우는 'markdown preview enhancement'를 설치하였다.

![image](https://user-images.githubusercontent.com/47516855/86532442-f803e280-bf04-11ea-96f5-007c5f06bd48.png)

이후 다음과 같이 편리하게 markdown을 편집할 수 있다.

![image](https://user-images.githubusercontent.com/47516855/86532471-2a154480-bf05-11ea-811f-2e99a026245b.png)

# Google search console 등록

아쉽게도 내가 따라한 블로그에는 제대로 설정되어있지가 않다. 다음 링크를 참고하여 설정해보자.

[구글 검색 등록하기](http://jinyongjeong.github.io/2017/01/13/blog_make_searched/)

# 에러 발생 시 대처방법

## jekyll serve 에러

다음과 같은 에러가 발생했다.

```
C:/Ruby27-x64/lib/ruby/2.7.0/bundler/runtime.rb:312:in `check_for_activated_spec!': You have already activated rouge 3.21.0, but your Gemfile requires rouge 3.20.0. Prepending `bundle exec` to your command may solve this. (Gem::LoadError)
```

`bundle exec jekyll serve`로 해결 가능하다길래 해봤더니 다른 에러를 뱉어냈다.

```
Configuration file: C:/Users/mkult/inhyeokyoo.github.io/_config.yml
  Dependency Error: Yikes! It looks like you don't have tzinfo or one of its dependencies installed. In order to use Jekyll as currently configured, you'll need to install this gem. If you've run Jekyll with `bundle exec`, ensure that you have included the tzinfo gem in your Gemfile as well. The full error message from Ruby is: 'cannot load such file -- tzinfo' If you run into trouble, you can find helpful resources at https://jekyllrb.com/help/!
```

찾아봤더니 time zone 문제라고 한다. 따라서 다음 명령어를 실행하고, 

```
gem install tzinfo-data
```

github pages 루트 디렉토리에 있는 gemfile을 메모장으로 열고 `gem 'tzinfo-data'`를 추가해준다. 그러면 [http://127.0.0.1:4000/](http://127.0.0.1:4000/)에 접속이 가능하다.

# facebook 코멘트 등록

다음을 보고 따라하자.

https://shantoroy.com/jekyll/facebook-comment-plugin-jekyll-minimal-mistakes-blog-posts/