---
title:  "Python사용 시 도움되는 VSCode 팁 모음"
toc: true
toc_sticky: true

categories:
  - Python
tags:
  - VSCode
use_math: true
last_modified_at: 2021-10-28
---

## 들어가며

이번 시간에는 VScode에서 유용하게 사용할 수 있는 팁을 공유해보도록 하자.

## Debugging

VScode로 디버깅하는 방법을 알아보자. 

우선 좌측 메뉴바에서 실행 및 디버그(`Ctrl+<Shift>+D`)를 눌러보자. 그러면 `launch.json`파일 만들기를 할 수 있을 것이다. 아래의 형식처럼 이름(`name`)과 파이썬파일(`program`), argparse로 받는 인자(`args`)를 넣어주도록 하자. 나의 경우엔 wikipedia 데이터 파싱 코드를 디버깅해보았다. 또한, working directory를 세팅해야 되는 경우도 있으므로, 이를 `cwd`에다가 넣어주도록 하자

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "create_wikicorpus test",
            "type": "python",
            "request": "launch",
            "program": "source_code/path/file_name.ext",
            "console": "integratedTerminal",
            "cwd": "source_code/path/",
            "args": [
                "--output_path",
                "../datasets/pre-train/wiki.txt"
            ]
        }
    ]
}
```

## Formatter, Linter

PEP8를 지키는데 도움을 주고, 이를 자동으로 수정하는 기능을 완성해보자. 원격환경이나 로컬환경의 config파일을 다음과 같이 수정하면 된다.

```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.banditEnabled": false,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.formatting.blackArgs": [
        "--line-length",
        "140"
    ],
    "python.linting.lintOnSave": true,
    "python.linting.flake8Args": [
        "--max-line-length=140",
        "--ignore=W291",
    ],
    "git.ignoreLegacyWarning": true,
    "python.languageServer": "Pylance",
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

