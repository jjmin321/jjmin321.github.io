---
title: "Translate api 사용법 2020-05-15"
date: 2020-05-15 11:15:28 -0400
categories: Development
---
🖲️TRANSLATE_API - 5/14/2020 ~ 5/16/2020🖲️
🚆사용자가 원본 언어, 사용할 언어, 텍스트를 보내주면 그에 맞는 번역값을 리턴해주는 API🚆
## /translate
1. 파라미터를 통해 source, target, text 값을 발신받아 저장.
```go
source, target, text := c.Param("source"), c.Param("target"), c.Param("text")
data := url.Values{}
data.Set("source", source)
data.Set("target", target)
data.Set("text", text)
```

2. 네이버 API 클라이언트 ID, Secret을 본인의 것으로 등록
Default: 없음 (env 파일을 통해 사용 추천)
```go
req.Header.Add("X-Naver-Client-Id", clientID)
req.Header.Add("X-Naver-Client-Secret", clientSecret)
```

3. 반환할 값 형식 설정 
Default: JSON - 번역된 값 
```go
return c.JSON(http.StatusOK, map[string]string{
		"result": translator.Message.Result.TranslatedText,
	})
```