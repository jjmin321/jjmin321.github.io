---
title:  "Minimal Mistakes: github pages를 잘 꾸며보자"
excerpt: "Minimal Mistakes를 내 입맛대로 수정하기"
toc: true
toc_sticky: true

categories:
  - Github Pages
tags:
  - jekyll
  - minimal mistakes
use_math: true
last_modified_at: 2020-10-11
---

내 입맛대로 Minimal mistakes를 수정하는 과정에서 생기는 문제와 해결방안을 정리해보았다.

# TOC 폰트 사이즈 수정

TOC 폰트 사이즈에 대한 지정은 `_sass\minimal-mistakes\_navigation.scss` 파일에서 할 수 있다.
이를 보면 다음과 같은 항목이 있는데,

```scss
.toc__menu {
  margin: 0;
  padding: 0;
  width: 100%;
  list-style: none;
  font-size: $type-size-6;

  @include breakpoint($large) {
    font-size: 0.8em; //$type-size-6;
  }
```

여기서 breakpoint의 `font-size`를 적절하게 수정해주면 된다. 각 type-size는 `_sass\minimal-mistakes\_variables.scss`에 다음과 같이 할당되어 있다.

```scss
/* type scale */
$type-size-1: 2.441em !default; // ~39.056px
$type-size-2: 1.953em !default; // ~31.248px
$type-size-3: 1.563em !default; // ~25.008px
$type-size-4: 1.25em !default; // ~20px
$type-size-5: 1em !default; // ~16px
$type-size-6: 0.75em !default; // ~12px
$type-size-7: 0.6875em !default; // ~11px
$type-size-8: 0.625em !default; // ~10px
```

개인적으로는 5는 너무 크고, 6은 너무 작아서 `0.8em`값으로 따로 주었다.

# Notice 사용하기

다음과 같은 **notice**를 사용할 수 있다. 사용법은 문단의 끝에 `{: .notice}`를 사용하는 것이다. 직접 [깃헙](https://github.com/InhyeokYoo/inhyeokyoo.github.io/blob/master/_posts/etc/2020-10-11-minimal%20mistakes.md)문서를 확인해보자.

**서비스 변경:** 서비스 변경과 같이 간단한 안내는 `{: .notice}`를 문단의 끝에 첨부함으로서 사용할 수 있다.
{: .notice}

**중요한 노트:** 조금 더 중요한 문구는 `{: .notice--primary}`를 통해 사용할 수 있다.
{: .notice--primary}

**정보 안내:** 정보 안내는 `{: .notice--info}`를 이용한다.
{: .notice--info}

**경고 안내:** 경고 안내는 `{: .notice--warning}`를 이용한다.
{: .notice--warning}

**위험 안내:** 위험 안내는 `{: .notice--danger}`를 이용한다.
{: .notice--danger}

**성공 안내:** 성공 안내는 `{: .notice--success}`를 이용한다.
{: .notice--success}

**블록으로 감싸기**: 리스트나 수학공식과 같이 본 notice를 같이 쓸 수 있다.

<div class="notice--primary" markdown="1">
**중요한 노트와 코드블록:** 조금 더 중요한 문구와 함께 코드를 사용하는 것은 아래 코드블록을 통해 사용할 수 있다.

```html
<div class="notice--primary" markdown="1">
**중요한 노트와 코드블록:** 조금 더 중요한 문구와 함께 코드를 사용하는 것은 아래 코드블록을 통해 사용할 수 있다.

// 이하 코드블록

</div>
```
</div>

## Notice 폰트 사이즈 변경하기

`_sass/minimal-mistakes/_notices.scs`에서 변경할 수 있다.

```css
@mixin notice($notice-color) {
  margin: 2em 0 !important;  /* override*/
  padding: 1em;
  color: $dark-gray;
  font-family: $global-font-family;
  font-size:  10 !important; /* <-- 변경 */
  text-indent: initial; /* override*/
  background-color: mix(#fff, $notice-color, 90%);
  border-radius: $border-radius;
  box-shadow: 0 1px 1px rgba($notice-color, 0.25);
```

# Image

## Image alignment

Minimal Mistakes [Uitlity Classes](https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#image-alignment)에는 이미지를 정렬하는 유용한 기능을 제공한다.

```markdown
![image-right](/assets/images/filename.jpg){: .align-right}
![image-right](/assets/images/filename.jpg){: .align-left}
![image-right](/assets/images/filename.jpg){: .align-center}
```

**가운데 정렬**을 하면 다음과 같이 된다.  

![가운데 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg){: .align-center}

---

![우측 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg){: .align-right} 이 문서의 **우측 정렬 결과**는 다음과 같다. 보다시피 오른쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 

---


![좌측 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg){: .align-left} 이 문서의 **좌측 정렬 결과**는 다음과 같다. 보다시피 왼쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 

## Image size

html 사용하거나,
```html
<img src=imgurl width=300 height=500>
```

markdown을 이용한다.
```markdown
![image-name](image-url){: width="400" height="200"}
```

아래는 원본 이미지와 사이즈가 변경 된 이미지다. 둘 다 이쁘라고 가운데 정렬을 해주었다.

![큰 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg){: .align-center}


![큰 이미지](https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg){: .align-center}{: width="400"}

## Markup

[마크업](https://mmistakes.github.io/minimal-mistakes/markup/markup-image-alignment/)을 통해서도 진행할 수 있다. 이 경우 장점은 정렬과 크기조절, figcaption을 동시에 할 수 있다는 점이다.

<figure class="align-left">
  <img src='https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg' alt="">
  <figcaption>왼쪽 정렬한 모습</figcaption>
</figure>
위에서 진행한 것을 다시 한 번 해보자. 이 문서의 **좌측 정렬 결과**는 다음과 같다. 보다시피 왼쪽으로 가게 된다. 이 문단의 텍스트는 자동으로 그림 왼쪽에 정렬된다. 그러나 주의할 점이 있는데, 아래와 위, 그리고 옆에 충분한 공간이 있어야 한다는 점이다. 만일 그렇지 않으면 레이아웃이 깨지게 된다. 여기서부터는 충분한 공간을 채우기 위해 막 지껄였다. 그래야지만 레이아웃이 안 깨지니까. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다. 그냥 계속 말하는 중이다. 아직도 안 채워졌다. 여전히 안 채워졌다.그냥 계속 말하는 중이다. 아직도 안 채워졌다. 아직도 안 채워졌다. 여전히 안 채워졌다. 이제 됐나? 됐다. 여기서부턴 넘어가기 시작한다. 사진의 아래에 텍스트가 위치한 것을 확인할 수 있다.

이 경우에는 아까전에 utility class보다 좀 더 여백이 있는 모습을 확인할 수 있다. 

코드는 다음과 같다.

```html
<figure class="align-left">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-150x150.jpg" alt="">
  <figcaption>왼쪽 정렬한 모습</figcaption>
</figure>
```

---

이번엔 크기 조절을 해보자. 

<figure style="width: 300px" class="align-center">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg" alt="">
  <figcaption>크기는 아까와 같이 300으로 주었다. </figcaption>
</figure> 

코드는 다음과 같다.

```html
<figure style="width: 300px" class="align-left">
  <img src="https://mmistakes.github.io/minimal-mistakes/assets/images/image-alignment-580x300.jpg" alt="">
  <figcaption>크기는 아까와 같이 300으로 주었다. </figcaption>
</figure>
```