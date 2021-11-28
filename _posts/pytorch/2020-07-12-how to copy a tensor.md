---
title:  "PyTorch에서 tensor를 copy하는 법"
excerpt: "PyTorch에서 tensor를 복사하는 방법은 여러가지가 있다."
toc: true
toc_sticky: true
permalink: /pytorch/how-to-copy-a-tensor/
categories:
  - PyTorch
use_math: true
last_modified_at: 2021-10-17
---

[출처](https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor)

PyTorch에서 tensor를 복사하는 방법은 여러가지가 있다.

```python

# x = some tensor

y = tensor.new_tensor(x) # 1

y = x.clone().detach() # 2

y = torch.empty_like(x).copy_(x) # 3

y = torch.tensor(x) # 4
```

과연 어떠한 방법이 올바르게 tensor를 복사하는 방법일까?

## 1. y = tensor.new_tensor(x)

`new_tensor()`는 parameter가 뭐든간에 이를 읽어서 leaf variable을 생성한다. 따라서 근본적으로 b와 동일한 코드이다. 만약 grad가 흐르게 하고 싶다면, require_grad 옵션을 따로 넣어주면 된다.

```python
y = tensor.new_tensor(x, requires_grad=True)
```

## 2. y = x.clone().detach()

이 방법의 경우 computational graph에서 더 이상 필요하지 않을 때 사용할 수 있다. 즉, 다음과 같이 tensor가 있다고 할 때,

```python
x = torch.rand(3, requires_grad=True)
print(x.requires_grad) # True
y = x.clone().detach()

print(y.requires_grad) # True
```

y를 통해 어떠한 연산을 진행하더라도 x에 미치는 영향은 없다. 따라서 weight를 통해 특정 작업을 하고 싶다면 이를 이용하면 된다.

이는 a와 동일한 코드이며, computation graph에서 분리하고 싶을 때 권장된다. `clone`후에 `detach`를 하는 이유는 밑을 참고.

> Unlike copy_(), this function is recorded in the computation graph. Gradients propagating to the cloned tensor will propagate to the original tensor.

### 3. y = torch.empty_like(x).copy_(x)

이는 y에 gradient가 흐를 수 있으므로, 나머지 셋과 성격이 가장 다르다.

### 4. y = torch.tensor(x)

~a와 동일.하다. 근데 근본적으로 a와 같은데, 왜 이 방법이 추천되지 않는지는 모르겠다.

#### 추가:

![image](https://user-images.githubusercontent.com/47516855/87240947-0009d800-c459-11ea-84ba-50984fe4ab10.png)


![](https://i.stack.imgur.com/5QjuT.png)

결론:

> Since in general for a copy operation one wants a clean copy which can't lead to unforeseen side effects the preferred way to copy a tensors is .clone().detach().