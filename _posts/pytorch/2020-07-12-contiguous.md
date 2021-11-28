---
title:  "PyTorch contiguous에 대해 알아보자"
excerpt: "Attention을 활용한 NMT를 학습하던 도중 다음과 같은 에러가 발생했다."
toc: true
toc_sticky: true
permalink: /pytorch/contiguous/
categories:
  - PyTorch
use_math: true
last_modified_at: 2020-07-12
---

Attention을 활용한 NMT를 학습하던 도중 다음과 같은 에러가 발생했다.

```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-18-ead04f0ff023> in <module>()
    108 for epoch in range(N_EPOCHS):
    109     start_time = time.time()
--> 110     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    111     valid_loss = evaluate(model, valid_iterator, criterion)
    112     end_time = time.time()

6 frames
/usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py in forward(self, input, hx)
    725         if batch_sizes is None:
    726             result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
--> 727                              self.dropout, self.training, self.bidirectional, self.batch_first)
    728         else:
    729             result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,

RuntimeError: rnn: hx is not contiguous
```

conitguous라니? 대체 뭔 이야긴가 싶어서 좀 찾아봤다.

### Contiguous란?

[출처](https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays?fbclid=IwAR21BZEIAtBzlIR163wQxvPaKE9e0oHpXgB4qEGt9LSsG5an8avYiRfw8as)

> A contiguous array is just an array stored in an unbroken block of memory: to access the next value in the array, we just move to the next memory address.

즉, array의 idx=t에서 t+1로 넘어가고 싶다면, 다음 memory address로 옮기기만 하면 되는 것을 contiguous array라 한다. 예를 들어 다음과 같은 `3, 4` 2D-array는 다음과 같이 생겼는데,

![](https://i.stack.imgur.com/BJIVL.png)

실제 memory 상으로는 다음과 같이 저장되어있다.

![](https://i.stack.imgur.com/MXrA6.png))

이 경우 **C contiguous**하다고 한다. row는 앞서 말한 그대로고, column으로 움직이고 싶을 경우 단순히 3개의 블록을 점프하면 그만이기 때문이다.

이를 transporting하면 C contiguous는 파괴되는데, 인접한 row는 더 이상 메모리 상으로 인접하지 않기 때문이다. 그러나 이 경우엔 **Fortran contiguous** 하다고 표현하는데, 이는 column이 메모리 상으로 인접하기 때문이다.

### 그래서 어쩌라고?

contiguous할 경우 퍼포먼스 측면에서 매우 이득이 많다. array가 메모리 전체적으로 spread-out하다면, 접속하는데 느릴 것이라는게 자명하다. (fetching a value from RAM could entail a number of neighbouring addresses being fetched and cached for the CPU.)

C contiguous는 row의 접근이 빠를 것이고, Fortran contiguous의 경우 column접근이 빠를 것이다.

따라서 C contiguous의 경우 `np.sum(arr, axis=1)`는 `np.sum(arr, axis=0)`보다 조금 더 빠르다.

Fortran의 경우 새로운 shape 값을 주는 방식으로 flatten하는게 불가능하다. 이는 다음 그림처럼 메모리 순서가 뒤죽박죽이 되기 때문이다.

![](https://i.stack.imgur.com/GhErW.png)

그러나 `np.reshape`은 안에 있는 값을 새롭게 복사하여 메모리를 저장하기 때문에 contiguous에 영향을 받지 않는다.

### 언제 contiguous하고, non-contiguous한가?

> There are few operations on Tensor in PyTorch that do not really change the content of the tensor, but only how to convert indices in to tensor to byte location. These operations include: `narrow()`, `view()`, `expand()` and `transpose()`

위에 적힌 연산을 사용할 경우, PyTorhc는 새로운 메모리를 할당하는 것이 아닌, indices와 같은 meta information만 변경한다. 이는 메모리를 공유하므로, call by reference와 비슷하다고 볼 수 있겠다. 아래의 코드는 이러한 개념의 예시이다. x의 값을 변경하였으나, 같은 메모리를 공유하므로 y값도 변화한 것을 알 수 있다.

```python
x = torch.randn(3,2)
y = torch.transpose(x, 0, 1)
x[0, 0] = 42
print(y[0,0])
# prints 42
```

이 경우, y값은 contiguous하지 않다. 이는 처음부터 `(2, 3)`으로 만들어진 tensor와 memory layout이 다르기 때문이다. 또한, memory bolck은 붙어있지만 순서가 다르다.

### 내 코드의 문제점은 무엇이었는가?

내 경우에는 아래의 코드에서 contiguous 에러가 발생했다.

```python
dec_output, s = self.decoder(input_, prev_s)
```

따라서 `.stride()`를 이용하여 각 차원으로 1만큼 이동하는데 몇 step이 걸리는지 확인해봤다.

```
torch.Size([1, 128, 160]) torch.Size([2, 128, 64])
(20480, 160, 1) (64, 128, 1)
```

첫번째 줄은 input과 h의 사이즈, 두번째 줄은 각 tensor의 stride값이다. input의 경우 첫번째 차원은 $128 \\times 160$이 20480, 두번째 차원은 160만큼 이동할거고, 마지막 차원은 1만큼 이동할 것이니 옳게 나왔다 (사실 에러메세지가 여기서 나온 것이 아니기 때문에 당연히 맞다). h의 경우, $128 \\times 64$인 8192, 64, 1이 나와야 하는데, 뭔가 값이 꼬여있다. 물론 `prev_s.contiguous()`를 통해 간단하게 해결할 수 있지만, 그래도 궁금하니까 한번 찾아보기로 했다.

```python
print(h.size(), h.stride())
# torch.Size([2, 2, 128, 64]) (16384, 8192, 64, 1)

h = torch.cat([h[:, -1, :, :], h[:, 0, :, :]], dim=2)
print(h.size(), h.stride())
# torch.Size([2, 128, 128]) (16384, 128, 1)

h = h.permute(1, 0, 2)
print(h.size(), h.stride())
# torch.Size([128, 2, 128]) (128, 16384, 1)

h = torch.tanh(self.fc_h(h))
print(h.size(), h.stride())
# torch.Size([128, 2, 64]) (128, 64, 1)
```

3번째 `print`부터 모양이 이상해진다. `permute()`에 대해서는 한 마디도 언급이 없어서 당연히 알아서 해주는줄 알았더니 그게 아니였던 모양이다.