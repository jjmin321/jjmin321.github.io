---
title:  "PyTorch RNN"
excerpt: "PyTorch RNN 이용하면서 중요한 점을 공유해보자"
permalink: /pytorch/rnn/
toc: true
toc_sticky: true
categories:
  - PyTorch
  - Deep Learning
use_math: true
last_modified_at: 2020-07-12
---

그냥 PyTorch RNN 이용하면서 중요한 점 간단하게 공유해봄. 실제 코드에서의 이야기이니 RNN을 모르는 사람은 이론적인 배경을 공부하고 오면 좋을듯 싶다.

### 기본 구조

우리가 생각하는 것과는 다르게, RNN은 $x_1, ..., x_{t}$]의 sequence 데이터 전부를 필요로 한다. 즉, iteratively $x=1, 2, 3, ...$을 하나씩 넣는 구조가 아니다. 기본 구조는 다음과 같이 생겼다.

```python
model = RNN(input_size: int, hidden_size: int, num_layers: int=1, bias=True, batch_first: bool, dropout: float, bidirectional: bool=False, nonlinearity: str='tanh')
```

-   input_size: 말 그대로 input의 size. 자연어 처리에서는 embedding 차원이 된다. 일반적인 sequence 데이터라면 # features가 될 것이다.
-   'hidden_size': hidden state의 dimension.
-   'num_layers': layer의 개수를 의미한다. 우리가 알고있는 RNN은 보통 layer 1개 짜리다. 밑은 2개짜리 RNN이다.

![](https://www.researchgate.net/profile/Matt_Bianchi/publication/318720785/figure/fig2/AS:520568544137216@1501124620230/Multi-layer-RNN-for-SLEEPNET.png)

-   batch_first: 이 옵션을 주면, input의 데이터 구조가 batch first로 바뀐다. 밑에서 후술.
-   nonlinearity: non-linear 함수. 디폴트는 tanh고, relu를 줄 수가 있다.

RNN에 feed할 때 input의 차원은 `[Seq_len, Batch_size, Hidden_size]`가 된다. 만일 `batch_first=True`라면, `[Batch_size, Seq_len, Hidden_size]` 형태로 feed하면 된다. 또 다른 input인 `hidden`의 경우, `[num_layers * num_directions, batch, hidden_size]`이 된다. 이는 `batch_first=True` 유무와 무관하다. 이는 초기 hidden state $h\_0$를 의미한다.

RNN의 결과로 나오는 것은 `output`과 `h_n`이다. `output`의 경우 `[seq_len, batch, num_directions * hidden_size]`의 차원이 나오게 된다. `batch_first=True`라면, `[Batch_size, Seq_len, num_directions * hidden_size]` 형태가 된다. 이것이 의미하는 것은 **마지막 layer**의 hidden state의 총 집합이다. `h_n`의 경우, t=T일 때의 hidden state가 되며, 차원은 `[num_layers * num_directions, batch, hidden_size]`가 된다. 다음 그림을 보면 아마 이해하기 편할 것이다.

![](https://i.stack.imgur.com/SjnTl.png)

Bi-directional의 경우, 다음 코드를 통해서 split 가능하다.

```python
output.view(seq_len, batch, num_directions, hidden_size)
```

반면, multi layer의 경우, 다음과 같이 하면 된다.

```python
h_n.view(num_layers, num_directions, batch, hidden_size)
```

LSTM구조.

```python
model = LSTM(input_size: int, hidden_size: int, num_layers: int=1, bias=True, batch_first: bool, dropout: float, bidirectional: bool=False, nonlinearity: str='tanh')
```

RNN과 동일하지만, LSTM에는 cell state가 하나 더 있다. 따라서 model의 inputs는 `input, (h_0, c_0)`가 되고, 결과물은 `output, (h_n, c_n)`이 된다. 이들의 차원은 위에서 언급한 것과 동일하다.

다음은 GRU 구조.

```python
model = GRU(input_size: int, hidden_size: int, num_layers: int=1, bias=True, batch_first: bool, dropout: float, bidirectional: bool=False, nonlinearity: str='tanh')
```

여태까지 설명했던 것과 차이 없이 동일하다.

### LSTM의 Bias term

![image](https://user-images.githubusercontent.com/47516855/87240624-11051a00-c456-11ea-9eef-ab4106900600.png)

위 그림과 같이 bias term은 `RNN.bias_ih_l[k]`로 가져올 수 있다. 근데 이게 왜 중요한가? 바로 LSTM의 forget gate의 bias를 초기화 해줄 필요가 있기 때문이다. 유도과정은 직접 쓰기는 귀찮고.. 한번 backpropagation을 해보면, forget gate의 bias term에서 vanishing gradient 문제가 발생할 수 있음을 확인할 수 있을 것이다. 이에 대한 중요성은 다음 인용구로 언급을 마치겠다 (Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever. 2015).

> There is an important technical detail that is rarely mentioned in discussions of the LSTM, and that is the initialization of the forget gate bias bf. Most applications of LSTMs simply initialize the LSTMs with small random weights which works well on many problems. But this initialization effectively sets the forget gate to 0.5. This introduces a vanishing gradient with a factor of 0.5 per timestep, which can cause problems whenever the long term dependencies are particularly severe (such as the problems in Hochreiter & Schmidhuber (1997) and Martens & Sutskever (2011)).

> This problem is addressed by simply initializing the forget gates bf to a large value such as 1 or 2. By doing so, the forget gate will be initialized to a value that is close to 1, enabling gradient flow. This idea was present in Gers et al.(2000), but we reemphasize it since we found many practitioners to not be familiar with it.

> If the bias of the forget gate is not properly initialized, we may erroneously conclude that the LSTM is incapable of learning to solve problems with long-range dependencies, which is not the case.

LSTM의 bias term은 `[b_ig | b_fg | b_gg | b_og]` 형태로 되어 있다. 따라서 두번째 `b_fg`를 적당히 큰 값(1 or 2)으로 초기화 해주면 된다. 이는 다음을 통해서 실행 가능하다. (출처: [https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4](https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4))

<script src="https://gist.github.com/InhyeokYoo/21f2da7e29c723a26167ac42c7533b34.js"></script>

### Dropout

RNN의 dropuout은 매 time step에서 layer에 적용된다. unfold를 해보면, 각 time step에서 RNN은 FC처럼 동작하므로, dropout이 작용한다는 것을 쉽게 알 수 있을 것이다. 그러나 stacked RNN에서는 마지막 layer를 빼고 작동한다. 따라서 `n_layer=1`일 경우 dropout은 동작하지 않는다. 아래 그림은 Zaremba et al., 2015 의 논문에서 소개된 RNN dropout의 예시이다. 아마 PyTorhc또한 이러한 방법으로 구현되어 있을 것 같다.

![image](https://user-images.githubusercontent.com/47516855/87240821-e9af4c80-c457-11ea-82f4-3c36e9246c57.png)

여기서 점선은 dropout이 적용되는 layer다.

[(Gal and Ghahramani, 2016)](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf) 에 따르면, [variational dropout](https://becominghuman.ai/learning-note-dropout-in-recurrent-networks-part-1-57a9c19a2307)을 적용하는게 더 좋은 것으로 알려져 있는데, pytorch에선 구현이 안되어 있는 것으로 보인다.

### Initial hidden state

RNN의 맨 처음 hidden state는 영벡터로 초기화한 후 이를 학습하는 식으로 진행한다. Seq2seq이나 language model같이 initial state의 영향을 적게 받는 구조에서는 이러한 방법이 잘 통하지만, 몇몇 특별한 케이스에서는 특별한 전략을 고려하기도 한다.
  
[Zimmerman et al. (2012)](http://www.scs-europe.net/conf/ecms2015/invited/Contribution_Zimmermann_Grothmann_Tietz.pdf)에 따르면 잘못된 initial state를 RNN을 통과함에 따라 바로잡을 수 있도록 충분힌 time step이 포함되어야 한다고 한다. 혹은 initial state에 민감하지 않게 모델을 만드는 방법도 구사할 수 있는데, 이는 특정 noise term을 더하는 방식으로 구현한다.