# 15. 트랜스포머로 만드는 대화형 챗봇

챗봇은 일상에 많이 녹아들어 있는 서비스이다. 챗봇에는 인간과 자연어로 대화를 주고받는 대화형 챗봇 이외에도, 정해진 트리형 메뉴 구조를 따라가는 트리형 (버튼) 챗봇, 추천형 챗봇, 시나리오형 챗봇이 있고, 이들을 결합한 결합형 챗봇, 5 가지 유형이 있다.

하지만 대화형 챗봇을 제외하면 사실상 검색엔진이거나, 혹은 음성ARS를 대화형 UX에 옮겨놓은 것과 같다. 즉 규칙 기반으로 구현된 챗봇은 사전에 정해진 말만 알아듣고 반응한다는 한계가 있다.

+ 참고 : [챗봇의 5가지 대표 유형](https://tonyaround.com/%ec%b1%97%eb%b4%87-%ea%b8%b0%ed%9a%8d-%eb%8b%a8%ea%b3%84-%ec%b1%97%eb%b4%87%ec%9d%98-5%ea%b0%80%ec%a7%80-%eb%8c%80%ed%91%9c-%ec%9c%a0%ed%98%95-%ec%a2%85%eb%a5%98/)
<br></br>

### 챗봇과 딥러닝

+ 참고 : [챗봇 역사의 모든 것](https://blog.performars.com/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EC%B1%97%EB%B4%87chatbot-%EC%B1%97%EB%B4%87-%EC%97%AD%EC%82%AC%EC%9D%98-%EB%AA%A8%EB%93%A0-%EA%B2%83)
<br></br>

위 참고자료를 읽어보면, 챗봇에 대한 초창기의 기대, 한계점, 최근 BERT 등의 pretrained model 의 발전 이후의 새로운 기대감으로 이어지는 챗봇의 간략한 역사를 확인해 볼 수 있다. 

인간보다 정확하게 퀴즈를 풀어내는 BERT, ALBERT 등은 모두 트랜스포머 (Transformer) 라는 모델을 활용하여 pretrain 을 적용한 것이며, 트랜스포머 이전에도 LSTM 등 RNN 기반의 딥러닝 모델, 그리고 이를 인코더-디코더 구조로 엮은 seq2seq 모델 등을 활용하여 챗봇 제작을 시도해 왔다.

그러나 2017 년에 발표된 트랜스포머는 병렬처리에 불리한 LSTM 에 비해 훨씬 뛰어난 처리속도를 보이면서도 LSTM 등 RNN 모델이 가지는 장기의존성에 강건한 특징 때문에 매우 긴 길이의 문장을 처리하는 데 유리하다는 좋은 특징을 보여주었고, 이후 자연어처리 분야의 혁신을 가져온 발판이 되어 주었다.
<br></br>

## 학습 목표

1.  트랜스포머의 인코더 디코더 구조 이해하기
2.  내부 단어 토크나이저 사용하기
3.  셀프 어텐션 이해하기
4.  한국어에도 적용해보기

## 사전 준비

> 작업 디렉토리 생성
```bash
$ mkdir -p ~/aiffel/songys_chatbot
```
<br></br>

## 인코더와 디코더 구조 되짚어보기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_UcFQAjh.max-800x600.png)
<br></br>

번역기는 인코더와 디코더 두 가지 아키텍처로 구성되어 있으며, 인코더에는 입력 문장이 들어가며, 디코더는 입력 문장에 상응하는 출력 문장을 생성한다.

즉, 이러한 모델을 훈련한다는 것은 입력 문장과 출력 문장과 같이 병렬 구조로 구성된 데이터셋을 훈련하는 것이다.
<br></br>
### 훈련 데이터셋의 구성 (번역)

-   입력 문장 : '저는 학생입니다.'
-   출력 문장 : 'I am a student'

<br></br>
위와 같은 병렬 데이터를 통해 인코더와 디코더로 학습한다면 질문에 대해 대답을 하도록 구성된 데이터셋을 인코더와 디코더에 학습 시킴으로써 챗봇을 만들 수 있다.
<br></br>
### 훈련 데이터셋의 구성 (질문 - 답변) 

-   입력 문장 : '오늘의 날씨는 어때?'
-   출력 문장 : '오늘은 매우 화창한 날씨야'
<br></br>

### 트랜스포머의 인코더와 디코더

트랜스포머 역시 인코더와 디코더의 구조를 가지고 있으며, 따라서 입력 문장을 넣으면 출력 문장은 출력한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_1_kxflIxg.max-800x600.png)
![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_2_EnQyi4S.max-800x600.png)
<br></br>

위 그림은 트랜스포머의 구조와, 블랙박스로 가려져 있는 그림 (첫 번째 그림) 의 내부구조 (두 번째 그림) 을 나타낸 것이다.

초록색 색깔의 도형을 인코더 층 (Encoder layer), 핑크색 색깔의 도형을 디코더 (Decoder layer) 라고 하였을 때, 입력 문장은 누적되어져 쌓아 올린 인코더의 층을 통해서 정보를 뽑아내고, 디코더는 누적되어져 쌓아 올린 디코더의 층을 통해서 출력 문장의 단어를 하나씩 만들어가는 구조를 가진다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_3_ddZedfW.max-800x600.png)
<br></br>

위 그림과 같이 트랜스포머의 내부 구조를 보다 자세히 보면 여러 모듈들로 구성되어있음을 볼 수 있다.

## 트랜스포머 입력 이해하기

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_4_fuzN6PD.png)
<br></br>
대부분의 자연어 처리 모델은 텍스트 문장을 입력으로 받는다. 따라서 단어를 임베딩 벡터로 변환하는 벡터화 과정을 거치는데, 트랜스포머 모델 역시 예외는 아니다.

하지만 트랜스포머 모델의 입력 데이터 처리에는 임베딩 벡터에 특정 값을 더해준 뒤 입력으로 사용한다는 차별점이 있다.

위 그림은 트랜스포머의 포지셔널 인코딩 (positional encoding) 에 해당하는 부분이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_5_kH52kQN.png)
<br></br>
위 그림은 포지셔널 인코딩을 좀 더 확대한 것이다. 

트랜스포머의 입력 데이터 처리에서 특정 값을 더해주는 이유는 입력을 받을 때, 문장에 있는 단어들을 1 개 씩 순차적으로 처리하는 것이 아니라 모든 단어를 한번에 입력으로 받기 때문이다. 이는 트랜스포머와 RNN 의 결정적인 차별점이다.

RNN 은 문장을 구성하는 단어들이 어순에 따라 모델에 입력되기 때문에 어순 정보가 필요없다.

하지만 트랜스포머는 모든 단어를 한번에 입력 받기 때문에 어순 정보가 필요하다.

따라서 단어의 어순을 알려주기 위해 단어의 임베딩 벡터에 어순에 해당하는 위치 정보를 가진 벡터 (positional encoding) 값을 더해 모델의 입력으로 사용한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_6_DyxB6Ax.png)
<br></br>

포지셔널 인코딩 벡터값은 위 그림의 수식과 같이 사인 함수와 코사인 함수의 값을 임베딩 벡터에 더해 단어의 순서 정보를 더하는 것이다.

수식의 두 함수에서 다소 생소한 $pos, i, d_model$ 등의 변수를 볼 수 있는데, 이를 이해하기 위해서는 임베딩 벡터와 포지셔널 인코딩의 덧셈은 임베딩 벡터가 모여 만들어진 문장 벡터 행렬과 포지셔널 인코딩 행렬의 덧셈 연산을 통해 이뤄진다는 것을 이해해야한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_7_3Rneu0P.png)
<br></br>
$d_model$ 은 임베딩 벡터의 차원을 의미하며, $pos$ 는 입력 문장에서의 임베딩 벡터의 위치를 나타내며, $i$ 는 임베딩 벡터 내의 차원의 인덱스를 의미한다.

즉, 임베딩 행렬과 포지셔널 행렬을 더해줌으로써 각 단어 벡터에 위치 정보를 추가해주는 것이다.

<br></br>
> 필요 라이브러리 임포트
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt
```
<br></br>
> 포지셔널 행렬 구현
```python
# 포지셔널 인코딩 레이어
class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # 배열의 짝수 인덱스에는 sin 함수 적용
    sines = tf.math.sin(angle_rads[:, 0::2])
    # 배열의 홀수 인덱스에는 cosine 함수 적용
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
```
<br></br>
> 시각화를 통해 포지셔널 행렬 확인
> (행의 크기가 50, 열의 크기가 512 인 행렬로 설정)
```python
sample_pos_encoding = PositionalEncoding(50, 512)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
```
행의 크기가 50, 열의 크기가 512 인 행렬은 문장 최대 길이가 50 이고 워드 임베딩 차원을 512 로 하는 모델의 입력 벡터 모양과 같다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_9_l58gVWT.max-800x600.png)
<br></br>
위 그림은 실제 논문에서 포지셔널 인코딩을 표현한 것이다.
<br></br>
## 어텐션? 어텐션!

### 어텐션이란?

트랜스포머의 인코더와 디코더에서는 어텐션이라는 개념을 사용한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_10_AaCfqrY.png)
<br></br>
위 그림은 어텐션 매커니즘을 표현한 것이다.

어텐션 함수는 주어진 '쿼리(Query)' 에 대해서 모든 '키(Key)' 와의 유사도를 각각 구한다. 이렇게 구해낸 유사도를 키(Key) 와 맵핑되어있는 각각의 '값(Value)' 에 반영하며, 유사도가 반영된 '값(Value)' 을 모두 더해서 뭉쳐주면 이를 최종 결과인 어텐션 값(Attention Value) 이 된다.

<br></br>
### 트랜스포머에서 사용된 어텐션

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_11_tFFhFjx.png)
<br></br>
위 그림은 트랜스포머에서는 총 3 가지의 어텐션을 나타낸 것이다.

첫 번째 그림은 인코더 셀프 어텐션으로, 인코더에서 이뤄지는 어텐션이며, 인코더의 입력으로 들어간 문장 내 단어들 간의 유사도를 구하는 역할을 한다.

두 번째 그림은 디코더 셀프 어텐션으로 디코더에서 이뤄지는 어텐션이며, 단어를 1 개 씩 생성하는 디코더가 이미 앞에서 생성된 단어들과의 유사도를 구하는 역할을 한다. 

세 번째 그림은 인코더 - 디코더 어텐션으로 디코터에서 이뤄지며, 디코더가 잘 예측하기 위해 인코더에서 입력된 단어들과의 유사도를 구하는 역할을 한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_12_SIe2V15.png)
<br></br>
위 그림은 트랜스포머의 전체적인 아키텍쳐에서 어텐션이 위치한 곳을 나타낸다.

트랜스포머의 어텐션 함수에 사용되는 쿼리 (Query), 키 (Key), 밸류 (Value) 는 기본적으로 단어 (정보를 함축한) 벡터' 이다.

이때 '단어 벡터' 란 초기 입력으로 사용된 임베딩 벡터가 아니라 트랜스포머의 연산을 거친 후의 '단어 벡터' 를 의미한다.
<br></br>

### 셀프 어텐션 (self Attention)

셀프 어텐션이란 유사도를 구하는 대상이 다른 문장의 단어가 아니라 현재 문장 내의 단어들이 서로 유사도를 구하는 경우를 의미한다.

위에서 인코더 - 디코더 어텐션은 서로 다른 단어 목록 (인코더 내 단어와 디코더 내 단어) 사이에서 유사도를 구하므로 셀프 어텐션이 아니다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_13_hjMyZwL.png)<br></br>
위 그림은 구글 AI 블로그 포스트에서 가져온 것으로 예시 문장을 번역하면 '그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤하였기 때문이다.' 이다. 하지만 여기서 it 에 해당하는 것을 컴퓨터는 어떻게 알 수 있을까?

바로 셀프 어텐션을 통해서 알 수 있다. 셀프 어텐션은 문장 내의 단어들 끼리 유사도를 구하여 it 이 동물과 연관되어 있을 확률이 높다는 것을 찾아내는 것이다.
<br></br>

## 스케일드 닷 프로덕트 어텐션

어텐션은 단어 간 유사도를 구하는 방법이다. 그렇다면 유사도는 어떻게 구할까?

어텐션에서 유사도를 구하는 수식은 다음과 같다.

$$Attention(Q,K,V) = softmax \left( \frac {QK^T} {\sqrt{d_k}} \right) V$$

Q, K, V 는 각각 쿼리 (Query), 키 (Key), 값 (Value)를 나타낸다. 어텐션 함수는 주어진 '쿼리 (Query)'에 대해서 모든 '키 (Key)'와의 유사도 를 각각 구한다.

또한 구해낸 이 유사도를 키와 맵핑되어있는 각각의 '값 (Value)' 에 반영해준다. 그리고 유사도가 반영된 '값(Value)' 을 모두 더해서 뭉쳐주면 이를 최종 결과인 어텐션 값(Attention Value) 라고한다.

즉, 다음과 같이 정리할 수 있다.

1.  Q, K, V 는 단어 벡터를 행으로 하는 문장 행렬이다.
2.  벡터의  내적 (dot product) 은 벡터의  유사도 를 의미한다.
3.  특정 값을 분모로 사용하는 것은 값의 크기를 조절하는 스케일링 (Scaling) 을 위함이다.

Q 와 K 의 전치 행렬을 곱하는 것을 그림으로 표현하면 다음과 같다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_15_pUfIgKn.png)
<br></br>
문장 행렬 Q 와 문장 행렬 K 를 곱하면 위의 그림과 같은 초록색 행렬을 얻을 수 있다.

위 초록색 행렬이 의미하는 값은 예를 들어 'am' 행과 'student' 열의 값은 Q 행렬에 있던 'am' 벡터와 K 행렬에 있던 'student 벡터'의 내적값을 의미한다. 결국 각 단어 벡터의 유사도가 모두 기록된 유사도 행렬이 된다.

이 유사도 값을 스케일링 해주기 위해서 행렬 전체를 특정 값으로 나눠주고, 유사도를 0 과 1 사이의 값으로 Normalize 해주기 위해서 소프트맥스 함수를 사용한다.

여기까지가 Q 와 K 의 유사도를 구하는 과정이며 여기에 문장 행렬 V 와 곱하면 어텐션 값 (Attention Value) 를 얻을 수 있다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_16_neA52rZ.png)
<br></br>
위 그림을 수식으로 표현하면 다음과 같다.

$$Attention(Q,K,V) = softmax \left( \frac {QK^T} {\sqrt{d_k}} \right) V$$

내적 (dot product) 을 통해 단어 벡터 간 유사도를 구한 후에, 특정 값을 분모로 나눠주는 방식으로 Q 와 K 의 유사도를 구한다. 이렇게 유사도를 구하는 방법을 스케일드 닷 프로덕트 (scaled dot product) 이라고 하며 이러한 방식을 활용한 어텐션이기에 스케일드 닷 프로덕트 어텐션(Scaled Dot Product Attention) 이라고 한다.

### 구현하기

> 스케일드 닷 프로덕트 어텐션 함수 구현
```python
# 스케일드 닷 프로덕트 어텐션 함수
def scaled_dot_product_attention(query, key, value, mask):
  """어텐션 가중치를 계산. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output
```
<br></br>

##  머리가 여러 개인 어텐션

### 병렬로 스케일드 닷 프로덕트 어텐션 수행하기

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_18_nnOTx9p.png)
<br></br>
트랜스포머에는 `num_heads` 라는 변수가 있는데, 이는 병렬적으로 몇 개의 스케일드 닷 프로덕트 어텐션 연산을 수행할지를 결정하는 하이퍼파라미터이다.

포지셔널 인코딩에서 `d_model` 은 임베딩 벡터의 차원이며, 트랜스포머의 초기 입력 문장 행렬의 크기는 문장 길이를 행으로하는  `d_model` 을 열의 크기로 가진다.

트랜스포머는 이렇게 입력된 문장 행렬을 `num_heads` 의 수만큼 쪼개서 어텐션을 수행하고, 이렇게 얻은 `num_heads` 의 개수만큼의 어텐션 값 행렬을 다시 하나로 concatenate 한다.

위 그림은 `num_heads` 가 8 개인 경우를 나타낸 것으로 이를 다시 concatenate 하면 열의 크기가 `d_model` 이 된다.

즉, 여러 명이 문제를 풀고, 마지막에 결과를 합치는 것과 같다.
<br></br>

### 멀티 - 헤드 어텐션

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/Untitled_19_FwmaA3q.png)
<br></br>
위 그림은 `num_head` 의 값이 8 일 때, 병렬로 수행되는 스케일드 닷 프로덕트 어텐션이 서로 다른 셀프 어텐션 결과를 얻을 수 있음을 나타낸다.

즉, 병렬로 스케일드 닷 프로덕트 어텐션을 수행함으로써 각 각의 다른 관점에서 보며, 놓칠 수 있었을 정보를 놓치지 않게된다.

위 그림을 예로들면 `it_` 이라는 토큰이 `animal_` 과 유사하다고 보는 관점과 `street_` 과 유사하다고 보는 관점이 한꺼번에 모두 표현 가능하다.

이렇게 스케일드 닷 프로덕트 어텐션을 병렬로 수행하는 것을 멀티 헤드 어텐션이라고 한다.
<br></br>

> 멀티 헤드 어텐션 구현
```python
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # 병렬 연산을 위한 머리를 여러 개 만듭니다.
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # 스케일드 닷 프로덕트 어텐션 함수
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # 어텐션 연산 후에 각 결과를 다시 연결(concatenate)합니다.
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs
```
<br></br>

## 마스킹

마스킹 (Maskinig) 이란 특정 값들을 가려 실제 연산에 방해가 되지 않도록 하는 기법이다.

트랜스포머에서는 스케일드 닷 프로덕트 어텐션을 위해 패딩 마스킹 (Padding Masking), 룩 어헤드 마스킹 (Look - ahead Masking), 2 가지 마스킹을 사용한다.
<br></br>

### 패딩 마스킹 (Padding Masking)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/1365906-20200410103623697-871078599.max-800x600.png)
<br></br>
패딩이란 문장의 길이가 서로 다를 때, 모든 문장의 길이를 통일 시켜주기 위해 정해준 길이보다 짧은 문장의 뒤에 0 으로 채워 문장의 길이를 맞춰주는 것이다.

위 그림은 케라스의 `pad_sequences()`를 사용하여 패딩을 하는 과정을 시각화한 그림이다.

이렇게 추가된 0 은 의미있는 단어가 아니기 때문에 실제 스케일드 닷 프로덕트 어텐션 등과 같은 연산에서는 제외해 줘야한다.

패딩 마스킹은 이를 위해 0 의 위치를 체크하는 역할을 하며 스케일드 닷 프로덕트 어텐션 연산 시 패딩 마스크를 참고하여 불필요한 숫자 0 을 참고하지 않도록 한다.

<br></br>
> 패딩 마스킹 구현
```python
def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]
```
패딩 마스킹을 실행하는 함수에 정수 시퀀스를 입력으로 하면 숫자가 0 인 부분을 체크한 벡터를 출력한다.
<br></br>
> 정수 시퀀스를 입력으로 했을 때 패딩마스크의 출력 확인
```python
print(create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]])))
```
숫자가 0 인 위치에서만 숫자 1 이 나오고 숫자 0 이 아닌 위치에서는 숫자 0 인 벡터를 출력한다.
<br></br>

## 룩 어헤드 마스킹 (Look - ahead Masking, 다음 단어 가리기)

RNN 과 트랜스포머는 문장을 입력받을 때 입력 방식이 전혀 다르다.

RNN은 step 이라는 개념이 존재해서 각 step 마다 단어가 순서대로 입력으로 들어가는 구조인 반면 트랜스포머의 경우에는 문장 행렬을 만들어 한 번에 행렬 형태로 입력으로 들어간다는 특징이 있습니다. 그리고 이 특징 때문에 추가적인 마스킹(Masking) 을 필요로 한다.
<br></br>

### RNN

![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_20_NAntZnv.max-800x600.png)
<br></br>
위 그림은 RNN 으로 다음 단어를 예측하며 문장을 생성하는 과정을 나타낸 것이다.

즉, RNN 을 활용해 디코더를 구현한 것인데 구조상 다음 단어를 만들 때, 자신보다 앞에 있는 단어만을 참고해 다음 단어를 예측한다.

즉, 예측 과정은 첫 번재 step 까지의 입력은 what 이므로 is 를 출력하고, 두 번째 step 까지의 입력은 what is 이므로 the 를 출력, 세 번째 step 까지의 입력은 what is the 이므로, problem 을 출력하는 것이다.
<br></br>

###  트랜스포머

트랜스포머의 경우 RNN 과 달리 전체 문장이 행렬로 들어가기 때문에 위치와 상관없이 모든 단어를 참고해서 다음 단어를 예측한다.

우리가 최종적으로 원하는 목표는 RNN 처럼 이전 단어들로부터 다음 단어를 예측하는 것이므로 트랜스포머가 RNN 과 같은 방식으로 수행할 수 있도록 다음에 나올 단어를 참고하지 않도록 가리는 기법이 바로 룩 어헤드 마스킹 기법이다.

룩 어헤드 마스킹 기법은 스케일드 닷 프로덕트 어텐션을 수행할 때, Query 단어 뒤에 나오는 Key 단어들에 대해서는 마스킹한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/_.max-800x600.png)
<br></br>
위 그림에서 빨간색으로 색칠된 부분은 마스킹을 표현한 것이다. 빨간색은 실제 어텐션 연산에서 가리는 역할을 하여 어텐션 연산 시에 현재 단어를 기준으로 이전 단어들하고만 유사도를 구할 수 있으며 행을 Query, 열을 Key 로 표현된 행렬이다.

예를 들어 Query 단어가 '찾고'라고 한다면, 이 '찾고'라는 행에는 `<s>`, `<나는>`, `<행복을>`, `<찾고>` 까지의 열만 보이고 그 뒤 열은 아예 빨간색으로 칠해져 있다.
<br></br>
> 룩 어헤드 마스킹 구현
```python
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)
```
<br></br>
> 룩 어헤드 마스킹 테스트
```python
print(create_look_ahead_mask(tf.constant([[1, 2, 3, 4, 5]])))
```
대각선 형태로 숫자 1 이 채워지는 것을 볼 수 있다.

룩 어헤드 마스킹 시 숫자 0 인 단어가 있다면 같이 가려줘야 한다. 따라서 `create_look_ahead_mask()` 함수는 내부적으로 앞서 구현한 패딩 마스크 함수도 호출하고 있다.
<br></br>
> 숫자 0 이 포함된 경우 룩 어헤드 마스킹 테스트
```python
print(create_look_ahead_mask(tf.constant([[0, 5, 1, 5, 5]])))
```
<br></br>

## 인코더

### 인코더 층 만들기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_21_Y7Cy8sm.max-800x600.png)
<br></br>

트랜스포머의 하나의 인코더층은 셀프 어센텐션, 피드 포워드 신경망, 2 개의 서브 층 (sub layer) 로 이루어져있다.
<br></br>
> 하나의 인코더 층 구현
```python
# 인코더 하나의 레이어를 함수로 구현.
# 이 하나의 레이어 안에는 두 개의 서브 레이어가 존재합니다.
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 첫 번째 서브 레이어 : 멀티 헤드 어텐션 수행 (셀프 어텐션)
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })

  # 어텐션의 결과는 Dropout과 Layer Normalization이라는 훈련을 돕는 테크닉을 수행
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  # 두 번째 서브 레이어 : 2개의 완전연결층
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 완전연결층의 결과는 Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
```
<br></br>

### 인코더 층을 쌓아 인코더 만들기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_22_teJgoCi.max-800x600.png)
<br></br>
인코더 층을 임베딩 층 (Embedding layer) 과 포지셔널 인코딩 (Positional Encoding) 을 연결하고 원하는 만큼 인코더 층을 쌓으면 트랜스포머의 인코더가 완성된다.

인코더 층을 몇 개를 쌓을지는 사용자가 임의로 정할 수 있다.

인코더와 디코더 내부에서는 각 서브층 이후에 훈련을 돕는 Layer Normalization 이을 적용하며 위 기름에서 Normalize 라고 표시된 부분이 Layer Normalization 을 적용한 부분이다.

트랜스포머는 하이퍼파라미터인 num_layers 개수 만큼 인코더 층을 쌓는다고 하였다. 논문에서는 6 개의 인코더 층을 사용하였다. 하지만 이는 사용자에 따라 달리 설정할 수 있다.

<br></br>
> 인코더 층 구현
```python
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 임베딩 레이어
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

  # 포지셔널 인코딩
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # num_layers만큼 쌓아올린 인코더의 층.
  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
```
<br></br>

## 디코더

### 디코더 층

![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_23_vBHZ3i0.max-800x600.png)
<br></br>
트랜스포머의 디코더는 셀프 엍텐션, 인코더 - 디코더 어텐션, 피드 포워드 신경망, 총 3 개의 서브 층으로 구성된다.

인코더 - 디코더 어텐션은 셀프 어텐션과는 달리, Query 가 디코더의 벡터인 반면에 Key 와 Value 가 인코더의 벡터라는 특징이 있으며, 인코더 - 디코더 어텐션이 인코더의 입력 문장으로 부터 정보를 디코더에 전달하는 과정을 담당한다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/Untitled_24_Kj9egLY.max-800x600.png)
<br></br>

인코더의 셀프 어텐션과 동일하게 디코더의 셀프 어텐션 과 인코더 - 디코더 어텐션, 2 개의 어텐션 모두 스케일드 닷 프로덕트 어텐션을 멀티 헤드 어텐션으로 병렬 처리한다.
<br></br>
> 디코더 구현
```python
# 디코더 하나의 레이어를 함수로 구현.
# 이 하나의 레이어 안에는 세 개의 서브 레이어가 존재합니다.
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 첫 번째 서브 레이어 : 멀티 헤드 어텐션 수행 (셀프 어텐션)
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })

  # 멀티 헤드 어텐션의 결과는 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  # 두 번째 서브 레이어 : 마스크드 멀티 헤드 어텐션 수행 (인코더-디코더 어텐션)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })

  # 마스크드 멀티 헤드 어텐션의 결과는
  # Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  # 세 번째 서브 레이어 : 2개의 완전연결층
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 완전연결층의 결과는 Dropout과 LayerNormalization 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
```
<br></br>

### 디코더 층을 쌓아 디코더 만들기

디코더 층은 임베딩 층과 포지셔널 인코딩을 연결하며, 인코더 층과 동일하게 num_layers 의 개수를 설정해 사용자가 원하는 만큼 쌓으면 트랜스포머의 디코더가 완성된다.
<br></br>
> 디코더 층 구현
```python
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')

    # 패딩 마스크
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
    # 임베딩 레이어
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    # 포지셔널 인코딩
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    # Dropout이라는 훈련을 돕는 테크닉을 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
```
<br></br>

## 챗봇의 병렬 데이터 받아오기

챗봇을 구현해보기 위해 Cornell Movie-Dialogs Corpus 라는 영화 및 TV 프로그램에서 사용되었던 대화의 쌍으로 구성된 데이터셋을 사용하고자 한다.

대화의 쌍이란 말하는 사람의 대화 문장과 그 문장에 응답하는 대화 문장을 쌍으로 이루어진다.
<br></br>
> 데이터 다운
```python
path_to_zip = tf.keras.utils.get_file(
    'cornell_movie_dialogs.zip',
    origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

path_to_dataset = os.path.join(
    os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset,'movie_conversations.txt')
```
<br></br>
> 전체 데이터 중 일부 (5 만개) 만 가져오기
```python
# 사용할 샘플의 최대 개수
MAX_SAMPLES = 5000
```
<br></br>
> 질문과 답변의 쌍의 형태로 데이터셋 가공 (전처리) 하는 함수 구현
```python
# 전처리 함수
def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()

  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 예를 들어서 "I am a student." => "I am a student ."와 같이
  # student와 온점 사이에 거리를 만듭니다.
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)

  # (a-z, A-Z, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence
```
정규 표현식 (Regular Expression) 을 사용하여 구두점 (Punctuation) 을 제거하여 단어를 토크나이징하는 일에 방해가 되지 않도록 정제한다.
<br></br>
> 데이터 로드 및 전처리 함수를 호출하여 질문과 답변의 쌍을 전처리
```python
# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수
def load_conversations():
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_movie_conversations, 'r') as file:
    lines = file.readlines()

  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]

    for i in range(len(conversation) - 1):
            # 전처리 함수를 질문에 해당되는 inputs와 답변에 해당되는 outputs에 적용.
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))

      if len(inputs) >= MAX_SAMPLES:
        return inputs, outputs
  return inputs, outputs
```
<br></br>
> 다운받은 데이터의 샘플 수 확인
```python
# 데이터를 로드하고 전처리하여 질문을 questions, 답변을 answers에 저장합니다.
questions, answers = load_conversations()
print('전체 샘플 수 :', len(questions))
print('전체 샘플 수 :', len(answers))
```
데이터를 5 만개만 뽑아왔으므로 5 만개의 샘플이 저장된 것을 볼 수 있다.
<br></br>
> 임의의 샘플을 출력해 잘 저장되었는지 확인
```python
print('전처리 후의 22번째 질문 샘플: {}'.format(questions[21]))
print('전처리 후의 22번째 답변 샘플: {}'.format(answers[21]))
```
<br></br>

## 병렬 데이터 전처리하기

앞서 질문은 questions, 답변은 answers 에 저장하였다.

이제 다음과 같은 본격적인 전처리를 해보자.

1.  TensorFlow Datasets SubwordTextEncoder 를 토크나이저로 사용한다. 단어보다 더 작은 단위인 Subword 를 기준으로 토크나이징하고, 각 토큰을 고유한 정수로 인코딩한다.

2.  각 문장을 토큰화하고 각 문장의 시작과 끝을 나타내는  `START_TOKEN` 및  `END_TOKEN` 을 추가한다.

3.  최대 길이 MAX_LENGTH 인 40 을 넘는 문장들은 필터링한다.

4.  MAX_LENGTH 보다 길이가 짧은 문장들은 40 에 맞도록 패딩한다
<br></br>

### 단어장 (Vocabulary) 만들기

> 각 단어에 고유한 정수 인덱스를 부여하기 위한 단어장 생성
```python
# 질문과 답변 데이터셋에 대해서 Vocabulary 생성. (Tensorflow 2.2.0 이하)
#tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

# (주의) Tensorflow 2.3.0 이상의 버전에서는 아래 주석의 코드를 대신 실행해 주세요. 
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)
```
질문과 답변 데이터셋 모두를 이용하여 단어장을 생성한다.
<br></br>
> 디코더의 시작 토큰과 종료 토큰에 대해 정수부여
```python
# 시작 토큰과 종료 토큰에 고유한 정수를 부여합니다.

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
```
이미 생성된 단어장의 번호와 겹치지 않도록 각각 단어장의 크기와 그보다 1 이 큰 수를 번호로 부여한다.
<br></br>
> 시작 코튼과 종료 토큰에 부여된 정수 출력
```python
print('START_TOKEN의 번호 :' ,[tokenizer.vocab_size])
print('END_TOKEN의 번호 :' ,[tokenizer.vocab_size + 1])
```
각각 8,331 과 8,332 가 출력되며, 이는 현재 단어장의 크기가 8,331 (0 번부터 8,330 번) 이라는 의미이다.
<br></br>
> 추가된 토큰을 고려해 단어장의 크기를 명시
```python
# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기를 산정합니다.
VOCAB_SIZE = tokenizer.vocab_size + 2
print(VOCAB_SIZE)
```
<br></br>

### 각 단어를 고유한 정수로 인코딩 (Integer Encoding) & 패딩 (Padding)

위에서 `tensorflow_datasets` 의 `SubwordTextEncoder` 를 사용해서 tokenizer 를 정의하고 Vocabulary 를 만들었다면, `tokenizer.encode()` 로 각 단어를 정수로 변환할 수 있고 또는 `tokenizer.decode()` 를 통해 정수 시퀀스를 단어 시퀀스로 변환할 수 있다.

<br></br>
> 임의의 샘플을 `tokenizer.encode()` 의 입력으로 사용해서 변환 결과 확인
```python
# 임의의 22번째 샘플에 대해서 정수 인코딩 작업을 수행.
# 각 토큰을 고유한 정수로 변환
print('정수 인코딩 후의 21번째 질문 샘플: {}'.format(tokenizer.encode(questions[21])))
print('정수 인코딩 후의 21번째 답변 샘플: {}'.format(tokenizer.encode(answers[21])))
```
각 단어에 고유한 정수가 부여된 Vocabulary 를 기준으로 단어 시퀀스가 정수 시퀀스로 인코딩된 결과를 확인할 수 있다.
<br></br>
> 질문과 답변 데이터셋에 대해 전부 최대 길이 지정 및 패딩
```python
# 샘플의 최대 허용 길이 또는 패딩 후의 최종 길이
MAX_LENGTH = 40
print(MAX_LENGTH)
```
<br></br>
> 질문과 답변 데이터셋에 대해 전부 정수 인코딩 수행
```python
# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 40 이하인 경우에만 데이터셋으로 허용
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # 최대 길이 40으로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs
```
정수 인코딩 과정을 수행하면서 샘플의 길이가 40 을 넘는 경우는 샘플들을 필터링하였으므로 일부 샘플이 제외됨을 확인할 수 있다.
<br></br>
> 단어장의 크기와 샘플 개수 확인
```pytquestions, answers = tokenize_and_filter(questions, answers)
print('단어장의 크기 :',(VOCAB_SIZE))
print('필터링 후의 질문 샘플 개수: {}'.format(len(questions)))
print('필터링 후의 답변 샘플 개수: {}'.format(len(answers)))
```
<br></br>

### 교사 강요 (Teacher Forcing) 사용하기

tf.data.Dataset API 는 훈련 프로세스의 속도가 빨라지도록 입력 파이프라인을 구축하는 API 이며 이를 적극 활용하기 위해 질문과 답변의 쌍을 `tf.data.Dataset`의 입력으로 넣어주는 작업을 한다.

이때, 디코더의 입력과 실제값(레이블)을 정의해주기 위해서는 교사 강요 (Teacher Forcing) 이라는 언어 모델의 훈련 기법을 사용한다.

교사강요란 테스트 과정에서 t 시점의 출력이 t + 1 시점의 입력으로 사용되는 RNN 모델을 훈련시킬 때 사용하는 훈련 기법으로, t 시점에서 예측한 값을 t +1 시점의 입력으로 사용하지 않고, t 시점의 레이블, 즉, 실제 알고 있는 정답을 t + 1 시점의 입력으로 사용하는 것이다.

교사강요를 사용할 경우 사용하지 않을 경우보다 훈련 속도가 빠르다. 그 이유는 교사강요를 사용하지 않을 경우 한번 잘못 예측하면 뒤의 예측에도 영향을 끼치게되기 때문이다.

+ 참고 : [위키독스: RNN 언어 모델](https://wikidocs.net/46496)
<br></br>

정리하면 교사강요를 사용하지 않는 일반적인 RNN 모델의 경우 시계열 데이터에서 보았던 자기회귀 모델 (Auto - Regressive Model, AR) 과 같다. 트랜스포머의 디코더도 자기회귀 모델의 하나이며, 따라서 한번의 잘못이 뒤에까지 영향을 끼치지 않게 하기 위해 교사강요를 사용한다.
<br></br>

> 파이프라인 구성
```python
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
```
질문과 답변의 쌍을 tf.data.Dataset API 의 입력으로 사용하여 파이프라인을 구성하며, 교사 강요를 위해서 `answers[:, :-1]` 를 디코더의 입력값, `answers[:, 1:]` 를 디코더의 레이블로 사용한다.
<br></br>

## 모델 정의 및 학습하기

> 인코더 층과 디코더 층 함수를 사용하여 트랜스포머 모델을 구현하는 함수 생성
```python
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 인코더에서 패딩을 위한 마스크
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더에서 미래의 토큰을 마스크 하기 위해서 사용합니다.
  # 내부적으로 패딩 마스크도 포함되어져 있습니다.
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 두 번째 어텐션 블록에서 인코더의 벡터들을 마스킹
  # 디코더에서 패딩을 위한 마스크
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # 인코더
  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  # 디코더
  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # 완전연결층
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
```
<br></br>

### 모델 생성

> 모델 생성
```python
tf.keras.backend.clear_session()

# 하이퍼파라미터
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
DROPOUT = 0.1 # 드롭아웃의 비율

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

model.summary()
```
`num_layers`, `d-Model`, `units` 는 전부 사용자가 정할 수 있는 하이퍼파라미터값이며, 논문에서 `num_layers`는 6, `d-Model`은 512였지만, 빠르고 원활한 훈련을 위해 여기서는 각 하이퍼파라미터를 논문에서보다는 작은 값을 사용하였다.
<br></br>

### 손실 함수 (Loss Function)

> 손실함수 구현
```python
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)
```
레이블에 해당하는 시퀀스에 패딩이 되어 있기 때문에 Loss 를 계산할 때 패딩 마스크를 적용해야한다.
<br></br>

### 커스텀된 학습률 (Learning Rate)

Learning Rate 는 딥러닝 모델 학습에서 매우 중요하나 하이퍼파라미터이다.

최근 Learning Rate 를 학습 초기에는 높였다가 학습이 진행됨에 따라 서서히 낮춰가며 안정적으로 수렴하는 기법을 사용한다. 

이러한 방법을 커스텀 학습률 스케쥴링 (Custom Learning Rate Scheduling) 이라고 한다.

관련 논문에는 커스텀 학습률 스케쥴러를 통해 아담 옵티마이저를 사용하며, 이를 계산하는 공식은 다음과 같다.

$$lrate = d_{model}^{-0.5} \cdot min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$
<br></br>

> 커스텀 학습률 스케줄링 함수 구현
```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```
<br></br>
> 시각화를 통한 커스텀 학습률 스케쥴링 확인
```python
sample_learning_rate = CustomSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
```
커스텀 학습률 스케쥴링 수식은 $step_num^−0.5$ 에 비례하는 부분과 $step_num$ 에 비례하는 부분 중 작은 쪽을 택하도록 되어 있다. 

때문에 초기에는 Learning Rate 가 $step_num$ 에 비례하여 증가하다가 이후에는 감소하는 것을 볼 수 있다.
<br></br>

### 모델 컴파일

> 손실함수와 커스텀 된 학습률을 사용해 모델 컴파일
```python
learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
```
<br></br>
> 모델 훈련
```python
EPOCHS = 20
model.fit(dataset, epochs=EPOCHS, verbose=1)
```
<br></br>

## 챗봇 테스트하기

> 예측을 위한 decoder_inference() 함수 생성
```python
def decoder_inference(sentence):
  sentence = preprocess_sentence(sentence)

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
  output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
  for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
    predictions = model(inputs=[sentence, output_sequence], training=False)
    predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
    output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

  return tf.squeeze(output_sequence, axis=0)
```
예측 (inference) 단계는 다음과 같은 과정을 거친다.

1.  새로운 입력 문장에 대해서는 훈련 때와 동일한 전처리를 거친다.

2.  입력 문장을 토크나이징하고, `START_TOKEN`과 `END_TOKEN`을 추가한다.

3.  패딩 마스킹과 룩 어헤드 마스킹을 계산한다.

4.  디코더는 입력 시퀀스로부터 다음 단어를 예측한다.

5.  디코더는 예측된 다음 단어를 기존의 입력 시퀀스에 추가하여 새로운 입력으로 사용한다.

6.  `END_TOKEN`이 예측되거나 문장의 최대 길이에 도달하면 디코더는 동작을 멈춘다.
<br></br>
> 임의의 입력 문장에 대해 decoder_inference() 함수를 호출하여 챗봇의 대답을 얻는 sentence_generation() 함수 생성
```python
def sentence_generation(sentence):
  # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
  prediction = decoder_inference(sentence)

  # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('입력 : {}'.format(sentence))
  print('출력 : {}'.format(predicted_sentence))

  return predicted_sentence
```
<br></br>
> 임의의 문장으로부터 챗봇의 대답을 확인
```python
sentence_generation('Where have you been?')

sentence_generation("It's a trap")
```
<br></br>
