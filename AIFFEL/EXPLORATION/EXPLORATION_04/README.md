# 04. 영화리뷰 텍스트 감성분석하기

## 학습 목표

1.  텍스트 데이터를 머신러닝 입출력용 수치데이터로 변환하는 과정을 이해한다.
2.  RNN의 특징을 이해하고 시퀀셜한 데이터를 다루는 방법을 이해한다.
3.  1-D CNN으로도 텍스트를 처리할 수 있음을 이해한다.
4.  IMDB와 네이버 영화리뷰 데이터셋을 이용한 영화리뷰 감성분류 실습을 진행한다.

<br></br>
## 텍스트 감정분성의 유용성

IMDB, 네이버 영화 리뷰 텍스트를 활용해 감성분석 (Sentimental Analysis) 을 해보자.

감성 분석이란 텍스트에 담긴 사용자의 감성이 긍정적인지 부정적인지를 분류하는 딥러닝 모델이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-9-1.max-800x600.png)
<br></br>
텍스트 데이터는 소비자들의 개인적, 감성적 반응이 직접 담겨있을 뿐만 아니라 실시간 트렌트를 빠르게 반영하는 데이터이다.

딥러닝을 이용한 감성분석은 텍스트만 가지는 정보적 특성과 가치를 가진다. 

일반적인 데이터분석 업무는 범주화가 잘 된 정형데이터를 필요로 하는데, 이런 데이터를 큰 규모로 구축하기 위해서 많은 비용이 들지만, 쉽게 구할 수 있는 비정형데이터인 텍스트에 감성분석 기법을 적용하면 텍스트를 정형데이터로 가공하여 유용한 의사결정 보조자료로 활용할 수 있게 된다.

띠라서 데이터 분석 업무에 활용한다는 점에서 강점을 가질 수 있다.

하지만 텍스트 데이터 분석의 기술적인 어려움도 존재한다.

이런 텍스트 데이터를 활용한 감성분석은 기계학습 기반, 감성사전 기반의 2 가지 접근법이 있다.

+ 참고 : [텍스트 분류를 통한 감성분석 활용 사례](https://dbr.donga.com/article/view/1202/article_no/8891/ac/magazine)
<br></br>

## 텍스트 데이터의 특징

텍스트 데이터를 통해 인공지능 모델을 만들어본다고 가정했을 때, 숫자 값을 입력으로 받고, 출력 또한 숫자 값을 출력한다.

예를 들어, 텍스트 문장을 입력받아 긍정일 경우 1 을 출력, 부정일 경우 0 을 출력한다면 2 가지 문제를 해결해야 한다.

텍스트 데이터를 숫자로 입력해주어야한다는 문제와 텍스트에는 단어의 배열, 즉 순서가 중요한데 입력 데이터의 순서를 반영해야 한다는 문제이다.

<br></br>
## 텍스트 데이터의 특징 (01). 텍스트를 숫자로 표현하는 방법.

인공지능 모델의 입력은 0 과 1 의 비트로 표현 가능한 숫자로 이루어진 행렬이다.

A = 0, B = 1, …, Z = 25 라고 숫자를 임의로 부여할 때, A 와 B 는 1 만큼 멀고, A 와 Z 는 25 만큼 멀까? 

텍스트는 텍스트 자체로는 단순한 기호일 뿐이며 텍스트가 가지는 의미를 기호가 내포하지 않는다.

하지만 단어 사전을 구축하는 것 자체는 어렵지 않다.
마치 사전과 같이 단어와 단어의 의미를 나타내는 벡터를 짝지어 볼 수 있다.

이렇게 단어의 특성을 저차원 벡터값으로 표현할 수 있는 기법을 워드 임베딩 (Word Embedding) 기법이라 한다.

<br></br>
> 3개의 짧은 문장으로 이뤄진 텍스트 데이터를 처리
```python
# 처리해야 할 문장을 파이썬 리스트에 옮겨담았습니다.
sentences=['i feel hungry', 'i eat lunch', 'now i feel happy']

# 파이썬 split() 메소드를 이용해 단어 단위로 문장을 쪼개 봅니다.
word_list = 'i feel hungry'.split()
print(word_list)
```
<br></br>
> 문장을 단어 따위로 쪼갠후 딕셔너리 구조로 표현
```python
index_to_word={}  # 빈 딕셔너리를 만들어서

# 단어들을 하나씩 채워 봅니다. 채우는 순서는 일단 임의로 하였습니다. 그러나 사실 순서는 중요하지 않습니다. 
# <BOS>, <PAD>, <UNK>는 관례적으로 딕셔너리 맨 앞에 넣어줍니다. 
index_to_word[0]='<PAD>'  # 패딩용 단어
index_to_word[1]='<BOS>'  # 문장의 시작지점
index_to_word[2]='<UNK>'  # 사전에 없는(Unknown) 단어
index_to_word[3]='i'
index_to_word[4]='feel'
index_to_word[5]='hungry'
index_to_word[6]='eat'
index_to_word[7]='lunch'
index_to_word[8]='now'
index_to_word[9]='happy'

print(index_to_word)
```
<br></br>
> 텍스트 데이터를 숫자로 변환
```python
word_to_index={word:index for index, word in index_to_word.items()}
print(word_to_index)
```
텍스트를 숫자로 바꾸려면 위의 딕셔너리 {텍스트:인덱스} 구조여야 한다.
<br></br>
> 단어를 줄 경우 인덱스를 반환하는 방식으로 사용
```python
print(word_to_index['feel'])  # 단어 'feel'은 숫자 인덱스 4로 바뀝니다.
```
<br></br>
> 텍스트 데이터를 숫자로 바꿔 표현
```python
# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트로 변환해 주는 함수를 만들어 봅시다.
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

print(get_encoded_sentence('i eat lunch', word_to_index))
```
`get_encoded_sentence` 함수를 통해
-   `<BOS>`  -> 1
-   i -> 3
-   eat -> 6
-   lunch -> 7
와 같이 맵핑 된 것을 볼 수 있다.
<br></br>
> 어려 개 문장 리스트를 한꺼번에 변환
```python
# 여러 개의 문장 리스트를 한꺼번에 숫자 텐서로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# sentences=['i feel hungry', 'i eat lunch', 'now i feel happy'] 가 아래와 같이 변환됩니다. 
encoded_sentences = get_encoded_sentences(sentences, word_to_index)
print(encoded_sentences)
```
<br></br>
> 텍스트에서 숫자로 encode 된 벡터를 다시 텍스트 데이터로 복구
```python
# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외

print(get_decoded_sentence([1, 3, 4, 5], index_to_word))
```
<br></br>
> 여러개의 숫자 벡터로 encode 된 문장을 다시 텍스트로 복구
```python
# 여러개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]

# encoded_sentences=[[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]] 가 아래와 같이 변환됩니다.
print(get_decoded_sentences(encoded_sentences, index_to_word))
```
<br></br>

## 텍스트 데이터의 특징 (02). Embedding 레이어의 등장

감성분석을 위해서는 텍스트 데이터를 숫자로 변환하는 것 만으로는 부족하다. 이는 단어의 의미와 대응되는 것이 아니라 단순히 단어가 나열된 순서에 불과하기 때문이다.

감성분석을 위한 목적은 단어와 그 단어의 의미를 나타내는 벡터를 짝짓는 것이다.

따라서 단어의 의미를 나타내는 벡터를 훈련 가능한 파라미터로 놓고 이를 딥러닝을 통해 학습해서 최적화해야 한다.

Tensorflow, Pytorch 등의 딥러닝 프레임워크들은 이러한 의미벡터 파라미터를 구현한 Embedding 레이어를 제공한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-9-2.png)
<br></br>

위 그림에서 word_to_index('great') 는 1918 이다. 즉, 'great' 라는 단어의 의미공간상의 워드 벡터 (word vector) 는 Lookup Table 형태로 구성된 Embedding 레이어의 1919 번째 벡터가 된다. (위 그림에서는 [1.2, 0.7, 1.9, 1.5]가 된다.)

<br></br>
> 임베딩 레이어를 활용하여 텍스트 데이터를 워드 벡터 텐서 형태로 변환
```python
# 아래 코드는 그대로 실행하시면 에러가 발생할 것입니다. 

import numpy as np
import tensorflow as tf
from tensorflow import keras

vocab_size = len(word_to_index)  # 위 예시에서 딕셔너리에 포함된 단어 개수는 10
word_vector_dim = 4    # 위 그림과 같이 4차원의 워드벡터를 가정합니다. 

embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=word_vector_dim, mask_zero=True)

# 숫자로 변환된 텍스트 데이터 [[1, 3, 4, 5], [1, 3, 6, 7], [1, 8, 3, 4, 9]] 에 Embedding 레이어를 적용합니다. 
# list 형태의 sentences는 numpy array로 변환되어야 딥러닝 레이어의 입력이 될 수 있습니다.
raw_inputs = np.array(get_encoded_sentences(sentences, word_to_index))
output = embedding(raw_inputs)
print(output)
```
위 코드를 실행해보면 에러가 발생한다.

Embedding 레이어의 인풋이 되는 문장 벡터는 그 **길이가 일정**해야는데 이 길이가 일정하지 않기 때문이다.

Tensorflow 에서는 `keras.preprocessing.sequence.pad_sequences`라는 편리한 함수를 통해 문장 벡터 뒤에 패딩 (`<PAD>`) 을 추가하여 길이를 일정하게 맞춰주는 기능을 제공한다.
<br></br>
> `keras.preprocessing.sequence.pad_sequences` 를 통해 패딩 추가
```python
raw_inputs = keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                       value=word_to_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=5)
print(raw_inputs)
```
`<PAD>` 가 0에 맵핑되어 있어, 짧은 문장 뒤쪽이 0으로 채워지는 것을 확인할 수 있다.
<br></br>
> 패딩 후 `output = embedding(raw_inputs)` 다시 시도
```python
import numpy as np
import tensorflow as tf

vocab_size = len(word_to_index)  # 위 예시에서 딕셔너리에 포함된 단어 개수는 10
word_vector_dim = 4    # 그림과 같이 4차원의 워드벡터를 가정합니다.

embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=word_vector_dim, mask_zero=True)

# keras.preprocessing.sequence.pad_sequences를 통해 word vector를 모두 일정길이로 맞춰주어야 
# embedding 레이어의 input이 될 수 있음에 주의해 주세요. 
raw_inputs = np.array(get_encoded_sentences(sentences, word_to_index))
raw_inputs = keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                       value=word_to_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=5)
output = embedding(raw_inputs)
print(output)
```
output 의 shape 의 인자는 각각 문장의 개수, 문장에 포함된 단어의 수 (문장의 최대 길이), 해당 벡터의 차원을 의미한다.
<br></br>

## 시퀀스 데이터를 다루는 RNN

딥러닝에서 텍스트 데이터를 주로 다루는 모델은 RNN (Recurrent Neural Netowrk) 모델 이다.

RNN 은 시퀀스 형태의 데이터를 처리하는데 있어 최적의 모델이다.

시퀀스 데이터란 시간의 흐름에 따라 발생하는 데이터로 그 순서가 영향을 미치는 데이터를 의미한다.

즉, 텍스트 데이터도 시퀀스 데이터이지만, 음성 데이터가 보다 더 시퀀스 데이터에 가깝다.

RNN 모델은 이렇게 시간의 흐름에 따라 새롭게 들어오는 입력에 따라 변하는 현재 상태를 묘사하는 state machine 으로 설계되었다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-9-3.max-800x600.png)
<br></br>

+ 참고 : [RNN 의 기본 개념 (김성훈 교수님의 딥러닝 강좌)](https://youtu.be/-SHPG_KMUkQ)
<br></br>

> RNN 모델을 활용하여 텍스트 데이터를 처리
```python
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4  # 단어 하나를 표현하는 임베딩 벡터의 차원수입니다. 

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경가능)
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
<br></br>

## 꼭 RNN 이어야 할까?

텍스트 데이터 처리를 위해서는 RNN 뿐만 아니라 `1-D Convolution Neural Network(1-D CNN)` 를 사용할 수도 있다.

`1-D CNN` 은 문장 전체를 한꺼번에 한 방향으로 길이 7짜리 필터로 스캐닝하면서 7단어 이내에서 발견되는 특징을 추출하여 그것으로 문장을 분류하는 방식으로 사용된다.

이 방식 역시 RNN 과 같이 높은 효율을 보여주며, CNN 은 병렬 처리가 가능하기에 학습 속도가 빠르다는 장점이 존재한다.

또 다른 방법으로는 `GlobalMaxPooling1D()` 레이어 하나만 사용하는 방법도 있다.

이는 전체 문장 중에서 단 하나의 가장 중요한 단어만 피처로 추출하여 그것으로 문장의 긍정/부정을 평가하는 방식이라고 생각할 수 있는데, 의외로 성능이 잘 나올 수도 있다.
<br></br>
> `1-D Convolution Neural Network(1-D CNN)` 를 통한 텍스트 데이터 처리
```python
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4   # 단어 하나를 표현하는 임베딩 벡터의 차원수입니다. 

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
<br></br>
> `GlobalMaxPooling1D()` 레이어 하나만을 활용한 텍스트 데이터 처리
```python
vocab_size = 10  # 어휘 사전의 크기입니다(10개의 단어)
word_vector_dim = 4   # 단어 하나를 표현하는 임베딩 벡터의 차원수입니다. 

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
<br></br>

## IMDB 영화리뷰 감성분석 (01). IMDB 데이터셋 분석

이제 IMDB 영화리뷰 감성분석을 해보자.

+ 참고 : [IMDB 데이터셋 소개 논문](https://aiffelstaticprd.blob.core.windows.net/contents/[https://www.aclweb.org/anthology/P11-1015.pdf](https://www.aclweb.org/anthology/P11-1015.pdf)
<br></br>

IMDB 데이터셋은 50000개의 리뷰 중 절반인 25000개가 훈련용 데이터, 나머지 25000개를 테스트용 데이터로 사용하도록 지정되어 있다.

이 데이터셋은 tensorflow Keras 데이터셋 안에 포함되어 있어서 손쉽게 다운로드하여 사용할 수 있다.
<br></br>
> 필요 라이브러리 임포트 및 IMDB 데이터셋 다운
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)
imdb = keras.datasets.imdb

# IMDB 데이터셋 다운로드 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print("훈련 샘플 개수: {}, 테스트 개수: {}".format(len(x_train), len(x_test)))
```
<br></br>
> 다운받은 데이터 확인
```python
print(x_train[0])  # 1번째 리뷰데이터
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
print('1번째 리뷰 문장 길이: ', len(x_train[0]))
print('2번째 리뷰 문장 길이: ', len(x_train[1]))
```
다운 받은 데이터는 텍스트 데이터가 아닌 숫자로 이미 encode 된 텍스트 데이터임을 알 수 있다.
<br></br>

IMDb 데이터셋에는 encode에 사용한 딕셔너리까지 함께 제공한다.
<br></br>
> encode 에 사용한 딕셔너리 확인
```python
word_to_index = imdb.get_word_index()
index_to_word = {index:word for word, index in word_to_index.items()}
print(index_to_word[1])     # 'the' 가 출력됩니다. 
print(word_to_index['the'])  # 1 이 출력됩니다.
```
<br></br>
> 텍스트 인코딩을 위한 `word_to_index`, `index_to_word` 보정
```python
#실제 인코딩 인덱스는 제공된 word_to_index에서 index 기준으로 3씩 뒤로 밀려 있습니다.  
word_to_index = {k:(v+3) for k,v in word_to_index.items()}

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word[0] = "<PAD>"
index_to_word[1] = "<BOS>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

index_to_word = {index:word for word, index in word_to_index.items()}

print(index_to_word[1])     # '<BOS>' 가 출력됩니다. 
print(word_to_index['the'])  # 4 이 출력됩니다. 
print(index_to_word[4])     # 'the' 가 출력됩니다.
```
`word_to_index` 는 IMDB 텍스트 데이터셋의 단어 출현 빈도 기준으로 내림차수 정렬되어 있다.
<br></br>
> encode 된 텍스트가 정상적으로 decode 되는지 확인
```python
print(get_decoded_sentence(x_train[0], index_to_word))
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
```
<br></br>
> `pad_sequences`를 통해 데이터셋 상의 문장의 길이를 통일
```python
total_data_text = list(x_train) + list(x_test)
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다. 
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,  
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))
```
문장 최대 길이 `maxlen`의 값 설정도 전체 모델 성능에 영향을 미치며, 이 길이으이 적절한 값을 찾기 위해서는 데이터셋 분포를 확인해봐야 한다.
<br></br>
> RNN 학습
```python
x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=word_to_index["<PAD>"],
                                                        padding='post', # 혹은 'pre'
                                                        maxlen=maxlen)

x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_to_index["<PAD>"],
                                                       padding='post', # 혹은 'pre'
                                                       maxlen=maxlen)

print(x_train.shape)
```
padding 방식을 문장 뒤쪽 ('post') 과 앞쪽 ('pre') 중 어느쪽으로 하느냐에 따라 RNN 을 이용한 딥러닝 적용 시 성능 차이가 발생한다.

RNN 은 입력데이터가 순차적으로 처리되어, 가장 마지막 입력이 최종 state 값에 가장 영향을 많이 미치게 됩니다. 그러므로 마지막 입력이 무의미한 padding 으로 채워지는 것은 비효율적이다.
<br></br>

## IMDB 영화리뷰 감성분석 (02). 딥러닝 모델 설계와 훈련

> RNN 모델 설계
```python
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model = keras.Sequential()
# [[YOUR CODE]]
model.add(keras. layers. Embedding(vocab_size, word_vector_dim, input_shape = (None,)))
model.add(keras. layers. LSTM(8))
model.add(keras. layers.Dense(8, activation ='relu'))
model.add(keras. layers.Dense(1, activation = 'sigmoid'))

model.summary()
```
<br></br>
> 모델 훈련에 앞서 데이터셋 분리
```python
# validation set 10000건 분리
x_val = x_train[:10000]   
y_val = y_train[:10000]

# validation set을 제외한 나머지 15000건
partial_x_train = x_train[10000:]  
partial_y_train = y_train[10000:]

print(partial_x_train.shape)
print(partial_y_train.shape)
```
<br></br>
> 모델 학습
```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```
<br></br>
> 테스트셋을 통한 모델 평가
```python
results = model.evaluate(x_test,  y_test, verbose=2)

print(results)
```
<br></br>

`model.fit()` 과정 중의 train / validation loss, accuracy 등이 매 epoch 마다 history 변수에 저장되며 이를 통해 학습 과정을 시각화하여 학습이 잘 되었는지, 오버피팅이 발생하였는지 등을 고민해 볼 수 있다.
<br></br>
> 학습과정 시각화
```python
history_dict = history.history
print(history_dict.keys()) # epoch에 따른 그래프를 그려볼 수 있는 항목들

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```
Training and validation loss 를 그려 보면, 몇 epoch 까지의 트레이닝이 적절한지 최적점을 추정해 볼 수 있다.

validation loss 의 그래프가 train loss 와의 이격이 발생하게 되면 더 이상의 트레이닝은 무의미하다.
<br></br>
> Training and Validation accuracy 시각화
```python
plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```
<br></br>

## IMDB 영화리뷰 감성분석 (03). Word2Vec 의 적용

Word2Vec 은 무려 1억 개의 단어로 구성된 Google News dataset 을 바탕으로 학습되었으며, 임베딩 레이어와 원리는 동일하다.

앞서 사용한 모델의 첫 번째 레이어가 임베딩 레이어였다. 이 레이어는 사전의 단어 개수 X 워드 벡터 사이즈만큼의 크기를 가진 학습 파라미터이다.

학습을 성공적으로 마쳤다면 임베딩 레이어에 학습된 워드 벡터들이 의미 공간상에 유의미한 형태로 학습되어야 한다.
<br></br>
> 작업 디렉토리 구성 및 워드 벡터를 다루기 위한 `gensim` 패키지 설치
```bash
$ mkdir -p ~/aiffel/sentiment_classification $ pip install gensim
```
<br></br>
> 임베딩 레이어에 학습된 워드 벡터를 확인 (Word2Vec)
```python
import os

# 학습한 Embedding 파라미터를 파일에 써서 저장합니다. 
word2vec_file_path = os.getenv('HOME')+'/sentiment_classification/word2vec.txt'
f = open(word2vec_file_path, 'w')
f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))  # 몇개의 벡터를 얼마 사이즈로 기재할지 타이틀을 씁니다.

# 단어 개수(에서 특수문자 4개는 제외하고)만큼의 워드 벡터를 파일에 기록합니다. 
vectors = model.get_weights()[0]
for i in range(4,vocab_size):
    f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()
```
<br></br>
> 임베딩 파라미터를 읽어 word vector 로 활용
```python
from gensim.models.keyedvectors import Word2VecKeyedVectors

word_vectors = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
vector = word_vectors['computer']
vector
```
`gensim` 에서 제공하는 패키지를 통해 임베딩 벡터를 읽고 word vector 로 활용이 가능하다.
<br></br>
> 주어진 임의의 단어와 유사한 단어를 출력하고, 출력된 단어와 유사도를 확인
```python
word_vectors.similar_by_word("love")
```
워드 벡터가 의미벡터 공간상에 유의미하게 학습되었는지 확인하는 방법 중 하나로 임의의 단어를 주었을 때 이와 유사한 단어를 출력하고, 출력된 단어와 제공된 단어간 유사도를 확인할 수 있다.
<br></br>

+ 참고 : [한국어 임베딩 서문](https://ratsgo.github.io/natural%20language%20processing/2019/09/12/embedding/)
<br></br>

+ 참고 : [Google Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

> 작업 디렉토리 설정
```bash
$ mv ~/Downloads/GoogleNews-vectors-negative300.bin.gz ~/aiffel/sentiment_classification
```
<br></br>
> Google Word2Vec 모델 가져오기
```python
from gensim.models import KeyedVectors
word2vec_path = os.getenv('HOME')+'/sentiment_classification/GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=1000000)
vector = word2vec['computer']
vector     # 무려 300dim의 워드 벡터입니다.
```
Google Word2Vec 는 300 차원의 벡터로 이루어져 있기에 300 만 개의 모든 단어를 로딩하기에는 필요 컴퓨터 자원이 만만치 않다.

`KeyedVectors.load_word2vec_format` 메소드로 워드 벡터를 로딩할 때 가장 많이 사용되는 상위 100만 개만 `limt`으로 조건을 주어 로딩했다.
<br></br>
> Word2Vec 를 통한 단어간 유사도 확인
```python
# 메모리를 다소 많이 소비하는 작업이니 유의해 주세요.
word2vec.similar_by_word("love")
```
<br></br>
> 임베딩 행렬에 Word2Vec 단어를 복사
```python
vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

# embedding_matrix에 Word2Vec 워드벡터를 단어 하나씩마다 차례차례 카피한다.
for i in range(4,vocab_size):
    if index_to_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_to_word[i]]
```
<br></br>
> 앞서 학습한 모델의 임베딩 레리어를 Word2Vec 로 교체하여 학습
```python
from tensorflow.keras.initializers import Constant

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수 (변경가능한 하이퍼파라미터)

# 모델 구성
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 
                                 word_vector_dim, 
                                 embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                 input_length=maxlen, 
                                 trainable=True))   # trainable을 True로 주면 Fine-tuning
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid')) 

model.summary()
```
<br></br>
> 모델 학습
```python
# 학습의 진행
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```
<br></br>
> 테스트셋을 통한 모델 평가
```python
# 테스트셋을 통한 모델 평가
results = model.evaluate(x_test,  y_test, verbose=2)

print(results)
```
Word2Vec 를 적용하면 임베딩 레이어 모델보다 약 5% 이상의 성능이 향상됨을 알 수 있다.
<br></br>
