# 06. 작사가 인공지능 만들기

## 학습 목표

- 인공지능이 문장을 이해하는 방식의 이해
- 인공지능이 모델에 작문을 가르치는 법 이해
- 나만의 인공지능 작사가 모델 구현
<br></br>

## 시퀀스? 시퀀스!

시퀀스 (Squence) 데이터란 나열된 데이터를 의미한다.
데이터의 각 요소가 동일한 속성을 띌 필요는 없지만 인공지능을 통해 예측을 하기 위해서는 어느 정도 연관성이 있어야 한다.

영화, 전기, 수열, 문장, 주가, 날짜, 드라마 등 많은 유형의 데이터가 시퀀스데이터이다.

문장을 구성하는 단어들은 문법이라는 규칙에 맞춰 배열되어있다. 

하지만 문법을 인공지능으로 학습하기란 매우 어렵기 때문에 통계에 기반한 방법으로 접근한다.

<br></br>
## I 다음 am 을 쓰면 반 이상은 맞더라

`나는 밥을 [ ]` 에서 빈 칸에 들어갈 말이 `먹는다` 라는 것을 쉽게 알 수 있다.

또한 `알바생이 커피를 [ ]` 라면 아마도 `만든다` 가 정답일 것이다.

이 둘은 통계적으로 해당 상황에서 많이 쓰일 확률이 `먹는다` 와 `만든가` 로 접근할 수 있다.

인공지능에게 글을 이해하는 방식도 역시 통계에 기반한 방법으로 접근이 가능하다.

문법을 학습시키는 것이 아닌, 많은 양의 글 을 학습시키고, 뒤에 올 말을 통계적으로 맞추는 것이다.

때문에 많은 데이터가 좋은 결과로 직결된다.

이러한 방식을 가장 잘 처리하는 모델은 순환신경망 (RNN) 모델이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-12-RNN2.max-800x600.png)
<br></br>

위 그림은 순환신경망 RNN 을 단순히 표현한 그림이다.

해당 그림에서 시작인 `나는` 은 어떻게 만들까?

바로 `<start>` 라는 특수한 토큰을 맨 앞에 추가해주므로써 해결할 수 있다. `<start>` 토큰은 어떤 문장이든 `<start>` 토큰에 해당하는 값으로 시작하라는 의미이다.

즉, 순환 신경망은 이렇게 시작은 정해주고, 시작을 다음 단계의 입력으로 재사용한다. 이러한 재사용이 순환신경망이라는 이름을 붙이는 이유이다.

그렇다면 다 만들었다면 어떻게 끝내야 할까?  `<start>` 토큰과 같이 `<end>` 토큰을 생성한다.

`<start>` 가 문장의 시작에 더해진 입력 데이터(문제지)와, `<end>` 가 문장의 끝에 더해진 출력 데이터(답안지)가 필요하며, 이는 **문장 데이터만 있으면 만들어낼 수 있다.**
<br></br>
> `<start>`, `<end>` 토큰을 활용한 문장 생성
```python
sentence = " 나는 밥을 먹었다 "

source_sentence = "<start>" + sentence
target_sentence = sentence + "<end>"

print("Source 문장:", source_sentence)
print("Target 문장:", target_sentence)
```
<br></br>

### 언어 모델 (Language Model)

인공지능 모델을 통해 작문을 할 경우 통계에 기반한 접근법을 사용한다고 말했다.

예를 들어 '나는 밥을 먹었다.' 라는 문장을 만들고자 할 때, '나는 밥을' 다음에 '먹었다' 가 나올 확률을 $p(먹었다|나는,밥을)$ 이라고 한다면 이 확률은 '나는' 뒤에 '밥이' 가 나올 확률인 $p(밥을|나는)$ 보다는 높게 나올 것이다.

이렇게 높은 확률을 가지게 될 때 자연스럽다는 것을 알 수 있다. 하지만 확률값이 낮다고 틀린 것은 아니다. 단지  조금 부자연스러운 확률이 높다는 것이다.

n - 1 개의 단어 시퀀스 $w_1,⋯,w_n − 1$ 가 주어졌을 때, n 번째 단어 $w_n$ 으로 무엇이 올지를 예측하는 확률 모델을 **언어 모델(Language Model)**이라고 부른다.

파라미터 $θ$ 로 모델링하는 언어 모델을 다음과 같이 표현할 수 있다.

$$P(w_n | w_1, …, w_{n-1};\theta )$$

이러한 모델은 어떻게 학습 시킬 수 있을까? 복잡해 보일 수 있지만 간단하다. 어떤한 텍스트도 언어 모델의 학습 데이터가 될 수 있기 때문이다.

n - 1 번째까지의 단어 시퀀스가 x_train이 되고 n 번째 단어가 y_train 이 되는 데이터셋은 무궁무진하게 만들 수 있기 때문이다.

이렇게 학습된 언어 모델을 테스트 모드로 가동하게 되면 일정한 단어 시퀀스가 주어질 때 다음 단어를 계속해서 출력해 내게되며, 이러한 연유로 잘 학습된 언어 모델은 문장 생성기로 사용할 수 있다.
<br></br>

## 실습 (01). 데이터 다듬기

> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/lyricist/data 
$ mkdir -p ~/aiffel/lyricist/models
```
<br></br>
> 사용할 데이터 다운 
> (Tensorflow 에서 제공하는 셰익스피어의 연극 대본)
```python
$ wget https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt 
$ mv shakespeare.txt ~/aiffel/lyricist/data
```
<br></br>
> 사용할 라이브러리 가져오기
```python
import re                  # 정규표현식을 위한 Regex 지원 모듈 (문장 데이터를 정돈하기 위해) 
import numpy as np         # 변환된 문장 데이터(행렬)을 편하게 처리하기 위해
import tensorflow as tf    # 대망의 텐서플로우!
import os

# 파일을 읽기모드로 열어 봅니다.
file_path = os.getenv('HOME') + '/lyricist/data/shakespeare.txt'
with open(file_path, "r") as f:
    raw_corpus = f.read().splitlines()   # 텍스트를 라인 단위로 끊어서 list 형태로 읽어옵니다.

print(raw_corpus[:9])    # 앞에서부터 10라인만 화면에 출력해 볼까요?
```
<br></br>

데이터를 불러오면 연극 대본임을 알 수 있는데, 우리가 필요한 데이터는 문장만 원하며, 공백, 이름 등의 정보는 필요가 없다.

데이터를 확인해보면 화자가 표기된 문장은 문장의 끝이 `:` 로 끝나게 되어 있음을 알 수 있다.

따라서 `:` 를 기준으로 문장을 제외시키며, 대사가 없다면 문장 길이가 0 일 테니 이 또한 제외시키도록 한다.
<br></br>
> 문장 길이가 0 인 것과, 문장 끝이  `:` 인 문장 제외
> (10개만 우선 확인)
```python
for idx, sentence in enumerate(raw_corpus):
    if len(sentence) == 0: continue   # 길이가 0인 문장은 건너뜁니다.
    if sentence[-1] == ":": continue  # 문장의 끝이 : 인 문장은 건너뜁니다.

    if idx > 9: break   # 일단 문장 10개만 확인해 볼 겁니다.
        
    print(sentence)
```
모델에 필요한 문장이 있는 데이터만 성공적으로 출력됨을 알 수 있다.
<br></br>

이렇게 문장만으로 데이터를 만들고, 이를 활용해 단어 사전을 만들어야 한다.

이렇게 단어 사전을 만드는 과정을 토큰화 (Tokenize) 라고 한다.

토큰화를 하는 가장 대표적인 방법은 띄워쓰기를 기준으로 나누는 방법이다.

하지만 다음과 같은 문제가 발생할 수 있다.

1.  Hi, my name is John. *("Hi," "my", …, "john." 으로 분리됨) - 문장부호
    
2.  First, open the first chapter. *(First와 first를 다른 단어로 인식) - 대소문자
    
3.  He is a ten-year-old boy. *(ten-year-old를 한 단어로 인식) - 특수문자

첫 번째 문제를 방지하기 위해서는 문장 부호 양쪽에 공백을 추가해 줌으로써 방지할 수 있다.

두 번째 문제는 모든 문자들을 소문자로 변환함으로써 예방할 수 있다.

마지막으로 세 번째 문제는 모든 특수문자를 제거함으로써 해결할 수 있다.

이러한 전처리 과정은 정규표현식 (Regex) 를 통해 필터링 할 수 있다.
<br></br>
> 정규표현식을 통한 데이터 전처리 및 `<start>`, `<end>` 토큰 추가
```python
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()       # 소문자로 바꾸고 양쪽 공백을 삭제
  
    # 아래 3단계를 거쳐 sentence는 스페이스 1개를 delimeter로 하는 소문자 단어 시퀀스로 바뀝니다.
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)        # 패턴의 특수문자를 만나면 특수문자 양쪽에 공백을 추가
    sentence = re.sub(r'[" "]+', " ", sentence)                  # 공백 패턴을 만나면 스페이스 1개로 치환
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)  # a-zA-Z?.!,¿ 패턴을 제외한 모든 문자(공백문자까지도)를 스페이스 1개로 치환

    sentence = sentence.strip()

    sentence = '<start> ' + sentence + ' <end>'      # 이전 스텝에서 본 것처럼 문장 앞뒤로 <start>와 <end>를 단어처럼 붙여 줍니다
    
    return sentence

print(preprocess_sentence("This @_is ;;;sample        sentence."))   # 이 문장이 어떻게 필터링되는지 확인해 보세요.
```
<br></br>

자연어 처리 분야에서 모델에 입력되는 문장을 소스문장 (Source Sentence) 이라고 한다.

정답 역할을 하게되는 모델의 출력 문장은 타겟 문장 (Target Sentence) 라고 한다. 이는 각각  X_train, y_train 에 해당한다.

즉 위에서 만든 전처리 함수를 통해 토큰화를 수행하고, 끝 단어 `<end>` 를 없앤다면 소스 문장, `<start>` 를 없앤다면 타겟 문장이 된다.
<br></br>
> 전처리 함수를 활용해 정제된 데이터 구축
```python
corpus = []

for sentence in raw_corpus:
    if len(sentence) == 0: continue
    if sentence[-1] == ":": continue
        
    corpus.append(preprocess_sentence(sentence))
        
corpus[:10]
```
<br></br>

이로써 데이터 준비는 끝마쳤다.

하지만 사람이 언어를 습득할 때 모국어 표현을 통해 습득하듯이 컴퓨터도 마찬가지다.

즉, 준비한 데이터를 0 과 1 로 이루어진 데이터로 만들어 줄 필요가 있다.

이를 위해서는 `tf.keras.preprocessing.text.Tokenizer` 패키지를 활용하면 된다. 이를 활용하면 정제된 데이터를 토큰화하고, 단어 사전(vocabulary 또는 dictionary라고 칭함)을 만들어주며, 데이터를 숫자로 변환까지 한번에 수행해 준다.

이렇게 데이터를 모델이 학습할 수 있게 변환해 주는 과정을 **벡터화(vectorize)** 라 하며, 숫자로 변환된 데이터를 **텐서(tensor)** 라고 칭한다.

+ 참고 : [Tensor란 무엇인가?](https://rekt77.tistory.com/102)

<br></br>
> 벡터화 수행
```python
def tokenize(corpus):
    # 텐서플로우에서 제공하는 Tokenizer 패키지를 생성
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=7000,  # 전체 단어의 개수 
        filters=' ',    # 별도로 전처리 로직을 추가할 수 있습니다. 이번에는 사용하지 않겠습니다.
        oov_token="<unk>"  # out-of-vocabulary, 사전에 없었던 단어는 어떤 토큰으로 대체할지
    )
    tokenizer.fit_on_texts(corpus)   # 우리가 구축한 corpus로부터 Tokenizer가 사전을 자동구축하게 됩니다.

    # 이후 tokenizer를 활용하여 모델에 입력할 데이터셋을 구축하게 됩니다.
    tensor = tokenizer.texts_to_sequences(corpus)   # tokenizer는 구축한 사전으로부터 corpus를 해석해 Tensor로 변환합니다.

    # 입력 데이터의 시퀀스 길이를 일정하게 맞추기 위한 padding  메소드를 제공합니다.
    # maxlen의 디폴트값은 None입니다. 이 경우 corpus의 가장 긴 문장을 기준으로 시퀀스 길이가 맞춰집니다.
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  

    print(tensor,tokenizer)
    return tensor, tokenizer

tensor, tokenizer = tokenize(corpus)
```
<br></br>
> 벡터화를 마친 데이터에서 3 ~ 10 행까지만 출력
```python
print(tensor[:3, :10])
```
<br></br>
> 벡터화를 마친 데이터의 출력 값이 어떤 단어와 매칭되어 있는지 단어 사전을 통해 확인
```python
for idx in tokenizer.index_word:
    print(idx, ":", tokenizer.index_word[idx])

    if idx >= 10: break
```
2번 인덱스가 바로 `<start>` 임을 확인할 수 있다.
<br></br>

이제 생성된 텐서를 소스와 타켓으로 분리하여 모델 학습을 수행하면된다.

텐서 출력부에서 행 뒤쪽에 0 이 많이 나온 부분은 정해진 입력 시퀀스 길이보다 문장이 짧을 경우 0 으로 패딩(padding) 을 채워넣은 것이다.

패딩에 해당하는 단어는 사전에 없지만 0은 바로 패딩 문자 `<pad>` 가 된다.
<br></br>
> 소스 문장과 타겟 문장으로 데이터 분리
```python
src_input = tensor[:, :-1]  # tensor에서 마지막 토큰을 잘라내서 소스 문장을 생성합니다. 마지막 토큰은 <end>가 아니라 <pad>일 가능성이 높습니다.
tgt_input = tensor[:, 1:]    # tensor에서 <start>를 잘라내서 타겟 문장을 생성합니다.

print(src_input[0])
print(tgt_input[0])
```
<br></br>

corpus 내의 첫번째 문장에 대해 생성된 소스와 타겟 문장을 확인해보았다.

소스 문장은 2(`<start>`) 에서 시작해서 3(`<end>`) 으로 끝난 후 0(`<pad>`) 로 채워져 있으며, 타겟 문장은 2로 시작하지 않고 소스를 왼쪽으로 한칸 시프트한 형태를 가지고 있음을 확인할 수 있다.

앞서 model.fit(x_train, y_train, …) 형태로 Numpy Array 데이터셋을 생성하여 model 에 제공하는 형태의 학습을 많이 진행했었다.

하지만 텐서플로우를 활용할 경우 생성된 데이터를 이용해 `tf.data.Dataset` 객체를 생성하는 방법을 흔히 사용한다. 이를 데이터 객체 생성이라 한다.

`tf.data.Dataset`객체는 텐서플로우에서 사용할 경우 데이터 입력 파이프라인을 통한 속도 개선 및 각종 편의기능을 제공한다.
<br></br>
> `tf.data.Dataset.from_tensor_slices()` 메소드를 이용해 `tf.data.Dataset` 객체 생성
```python
BUFFER_SIZE = len(src_input)
BATCH_SIZE = 256
steps_per_epoch = len(src_input) // BATCH_SIZE

VOCAB_SIZE = tokenizer.num_words + 1    # tokenizer가 구축한 단어사전 내 7000개와, 여기 포함되지 않은 0:<pad>를 포함하여 7001개

dataset = tf.data.Dataset.from_tensor_slices((src_input, tgt_input)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset
```
<br></br>

위 과정을 정리해보면 다음과 같다.

-   정규표현식을 이용한 corpus 생성

-   `tf.keras.preprocessing.text.Tokenizer`를 이용해 corpus를 텐서로 변환

-   `tf.data.Dataset.from_tensor_slices()`를 이용해 corpus 텐서를  `tf.data.Dataset`객체로 변환

이러한 과정을 데이터 전처리 과정이라고 한다.
<br></br>

## 실습 (02). 인공지능 학습시키기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-12-4.max-800x600.png)
<br></br>

지금부터 만들 모델의 구조는 위 그림과 같다.

이러한 구조를 tf.keras.Model을 Subclassing하는 방식으로 만들고자 하며, 1 개의 Embedding 레이어, 2 개의 LSTM 레이어, 1 개의 Dense 레이어로 구성되어 있다.
<br></br>
> 모델 생성
```python
class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(TextGenerator, self).__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)
        
    def call(self, x):
        out = self.embedding(x)
        out = self.rnn_1(out)
        out = self.rnn_2(out)
        out = self.linear(out)
        
        return out
    
embedding_size = 256
hidden_size = 1024
model = TextGenerator(tokenizer.num_words + 1, embedding_size , hidden_size)
```
<br></br>

모델을 간단하게 살펴보자면 입력 텐서에 단어 사전의 인덱스가 들어 있고, Embedding 레이어는 이 인덱스 값을 해당 인덱스 번째의 워드 벡터로 바꿔 주는 역할을 한다.

이 워드 벡터는 의미 벡터 공간상에서 단어의 추상적 표현 (Representation) 으로 사용한다.

위 코드에서 `embedding_size` 는 워드 벡터의 차원수, 즉 단어가 추상적으로 표현되는 크기이다.

값이 커질수록 단어의 추상적인 특징을 더 잘 표현할 수 있지만 충분한 데이터가 없다면 혼란만을 야기할 뿐이다.

LSTM 레이어의 hidden state 의 차원수인 `hidden_size` 도 같은 맥락이며, 쉽게 얼마나 많은 일꾼을 모델에서 사용하겠는가? 에 해당한다.

이 일꾼도 워드 벡터의 차원 수와 같이 충분한 데이터가 주어지지 않는다면 혼란만 야기할 뿐이다.

모델 생성을 마치기 전에 model에 데이터를 아주 조금 태워 보는 것도 방법이다. model의 input shape가 결정되면서 model.build()가 자동으로 호출된다.
<br></br>
> 모델에 데이터를 조금 태워보기
```python
for src_sample, tgt_sample in dataset.take(1): break
model(src_sample)
```
모델의 최종 출력 텐서 shape를 유심히 보면 `shape=(256, 20, 7001)` 임을 알 수 있다.

7001은 Dense 레이어의 출력 차원수이며, 256은 이전 스텝에서 지정한 배치 사이즈이다. 20 은 LSTM은 자신에게 입력된 시퀀스의 길이만큼 동일한 길이의 시퀀스를 출력한다는 의미이다.

즉, 7001개의 단어 중 어느 단어의 확률이 가장 높을지를 의미하며, `dataset.take(1)`를 통해서 1개의 배치, 즉 256개의 문장 데이터를 가져옴을 의미한다.

마지막으로 20 에 해당하는 부분이 만약 `return_sequences=False` 였다면 LSTM 레이어는 1개의 벡터만 출력한다.

하지만 모델을 구성할 때 입력 데이터의 시퀀스 길이를 설정한 적이 없었다. 어떻게 20 이라는 시퀀스 길이가 나온 것일까?

바로 데이터를 입력할 때 데이터셋의 max_len 값을 통해 나온 결과이다.
<br></br>
> 생성한 모델 확인 (요약)
```python
model.summary()
```
<br></br>

위 모들에서 Output Shape를 정확하게 알려주지 않았다.

이는 위에서 말한 것처럼 입력 데이터의 시퀀스 길이를 알려주지 않았기 때문에 Output Shape 를 특정할 수 없는 것이며, 이는 데이터를 입력받았을 때 비로소 정해진다.
<br></br>
> 모델 학습
```python
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)

model.compile(loss=loss, optimizer=optimizer)
model.fit(dataset, epochs=30)
```
<br></br>

## 실습 (03). 잘 만들어졌는지 평가하기

우리가 만든 모델을 가장 잘 평가하는 방법은 바로 직접 작문을 시켜보는 것이다.

> 모델에게 시작 문장을 전달하는 `generate_text` 함수 생성
```python
def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    # 테스트를 위해서 입력받은 init_sentence도 일단 텐서로 변환합니다.
    test_input = tokenizer.texts_to_sequences([init_sentence])
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    # 텍스트를 실제로 생성할때는 루프를 돌면서 단어 하나씩 생성해야 합니다. 
    while True:
        predict = model(test_tensor)  # 입력받은 문장의 텐서를 입력합니다. 
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1]   # 우리 모델이 예측한 마지막 단어가 바로 새롭게 생성한 단어가 됩니다. 

        # 우리 모델이 새롭게 예측한 단어를 입력 문장의 뒤에 붙여 줍니다. 
        test_tensor = tf.concat([test_tensor, 
                                                                 tf.expand_dims(predict_word, axis=0)], axis=-1)

        # 우리 모델이 <end>를 예측했거나, max_len에 도달하지 않았다면  while 루프를 또 돌면서 다음 단어를 예측해야 합니다.
        if predict_word.numpy()[0] == end_token: break
        if test_tensor.shape[1] >= max_len: break

    generated = ""
    # 생성된 tensor 안에 있는 word index를 tokenizer.index_word 사전을 통해 실제 단어로 하나씩 변환합니다. 
    for word_index in test_tensor[0].numpy():
        generated += tokenizer.index_word[word_index] + " "

    return generated   # 이것이 최종적으로 모델이 생성한 자연어 문장입니다.
```
위 코드를 살펴보면 while 반복문이 있다.

학습에서 while 반복문은 없었는데 왜 있을까?
학습에서는 소스 문장을 모델에 입력하고 출력되는 문장을 타겟 문장과 비교하면 되었지만 텍스트를 실제로 생성할 때 소스 문장과 타켓 문장이 없기 때문이다.

즉, 앞서 테스트용 데이터셋을 따로 구축한 적이 없기 때문이다.

generate_text() 함수에서 `init_sentence`를 인자로 받아 텐서로 만들고 있다. 기본값으로는 `<start>` 단어 하나만 받는다.

이때 반복문에서는 다음과 같은 작업을 수행한다.

-   while의 첫번째 루프에서 test_tensor에  `<start>`  하나만 들어갔다고 가정할 때, 모델이 출력으로 7001개의 단어 중  `A`를 골랐다고 하자.

-   while의 두번째 루프에서 test_tensor에는  `<start> A`가 들어가며, 그래서 우리의 모델이 그다음  `B`를 골랐다고 하자.

-   while의 세번째 루프에서 test_tensor에는  `<start> A B`가 들어가며, 그래서….. (이하 후략)

위와 같은 과정이 반복되는 것이다.
<br></br>
> 문장 생성 함수를 실행
```python
generate_text(model, tokenizer, init_sentence="<start> he")
```
 정상적으로 문장을 출력하는 것을 볼 수 있다.
<br></br>
