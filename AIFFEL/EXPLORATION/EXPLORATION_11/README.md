# 11. 뉴스 요약봇 만들기

## 학습 목표

-   Extractive/Abstractive summarization 이해하기
-   단어장 크기를 줄이는 다양한 text normalization 적용해보기
-   seq2seq의 성능을 Up시키는 Attention Mechanism 적용하기
<br></br>

## 텍스트 요약 (Text Summarizaion) 이란?

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-21-1.png)
<br></br>
텍스트 요약 (Text Summarization) 이란 위 그림과 같이 긴 길이의 문서 원문을 핵심 주제로 짧게 요약한 문장으로 변환하는 것을 의미한다.

텍스트 요약의 중요한 부분은 요약 전후에 정보 손실 발생이 최소화되어야 한다는 것이다. 이는 정보를 압축하는 과정과 동일하다.

긴 문장을 읽고 정확히 이해한 후, 의미를 손상하지 않는 짧은 다른 표현으로 원문을 번역해야 하는 것이다. 따라서 원문의 길이가 길수록 어려운 작업이 된다.

텍스트 요약은 추출적 요약 (Extractive Summarization) 과 추상적 요약 (Abstractive Summarizaion) 2 가지 접근법으로 나눌 수 있다.
<br></br>

### 추출적 요약 (Extractive Summarization)

추출적 요약은 원문에서 문장들을 추출해서 요약하는 방식으로, 원문에서 등장하는 문장을 그대로 활용한다.

하지만 원문에서 중요한 문장일 순 있지만, 이 문장간의 연결이 자연스럽지 않을 수 있다.

이 추출적 요약은 딥러닝보다는 주로 전통적인 머신러닝 방식에 속하는 텍스트랭크와 같은 알고리즘을 사용해서 이 방법을 사용한다. 

다른 시각에서 보면 요약문에 들어갈 핵심문장인지를 판별하는 점에서 문장 분류 (Text Classification) 문제로 볼 수 있다.

+ 참고 : [텍스트 랭크](https://www.aclweb.org/anthology/W04-3252.pdf)
<br></br>

### 추상적 요약 (Abstractive Summarization)

추상적 요약은 추출적 요약과는 다른 방식으로 접근하는데, 원문으로 내용이 요약된 새로운 문장을 생성하는 방식이다.

새로운 문장이란 추출적 요약의 요약본은 원문에 동일한 문장이 있었던 것과 달리 요약본은 원문에 원래 없던 문장일 수도 있다는 것이다.

자연어 처리 분야 중 자연어 생성 (Natural Language Generation, NLG) 의 영역인 것이다.

자연어 처리에는 대표적으로 RNN 모델을 사용한다. 하지만 RNN 은 학습 데이터의 길이가 길수록 먼 과거의 정보를 현재로 전달하기가 어렵다. 이를 장기 의존성 (Long Term Dependency) 문제라고 한다.

이러한 문제로 인해 단순히 RNN 을 이용해 언어 생성 모델을 만든다고 해서 원문을 읽고 요약본을 만들어 내기란 쉽지않다.
<br></br>

## 인공 신경망으로 텍스트 요약 훈련시키기

seq2seq 모델을 통해 추상적 요약 방식의 텍스트 요약 모델을 만들어보고자 한다.

seq2seq 모델이란 두 개의 RNN 아키텍처를 사용하여 입력 시퀀스로부터 출력 시퀀스를 생성해내는 자연어 생성 모델로, 주로 뉴럴 기계번역에 사용되는 모델이다.
<br></br>

### seq2seq 개요

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-21-2.max-800x600.png)
<br></br>
요약하고자 하는 원문을 첫 번째 RNN 인 인코더로 입력하면 인코더는 이를 하나의 고정된 벡터로 변환한다.

이 벡터를 문맥 정보를 가지고 있는 벡터라고 하며, 컨텍스트 벡터 (context vector) 라고 한다.

두 번째 RNN 인 디코더는 이 컨텍스트 벡터를 전달받아 한 단어씩 생성해내서 요약 문장을 완성한다.
<br></br>

### LSTM 과 컨텍스트 벡터

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-21-3.max-800x600.png)
<br></br>
seq2seq 를 구현할 때 인코더 / 디코더로 바닐라 RNN 이 아닌 LSTM 을 사용하고자 한다.

LSTM 과 바닐라 RNN 이 다른 점은 다음 Time Step 의 셀에 Hidden State 뿐만 아니라, Cell State 도 함께 전달한다는 점이다.

즉, 인코더가 디코더에 전달하는 컨텍스트 벡터 또한 Hidden State h 와 Cell State C 두 개의 값 모두 존재해야한다.
<br></br>

### 시작 토큰과 종료 토큰

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-21-4.png)
<br></br>
seq2seq 구조에서 디코더는 시작 토큰 SOS 가 입력되면, 각 시점마다 단어를 생성하고 이 과정을 종료 토큰 EOS 를 예측하는 순간까지 멈추지 않는다.

즉, 훈련 데이터의 예측 대상 시퀀스의 앞, 뒤에는 시작 토큰과 종료 토큰을 넣어주는 전처리를 통해 멈춰야할 위치를 알려줘야한다.
<br></br>

### 어텐션 메커니즘을 통한 새로운 컨텍스트 벡터 사용하기

기존의 seq2seq 를 수정하고 새로운 모듈을 붙여 모델의 성능을 높여보자. 기존의 seq2seq 는 인코더의 마지막에 time step 의 hidden state 를 컨텍스트 벡터로 사용한다.

하지만 RNN 계열의 인공 신경망 (바닐라 RNN, LSTM, GRU) 의 한계로 인해 컨텍스트 정보에는 이미 입력 시퀀스의 많은 정보가 손실된 상태가 된다.

어텐션 메커니즘 (Attention Mechanism) 은 이러한 한계점을 극복하여 인코더의 모든 step 의 hidden state 의 정보가 컨텍스트 벡터에 전부 반영되도록 한다.

하지만 인코더의 모든 hidden state 가 동일한 비중으로 반영되는 것이 아닌 디코더의 현재 time step 의 예측에 인코더의 각 step 이 얼마나 영향을 미치는가에 따른 가중합으로 계산된다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-21-5.png)
<br></br>
위 그림에서 seq2seq 모델이라면 디코더로 전달되는 인코더의 컨텍스트 벡터는 인코더의 마지막 스텝의 hidden state 인 $h_5$ 가 되겠지만, 어텐션 메커니즘이 적용된 seq2seq 인 Attentional seq2seq 이라면 인코더의 컨텍스트 벡터는 예를 들어 $0.2h_1+0.3h_2+0.1h_3+0.15h_4+0.25h_5$ 가 될 수도 있는 것이다.

컨텍스트 벡터를 구성하기 위한 인코더 hidden state 의 가중치 값은 디코더의 현재 스텝이 어디냐에 따라 계속 달라진다 는 점을 주의해야 한다. 즉, 디코더의 현재 문장 생성 부위가 주어부인지 술어부인지 목적어인지 등에 따라 인코더가 입력 데이터를 해석한 컨텍스트 벡터가 다른 값이 된다.

(일반 seq2seq 모델은 컨텍스트 벡터는 디코더의 현재 step 위치에 무관하게 한번 계산되면 불변하며, 고정된 값을 가진다.)

이렇게 디코더의 현재 step 에 따라 유동적으로 변하는 인코더의 컨텍스트 벡터를 사용해서 현재의 예측에 활용하면 디코더가 보다 정확한 예측을 할 수 있다.
<br></br>

## 데이터 준비하기

> 작업환경 구성
```bash
$ mkdir -p ~/aiffel/news_summarization/data
```
<br></br>

텍스트 요약 모델 학습에 사용할 데이터셋은 캐글에서 제공된 `아마존 리뷰 데이터셋` 이다.
<br></br>
+ [아마존 리뷰 데이터셋](https://www.kaggle.com/snap/amazon-fine-food-reviews)
<br></br>
> 학습 데이터 다운
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/Reviews.csv.zip 
$ mv Reviews.csv.zip ~/aiffel/news_summarization/data 
$ cd ~/aiffel/news_summarization/data && unzip Reviews.csv.zip
```
[데이터 : Reviews.csv.zip](https://aiffelstaticprd.blob.core.windows.net/media/documents/Reviews.csv.zip)
<br></br>
> NLTK 설치 및 NTLK 데이터셋 다운
```python
pip install nltk
```
NLTK의 불용어(stopwords) 를 사용하고자 한다. NLTK는 Natural Language Toolkit 의 축약어로 영어 기호, 통계, 자연어 처리를 위한 라이브러리로, NLTK에는 I, my, me, over, 조사, 접미사와 같이 문장에는 자주 등장하지만, 의미를 분석하고 요약하는데는 거의 의미가 없는 100여개의 불용어가 미리 정리되어있다.
<br></br>
> BeautifulSoup 라이브러리 설치
```python
pip install beautifulsoup4
```
문서 파싱을 위해 BeautifulSoup 라이브러리를 활용하고자 한다.
<br></br>
> NLTK 패키지에서 불용어 사전 다운 및 사용할 라이브러리 가져오기
```python
import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
```
<br></br>
> 다운 받은 데이터셋 가져오기
```python
data = pd.read_csv(os.getenv("HOME")+"/news_summarization/data/Reviews.csv", nrows = 100000)
print('전체 샘플수 :',(len(data)))
```
전체 데이터를 활용하지 않고, 10 만개의 데이터만 활용하고자 한다.
<br></br>
> 5개의 데이터만 출력]
```python
data.head()
```
<br></br>
> 사용할 Row 만 별도로 저장 후 다시 출력
```python
data = data[['Text','Summary']]
data.head()

#랜덤한 15개 샘플 출력
data.sample(15)
```
전체 데이터 중 Summary 열과 Text 열만 훈련에 사용하므로 별도로 저장하고 저장한 데이터만 다시 출력해준다.
<br></br>

## 데이터 전처리하기 (01). 데이터 정리하기

> 데이터 중 중복된 샘플 유무 확인
```python
print('Text 열에서 중복을 배제한 유일한 샘플의 수 :', data['Text'].nunique())
print('Summary 열에서 중복을 배제한 유일한 샘플의 수 :', data['Summary'].nunique())
```
중복을 제외한다면 Text 에는 88,426 개, Summary 에는 72,348 개의 의 유니크한 데이터가 존재한다. 이중 텍스트 자체가 중복되는 경우는 중복 샘플이므로 제거한다.
<br></br>
> 중복 샘플 제거
> (데이터프레임의 `drop_duplicates()`를 활용)
```python
data.drop_duplicates(subset = ['Text'], inplace = True)
print('전체 샘플수 :',(len(data)))
```
중복이 제거되면서 샘플수가 88,426 개로 줄어듬을 확인할 수 있다.
<br></br>
> 데이터중 공백으로 채워져있는 NULL 데이터 유무 확인
> (`.isnull().sum()`을 활용)
```python
print(data.isnull().sum())
```
Summary 에 1 개의 Null 값이 있음을 확인할 수 있다.
<br></br>
> NULL 값 제거
> (`dropna()` 함수 활용)
```python
data.dropna(axis = 0, inplace = True)
print('전체 샘플수 :',(len(data)))
```
<br></br>

### 텍스트 정규화 및 불용어 제거

간단한 전처리를 마친 데이터에는 많은 단어들이 있는데, 이 중에는 같은 의미지만 다른 표현으로 쓰이는 단어가 많다. 이러한 단어는 마치 다른 단어로 간주되기 때문에 같은 표현으로 통일시켜 줄 필요가 있다. (예 : i'm = i am 등)

이러한 과정을 텍스트 정규화 (Text Normalization) 이라고 한다.

+ 참고 : [텍스트 정규화](https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python)
<br></br>
> 텍스트 정규화
> 같은 의미이지만 다르게 표현되는 단어들을 통일시켜준다.
```python
contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

print("정규화 사전의 수: ",len(contractions))
```
<br></br>
> 불용어 (StopWords) 제거
> NLTK 에서 제공하는 불용어 리스트를 참조하여 불용어 제거
```python
print('불용어 개수 :', len(stopwords.words('english') ))
print(stopwords.words('english'))
```
불용어란 자주 등장사지만 자연어 처리를 할 때 의마가 없는 단어들을 의미한다.
<br></br>
> 모든 단어를 소문자로 변환, 특수문자 제거
```python
#데이터 전처리 함수
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","",sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```
함수의 하단을 보면, NLTK 를 이용해 불용어를 제거하는 파트가 있는데, 이는 Text 전처리 시에서만 호출하고 이미 상대적으로 문장 길이가 짧은 Summary 전처리할 때는 호출하지 않는다.

Abstractive 한 문장 요약 결과문이 자연스러운 문장이 되려면 이 불용어들이 Summary 에는 남아 있는게 더 좋기 때문에 이를 위해 함수의 인자로 remove_stopwords 를 추가하고, if 문을 추가하였다.
<br></br>

> 파싱을 위한 lxml 라이브러리 설치
```python
pip install lxml
```
<br></br>
> 전처리 함수가 잘 작동하는지 확인
```python
temp_text = 'Everything I bought was great, infact I ordered twice and the third ordered was<br />for my mother and father.'
temp_summary = 'Great way to start (or finish) the day!!!'

print(preprocess_sentence(temp_text))
print(preprocess_sentence(temp_summary, False))  # 불용어를 제거하지 않습니다.
```
모든 알파벳이 소문자로 변환되고, 특수문자, 
html 태그가 제거 및 괄호로 묶였던 단어 시퀀스가 제거된 것도 확인할 수 있다.
<br></br>
> 훈련 데이터 전체에 대해 전처리 함수를 적용
```python
clean_text = []

# 전체 Text 데이터에 대한 전처리 : 10분 이상 시간이 걸릴 수 있습니다. 
for s in data['Text']:
    clean_text.append(preprocess_sentence(s))

# 전처리 후 출력
clean_text[:5]
```
<br></br>
> Summary 에 대해 전처리 함수를 적용
> (불용어 제거를 수행하지 않기 때문에 두 번째 인자로 False 를 넣어준다.)
```python
clean_summary = []

# 전체 Summary 데이터에 대한 전처리 : 5분 이상 시간이 걸릴 수 있습니다. 
for s in data['Summary']:
    clean_summary.append(preprocess_sentence(s, False))

clean_summary[:5]
```
Text의 경우에는 불용어를 제거하고, Summary의 경우에는 불용어를 제거하지 않기 위해 따로 호출하여 진행하였다.
<br></br>
> 전처리를 마친 후 샘플을 통해 데이터 확인
```python
data['Text'] = clean_text
data['Summary'] = clean_summary

# 빈 값을 Null 값으로 변환
data.replace('', np.nan, inplace=True)
```
전처리 과정에서 문장의 모든 단어가 삭제되는 경우가 발생할 수 있으니 샘플 확인을 꼭 해보자.

이 경우 샘플 자체는 빈 값을 가지기 때문에 빈 값을 NULL 값으로 변환시켜주었다.
<br></br>
> NULL 값이 생겼는지 확인
> (`.isnull().sum()` 을 활용)
```python
data.isnull().sum()
```
Summary 열에서 70 개의 Null 값 이 생긴 것을 확인할 수 있다. 이는 제거해줄 필요가 있다.
<br></br>
> 새로 생긴 Summary 의 NULL 값을 제거
```python
data.dropna(axis=0, inplace=True)
print('전체 샘플수 :',(len(data)))#데이터 전처리 함수
```
<br></br>

## 데이터 전처리하기 (02). 훈련데이터와 테스트데이터 나누기

학습을 진행하기에 앞서 학습에 사용할 데이터의 크기를 결정하고 문장의 시작과 끝을 표시해 줘야한다.
<br></br>

### 샘플의 최대 길이 정하기

> Text 와 Summary 의 최소, 최대, 평군 길이를 확인하고 분포를 시각화
```python
# 길이 분포 출력
import matplotlib.pyplot as plt

text_len = [len(s.split()) for s in data['Text']]
summary_len = [len(s.split()) for s in data['Summary']]

print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

plt.subplot(1,2,1)
plt.boxplot(summary_len)
plt.title('Summary')
plt.subplot(1,2,2)
plt.boxplot(text_len)
plt.title('Text')
plt.tight_layout()
plt.show()

plt.title('Summary')
plt.hist(summary_len, bins = 40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('Text')
plt.hist(text_len, bins = 40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
많은 양의 데이터를 다룰때는 데이터를 시각화하여 보는 것이 많은 도움이 된다. 

차례대로 그래프는 각각 요약과 실제 텍스트의 길이 분포, 요약본 샘플 길이별 갯수, 실제 텍스트 샘플 길이별 갯수를 나타낸다.

Text 의 경우 최소 길이가 2, 최대 길이가 1,235 으로 그 차이가 굉장히 크며, 평균 길이는 38 로 시각화 된 그래프로 봤을 때는 대체적으로는 100 내외의 길이를 가진다는 것을 확인할 수 있다.

Summary 의 경우 최소 길이가 1, 최대 길이가 28, 그리고 평균 길이가 4 로 Text 에 비해 상대적으로 길이가 매우 짧은 것을 볼 수 있다.
<br></br>
> Text 와 Summary 의 적절한 최대 길이를 임의로 설정
```python
text_max_len = 50
summary_max_len = 8
```
<br></br>
> 얼마나 많은 샘플이 포함되는지 통계로 확인
> (훈련 데이터와 샘플의 길이를 입력했을 때 데이터츼 몇 % 가 포함되는지 계산하는 함수 생성)
```python
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))
```
<br></br>
> 위에서 구현한 함수를 Text 와 Summary 에 적용
```python
below_threshold_len(text_max_len, data['Text'])
below_threshold_len(summary_max_len,  data['Summary'])
```
각각 50 과 8 로 패딩을 하게되면 해당 길이보다 긴 샘플들은 내용이 잘리게 되는데, Text 열의 경우에는 약 23% 의 샘플들이 내용이 사진다는 것을 알 수 있다.
<br></br>
> 정해진 길이에 맞춰 자르는 것이 아닌 정해진 길이보다 길면 제외하는 방법으로 데이터 정제
```python
data = data[data['Text'].apply(lambda x: len(x.split()) <= text_max_len)]
data = data[data['Summary'].apply(lambda x: len(x.split()) <= summary_max_len)]
print('전체 샘플수 :',(len(data)))
```
<br></br>

### 시작 토큰과 종료 토큰 추가하기

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-21-4.png)
<br></br>
위 그림에서 볼 수 있듯이 디코더는 시작 토큰을 입력받아 문장을 생성하고, 종료 토큰을 예측한 순간에 문장 생성을 멈춘다.

따라서 seq2seq 훈련을 위해서는 디코더의 입력과 레이블에 시작 토큰과 종료 토큰을 추가해 줘야한다.

시작 토큰은 'sostoken', 종료 토큰은 'eostoken'이라 임의로 명명하고 앞, 뒤로 추가해 주자.
<br></br>
> 디코더에 시작 토큰과 종료 토큰 추가
```python
#요약 데이터에는 시작 토큰과 종료 토큰을 추가한다.
data['decoder_input'] = data['Summary'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['Summary'].apply(lambda x : x + ' eostoken')
data.head()
```
디코더의 입력에 해당하면서 시작 토큰이 맨 앞에 있는 문장의 이름을 decoder_input, 디코더의 출력 또는 레이블에 해당되면서 종료 토큰이 맨 뒤에 붙는 문장의 이름을 decoder_target 이라고 이름을 정하고 두 개의 문장 모두 Summary 열로부터 만든다.
<br></br>
> 인코더의 입력과 디코더의 입력 레이블을 넘파이 타입으로 저장
```python
encoder_input = np.array(data['Text']) # 인코더의 입력
decoder_input = np.array(data['decoder_input']) # 디코더의 입력
decoder_target = np.array(data['decoder_target']) # 디코더의 레이블
```
<br></br>
> 코딩을 통해 훈련 데이터와 테스트 데이터 분리하기 위해 encoder_input 과 크기와 형태가 같은 순서가 섞인 정수 시퀀스 생성
```python
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print(indices)
```
데이터를 분리하는 방법은 패키지를 이용하는 방법, 직접 코딩을 통해 분리하는 방법 등 여러가지가 있지만 직접 코딩을 통해 데이터를 분리하기위해 encoder_input 과 크기와 형태가 같은 순서가 섞인 정수 시퀀스를 생성하였다.
<br></br>
> 정수 시퀀스를 통해 데이터의 샘플 순서를 정의
```python
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]
```
<br></br>
> 훈련 데이터와 테스트 데이터를 8 : 2 의 비율로 분리하기위해 테스트 데이터의 크기 정의
```python
n_of_val = int(len(encoder_input)*0.2)
print('테스트 데이터의 수 :',n_of_val)
```
전체 데이터의 크기에서 0.2 를 곱해 테스트 데이터의 크기를 정의했다.
<br></br>
> 훈련 데이터와 테스트 데이터 분리
```python
encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

print('훈련 데이터의 개수 :', len(encoder_input_train))
print('훈련 레이블의 개수 :',len(decoder_input_train))
print('테스트 데이터의 개수 :',len(encoder_input_test))
print('테스트 레이블의 개수 :',len(decoder_input_test))
```
훈련 데이터와 테스트 데이터가 각각 52,655 개와 13,163 개로 잘 분리된 것을 볼 수 있다.
<br></br>

## 데이터 전처리하기 (03). 정수 인코딩

### 단어 집합 (Vocaburary) 만들기 및 정수 인코딩

이제 컴퓨터가 학습할 수 있도록 텍스트 데이터들을 모두 정수로 바꿔줘야한다.

이를 위해 각 단어에 고유한 정수를 매핑해야한다. 이러한 과정을 단어 집합 (Vocaburary) 을 만든다고 한다.
<br></br>
> 훈련 데이터에 대해 단어 집합 생성
> 원문에 해당하는 `encoder_input_train` 에 대해 단어 집합 생성
> (Keras 의 토크나이저를 활용)
```python
src_tokenizer = Tokenizer() # 토크나이저 정의
src_tokenizer.fit_on_texts(encoder_input_train) # 입력된 데이터로부터 단어 집합 생성
```
Keras 의 토크나이저를 사용하면, 입력된 훈련 데이터로부터 단어 집합을 만들 수 있다.

생성된 단어 집합은 `src_tokenizer.word_index` 에 저장된다.
<br></br>

위에서 만든 단어 집합에 있는 모든 단어를 사용하지 않고, 빈도수가 낮은 단어들은 훈련 데이터에서 제외하고자 한다.

따라서 빈도수가 7 회 미만인 단어들의 비중을 확인해보자.
<br></br>
> 단어 집합에서 단어의 등장 빈도를 확인
```python
threshold = 7
total_cnt = len(src_tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in src_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
`src_tokenizer.word_counts.items()`에는 단어와 각 단어의 등장 빈도수가 저장되있으며, 이를 통해 통계적 정보를 얻을 수 있다.

`encoder_input_train`에는 총 32,017 개의 단어가 있음을 알 수 있다.

등장 빈도가 threshold 값인 7회 미만, 즉, 6회 이하인 단어들은 단어 집합에서 무려 70% 이상을 차지하지만 훈련 데이터에서 등장 빈도로 차지하는 비중은 3.39% 밖에 되지 않는 것을 확인할 수 있다.
<br></br>
> 등장 빈도가 6회 이하인 단어는 정수 인코딩 과정에서 제외하기
```python
src_vocab = 8000
src_tokenizer = Tokenizer(num_words = src_vocab) # 단어 집합의 크기를 8,000으로 제한
src_tokenizer.fit_on_texts(encoder_input_train) # 단어 집합 재생성.
```
토크나이저를 정의할 때 num_words의 값을 정해주면, 단어 집합의 크기를 제한할 수 있다.
<br></br>
> 단어 집합에 기반하여 입력으로 주어진 텍스트 데이터의 단어들을 모두 정수로 변환하는 정수 인코딩 수행
```python
# 텍스트 시퀀스를 정수 시퀀스로 변환
encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train) 
encoder_input_test = src_tokenizer.texts_to_sequences(encoder_input_test)

#잘 진행되었는지 샘플 출력
print(encoder_input_train[:3])
```
`texts_to_sequences()`는 생성된 단어 집합에 기반하여 입력으로 주어진 텍스트 데이터의 단어들을 모두 정수로 변환하는 정수 인코딩을 수행한다.

따라서 8,000 으로 제한했기에 8,000 을 초과하는 숫자들은 정수 인코딩 후 데이터에 존재하지 않는다.

샘플을 확인해보면 텍스트가 아닌 정수로 출력됨을 확인할 수 있다.
<br></br>
> Summary 데이터에 대해 정수 인코딩 수행
```python
tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(decoder_input_train)
```
단어 집합 생성 및 각 단어의 고유한 정수를 부여하는 정수 인코딩이 수행되며, 이는 `tar_tokenizer.word_index`에 저장된다.
<br></br>
> 등장 빈도수가 6 회 미만인 단어들의 비중을 확인
```python
threshold = 6
total_cnt = len(tar_tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tar_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
`tar_tokenizer.word_counts.items()`에는 단어와 각 단어의 등장 빈도수가 저장되어져 있으며, 이를 통해 통계적 정보를 얻을 수 있다.

등장 빈도가 5 회 이하인 단어들은 단어 집합에서 약 77% 를 차지하지만, 실제로 훈련 데이터에서 등장 빈도로 차지하는 비중은 상대적으로 매우 적은 수치인 5.89% 밖에 되지 않음을 확인할 수 있다.
<br></br>
> 등장 빈도 5회 이하인 단어들을 모두 제거
```python
tar_vocab = 2000
tar_tokenizer = Tokenizer(num_words = tar_vocab) 
tar_tokenizer.fit_on_texts(decoder_input_train)
tar_tokenizer.fit_on_texts(decoder_target_train)

# 텍스트 시퀀스를 정수 시퀀스로 변환
decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train) 
decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)
decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)

#잘 변환되었는지 확인
print('input')
print('input ',decoder_input_train[:5])
print('target')
print('decoder ',decoder_target_train[:5])
```
모든 정수 인코딩 작업을 마쳤으며, 이제 `decoder_input_train` 과 `decoder_target_train` 에는 더 이상 숫자 2,000이 넘는 숫자들은 존재하지 않는다.
<br></br>

정수 인코딩 과정 후에는 패딩 작업을 수행해 줘야한다.

하지만 패딩을 하기 전에 빈도수가 낮은 단어를 삭제하면서 빈 샘플이 발생했을 수 있기 이를 확인 해줘야 한다.

> 길이가 0 인 샘플들의 인덱스를 받아 `drop_train`과 `drop_test`에 라는 변수에 저장
```python
drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]

print('삭제할 훈련 데이터의 개수 :',len(drop_train))
print('삭제할 테스트 데이터의 개수 :',len(drop_test))

encoder_input_train = np.delete(encoder_input_train, drop_train, axis=0)
decoder_input_train = np.delete(decoder_input_train, drop_train, axis=0)
decoder_target_train = np.delete(decoder_target_train, drop_train, axis=0)

encoder_input_test = np.delete(encoder_input_test, drop_test, axis=0)
decoder_input_test = np.delete(decoder_input_test, drop_test, axis=0)
decoder_target_test = np.delete(decoder_target_test, drop_test, axis=0)

print('훈련 데이터의 개수 :', len(encoder_input_train))
print('훈련 레이블의 개수 :',len(decoder_input_train))
print('테스트 데이터의 개수 :',len(encoder_input_test))
print('테스트 레이블의 개수 :',len(decoder_input_test))
```
요약문인 `decoder_input` 에는 `sostoken` 또는 `decoder_target` 에는 `eostoken` 이 추가된 상태이고, 이 두 토큰은 모든 샘플에서 등장하므로 빈도수가 샘플수와 동일하게 매우 높으므로 단어 집합 제한에도 삭제 되지 않는다. 

따라서 길이가 0 이 된 요약문의 실제 길이는 1 로 나타난다. 이는 `decoder_input` 에는 `sostoken`, `decoder_target` 에는 `eostoken`만 남아있기 때문이다.

`drop_train`과 `drop_test` 변수에 저장된 샘플을 모두 삭제해야 한다.
<br></br>

###  패딩하기

전처리가 완료된 샘플들은 모두 다른 길이를 가지고 있다. 모델 학습을 위해 이렇게 제각기 다른 길이의 샘플을 모두 동일한 길이로 맞춰줘야한다.

이때 최대 길이를 구하고, 최대 길이보다 짧은 샘플은 뒤에 0 을 붙여주는 패딩작업을 수행해 줘야한다.
<br></br>
> 패딩 작업을 통해 최대 길이를 맞춰준다.
```python
encoder_input_train = pad_sequences(encoder_input_train, maxlen = text_max_len, padding='post')
encoder_input_test = pad_sequences(encoder_input_test, maxlen = text_max_len, padding='post')
decoder_input_train = pad_sequences(decoder_input_train, maxlen = summary_max_len, padding='post')
decoder_target_train = pad_sequences(decoder_target_train, maxlen = summary_max_len, padding='post')
decoder_input_test = pad_sequences(decoder_input_test, maxlen = summary_max_len, padding='post')
decoder_target_test = pad_sequences(decoder_target_test, maxlen = summary_max_len, padding='post')
```
<br></br>

##  모델 설계하기

> 모델 설계
> (함수형 API 를 이용해 인코더 설계)
```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 인코더 설계 시작
embedding_dim = 128
hidden_size = 256

# 인코더
encoder_inputs = Input(shape=(text_max_len,))

# 인코더의 임베딩 층
enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

# 인코더의 LSTM 1
encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# 인코더의 LSTM 2
encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# 인코더의 LSTM 3
encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
```
임베딩 벡터의 차원은 128 로 정의하고, hidden state 의 크기를 256 으로 정의했다.

hidden state 는 LSTM 에서 얼만큼의 수용력 (capacity) 를 가질지를 정하는 파라미터이며, 이 파라미터는 LSTM 의 용량의 크기나, LSTM 에서의 뉴런의 갯수를 의미한다. 이 용량이 무조건 많다고 해서 성능이 반듯이 향상되는 것은 아니다.

인코더의 LSTM 은 총 3 개의 층으로 구성해서 모델의 복잡도를 높였으며, hidden state 의 크기를 늘리는 것이 LSTM 층 1 개의 용량을 늘린다면, 3 개의 층을 사용하는 것은 모델의 용량을 늘린것이다.

3개의 층을 지나서 인코더로부터 나온 출력 벡터는 디코더로 보내줘야 한다.
<br></br>
> 디코더 임베딩 층 및 LSTM 설계
```python
# 디코더 설계

decoder_inputs = Input(shape=(None,))

# 디코더의 임베딩 층
dec_emb_layer = Embedding(tar_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# 디코더의 LSTM
decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])
```
LSTM 의 입력을 정의할 때, `initial_state` 의 인자값으로 인코더의 hidden state 와 cell state 의 값을 넣어줘야 한다.
<br></br>
> 디코더의 출력층 설계
```python
# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation = 'softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs) 

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()
```
디코더의 출력층에서는 Summary 의 단어장인 `tar_vocab` 의 수많은 선택지 중 하나의 단어를 선택하는 다중 클래스 분류 문제를 풀어야 한다.

때문에 Dense 의 인자로 `tar_vocab` 을 주고, 활성화 함수로 소프트맥스 함수를 사용한다.
<br></br>

### 어텐션 매커니즘

위에서 구현한 모델은 인코더의 hidden state와 cell state를 디코더의 초기 state로 사용하는 가장 기본적인 seq2seq 모델이다. 

여기서 디코터 출력층을 변형해줘 모델 성능을 향상시키는 어텐션 매커니즘을 적용해보자.
<br></br>
> 깃허브에 공개되어 있는 어텐션 함수 다운
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/src/layers/attention.py", filename="attention.py")
from attention import AttentionLayer
```
작업 디렉토리에 attention.py 파일이 다운되어 생긴다.
<br></br>
> 디코더의 출력층을 수정하여 어텐션 매커니즘 적용
```python
# 어텐션 층(어텐션 함수)
attn_layer = AttentionLayer(name='attention_layer')
# 인코더와 디코더의 모든 time step의 hidden state를 어텐션 층에 전달하고 결과를 리턴
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis = -1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()
```
<br></br>

## 모델 훈련하기
<br></br>
> 모델 훈련
```python
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
history = model.fit(x = [encoder_input_train, decoder_input_train], y = decoder_target_train, \
          validation_data = ([encoder_input_test, decoder_input_test], decoder_target_test),
          batch_size = 256, callbacks=[es], epochs = 50)
```
EarlyStopping 은 한국어로 해석 하면 '조기 종료'의 뜻을 가지고 있는데, 특정 조건이 충족되면 모델의 훈련을 멈추는 역할을 한다.

`val_loss` (검증 데이터의 손실) 을 모니터링 하면서, 검증 데이터의 손실이 줄어들지 않고 증가하는 현상이`patiensce =2` 2회 관측되면 학습을 멈추도록 설정하였다.
<br></br>
> 훈련 데이터의 손실과 검증 데이터의 손실 변화 과정을 시각화
```python
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
```
<br></br>

## 인퍼런스 모델 구현하기
<br></br>
> 테스트 단계에서 정수 인덱스 행렬로 존재하던 텍스트 데이터를 실제 데이터로 복원하기 위한 3 개의 사전을 준비
```python
src_index_to_word = src_tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
tar_word_to_index = tar_tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음
tar_index_to_word = tar_tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음
```
<br></br>

seq2seq는 훈련할 때와 실제 동작할 때 (인퍼런스 단계) 의 방식이 다르므로 그에 맞게 모델 설계를 별개로 진행해야 한다.
<br></br>
> 인코더 모델과 디코더 모델을 분리해서 설계
```python
# 인코더 설계
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))

dec_emb2 = dec_emb_layer(decoder_inputs)
# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
```
훈련 단계에서는 디코더의 입력부에 정답이 되는 문장 전체를 한꺼번에 넣고 디코더의 출력과 한번에 비교할 수 있으므로, 인코더와 디코더를 엮은 통짜 모델 하나만 준비하였다.

 하지만 정답 문장이 없는 인퍼런스 단계에서는 만들어야 할 문장의 길이만큼 디코더가 반복 구조로 동작해야 하기 때문에 부득이하게 인퍼런스를 위한 모델 설계를 별도로 해주어야 한다.
 <br></br>
 > 어텐션 매커니즘을 사용하는 출력층 설계
```python
# 어텐션 함수
decoder_hidden_state_input = Input(shape=(text_max_len, hidden_size))
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# 디코더의 출력층
decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat) 

# 최종 디코더 모델
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])
```
 <br></br>
 > 인퍼런스 단계에서 단어 시퀀스를 완성하는 함수 생성
```python
def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)

     # <SOS>에 해당하는 토큰 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tar_index_to_word[sampled_token_index]

        if(sampled_token!='eostoken'):
            decoded_sentence += ' '+sampled_token

        #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (summary_max_len-1)):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence
```
 <br></br>

## 모델 테스트 하기

테스트 단계에서는 정수 시퀀스를 텍스트 시퀀스로 변환하여 결과를 확인하는 것이 편리하다.

 <br></br>
 > 주어진 정수 시퀀스를 텍스트 시퀀스로 변환하는 함수 생성
```python
# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2text(input_seq):
    temp=''
    for i in input_seq:
        if(i!=0):
            temp = temp + src_index_to_word[i]+' '
    return temp

# 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2summary(input_seq):
    temp=''
    for i in input_seq:
        if((i!=0 and i!=tar_word_to_index['sostoken']) and i!=tar_word_to_index['eostoken']):
            temp = temp + tar_index_to_word[i] + ' '
    return temp
```
Text 의 정수 시퀀스에서는 패딩을 위해 사용되는 숫자 0 을 제외시키고 Summary 의 정수 시퀀스에서는 숫자 0, 시작 토큰의 인덱스, 종료 토큰의 인덱스를 출력에서 제외시키도록 함수를 생성하였다.
 <br></br>
 > 테스트 데이터 50 개 샘플에 대해 실제 요약과 예측된 요약을 비교
```python
for i in range(50, 100):
    print("원문 :", seq2text(encoder_input_test[i]))
    print("실제 요약 :", seq2summary(decoder_input_test[i]))
    print("예측 요약 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
    print("\n")
```
잘 된 요약과 그렇지 않은 요약 모두를 확인할 수 있다.

성능을 개선하기 위해서는 seq2seq와 어텐션의 자체의 조합을 좀 더 좋게 수정하는 방법도 있고, 빔 서치 (beam search), 사전 훈련된 워드 임베딩 (pre-trained word embedding), 또는 인코더 - 디코더 자체의 구조를 새로이 변경한 하는 트랜스포머 (Transformer)와 같은 여러 개선 방안들이 존재한다.
 <br></br>

## 추출적 요약 해보기

`Summa` 패키지에서는 추출적 요약을 위한 모듈인 summarize 를 제공하기 때문에 간단하게 추출적 요약을 만들어볼 수 있다.

영화 매트릭스 시놉시스에 summarize 를 활용하여 추출적 요약을 해보자.
 <br></br>
 
###  패키지 설치

> 패지키 설치
```python
pip install summa
```
 <br></br>

### 데이터 다운로드

> 필요 라이브러리 가져오기
```python
import requests
from summa.summarizer import summarize
```
 <br></br>
> 영화 매트릭스 시놉시스 데이터 다운로드
```python
text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
```
 <br></br>
 > 다운 받은 데이터의 일부만 확인
```python
print(text[:1500])
```
시놉시스는 아주 긴 출력 결과를 가지기 때문에 1,500 자 만큼만 출력해본다.
 <br></br>

### summarize 사용하기

Summa 의 summarize() 의 인자로 사용되는 값들은 다음과 같다.

+ **text (str)** : 요약할 테스트.  

+ **ratio (float, optional)** – 요약문에서 원본에서 선택되는 문장 비율. 0 ~ 1 사이값  

+ **words (int or None, optional)** – 출력에 포함할 단어 수. 만약, ratio 와 함께 두 파라미터가 모두 제공되는 경우 ratio 는 무시한다.  

+ **split (bool, optional)** – True 면 문장 list / False는 조인 (join) 된 문자열을 반환

Summa 의 summarize 는 문장 토큰화를 별도로 하지 않더라도 내부적으로 문장 토큰화를 수행한다.

때문에 문장 구분이 되어있지 않은 원문을 바로 입력으로 넣을 수 있다. 
 <br></br>
> 원문의 0.005 % 만을 출력하도록 설정
```python
print('Summary:')
print(summarize(text, ratio=0.005))
```
 <br></br>
 > 리스트로 출력 결과를 받도록 설정
```python
print('Summary:')
print(summarize(text, ratio=0.005, split=True))
```
리스트로 출력 결과를 받고 싶다면 split 인자의 값을 True 로 하면된다.
 <br></br>
 > 단어를 50 개만 선택
```python
print('Summary:')
print(summarize(text, words=50))
```
단어의 수로 요약문의 크기를 조절할 수도 있다.
 <br></br>

