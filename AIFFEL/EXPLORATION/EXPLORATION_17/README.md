# 17. 다음에 볼 영화 예측하기

### Session - Based Recommendation

Session - Based Recommendation 란 세션 데이터를 기반으로 유저가 마음에 들어할 만한 혹은 다음에 클릭 또는 구매할만한 아이템을 예측하여 추천해주는 것을 의미한다.

요즘은 인터넷을 통한 전자상거래가 빈번하게 이루어지고 있다. 이렇게 유저가 서비스를 이용하면서 발생하는 정보를 담은 데이터를 Session 이라고 한다.

유저의 행동 데이터는 유저측 브라우저를 통해 쿠키 형태로 저장되며, 쿠키는 세션과 상화작용하며 정보를 주고 받는다.

+ 참고 : 
	+ [쿠키, 세션, 캐시가 뭔가요?](https://www.youtube.com/watch?v=OpoVuwxGRDI&ab_channel=%EC%96%84%ED%8C%8D%ED%95%9C%EC%BD%94%EB%94%A9%EC%82%AC%EC%A0%84)
	+ [쿠키, 세션이란?](https://medium.com/@chrisjune_13837/web-%EC%BF%A0%ED%82%A4-%EC%84%B8%EC%85%98%EC%9D%B4%EB%9E%80-aa6bcb327582)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/session.max-800x600.png)
<br></br>

위 그림은 사용자의 행동 정보를 담은 세션과 어떤 행동을 나타냈는지에 대한 것이다.

9194111 세션의 유저는 약 8 분간 4 개의 아이템을 본 상황인 것이다. 이러한 유저의 행동 정보 데이터를 통해 유저에게 알맞는 아이템을 추천해 줄 수 있다.
<br></br>

## 학습 준비

> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/yoochoose-data
```
<br></br>
> 데이터 준비
```bash
$ cd ~/aiffel  
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/yoochoose-data.7z

$ sudo apt install p7zip-full
$ 7z x yoochoose-data.7z -oyoochoose-data
```
추천 엔진 솔루션 회사인 YOOCHOOSE 에서 공개한 E - Commerce 데이터를 활용한다.

+ 참고 : [E-Commerce 데이터](https://2015.recsyschallenge.com/challenge.html)
<br></br>
> 데이터 설명 읽어보기
```python
# 데이터 설명(README)를 읽어 봅니다. 
import os
f = open(os.getenv('HOME')+'/yoochoose-data/dataset-README.txt', 'r')
while True:
    line = f.readline()
    if not line: break
    print(line)
f.close()
```
데이터 설명을 확인해보면 몇 가지 특징이 있다.

1. 유저에 관한 정보를 알 수 없다. (성별, 나이, 장소, 최종 접속 날짜, 이전 구매 내역 등)

2. 아이템에 대한 정보를 알 수 없다. (아이템번호 외에 실제로 어떤 아이템인지에 관한 사진 혹은 설명, 가격 등)

이렇게 세세한 정보를 알 수 없는 데이터와 달리 정보를 알 수 있는 경우 Sequential Recommendation 으로 구별해서 부른다.

유저 및 아이템의 세부 정보를 Sequential Recommendation 모델에 적용한 분야는 Context - Aware 라는 키워드로 연구되고 있다.

위 다운받은 데이터인 E - Commerce 데이터의 경우 다음과 같은 특징을 가지고 있다.

1. 비로그인 상태로 탐색하는 유저가 많다.

2. 로그인 상태로 탐색한다고해도 매 접속마다 탐색하는 의도가 뚜력하게 다르다.

때문에 이전에 키보드를 검색한 이력은 다음 유저 탐색시 도움이 되지 않는다.
<br></br>

## Data PreProcess

### Data Load

> 라이브러리 설치
```python
pip list | grep pathlib

pip install pathlib
```
<br></br>
> 사용할 라이브러리 가져오기
```python
import datetime as dt
from pathlib import Path
import os

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```
<br></br>
> 데이터 경로 설정
```python
data_path = Path(os.getenv('HOME')+'/yoochoose-data') 
train_path = data_path / 'yoochoose-clicks.dat'
train_path
```
<br></br>
> 데이터 가져오기
```python
def load_data(data_path: Path, nrows=None):
    data = pd.read_csv(data_path, sep=',', header=None, usecols=[0, 1, 2],
                       parse_dates=[1], dtype={0: np.int32, 2: np.int32}, nrows=nrows)
    data.columns = ['SessionId', 'Time', 'ItemId']
    return data
```
데이터의 `SessionId`, `Time`, `ItemId` 3 개의 컬럼만 가져와 사용한다.
<br></br>
> 데이터를 시간순 정렬
```python
# 시간이 좀 걸릴 수 있습니다. 메모리도 10GB 가까이 소요될 수 있으니 메모리 상태에 주의해 주세요.  

data = load_data(train_path, None)
data.sort_values(['SessionId', 'Time'], inplace=True)  # data를 id와 시간 순서로 정렬해줍니다.
data
```
<br></br>
> 유저수 (세션수) 와 아이템 수 확인
```python
data['SessionId'].nunique(), data['ItemId'].nunique()
```
900 만개의 세션, 5 만개의 아이템을 가지는 것을 알 수 있다.

한 유저가 여러 개의 세션을 만들 수 있기 때문에 세션이 900 만개라고해서 유저 수가 900 만 명이라는 의미는 아니다.
<br></br>

### Session Length

> 각 세션이 몇 개의 클릭 데이터를 갖는지 확인
```python
session_length = data.groupby('SessionId').size()
session_length
```
`session_length` 란 같은 `SessionId` 를 공유하는 데이터 row 의 개수를 의미하며, `SessionId` 란 브라우저로 접속할 때 항상 포함하게되는 유저의 구분자이다.

유저의 세부 정보는 알 수 없지만 `SessionId` 를 기준으로 특정 행동들을 분류해 낼 수 있으며, 따라서 `session_length` 란 해당 세션의 사용자가 한 세션 동안 몇 번의 액션을 취했는지를 나타낸다.
<br></br>
> session_length 의 중앙값 및 평균 산출
```python
session_length.median(), session_length.mean()
```
<br></br>
> session_length 의 최대값 및 최소값 산출
```python
session_length.min(), session_length.max()
```
<br></br>
> 99% 에 해당하는 session_length 확인
```python
session_length.quantile(0.999)
```
평균, 중앙값, 최대 / 최소값을 확인해 봤을 때, 각 세션의 길이는 보통 2 ~ 3 이며, 전체 데이터의 99.9 % 의 세션 길이는 41 이하이다.

하지만 최대값인 200 세션은 이상치로써 뭔가 이상하다는 것을 느낄 수 있다.
<br></br>
> session_length 가 200 인 항목을 확인
```python
long_session = session_length[session_length==200].index[0]
data[data['SessionId']==long_session]
```
매우 짧은 간격으로 지속적으로 클릭을 1 시간 30 분 가량 지속하고 있음을 확인할 수 있다.

해당 정보는 여러 경우가 있을 것이다. 나쁘게 생각하면 상품 조회수 및 평점 조작 등이 있을 것이고, 다르게 생각하면 음악 서비스의 랜덤 재생을 계속 이용하는 등이 있다.

이러한 이상치 데이터는 제거하거나 포함시키는 것은 사용자의 주관이나 이를 제거 혹은 변형하여 정제해주지 않는다면 예측에 혼란 및 오류를 끼칠 수 있다.
<br></br>
> session_length 기준 하위 99.9% 의 분포 누적합을 시각화
```python
length_count = session_length.groupby(session_length).size()
length_percent_cumsum = length_count.cumsum() / length_count.sum()
length_percent_cumsum_999 = length_percent_cumsum[length_percent_cumsum < 0.999]

length_percent_cumsum_999

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.bar(x=length_percent_cumsum_999.index,
        height=length_percent_cumsum_999, color='red')
plt.xticks(length_percent_cumsum_999.index)
plt.yticks(np.arange(0, 1.01, 0.05))
plt.title('Cumsum Percentage Until 0.999', size=20)
plt.show()
```
<br></br>

일반적으로 추천 시스템은 유저 - 상품 관계 매트릭스를 유저 행렬과 상품 행렬의 곱으로 표현하는 Matrix Factorization 모델을 사용한다.

하지만 위 데이터의 경우 Matrix Factorization 모델을 사용하면 제대로 학습되지 않을 가능성이 크다.

위 데이터는 유저 및 상품에 관한 정보가 누락되어 있는 상태이므로, User * Item Matrix 의 빈칸이 많다 (Data Sparsity). 

이러한 세션정보는 유저 ID 기반으로 정리할 수 없으며, 때문에 세션 하나를 유저 하나로 본다면 기존의 유저 - 상품 정보 행렬 보다 훨씬 빈칸이 많은 (Sparse 한) 형태가 된다.
<br></br>

### Session Time

추천 시스템에서 시간 관련 데이터는 데이터의 생성 날짜 외에 접속 요일, 시간대, 머문 시간, 마지막 접속 시간 등의 다양한 요소가 있다.
<br></br>
> 데이터의 시간 관련 정보 확인
```python
oldest, latest = data['Time'].min(), data['Time'].max()
print(oldest) 
print(latest)
```
6 개 월치 데이터가 있음을 확인할 수 있다.
<br></br>
> 데이터 타입 확인
```python
type(latest)
```
데이터는 Timestamp 객체인 것을 확인할 수 있다.
<br></br>
> 1 달치 데이터만 추출
```python
month_ago = latest - dt.timedelta(30)     # 최종 날짜로부터 30일 이전 날짜를 구한다.  
data = data[data['Time'] > month_ago]   # 방금 구한 날짜 이후의 데이터만 모은다. 
data
```
 Timestamp 객체 int 객체와 사칙연산이 불가능하다.

따라서 날짜끼리의 차이를 구하려면 datetime 라이브러리의  timedelta 객체를 활용해야한다.

1 개월치 데이터만 사용하기 위해 최종 날짜로부터 30 일 이전 날짜를 구하고, 해당 날짜 이후의 데이터만 모았다.
<br></br>

### Data Cleansing

> session_length 이 1이거나 너무 적은 세션 제거하는 함수 생성
```python
# short_session을 제거한 다음 unpopular item을 제거하면 다시 길이가 1인 session이 생길 수 있습니다.
# 이를 위해 반복문을 통해 지속적으로 제거 합니다.
def cleanse_recursive(data: pd.DataFrame, shortest, least_click) -> pd.DataFrame:
    while True:
        before_len = len(data)
        data = cleanse_short_session(data, shortest)
        data = cleanse_unpopular_item(data, least_click)
        after_len = len(data)
        if before_len == after_len:
            break
    return data


def cleanse_short_session(data: pd.DataFrame, shortest):
    session_len = data.groupby('SessionId').size()
    session_use = session_len[session_len >= shortest].index
    data = data[data['SessionId'].isin(session_use)]
    return data


def cleanse_unpopular_item(data: pd.DataFrame, least_click):
    item_popular = data.groupby('ItemId').size()
    item_use = item_popular[item_popular >= least_click].index
    data = data[data['ItemId'].isin(item_use)]
    return data
```
우리의 목적은 유저가 최소 1 개 이상 클릭했을 때 다음 클릭을 예측하는 것이므로 세션 길이가 1 인 세션은 제거해주며, 세션의 길이가 너무 적은 아이템들은 이상한 아이템이거나 잘 못 누른 아이템일 확률이 높기 때문에 함께 제거해준다.
<br></br>
> session_length 의 길이가 1 이거나 너무 적은 데이터 제거
```python
data = cleanse_recursive(data, shortest=2, least_click=5)
data
```
<br></br>

### Train / Valid / Test Split

> Test set 데이터 확인하기
```python
test_path = data_path / 'yoochoose-test.dat'
test= load_data(test_path)
test['Time'].min(), test['Time'].max()
```
Test 셋을 살펴보니 Training 셋과 기간이 겹치는 것을 알 수 있다.
<br></br>
> 기준에 따라 test / validation 데이터를 나눌 함수 생성 
```python
def split_by_date(data: pd.DataFrame, n_days: int):
    final_time = data['Time'].max()
    session_last_time = data.groupby('SessionId')['Time'].max()
    session_in_train = session_last_time[session_last_time < final_time - dt.timedelta(n_days)].index
    session_in_test = session_last_time[session_last_time >= final_time - dt.timedelta(n_days)].index

    before_date = data[data['SessionId'].isin(session_in_train)]
    after_date = data[data['SessionId'].isin(session_in_test)]
    after_date = after_date[after_date['ItemId'].isin(before_date['ItemId'])]
    return before_date, after_date
```
추천 시스템은 지금을 잘 예측하는게 중요하므로 사용하고자 하는 1 개월치 데이터 중 마지막 1일(30일) 은 Test 로, 2 일 전부터 1 일전 까지 (29일) 를 Valid set 으로 나눠주었다. (Train set : 1 ~ 28일, 총 30일 / 1개월)
<br></br>
> 데이터 분리
```python
tr, test = split_by_date(data, n_days=1)
tr, val = split_by_date(tr, n_days=1)
```
<br></br>
> data 에 해당하는 정보를 확인하는 함수 생성
```python
# data에 대한 정보를 살펴봅니다.
def stats_info(data: pd.DataFrame, status: str):
    print(f'* {status} Set Stats Info\n'
          f'\t Events: {len(data)}\n'
          f'\t Sessions: {data["SessionId"].nunique()}\n'
          f'\t Items: {data["ItemId"].nunique()}\n'
          f'\t First Time : {data["Time"].min()}\n'
          f'\t Last Time : {data["Time"].max()}\n')
```
<br></br>
> train / test / val 데이터의 정보 확인
```python
stats_info(tr, 'train')
stats_info(val, 'valid')
stats_info(test, 'test')
```
<br></br>
> train  set 을 기준으로 test, val set 의 아이템 정리
> (train set 에는 없고 test, val set 에는 있는 아이템 처리 )
```python
# train set에 없는 아이템이 val, test기간에 생길 수 있으므로 train data를 기준으로 인덱싱합니다.
id2idx = {item_id : index for index, item_id in enumerate(tr['ItemId'].unique())}

def indexing(df, id2idx):
    df['item_idx'] = df['ItemId'].map(lambda x: id2idx.get(x, -1))  # id2idx에 없는 아이템은 모르는 값(-1) 처리 해줍니다.
    return df

tr = indexing(tr, id2idx)
val = indexing(val, id2idx)
test = indexing(test, id2idx)
```
<br></br>
> 데이터 저장
```python
save_path = data_path / 'processed'
save_path.mkdir(parents=True, exist_ok=True)

tr.to_pickle(save_path / 'train.pkl')
val.to_pickle(save_path / 'valid.pkl')
test.to_pickle(save_path / 'test.pkl')
```
<br></br>

## 논문소개 (GRU4REC)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/model.max-800x600.png)
<br></br>
GRU4REC 모델은 Session Data 에서 처음으로 RNN 계열 모델을 적용한 것으로 위 그림과 같은 구조를 가지고 있다.

여러 RNN 계열의 모델 중 GRU 를 활용하며 Embedding Layer 를 사용하지 않는다는 특징을 가지고 있다.

<br></br>
### Session - Parallel Mini - Batches 

![](https://aiffelstaticprd.blob.core.windows.net/media/images/input1.max-800x600.png)
<br></br>
데이터의 세션의 길이는 매우 짧은 것들이 대부분을 이루고 있는 반면에 매우 긴 것들도 있었다.

이러한 세션들을 데이터 샘플 하나로 봤을 때, mini - batch 를 구성하여 input 으로 넣는다면 길이가 긴 세션의 연산이 끝날 때 까지 짧은 세션들이 기다려야한다.

위 그림에서 Session 1, 2, 3 을 하나의 mini - batch 로 만든다면 길이가 제일 긴 Session 3 의 연산이 끝나야 미니배치의 연산이 끝나는 것이다.

이러한 시간적 비효율성을 개선하기 위해 논문에서는 Session - Parallel Mini - Batches 를 제안한다.

이를 병렬적으로 계산하도록 처리함으로써 불필요하게 기다리는 시간을 없앤 것이다.

위 그림에서 session 2 가 끝나면 session 4 가 시작하는 방식이다.

이렇게 구성한 Mini - Batch 의 shape 는 (3, 1, 1) 이 되고, RNN cell 의 state 가 1 개로만 이루어진다.

또한 텐서플로우를 기준으로 RNN 을 만들 때 stateful = True 옵션을 사용하고 2 처럼 세션이 끝나면 state 를 0으로 만들어 준다.

+ 참고 : [RNN API 보기](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/input2.max-800x600.png)
<br></br>

위 그림은 Session-Parallel Mini-Batches 에 관한 구조를 나타낸 것이다.

SAMPLING ON THE OUTPUT : Item 수가 많기 때문에 Loss 를 계산할 때 모든 아이템을 비교하지 않고 인기도를 고려하여 Sampling 하는 방법으로 Negative Sampling 과 같은 개념이다.

이번 실습에서는 구현하지 않는다.
<br></br>
Ranking Loss : Session-Based Recommendation Task를 여러 아이템 중 다음 아이템이 무엇인지 Classification 하는 Task 로 생각할 수도 있지만 여러 아이템을 관련도 순으로 랭킹을 매겨서 높은 랭킹의 아이템을 추천하는 Task 로도 생각할 수 있다.

추천 시스템 연구 분야에서는 이렇게 Ranking을 맞추는 objective function에 대한 연구가 있었고 논문의 저자 역시 이런 Loss를 사용했다.

하지만 이번에는 Classification Task 로 보고 Cross-Entropy Loss 를 사용한다.

+ 참고 : [논문 : SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1511.06939v4.pdf)
<br></br>

## Data Pipeline

Session - Parallel Mini - Batch 를 구현해본다.
<br></br>
### SessionDataser

> 데이터가 주어지면 세션이 시작되는 인덱스를 담는 값과 세션을 새로 인덱싱한 값을 갖는 클래스 생성
```python
class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data):
        self.df = data
        self.click_offsets = self.get_click_offsets()
        self.session_idx = np.arange(self.df['SessionId'].nunique())  # indexing to SessionId

    def get_click_offsets(self):
        """
        Return the indexes of the first click of each session IDs,
        """
        offsets = np.zeros(self.df['SessionId'].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby('SessionId').size().cumsum()
        return offsets
```
<br></br>
> train데이터로 `SessionDataset` 객체를 생성
```python
tr_dataset = SessionDataset(tr)
tr_dataset.df.head(10)
```
<br></br>
> 인스턴스 변수인 `click_offsets`  변수 확인
```python
tr_dataset.click_offsets
```
`click_offsets` 변수는 각 세션이 시작된 인덱스를 담고 있다.
<br></br>
> 인스턴스 변수인 `session_idx` 변수 확인
```python
tr_dataset.session_idx
```
`session_idx` 변수는 각 세션을 인덱싱한 `np.array` 이다.
<br></br>

### Session Data Loader

> `SessionDataset` 객체를 받아서 Session-Parallel mini-batch를 만드는 클래스 생성
```python
class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, dataset: SessionDataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        start, end, mask, last_session, finished = self.initialize()  # initialize 메소드에서 확인해주세요.
        """
        start : Index Where Session Start
        end : Index Where Session End
        mask : indicator for the sessions to be terminated
        """

        while not finished:
            min_len = (end - start).min() - 1  # Shortest Length Among Sessions
            for i in range(min_len):
                # Build inputs & targets
                inp = self.dataset.df['item_idx'].values[start + i]
                target = self.dataset.df['item_idx'].values[start + i + 1]
                yield inp, target, mask

            start, end, mask, last_session, finished = self.update_status(start, end, min_len, last_session, finished)

    def initialize(self):
        first_iters = np.arange(self.batch_size)    # 첫 배치에 사용할 세션 Index를 가져옵니다.
        last_session = self.batch_size - 1    # 마지막으로 다루고 있는 세션 Index를 저장해둡니다.
        start = self.dataset.click_offsets[self.dataset.session_idx[first_iters]]       # data 상에서 session이 시작된 위치를 가져옵니다.
        end = self.dataset.click_offsets[self.dataset.session_idx[first_iters] + 1]  # session이 끝난 위치 바로 다음 위치를 가져옵니다.
        mask = np.array([])   # session의 모든 아이템을 다 돌은 경우 mask에 추가해줄 것입니다.
        finished = False         # data를 전부 돌았는지 기록하기 위한 변수입니다.
        return start, end, mask, last_session, finished

    def update_status(self, start: np.ndarray, end: np.ndarray, min_len: int, last_session: int, finished: bool):  
        # 다음 배치 데이터를 생성하기 위해 상태를 update합니다.
        
        start += min_len   # __iter__에서 min_len 만큼 for문을 돌았으므로 start를 min_len 만큼 더해줍니다.
        mask = np.arange(self.batch_size)[(end - start) == 1]  
        # end는 다음 세션이 시작되는 위치인데 start와 한 칸 차이난다는 것은 session이 끝났다는 뜻입니다. mask에 기록해줍니다.

        for i, idx in enumerate(mask, start=1):  # mask에 추가된 세션 개수만큼 새로운 세션을 돌것입니다.
            new_session = last_session + i  
            if new_session > self.dataset.session_idx[-1]:  # 만약 새로운 세션이 마지막 세션 index보다 크다면 모든 학습데이터를 돈 것입니다.
                finished = True
                break
            # update the next starting/ending point
            start[idx] = self.dataset.click_offsets[self.dataset.session_idx[new_session]]     # 종료된 세션 대신 새로운 세션의 시작점을 기록합니다.
            end[idx] = self.dataset.click_offsets[self.dataset.session_idx[new_session] + 1]

        last_session += len(mask)  # 마지막 세션의 위치를 기록해둡니다.
        return start, end, mask, last_session, finished
```
`__iter__` 메소드는 모델 인풋, 라벨, 세션이 끝나는 곳의 위치를 `yield` 하며, mask는 후에 RNN Cell State를 초기화 하는데 사용한다.

+ 참고 : [파이썬 문법 : iterator, generator](https://dojang.io/mod/page/view.php?id=2405)
<br></br>
> data loadr  정의
```python
tr_data_loader = SessionDataLoader(tr_dataset, batch_size=4)
tr_dataset.df.head(15)
```
<br></br>
> 
```python
iter_ex = iter(tr_data_loader)
```
<br></br>
> `next` 를 통해 다음 데이터 생성 및 input, output, mask 확인
```python
inputs, labels, mask =  next(iter_ex)

print(f'Model Input Item Idx are : {inputs}')

print(f'Label Item Idx are : {"":5}  {labels}')

print(f'Previous Masked Input Idx are {mask}')
```
<br></br>

## Modeling

### Evaluation Metric

모델 성능 평가를 위한 대표적인 지표로는 precision, recall 이 있다.

Session-Based Recommendation Task 에서는 모델이 k 개의 아이템을 제시했을 때, 유저가 클릭 / 구매한 n 개의 아이템이 많아야 좋다.

때문에 recall 의 개념을 확장한 `recall@k` 지표, precision 의 개념을 확장한 `Mean Average Precision@k` 지표 등을 사용한다.

추천 모델에서는 몇 번째로 맞추는지도 중요한다. 따라서 순서에 민감한 지표인 `MRR`, `NDCG` 같은 지표도 사용한다.

이번 실습에서는 `MRR`과 `Recall@k` 를 사용하며 `MRR`은 정답 아이템이 나온 순번의 역수 값이다.

따라서 정답 아이템이 추천 결과의 앞부분에 있다면 지표는 높아지고, 뒤쪽에 있거나 없다면 지표는 낮아진다.

+ 참고 : [`NDCG`, `MRR`, `MAP`](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)
<br></br>
> 모델 성능을 평가하는 지표 `MRR`과 `Recall@k` 구현
```python
def mrr_k(pred, truth: int, k: int):
    indexing = np.where(pred[:k] == truth)[0]
    if len(indexing) > 0:
        return 1 / (indexing[0] + 1)
    else:
        return 0


def recall_k(pred, truth: int, k: int) -> int:
    answer = truth in pred[:k]
    return int(answer)
```
<br></br>

### Model Architecture

> 라이브러리 설치
```python
pip install tqdm
```
<br></br>
> 사용할 라이브러리 임포트
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
```
<br></br>
> GRU4REC 모델 구현
```python
def create_model(args):
    inputs = Input(batch_shape=(args.batch_size, 1, args.num_items))
    gru, _ = GRU(args.hsz, stateful=True, return_state=True, name='GRU')(inputs)
    dropout = Dropout(args.drop_rate)(gru)
    predictions = Dense(args.num_items, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(loss=categorical_crossentropy, optimizer=Adam(args.lr), metrics=['accuracy'])
    model.summary()
    return model
```
<br></br>
> 모델에서 사용할 하이퍼파라미터 설정 
> 클래스로 구현하여 관리
```python
class Args:
    def __init__(self, tr, val, test, batch_size, hsz, drop_rate, lr, epochs, k):
        self.tr = tr
        self.val = val
        self.test = test
        self.num_items = tr['ItemId'].nunique()
        self.num_sessions = tr['SessionId'].nunique()
        self.batch_size = batch_size
        self.hsz = hsz
        self.drop_rate = drop_rate
        self.lr = lr
        self.epochs = epochs
        self.k = k

args = Args(tr, val, test, batch_size=2048, hsz=50, drop_rate=0.1, lr=0.001, epochs=20, k=20)
```
<br></br>
> 모델 정의
```python
model = create_model(args)
```
<br></br>

### Model Training

> 모델 학습
```python
# train 셋으로 학습하면서 valid 셋으로 검증합니다.
def train_model(model, args):
    train_dataset = SessionDataset(args.tr)
    train_loader = SessionDataLoader(train_dataset, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        total_step = len(args.tr) - args.tr['SessionId'].nunique()
        tr_loader = tqdm(train_loader, total=total_step // args.batch_size, desc='Train', mininterval=1)
        for feat, target, mask in tr_loader:
            reset_hidden_states(model, mask)  # 종료된 session은 hidden_state를 초기화합니다. 아래 메서드에서 확인해주세요.

            input_ohe = to_categorical(feat, num_classes=args.num_items)
            input_ohe = np.expand_dims(input_ohe, axis=1)
            target_ohe = to_categorical(target, num_classes=args.num_items)

            result = model.train_on_batch(input_ohe, target_ohe)
            tr_loader.set_postfix(train_loss=result[0], accuracy = result[1])

        val_recall, val_mrr = get_metrics(args.val, model, args, args.k)  # valid set에 대해 검증합니다.

        print(f"\t - Recall@{args.k} epoch {epoch}: {val_recall:3f}")
        print(f"\t - MRR@{args.k}    epoch {epoch}: {val_mrr:3f}\n")


def reset_hidden_states(model, mask):
    gru_layer = model.get_layer(name='GRU')  # model에서 gru layer를 가져옵니다.
    hidden_states = gru_layer.states[0].numpy()  # gru_layer의 parameter를 가져옵니다.
    for elt in mask:  # mask된 인덱스 즉, 종료된 세션의 인덱스를 돌면서
        hidden_states[elt, :] = 0  # parameter를 초기화 합니다.
    gru_layer.reset_states(states=hidden_states)


def get_metrics(data, model, args, k: int):  # valid셋과 test셋을 평가하는 코드입니다. 
                                             # train과 거의 같지만 mrr, recall을 구하는 라인이 있습니다.
    dataset = SessionDataset(data)
    loader = SessionDataLoader(dataset, batch_size=args.batch_size)
    recall_list, mrr_list = [], []

    total_step = len(data) - data['SessionId'].nunique()
    for inputs, label, mask in tqdm(loader, total=total_step // args.batch_size, desc='Evaluation', mininterval=1):
        reset_hidden_states(model, mask)
        input_ohe = to_categorical(inputs, num_classes=args.num_items)
        input_ohe = np.expand_dims(input_ohe, axis=1)

        pred = model.predict(input_ohe, batch_size=args.batch_size)
        pred_arg = tf.argsort(pred, direction='DESCENDING')  # softmax 값이 큰 순서대로 sorting 합니다.

        length = len(inputs)
        recall_list.extend([recall_k(pred_arg[i], label[i], k) for i in range(length)])
        mrr_list.extend([mrr_k(pred_arg[i], label[i], k) for i in range(length)])

    recall, mrr = np.mean(recall_list), np.mean(mrr_list)
    return recall, mrr
    
# 학습 시간이 다소 오래 소요됩니다. (예상시간 1시간)
train_model(model, args)
```
<br></br>

### Inference

> 모델 성능 평가
```python
def test_model(model, args, test):
    test_recall, test_mrr = get_metrics(test, model, args, 20)
    print(f"\t - Recall@{args.k}: {test_recall:3f}")
    print(f"\t - MRR@{args.k}: {test_mrr:3f}\n")

test_model(model, args, test)
```
<br></br>

