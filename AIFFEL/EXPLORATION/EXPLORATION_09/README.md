# 나의 첫 번째 캐글 경진대회, 무작정 따라해보기

## 학습 목표

-   데이터 사이언스 관련 최대 커뮤니티인 캐글의 경진대회에 직접 참여해서 문제를 해결해본다.
-   캐글에서 데이터를 내려받는 것으로부터 시작해서, 로컬 서버에서 자유롭게 다루어보며 문제 해결을 위한 고민을 해본다.
-   앙상블 기법의 개념과 강점을 이해하고, 여러 모델의 예측 결과를 Averaging 한 최종 결과로 캐글에 제출해본다.
-   하이퍼 파라미터 튜닝의 필요성과 의미를 이해하고, Grid Search, Random Search 등의 기법을 알아본다.
-   Grid Search 기법을 활용해서 직접 하이퍼 파라미터 튜닝 실험을 해보고, 모델의 성능을 최대한 끌어올려본다.
<br></br>

## 대회의 시작 (01). 참가 규칙, 평가 기준 살펴보기

캐글 경진대회는 데이터 사이언티스트들을 위한 경진대회로 사이트 캐글 (kaggle) 에서 진행된다.

우선 캐글에 가입을 하고, 아래 대회를 참여해보자.

-   [캐글 코리아와 함께하는 2nd ML 대회 - House Price Prediction](https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr)
<br></br>

### A . Description, 대회 소개

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-2.max-800x600.png)
<br></br>
캐글에 회원 가입 및 로그인을 하고 나면 위 그림과 같은 페이지를 볼 수 있다.

캐글에는 많은 대회가 있으며, 각 대회에 관한 소개, 데이터셋 소개, 규칙 설명 등 세부내용을 제공한다.

우리가 참여하고자 하는 대회는 특별한 링크를 눌러야만 참여가 가능하다.
<br></br>

### B. Evaluation, 점수 평가 기준

대회 페이지에서 `Evaluation` 을 누르면 평가 방식을 알 수 있다.

우리가 참여하고자 하는 대회의 평가 방식은 RMSE 라고 한다.

RMSE 는 Root Mean Squared Error 로 평균 제곱근 편차를 통한 평가 방식이다. 수식으로는 $\sqrt{( 1/N * Σ((Y_t - Y_pr)^2) )}$ 으로 나타내며, 실제 정답과 예측한 값의 차이의 제곱을 평균한 값의 제곱근을 통해 평가한다.

참여하는 대회의 문제는 집값을 예측하는 문제이기 때문에 예측값과 실제 값이 모두 실수로 구성되어 있다. 따라서 두 가지 값의 차이를 사용해 얼마나 떨어져 있는지 계산할 수 있는 RMSE 를 평가 척도로 사용하는 것이다.
<br></br>

### C. Prize, 상품

캐글 대회는 상위 리더보드 100 명, 즉 100 등까지 상품을 준다.

하지만 조건이 특이한데 대회가 마무리된 후 사용한 소스코드를 커널 항목에 공개해야하는 의무가 있다.

이는 지식 공유와 공개를 지향하는 캐글의 정신에서 정해진 규칙이라고 한다.
<br></br>

### D. Timeline, 대회 일정

대회 일정 역시 타임라인에 나와있으므로 잘 숙지하여 마감 전에 여러 실험을 통해 성능을 끌어올리자.
<br></br>

### E. Rules, 대회 규칙

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-5.max-800x600.png)
<br></br>
많은 사람이 참가하고, 또 상품이 걸려있는 만큼 엄격한 규칙들이 있으며, 부정행위를 통해 얻은 점수는 무효가 될 뿐만 아니라 향후 캐글의 다른 대회에 참가하는 것에도 불이익이 있을 수 있다.

이번 대회는 외부 데이터의 사용을 금지하며, 하루 최대 제출 횟수는 5 번으로 정해져있다.
<br></br>

## 대회의 시작 (02). 데이터 살펴보기

### F. Data Description, 데이터 설명

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-6.max-800x600.png)
<br></br>

복잡한 데이터를 다룰 수록 데이터 설명을 꼼꼼히 읽는 것이 중요하다. 이는 데이터를 더욱 잘 이해할 수록 좋은 결과를 낼 수 있도록 귀결되기 때문이다.

데이터는 다양한 컬럼을 가지고 있으며, 우리가 예측해야할 컬럼은 `price` 로 집 값이다.
<br></br>

### G. Data Explorer, 데이터 파일

데이터 분석을 위해서는 데이터셋 자체에 대한 설명 외에 데이터 파일의 형태를 살펴봐야한다.

이 대회에서는 `train.csv`라는 모델 학습용 파일과, `test.csv`라는 테스트용 파일, 그리고 `sample_submission.csv`라는 제출용 파일이 제공된다.

`train.csv`를 활용해서 데이터를 뜯어보고 모델을 학습시킨 후, `test.csv` 파일의 데이터에 대해 `price`를 예측해서 `sample_submission.csv`의 형식에 맞는 형태로 캐글에 제출해 보자.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-7.max-800x600.png)
<br></br>
위 그림에서 `Download` 버튼을 누르면 다운받을 수 있다.

다운 버튼을 클릭하면 `2019-2nd-ml-month-with-kakr.zip` 이라는 이름으로 압축 파일이 다운되며, 압출을 풀면 `2019-2nd-ml-month-with-kakr` 이름의 폴더 안에 `train.csv`, `test.csv`, `submission.csv` 가 들어있다.

이 대회는 중간에 데이커가 한번 변경되었기 때문에 다운받은 데이터로 예측해서 제출하면 데이터 길이가 맞지 않는다는 에러가 나타난다. 따라서 다음 데이터를 다운받아 사용하자.

+ [변경된 데이터](https://aiffelstaticprd.blob.core.windows.net/media/documents/kaggle-kakr-housing-data.zip)
<br></br>

> 변경된 데이터 다운 및 작업디렉토리로 이동
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/kaggle-kakr-housing-data.zip 
$ mv kaggle-kakr-housing-data.zip ~/aiffel/kaggle_kakr_housing 
$ cd ~/aiffel/kaggle_kakr_housing 
$ unzip kaggle-kakr-housing-data.zip
```
<br></br>
## 일단 제출하고 시작해! Baseline 모델 (01). Baseline 셋팅하기

이번 대회에서는 주최자 차원에서 Baseline 을 제공한다.

Baseline이라 함은 기본적으로 문제 해결을 시작할 때 쉽게 사용해볼 수 있는 샘플을 의미한다.

Baseline 커널은 다음 링크에 있다.

+ [Baseline 커널](https://www.kaggle.com/kcs93023/2019-ml-month-2nd-baseline)
<br></br>

### 다른 사람의 커널을 ipynb 파일로 다운받아 로컬에서 사용하기

캐글의 커널 (Kernel) 은 주피터 노트북 형태의 파일이 캐글 서버에서 실행될 때 그 프로그램을 일컫는 개념이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-8.max-800x600.png)
<br></br>
캐글 자체의 서버에서 baseline 노트북 파일을 돌리고 모델 학습을 시킬 수도 있으며, 이를 위해서는 위 그림에서 `Copy and Edit` 버튼을 클릭하면 웹상에서 코드를 돌려볼 수 있는 커널 창이 뜬다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-9.max-800x600.png)
<br></br>
웹상이 아닌 로컬 서버에서 사용하려면 위 그림에서 `File > Download` 를 통해 커널을 ipynb 파일로 다운받고, 해당 파일을 프로젝트 폴더로 옮긴 후 로컬 버서에서 파일을 열면된다.

+ 참고 : [코딩도장 - 주피터 노트북 사용하기](https://dojang.io/mod/page/view.php?id=2457)
<br></br>

### Baseline 커널 파일 실행 준비

다운받은 Baseline 의 모든 코드를 에러없이 돌리기 위해서는 몇 가지 준비가 필요하다.

1. 데이터를 노트북 파일과 같은 폴더에 위치시킨다.

2. 필요 라이브러리 설치를 한다.
	+ 이번 대회는 집 값을 예측하는 회귀 모델을 구현하는데 사용하는 `xgboost`와 `lightgbm` 라이브러리와, 결측 데이터를 확인하는 `missingno` 라이브러리가 필요하다.
<br></br>
> 필요 라이브러리 설치
```python
$ pip install xgboost 
$ pip install lightgbm 
$ pip install missingno
```
3. Jupyter Notebook 파일 실행 후 matplotlib 시각화를 위해 다음 셀 실행하기
<br></br>
> Baseline 커널에는 다양한 시각화 코드를 화면에 나타날 수 있도록 설정
```python
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```
<br></br>

## 일단 제출하고 시작해! Baseline 모델 (02). 라이브러리, 데이터 가져오기

> 필요 라이브러리 가져오기
```python
import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join

import pandas as pd
import numpy as np

import missingno as msno

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
```
<br></br>
> 데이터 경로 지정하기
```python
data_dir = os.getenv('HOME')+'/kaggle_kakr_housing/data'

train_data_path = join(data_dir, 'train.csv')
sub_data_path = join(data_dir, 'test.csv')      # 테스트, 즉 submission 시 사용할 데이터 경로

print(train_data_path)
print(sub_data_path)
```
우리 파일의 경로는 Baseline 커널과 다르다. Baseline 커널은 캐글 서버에서 돌아가도록 설계되었기 때문에 `../input` 이라는 디렉토리에 위치한다.
<br></br>
> Baseline 커널의 기존 코드
```python
train_data_path = join('../input', 'train.csv') 
sub_data_path = join('../input', 'test.csv')
```
<br></br>
이를 프로젝트 디렉토리(`~/aiffel/kaggle_kakr_housing` 등) 내 `data` 폴더에 있는 파일을 사용하기 위해 바꿔주어야 한다.

-   참고:  [파이썬 공식 문서 - os.path.join](https://docs.python.org/3/library/os.path.html#os.path.join)
<br></br>

## 일단 제출하고 시작해! Baseline 모델 (03). 데이터 이해하기

### 데이터 살펴보기

> Baseline 노트북에서 데이터가 가지고 있는 변수 정보
```python
1. ID :  집을  구분하는  번호  
2. date :  집을  구매한  날짜  
3. price :  타겟  변수인  집의  가격  
4. bedrooms :  침실의  수  
5. bathrooms :  침실당  화장실  개수  
6. sqft_living :  주거  공간의  평방  피트  
7. sqft_lot :  부지의  평방  피트  
8. floors :  집의  층수  
9. waterfront :  집의  전방에  강이  흐르는지  유무  (a.k.a.  리버뷰)  
10. view :  집이  얼마나  좋아  보이는지의  정도  
11. condition :  집의  전반적인  상태  
12. grade :  King  County  grading  시스템  기준으로  매긴  집의  등급  
13. sqft_above :  지하실을  제외한  평방  피트  
14. sqft_basement :  지하실의  평방  피트  
15. yr_built :  집을  지은  년도  
16. yr_renovated :  집을  재건축한  년도  
17. zipcode :  우편번호  
18. lat :  위도  
19. long :  경도  
20. sqft_living15 :  2015년  기준  주거  공간의  평방  피트(집을  재건축했다면,  변화가  있을  수  있음)  
21. sqft_lot15 :  2015년  기준  부지의  평방  피트(집을  재건축했다면,  변화가  있을  수  있음)
```
위 데이터 특징을 활용하여 집의 가격을 예측 해야한다.
<br></br>
> 데이터 불러오기
> (데이터를 `data`, `sub`이라는 변수로 불러온다)
```python
data = pd.read_csv(train_data_path)
sub = pd.read_csv(sub_data_path)
print('train data dim : {}'.format(data.shape))
print('sub data dim : {}'.format(sub.shape))
```
학습 데이터는 약 1만 5천 개, 테스트 데이터는 약 6천 개로 이루어져있으며, 테스트 데이터에는 집 값의 정보가 없기 때문에 컬럼이 하나 적다.
<br></br>
> 학습 데이터에서 라벨 제거하기
```python
y = data['price']
del data['price']

print(data.columns)
```
`price` 컬럼은 따로 `y` 라는 변수에 저장한 후 해당 컬럼은 지워준다.

-   참고:  [w3schools - python del keyword](https://www.w3schools.com/python/ref_keyword_del.asp)
-   참고로 데이터 분석 과정에서 칼럼을 없애고 싶다면  [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)도 사용할 수 있다.
<br></br>
> 학습 데이터와 테스트 데이터 합치기
> (`pd.concat` 을 활용)
```python
train_len = len(data)
data = pd.concat((data, sub), axis=0)

print(len(data))
```
모델 학습을 진행할 때에는 다시 분리해서 사용해야 하기 때문에 데이터를 합치기 전 `train_len` 에 `training data` 의 개수를 저장해서 추후에 학습데이터만 불러올 수 있는 인덱스로 사용한다.
<br></br>
> 합친 데이터 확인
```python
data.head()
```
<br></br>
> 빈 데이터와 전체 데이터의 분포를 확인
```python
msno.matrix(data)
```
결측치, 즉 빈 데이터가 있는지는 위에서 설치했던 `missingno` 라이브러리를 사용해서 확인한다.

missingno 라이브러리의 matrix 함수를 사용하면, 데이터의 결측 상태를 시각화를 통해 살펴볼 수 있다.

`data`라는 `DataFrame`을 매트릭스 모양 그대로 시각화한 것이 출력됨을 볼 수 있다.

만약 특정 row, col 에 NaN 이라는 결측치가 있었다면 해당 부분이 하얗게 나오며, 결측치가 없다면 매트릭스 전체가 까맣게 나온다.

+ 참고 : [데이터프레임 고급 인덱싱](https://datascienceschool.net/01%20python/04.03%20%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%94%84%EB%A0%88%EC%9E%84%20%EA%B3%A0%EA%B8%89%20%EC%9D%B8%EB%8D%B1%EC%8B%B1.html?highlight=%EB%8D%B0%EC%9D%B4%ED%84%B0%ED%94%84%EB%A0%88%EC%9E%84%20%EA%B3%A0%EA%B8%89%20%EC%9D%B8%EB%8D%B1%EC%8B%B1)
<br></br>
> 직접 결측치의 개수를 출력
```python
for c in data.columns:
    print('{} : {}'.format(c, len(data.loc[pd.isnull(data[c]), c].values)))
```
<br></br>
> id 변수 정리
```python
sub_id = data['id'][train_len:]
del data['id']

print(data.columns)
```
`id` 컬럼은 집 값을 예측하는데 필요없기 때문에 제거하며, 나중에 예측 결과를 제출할 때를 대비하여 `sub_id` 변수에 `id` 칼럼을 저장해두고 지운다.
<br></br>
> date 변수 정리
```python
data['date'] = data['date'].apply(lambda x : str(x[:6]))

data.head()
```
`date` 컬럼은 `apply`  함수로 필요한 부분만 남기고 잘라준다.

`str(x[:6])` 으로 처리한 것은 `20141013T000000` 형식의 데이터를 연 / 월 데이터만 사용하기 위해 `201410`까지 자르기 위한 것이다.

+ 참고 : [Pandas Lambda, apply를 활용하여 복잡한 로직 적용하기](https://data-newbie.tistory.com/207)
<br></br>
> 각 변수의 분포를 확인할 수 있는 그래프를 통한 시각화
```python
fig, ax = plt.subplots(9, 2, figsize=(12, 50))   # 가로스크롤 때문에 그래프 확인이 불편하다면 figsize의 x값을 조절해 보세요. 

# id 변수(count==0인 경우)는 제외하고 분포를 확인합니다.
count = 1
columns = data.columns
for row in range(9):
    for col in range(2):
        sns.kdeplot(data[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count += 1
        if count == 19 :
            break
```
지나치게 치우친 분포를 가지는 컬럼의 경우 모델이 결과를 예측하는데 악영향을 미치므로 다음는 작업을 수행해야 한다. 

`id` 컬럼을 제외한 19 개 컬럼에 대해 한 번에 모든 그래프를 그려주며, 10 행 2 열의 `subplot`에 그래프를 그리기 위해 2 중 for 문을 사용한다.

그래프의 종류는 `sns.kdeplot` 을 사용하며, `kdeplot` 은 이산 (discrete) 데이터의 경우에도 부드러운 곡선으로 전체 분포를 확인할 수 있도록 하는 시각화 함수이다.

-   참고:  [seaborn.kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)

시각화를 해보면 `bedrooms`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement` 변수가 한쪽으로 치우친 경향을 보임을 알 수 있다.

이렇게 한 쪽으로 치우친 분포의 경우에는 로그 변환 (log-scaling) 을 통해 데이터 분포를 정규분포에 가깝게 만들 수 있다.
<br></br>
> 로그 변환을 통해 데이터 분포를 변환
```python
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

for c in skew_columns:
    data[c] = np.log1p(data[c].values)
```
분포가 치우친 컬럼들을 `skew_columns` 리스트 안에 담고, 모두 `np.log1p()` 를 활용해서 로그 변환을 해준다.

`numpy.log1p()` 함수는 입력 배열의 각 요소에 대해 자연로그 log(1 + x) 을 반환해 주는 함수이다.
<br></br>
> 로그변환한 분포를 확인
```python
fig, ax = plt.subplots(3, 2, figsize=(12, 15))

count = 0
for row in range(3):
    for col in range(2):
        if count == 5:
            break
        sns.kdeplot(data[skew_columns[count]], ax=ax[row][col])
        ax[row][col].set_title(skew_columns[count], fontsize=15)
        count += 1
```
데이터의 분포가 치우침이 줄어듬을 알 수 있다.
<br></br>
> 로그 변환이 분포의 치우침을 줄어들게 하는 이유를 확인해보기 
> (로그함수의 특징 살펴보기)
```python
xx = np.linspace(0, 10, 500)
yy = np.log(xx)

plt.hlines(0, 0, 10)
plt.vlines(0, -5, 5)
plt.plot(xx, yy, c='r')
plt.show()
```
로그 함수의 특징은 다음과 같다.

- $0 < x < 1$ 범위에서 기울이가 매우 가파르다. 즉, $x$ 의 구간은  $(0, 1)$로 매우 짧은 반면,  $y$ 의 구간은  $(−∞,0)$으로 매우 크다.

- 따라서 0 에 가깝게 모여있는 값들이  $x$ 로 입력되면, 그 함수값인  $y$  값들은 매우 큰 범위로 벌어지게 된다. 즉, 로그 함수는 0 에 가까운 값들이 조밀하게 모여있는 입력값을, 넓은 범위로 펼칠 수 있는 특징을 가진다.

-   반면,  $x$ 값이 점점 커짐에 따라 로그 함수의 기울기는 급격히 작아진다. 이는 곧 큰  $x$ 값들에 대해서는  $y$ 값이 크게 차이나지 않게 된다는 뜻이고, 따라서 넓은 범위를 가지는 $x$ 를 비교적 작은 $y$ 값의 구간 내에 모이게 하는 특징을 가진다.

위와 같은 특성 때문에 한 쪽으로 몰려있는 분포에 로그 변환을 취하게 되면 넓게 퍼질 수 있는 것이다.
<br></br>
> `price` 의 분포를 확인
```python
sns.kdeplot(y)
plt.show()
```
대부분의 가격이 0 과 1사이에 몰려있고 소우의 짒 값이 매우 높은 것을 볼 수 있다.
<br></br>
> `price` 를 로그변환
```python
y_log_transformation = np.log1p(y)

sns.kdeplot(y_log_transformation)
plt.show()
```
로그변환을 통해 0 에 가깝게 몰려있는 데이터들은 넓게 퍼지며, 매우 크게 퍼져있는 소수의 데이터들은 작은 y 값으로 모이게 된다.
<br></br>
> 전체 데이터를 다시 나누기
```python
sub = data.iloc[train_len:, :]
x = data.iloc[:train_len, :]

print(x.shape)
print(sub.shape)
```
위에서 저장해두었던 `train_len` 을 인덱스로 활용해서 `:train_len` 까지는 학습 데이터, 즉 `x` 에 저장하고, `train_len:` 부터는 실제로 추론을 해야 하는 테스트 데이터, 즉 `sub` 변수에 저장한다.
<br></br>

## 일단 제출하고 시작해! Baseline 모델 (04). 모델 설계

### 모델링

Baseline 커널에서는 블렌딩 (blending) 이라는 기법을 통해, 여러가지 모델을 함께 사용해서 결과를 섞는 기법을 사용한다.

즉, 하나의 개별 모델을 사용하는 것이 아닌 다양한 모델을 함께 사용하여 종합된 결과를 얻는 것이다.

이러한 기법을 앙상블 (Ensemble) 이라고 하며, 여러개의 학습 알고리즘을 사용하고 그 예측을 결합함으로 보다 정확한 최종 예측을 도출하는 것으로 딥러닝 처럼 하나의 강력한 모델보다 여러개의 일반적인 모델이 낫다는 아이디어에 착안한 기법이다.

앙상블 기법의 가장 기본적인 것은 보팅 (Voting) 과 에버리징 (Averaging) 이다.

둘 모두 서로 다른 알고리즘을 가진 분류기를 결합하는 방식이며, 보팅은 분류에 사용 (카테고리 인 경우) 하며, 에버리징은 회귀에 사용 (수치데이터인 경우) 한다.

+ 참고 : 
	+ [Part 1. Introduction to Ensemble Learning](https://subinium.github.io/introduction-to-ensemble-1/#:~:text=%EC%95%99%EC%83%81%EB%B8%94(Ensemble)%20%ED%95%99%EC%8A%B5%EC%9D%80%20%EC%97%AC%EB%9F%AC,%EB%A5%BC%20%EA%B0%80%EC%A7%80%EA%B3%A0%20%EC%9D%B4%ED%95%B4%ED%95%98%EB%A9%B4%20%EC%A2%8B%EC%8A%B5%EB%8B%88%EB%8B%A4.)
	+ [Kaggle Ensemble Guide](https://gentlej90.tistory.com/73)
<br></br>
> 여러 모델의 결과를 산술평균하여 블렌딩 모델을 생성
> (모델은 부스팅 계열인 `gboost`, `xgboost`, `lightgbm` 세 가지를 사용)
```python
gboost = GradientBoostingRegressor(random_state=2019)
xgboost = xgb.XGBRegressor(random_state=2019)
lightgbm = lgb.LGBMRegressor(random_state=2019)

models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},
          {'model':lightgbm, 'name':'LightGBM'}]
```
<br></br>
> 교차검증 (Cross Validation) 을 통해 모델의 성능 평가
```python
def get_cv_score(models):
    kfold = KFold(n_splits=5, random_state=2019).get_n_splits(x.values)
    for m in models:
        print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'], x.values, y)), kf=kfold))

get_cv_score(models)
```
<br></br>
> 모델의 학습 정도를 파악하기 위한 측도 생성
```python
def AveragingBlending(models, x, y, sub_x):
    for m in models : 
        m['model'].fit(x.values, y)
    
    predictions = np.column_stack([
        m['model'].predict(sub_x.values) for m in models
    ])
    return np.mean(predictions, axis=1)
```
cross_val_score() 함수는 회귀모델을 전달할 경우 $R^2$ 점수를 반환한다.

$R^2$ 값은 1 에 가까울수록 모델이 잘 학습되었다는 것을 나타낸다.

+ 참고 : [결정계수 R squared](https://newsight.tistory.com/259)
<br></br>
> 집 값을 예측하는 함수 생성
```python
def AveragingBlending(models, x, y, sub_x):
    for m in models : 
        m['model'].fit(x.values, y)
    
    predictions = np.column_stack([
        m['model'].predict(sub_x.values) for m in models
    ])
    return np.mean(predictions, axis=1)
```
Baseline 모델에서는 다음과 같이 여러 모델을 입력하면 각 모델에 대한 예측 결과를 평균 내어 주는 `AgeragingBlending()` 함수를 만들어 사용한다.

`AgeragingBlending()` 함수는 `models` 딕셔너리 안에 있는 모델을 모두 `x`와 `y`로 학습시킨 뒤 `predictions`에 그 예측 결괏값을 모아서 평균한 값을 반환한다.
<br></br>
> 집 값을 예측
```python
y_pred = AveragingBlending(models, x, y, sub)
print(len(y_pred))
y_pred
```
<br></br>
> 제출을 위한 submission 파일을 만들기 전에 `sample_submission.csv` 파일 확인
```python
data_dir = os.getenv('HOME')+'/kaggle_kakr_housing/data'

submission_path = join(data_dir, 'sample_submission.csv')
submission = pd.read_csv(submission_path)
submission.head()
```
`id`와 `price`의 두 가지 열로 구성되어 있음을 확인할 수 있다. 따라서 submission 파일은 `id` 와 `price` 컬럼으로 구성된 데이터프레임 형식으로 만들어야 한다.
<br></br>
> `id`, `price` 컬럼으로 구성된 데이터프레임 생성
```python
result = pd.DataFrame({
    'id' : sub_id, 
    'price' : y_pred
})

result.head()
```
<br></br>
> `submission.csv` 파일로 예측 결과를 저장
```python
my_submission_path = join(data_dir, 'submission.csv')
result.to_csv(my_submission_path, index=False)

print(my_submission_path)
```
<br></br>

## 일단 제출하고 시작해! Baseline 모델 (05). 캐글에 첫 결과 제출하기

해당 캐글 대회는 이미 끝난 대회이기 때문에 `Late Submission`만 가능한 상태이다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-10-10.max-800x600.png)
<br></br>
위 그림은 `Late Submission` 버튼을 클릭하면 나오는 화면이다. 

위의 화면에서 `Step 1` 에 점선으로 보이는 제출 박스를 클릭하거나, `submission.csv` 파일을 해당 박스 안에 드래그앤드롭 하면 파일이 업로드된다.

제출하면 오른쪽에 `129560` 라는 점수도 확인할 수 있다.

아래의 `Jump to your position on the leaderboard`를 클릭하면 내 등수로 이동되며, 결과에 따른 등수를 확인할 수 있다.
<br></br>

## 랭킹을 올리고 싶다면? (01). 다시 한 번, 내 입맛대로 데이터 준비하기

### 최적의 모델을 찾아서, 하이퍼 파리미터 튜닝

이제 제출한 모델을 재가공하여 모델의 성능을 올려볼 수 있다.

가장 대포적인 방법으로는 하이퍼 파라미터를 튜닝해 보는 것이다.

파라미터와 하이퍼 파라미터는 다음과 같은 차이가 있다.

+ 파라미터 : 매개변수, 모델 내부에서 결정되는 변수로 데이터로부터 결정된다. 

	예를 들면 어떤 데이터의 평균과 표준편차에 해당하며 사용자에 의해 조정되지 않는다.

+ 하이퍼 파라미터 : 모델링 시 사용자가 직접 세팅해주는 값을 의미한다. 

	예를 들면 앞 선 예제들에서 에포크 값이 있다.
<br></br>

### 다시 한번, 내 입맛대로 데이터 준비하기

> 데이터 가져오기
```python
data_dir = os.getenv('HOME')+'/kaggle_kakr_housing/data'

train_data_path = join(data_dir, 'train.csv')
test_data_path = join(data_dir, 'test.csv') 

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
```
<br></br>

> 데이터 살펴보기
```python
train.head()
```
<br></br>
> Baseline 과 달리 `date` 컬럼을 `int`, 즉 정수형 데이터로 변환
```python
train['date'] = train['date'].apply(lambda i: i[:6]).astype(int)
train.head()
```
<br></br>
> `y` 변수에 `price` 를 넣어두고, `train` 에서 삭제
```python
y = train['price']
del train['price']

print(train.columns)
```
<br></br>
> `id` 컬럼 삭제
```python
del train['id']

print(train.columns)
```
<br></br>
> `test` 데이터에 대해 위와 같은 작업 (`date`, `id` 컬럼 전처리) 을 실행
> (훈련 데이터셋과는 달리 `price` 컬럼이 없으므로 해당 컬럼에 대한 처리는 하지 않아도 된다.)
```python
test['date'] = test['date'].apply(lambda i: i[:6]).astype(int)

del test['id']

print(test.columns)
```
<br></br>
> 타겟 데이터인 `y` 확인
```python
y
```
<br></br>
> 타겟 데이터 `y` 의 분포 확인
```python
"""
seaborn의 `kdeplot`을 활용해 `y`의 분포를 확인해주세요!
"""

#코드 작성

sns.kdeplot(y)
plt.show()
```
`price`는 왼쪽으로 크게 치우쳐 있는 형태를 보인다.
<br></br>
> 로그변환을 통해 `y` 를 변환
```python
y = np.log1p(y)
y
```
<br></br>
> 로그변환을 수행한 `y` 의 분포 확인
```python
sns.kdeplot(y)
plt.show()
```
<br></br>
> 전체 데이터의 자료형을 확인
> (`info()` 함수 활용)
```python
train.info()
```
모두 실수 및 정수 자료형이며, 모델 학습에 문제 없이 활용이 가능하다.
<br></br>

## 랭킹을 올리고 싶다면? (02). 다양한 실험을 위해 함수로 만들어 쓰자.

모델 성능 향상을 위해서는 다양한 실험을 해야하며, 따라서 실험을 위한 도구를 잘 준비해야한다.

이를 위해 반복적으로 사용되는 작업을 함수로 구현해두고 실험을 하는 것이 좋다.
<br></br>
> RMSE 계산을 위한 라이브러리 가져오기
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```
데이터셋을 훈련 데이터셋과 검증 데이터셋으로 나누기 위한 `train_test_split` 함수와, RMSE 점수를 계산하기 위한 `mean_squared_error` 를 가져온다.
<br></br>
> 대회 평가 척도인 RMSE 계산을 위한 함수 구현
```python
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
```
`y_test` 나 `y_pred` 는 위에서 `np.log1p()`로 변환이 된 값이기 때문에 원래 데이터의 단위에 맞게 되돌리기 위해 `np.expm1()` 를 추가해 줘야한다.

`exp` 로 다시 변환해서 `mean_squared_error` 를 계산한 값에 `np.sqrt` 를 취하여 RMSE 값을 산출한다.
<br></br>
> `XGBRegressor`, `LGBMRegressor`, `GradientBoostingRegressor`, `RandomForestRegressor` 네 가지 모델 가져오기
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
```
<br></br>
> 모델 인스턴스를 생성한 후 `models` 라는 리스트에 넣어준다.
```python
# random_state는 모델초기화나 데이터셋 구성에 사용되는 랜덤 시드값입니다. 
#random_state=None    # 이게 초기값입니다. 아무것도 지정하지 않고 None을 넘겨주면 모델 내부에서 임의로 선택합니다.  
random_state=2020        # 하지만 우리는 이렇게 고정값을 세팅해 두겠습니다. 

gboost = GradientBoostingRegressor(random_state=random_state)
xgboost = XGBRegressor(random_state=random_state)
lightgbm = LGBMRegressor(random_state=random_state)
rdforest = RandomForestRegressor(random_state=random_state)

models = [gboost, xgboost, lightgbm, rdforest]
```
모델 파라미터 초기화나 데이터셋 구성에 사용되는 랜덤 시드값인 `random_state` 값을 특정 값으로 고정시키거나, 아니면 지정하지 않고 None 으로 세팅할 수 있다.

`random_state`를 고정값으로 주면 모델과 데이터셋이 동일한 경우 머신러닝 학습결과도 항상 동일하게 재현된다.

하지만 이 값을 지정하지 않고 None 으로 남겨 두면 모델 내부에서 랜덤 시드값을 임의로 선택하기 때문에, 결과적으로 파라미터 초기화나 데이터셋 구성 양상이 달라져서 모델과 데이터셋이 동일하더라도 머신러닝 학습결과는 학습할 때마다 달라진다.

앞으로 베이스라인부터 시작해 다양한 실험을 통해 성능 개선의 여부를 검증해야 하기때문에 어떤 시도가 모델 성능 향상에 긍정적이었는지 여부를 판단하기 위해서는 랜덤적 요소의 변화 때문에 생기는 불확실성을 제거해야 한다.

따라서 아래와 같이 `random_state` 값을 특정 값으로 고정시킨다.
<br></br>
> 각 모델의 이름을 얻기
```python
gboost.__class__.__name__
```
각 모델의 이름은 다음과 같이 클래스의 `__name__` 속성에 접근해서 얻을 수 있다.
<br></br>
> `for` 반복문을 통해 각 모델별 학습 및 예측
```python
df = {}

for model in models:
    # 모델 이름 획득
    model_name = model.__class__.__name__

    # train, test 데이터셋 분리 - 여기에도 random_state를 고정합니다. 
    X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=random_state, test_size=0.2)

    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)

    # 예측 결과의 rmse값 저장
    df[model_name] = rmse(y_test, y_pred)
    
    # data frame에 저장
    score_df = pd.DataFrame(df, index=['RMSE']).T.sort_values('RMSE', ascending=False)
    
df
```
<br></br>
> 위 과정을 함수로 구현
```python
def get_scores(models, train, y):
    df = {}

    for model in models:
        model_name = model.__class__.__name__

        X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=random_state, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        df[model_name] = rmse(y_test, y_pred)
        score_df = pd.DataFrame(df, index=['RMSE']).T.sort_values('RMSE', ascending=False)

    return score_df
      
get_scores(models, train, y)
```
<br></br>

## 랭킹을 올리고 싶다면? (03). 하이퍼 파라미터 튜닝의 최강자, 그리드 탐색

> 실험을 위한 함수 가져오기
> (실험은 `sklearn.model_selection` 라이브러리 안에 있는 `GridSearchCV` 클래스를 활용한다.)
```python
from sklearn.model_selection import GridSearchCV
```

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E10-13.max-800x600.png)
<br></br>
하이퍼 파라미터를 최적화 하는 방법으로는 랜덤 탐색 (Random Search) 과 그리드 탐색 (Grid Search) 두 가지가 있다.

랜덤 탐색은 정해진 범위에서 난수를 생성하여 최적의 하이퍼 파라미터를 찾는 방법으로 중요한 하이퍼 파라미터를 더 많이 탐색한다. 언제든지 탐색을 중단할 수 있으며, 중간에 멈추더라도 편증된 탐색이 아니기 때문에 문제가 없다는 장점이 있다.

하지만 중요하지 않은 하이퍼 파라미터를 너무 많이 탐색한다고 볼 수도 있다.

그리드 탐색은 하이퍼 파라미터의 범위와 간격을 미리 정해 각 경우의 수를 모두 대입하여 최적의 경우의 수를 찾는 방법이며, 필요한 부분만 탐색하는 장점이 있다.

반면에 중간에 탐색을 중단하면 하이퍼 파라미터의 일부 범위를 제외하고 탐색한다는 단점이 있다.

+ 참고 : [Random Search vs Grid Search](https://shwksl101.github.io/ml/dl/2019/01/30/Hyper_parameter_optimization.html)
<br></br>
우리가 사용한 `GridSearchCV`에 입력되는 인자들은 다음과 같다.

-   `param_grid`  : 탐색할 파라미터의 종류 (딕셔너리로 입력)

-   `scoring`  : 모델의 성능을 평가할 지표

-   `cv`  : cross validation을 수행하기 위해 train 데이터셋을 나누는 조각의 개수

-   `verbose`  : 그리드 탐색을 진행하면서 진행 과정을 출력해서 보여줄 메세지의 양 (숫자가 클수록 더 많은 메세지를 출력한다.)

-   `n_jobs`  : 그리드 탐색을 진행하면서 사용할 CPU의 개수
<br></br>
> `param_grid` 에 탐색할 xgboost 관련 하이퍼 파라미터 준비
```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}
```
<br></br>
> 모델 준비
> (LightGBM(lgbm)를 사용)
```python
model = LGBMRegressor(random_state=random_state)
```
`model`, `param_grid` 와 함께 다른 여러 가지 인자를 넣어서 `GridSearchCV` 를 수행할 수 있다.
<br></br>
> `GridSearchCV`를 이용해서 `grid_model` 모델을 초기화하고, `train`과 `y` 데이터로 모델 학습
```python
grid_model = GridSearchCV(model, param_grid=param_grid, \
                        scoring='neg_mean_squared_error', \
                        cv=5, verbose=1, n_jobs=5)

grid_model.fit(train, y)
```
`param_grid` 내의 모든 하이퍼 파라미터의 조합에 대해 실험이 완료한다.
<br></br>
> 저장된 실험결과 확인
```python
grid_model.cv_results_
```
실험에 대한 결과는 `grid_model.cv_results_` 안에 저장된다.
<br></br>
많은 정보 중 원하는 값만 정제하여 확인하자.

관심있는 정보는 어떤 파라미터 조합일 때 점수가 어떻게 나오게 되는지에 관한 것이며, 파라미터 조합은 위 딕셔너리 중 `params` 에, 각각에 대한 테스트 점수는 `mean_test_score` 에 저장되어 있다.
<br></br>
> `params` 딕셔너리를 확인
```python
params = grid_model.cv_results_['params']
params
```
`params` 에는 각 파라미터의 조합이 들어있다.
<br></br>
> `mean_test_score` 확인
```python
score = grid_model.cv_results_['mean_test_score']
score
```
`score` 에는 각 조합에 대한 점수가 들어있다.
<br></br>

`params` 와 `score` 을 가지고 데이터프레임을 만들어 최적의 성능을 내는 하이퍼 파라미터 조합을 찾아보자.
<br></br>
> 데이터 프레임으로 결과를 출력
```python
# 여기에 코드를 작성하세요.

results = pd.DataFrame(params)
results['score'] = score

results
```
점수가 음수로 아노는 이유는 `GridSearchCV` 로 `grid_model` 모델을 초기화할 때, `scoring` 인자로 `neg_mean_squared_error` 를 넣었기 때문이다.

이는 MSE 에 음수를 취한 값이기 때문이다.

`GridSearchCV` 를 사용할 때에는 이 외에도 3 가지 점수 체계 (scoring) 를 사용할 수 있는데 Classification, Clustering, Regression 가 있다. 문제 유형에 따라 알맞는 점수 체계를 사용하면 된다.

+ 참고 : [사이킷런 - The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
<br></br>
> RMSE 변환 함수를 통해 원 점수로 변환
```python
results['RMSE'] = np.sqrt(-1 * results['score'])
results
```
음수로 된 MSE였으니, -1을 곱해주고 `np.sqrt`로 루트 연산을 한다.

위에서 보았던 10 만 단위의 RMSE 와는 값의 크기가 아주 다른데, 이는 `price`의 분포가 한쪽으로 치우쳐져 있는 것을 보고 log 변환을 했기 때문이다.

log 변환후 rmse 값을 계산하기 위한 함수에서는 `np.expm1` 함수를 활용해 다시 원래대로 복원한 후 RMSE 값을 계산했는데, 그리드 탐색을 하면서는 `np.expm1()`으로 변환하는 과정이 없었기 때문에 log 변환되어 있는 `price` 데이터에서 손실함수값을 계산한 것이다.

따라서 사실, 위의 데이터 프레임에 나타난 값은 정확히 말하면 RMSE 가 아니라 RMSLE, 즉 Root Mean Squared Log Error이다. 즉, log 를 취한 값에서 RMSE 를 구한 것이다.
<br></br>
> 컬럼의 이름을 RMSLE 로 변환
> (판다스에서 컬럼 이름 변환은 `rename` 을 활용한다.)
```python
results = results.rename(columns={'RMSE': 'RMSLE'})
results
```
<br></br>
> RMSLE 가 낮은 순서로 정렬
> (`sort_values` 를 활용)
```python
# 위의 표를 `RMSLE`가 낮은 순서대로 정렬해주세요.
results = results.sort_values('RMSLE')
results
```
+ 참고 : [pandas.DataFrame.sort_values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)
<br></br>
> 위의 그리드 탐색을 수행하는 과정을 하나의 함수로 구현
```python
"""
다음과 같은 과정을 진행할 수 있는 `my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)` 함수를 구현해 보세요.

1. GridSearchCV 모델로 `model`을 초기화합니다.
2. 모델을 fitting 합니다.
3. params, score에 각 조합에 대한 결과를 저장합니다. 
4. 데이터 프레임을 생성하고, RMSLE 값을 추가한 후 점수가 높은 순서로 정렬한 `results`를 반환합니다.
"""

# 코드 입력

def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    # GridSearchCV 모델로 초기화
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', \
                              cv=5, verbose=verbose, n_jobs=n_jobs)

    # 모델 fitting
    grid_model.fit(train, y)

    # 결과값 저장
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']

    # 데이터 프레임 생성
    results = pd.DataFrame(params)
    results['score'] = score

    # RMSLE 값 계산 후 정렬
    results['RMSLE'] = np.sqrt(-1 * results['score'])
    results = results.sort_values('RMSLE')

    return results
```
<br></br>

## 랭킹을 올리고 싶다면 (04). 제출하는 것도, 빠르고 깔끔하게!

> 앞어 만든 `my_GridSearch()` 함수를 통해 그리드 탐색
```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [1, 10],
}

model = LGBMRegressor(random_state=random_state)
my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)
```
가장 좋은 조합은 `max_depth=10`, `n_estimators=100` 임을 확인 가능하다.
<br></br>
> 최적의 하이퍼 파라미터 조합을 통해 예측값 산출
```python
model = LGBMRegressor(max_depth=10, n_estimators=100, random_state=random_state)
model.fit(train, y)
prediction = model.predict(test)
prediction
```
<br></br>
> 예측 결과에  `np.eapm1()` 을 씌워 원래 스케일로 조정
```python
prediction = np.expm1(prediction)
prediction
```
<br></br>
> `sample_submission,csv` 파일을 가져오기
```python
data_dir = os.getenv('HOME')+'/kaggle_kakr_housing/data'

submission_path = join(data_dir, 'sample_submission.csv')
submission = pd.read_csv(submission_path)
submission.head()
```
<br></br>
> 예측한 예측값을 덮어 씌우기
```python
submission['price'] = prediction
submission.head()
```
<br></br>
> 최종 결과를 `csv` 파일로 저장
```python
submission_csv_path = '{}/submission_{}_RMSLE_{}.csv'.format(data_dir, 'lgbm', '0.164399')
submission.to_csv(submission_csv_path, index=False)
print(submission_csv_path)
```
파일 이름에 모델의 종류와 위에서 확인했던 RMSLE 값을 넣어주면 제출 파일들이 깔끔하게 관리된다.
<br></br>
> 위 과정을 함수로 정리해두면 사용하기 편리하다.
```python
"""
아래의 과정을 수행하는 `save_submission(model, train, y, test, model_name, rmsle)` 함수를 구현해 주세요.
1. 모델을 `train`, `y`로 학습시킵니다.
2. `test`에 대해 예측합니다.
3. 예측값을 `np.expm1`으로 변환하고, `submission_model_name_RMSLE_100000.csv` 형태의 `csv` 파일을 저장합니다.
"""

# 코드 작성

def save_submission(model, train, y, test, model_name, rmsle=None):
    model.fit(train, y)
    prediction = model.predict(test)
    prediction = np.expm1(prediction)
    data_dir = os.getenv('HOME')+'/kaggle_kakr_housing/data'
    submission_path = join(data_dir, 'sample_submission.csv')
    submission = pd.read_csv(submission_path)
    submission['price'] = prediction
    submission_csv_path = '{}/submission_{}_RMSLE_{}.csv'.format(data_dir, model_name, rmsle)
    submission.to_csv(submission_csv_path, index=False)
    print('{} saved!'.format(submission_csv_path))
```
<br></br>
> 위에서 작성한 함수를 통해 한 줄로 모델을 학습시킨 후 예측 결과를 저장
```python
save_submission(model, train, y, test, 'lgbm', rmsle='0.0168')
```
<br></br>
