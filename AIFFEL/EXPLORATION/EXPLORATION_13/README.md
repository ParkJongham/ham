# 13. 어제 오늘 내 주식, 과연 내일은?

시계열 예측 (Time-Series Prediction) 을 다루는 여러 가지 통계적 기법 중에 가장 널리 알려진 ARIMA( Auto-regressive Integrated Moving Average) 에 대해 알아보고 이를 토대로 특정 주식 종목의 가격을 예측해 보는 실습을 진행해보자.

시계열 예측에 사용되는 모델은 ARIMA 외에 Prophet, LSTM 등 딥러닝을 활용하는 방법도 있다.

## 학습 목표

-   시계열 데이터의 특성과 안정적 (Stationary) 시계열의 개념을 이해한다.
-   ARIMA 모델을 구성하는 AR, MA, Diffencing 의 개념을 이해하고 간단한 시계열 데이터에 적용해 본다.
-   실제 주식 데이터에 ARIMA 를 적용해서 예측 정확도를 확인해 본다.
<br></br>

## 시계열 예측이란 (01). 미래를 예측한다는 것은 가능할까?

미래 예측 시나리오를 생각해보자.

1. 주가 곡선을 바탕으로 다음 주가 변동을 예측
2. 특정 지역의 기후 데이터를 바탕으로 내일의 온도변화 예측
3. 공장 센터데이터 변화이력을 토대로 이상 발생 예측

등등이 있다. 이런 데이터들의 시계열 (Time - Series) 데이터를 기반으로 한다는 것이다.

시계열 시간의 순서대로 발생한 데이터의 수열이라는 뜻이다. 즉, 시간의 축에 따라 발생한 데이터를 시계열 데이터라고 한다.

주식 데이터를 예로들면 `날짜 - 가격` 형태로 날짜순으로 모아둔 데이터가 있다면 날짜가 인덱스 역할을 하게 된다.

시계열은 수식으로 표현하자면 $$Y ={Y_t: t ∈ T }, where\ is\ the\ index\ set$$
<br></br>
하지만 꼭 일정 시간 간격으로 발생한 데이터일 필요는 없다.

하지만 많은 양의 시계열 데이터를 쌓았다고 미래 예측이 가능할까?

결론적으로 미래 예측은 불가능하다. 하지만 미래를 예측하려한다면 2 가지의 전제 조건이 필요하다.

1. 과거의 데이터에 일정한 패턴이 발견된다.  

2. 과거의 패턴은 미래에도 동일하게 반복될 것이다.
<br></br>

즉, 안정적 (Stationary) 데이터에 대해서만 미래 예측이 가능하다.

이때 안적적 이라는 것은 시계열 데이터의 통계적 특성이 변하지 않는다는 의미이다. 즉, 시간의 변화에 무관하게 일정한 패턴이 존재한다는 의미이다.

따라서 일정한 패턴이 반복되는 데이터라면 안정성이 뛰어난 데이터이며, 이 경우 어느정도 오차 범위 내에서 예측이 가능하다.

+ 참고 : [그림자로 원유재고 알아낸다](https://news.einfomax.co.kr/news/articleView.html?idxno=4082410)
<br></br>

시계열 데이터 분석은 완벽한 미래 예측은 불가능하지만 프로세스 내재적인 시간적 변화를 묘사하는데 뛰어나다.
<br></br>

## 시계열 예측이란 (02). Stationary 한 시계열 데이터

안정적인 시계열 데이터에서 시간의 추이와 관계없이 일정해야하는 통계적 특성 3 가지는 바로 평균, 분산, 공분산 (자기 공분산, Autocovariance) 이다.

+ 참고 :
	+ [Covariance와 Correlation](https://destrudo.tistory.com/15)
	+ [Autocovariance와 Autocorrelation](https://m.blog.naver.com/sw4r/221030974616)
<br></br>

우리는 과거 몇 개의 데이터를 통해 다음 데이터를 예측할때, 5 년치 판매량 X (t - 4), X (t - 3), X (t - 2), X (t - 1), X (t)를 가지고 X (t + 1)이 얼마일지 예측을 해보고 싶다고 할 때 예측에 의미가 있으려면 t 에 무관하게 예측이 맞아야한다는 것이다.

만약 t = 2010 일 때의 데이터를 가지고 X (2011) 을 정확하게 예측하는 모델이라면 이 모델에 t = 2020 을 대입해도 이 모델이 X (2021) 을 정확하게 예측할 수 있어야 한다.

이를위해서는 t에 무관하게 X (t - 4), X (t - 3), X (t - 2), X (t - 1), X (t) 의 평균과 분산이 일정 범위 안에 있어야하며 X (t - h)와 X (t)는 t에 무관하게 h 에 대해서만 달라지는 일정한 상관도를 가져야 한다.

그렇지 않으면 시계열 예측은 t에 따라 달라지게 된다. 즉, '과거의 패턴이 미래에도 반복될 것이다'라는 시계열 예측의 대전제를 무시하는 것이다.
<br></br>

## 시계열 예측이란 (03). 시계열 데이터 사례분석

> 데이터 다운로드
```bash
$ mkdir -p ~/aiffel/stock_prediction/data $ wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv 
$ wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv 
$ mv daily-min-temperatures.csv airline-passengers.csv ~/aiffel/stock_prediction/data
```
<br></br>

### 시계열 (Time Series) 생성

> 사용할 라이브러리 가져오기
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
```
<br></br>
> 사용할 데이터 가져오기
```python
dataset_filepath = os.getenv('HOME')+'/stock_prediction/data/daily-min-temperatures.csv' 
df = pd.read_csv(dataset_filepath) 
print(type(df))
df.head()
```
> Date 컬럼을 인덱스로 시계열 데이터 생성
```python
# 이번에는 Date를 index_col로 지정해 주었습니다. 
df = pd.read_csv(dataset_filepath, index_col='Date', parse_dates=True)
print(type(df))
df.head()
```
시계열(Time Series) 이란 것도 결국 시간 컬럼을 index 로 하는 Series 로 표현된다.

우리가 읽어 들인 데이터 파일은 Pandas 를 통해 2 개의 컬럼을 가진 DataFrame 으로 변환되며, 이것은 시계열 데이터 구조는 아니다.
<br></br>
> 시계여려 데이터 불러오기
```python
ts1 = df['Temp']
print(type(ts1))
ts1.head()
```
DataFrame 인 df 와 Series 인 df['Temp'] 는 index 구조가 동일하므로 Numpy, Pandas, Matplotlib 등 많은 라이브러리들이 호환해서 지원한다.

하지만 그렇지 않은 경우도 간혹 발생하므로 여기서는 명확하게 Series 객체를 가지고 진행하도록 한다.
<br></br>

### 시계열 안정성의 정성적 분석

> 시각화를 통해 안정성 여부 확인
```python
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 13, 6    # matlab 차트의 기본 크기를 13, 6으로 지정해 줍니다.

# 시계열(time series) 데이터를 차트로 그려 봅시다. 특별히 더 가공하지 않아도 잘 그려집니다.
plt.plot(ts1)
```
<br></br>
> 결측치 유무 확인
```python
ts1[ts1.isna()]  # 시계열(Time Series)에서 결측치가 있는 부분만 Series로 출력합니다.
```
결측치가 없음을 확인할 수 있다. 결측치가 있다면 이를 삭제하거나 결측치 양옆의 값들을 이용해 적절히 보간 대입해 주는 방법이 있다.
<br></br>
> 판다스에서 결측치 보간을 처리하는 메소드
```python
# 결측치가 있다면 이를 보간합니다. 보간 기준은 time을 선택합니다. 
ts1=ts1.interpolate(method='time')

# 보간 이후 결측치(NaN) 유무를 다시 확인합니다.
print(ts1[ts1.isna()])

# 다시 그래프를 확인해봅시다!
plt.plot(ts1)
```
+ 참고 : [결측치 보간](https://rfriend.tistory.com/264)
<br></br>
> 일정 시간 내 구간 통계치 (Rolling Statistics) 를 시각화하는 함수 생성
```python
def plot_rolling_statistics(timeseries, window=12):
    
    rolmean = timeseries.rolling(window=window).mean()  # 이동평균 시계열
    rolstd = timeseries.rolling(window=window).std()    # 이동표준편차 시계열

     # 원본시계열, 이동평균, 이동표준편차를 plot으로 시각화해 본다.
    orig = plt.plot(timeseries, color='blue',label='Original')    
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
```
현재 타임스텝부터 `window` 에 주어진 타임스텝 이전 사이 구간의 평균 (rolling mean, 이동평균) 과 표준편차 (rolling std, 이동표준편차) 를 원본 시계열과 함께 시각화해 보면 좀 더 뚜렷한 경향성을 확인할 수 있다.
<br></br>
> 일정 시간 내 구간 통계치 시각화
```python
plot_rolling_statistics(ts1, window=12)
```
<br></br>

### 다른 데이터에 대해서도 비교해 보자.

> `International airline passengers` 데이터셋 가져오기
```python
dataset_filepath = os.getenv('HOME')+'/stock_prediction/data/airline-passengers.csv' 
df = pd.read_csv(dataset_filepath, index_col='Month', parse_dates=True).fillna(0)  
print(type(df))
df.head()
```
<br></br>
> 승객 컬럼의 분포 확인
```python
ts2 = df['Passengers']
plt.plot(ts2)
```
<br></br>
> 일정 시간 내 구간 통계치 시각화
```python
plot_rolling_statistics(ts2, window=12)
```
시간의 추이에 따라 평균과 분산이 증가하는 패턴을 보인다. 따라서 안정적이라고 판단하기에는 어렵다.

시계열 데이터의 안정성을 시각화 방법을 통해 정성적으로 분석해볼 수 있으며, 시계열 데이터를 다루는 가장 기본적인 접근법이다. 이 외에 안정성을 평가하는데 있어 정량적인 방법도 있다.
<br></br>

## 시계열 예측이란 (04). Stationart 여부를 체크하는 통계적 방법

### Augmented Dickey - Fuller Test

Augmented Dickey-Fuller Test (ADF Test) 라는 시계열 데이터의 안정성을 테스트하는 통계적 방법을 알아보자.

`주어진 시계열 데이터가 안정적이지 않다` 라는 `귀무가설(Null Hypothesis)` 를 세운 후, 통계적 가설 검정 과정을 통해 이 귀무가설이 기각될 경우에 `이 시계열 데이터가 안정적이다` 라는 `대립가설(Alternative Hypothesis)`을 채택하는 것이다.

ADF Test 는 `statsmodels` 패키지에서 제공하는 `adfuller` 메소드를 이용해 쉽게 이용이 가능하다.

+ 참고 : 
	+ [ADF Test](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/)
	+ [유의확률-위키백과](https://ko.wikipedia.org/wiki/%EC%9C%A0%EC%9D%98_%ED%99%95%EB%A5%A0)  
	+ [P-value(유의확률)의 개념](https://m.blog.naver.com/baedical/10109291879)
<br></br>

### statsmodels 패키지와 adfuller 메소드

`statsmodels` 패키지는 `R`에서 제공하는 통계검정, 시계열분석 등의 기능을 파이썬에서도 이용가능한 통계 패키지이다.
<br></br>
> `statsmodels` 패키지 설치
```python
pip install statsmodels
```
<br></br>
> `statsmodels` 패키지에서 제공하는 `adfuller` 메소드를 이용해 주어진 `timeseries`에 대한 Augmented Dickey-Fuller Test를 수행 함수 생성
```python
from statsmodels.tsa.stattools import adfuller

def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메소드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메소드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
```
<br></br>
> 앞서 정상적으로 분석한  `ts1` 시계열에 대한 Augmented Dickey-Fuller Test를 수행
```python
augmented_dickey_fuller_test(ts1)
```
`Daily Minimum Temperatures in Melbourne` 시계열이 안정적이지 않다는 귀무가설은 p-value 가 거의 0 에 가깝게 나타남을 확인할 수 있다.

따라서 귀무가설은 기각되고, 이 시계열은 안정적 시계열이라는 대립가설이 채택한다.
<br></br>
> 앞서 정상적으로 분석한  `ts2` 시계열에 대한 Augmented Dickey-Fuller Test를 수행
```python
augmented_dickey_fuller_test(ts2)
```
`International airline passengers` 시계열이 안정적이지 않다는 귀무가설은 p-value 가 거의 1 에 가깝게 나타났음을 확인할 수 있다.

따라서 귀무가설이 옳다는 직접적인 증거가 되지는 않지만, 적어도 이 귀무가설을 기각할 수는 없게 되었으므로 이 시계열이 안정적인 시계열이라고 말할 수 없다.
<br></br>

## 시계열 예측의 기본 아이디어 : Stationary 하게 만들 방법은 없을까?

위에서 안정적이라고 말할 수 없게 된 `ts2` 데이터 (`International airline passengers`) 시계열을 조금 더 분석해 보자.

안정적이지 않은 시계열을 분석하려면 안정적인 시계열로 바꿔줘야한다.

이를 위한 방법은 2 가지가 있는데, 첫 째는 정성적인 분석을 통해 보다 안정적인 특성을 가지도록 기존의 시계열 데이터를 가공 / 변형하는 방법이며, 둘 째는 시계열 분해 (Time - Series Decomposition) 이라는 기법을 적용하는 것이다.

### 보다 안정적인 (Stationary) 한 시계열로 가공해 가기
<br></br>
#### 로그함수 변환

보다 안정적인 시계열로 가공하기 위해 가장 먼저 고려해볼 수 있는 방법으로는 시간 추이에 따라 분산이 점점 커지는 부분이다.

이렇게 분산이 점점 커질 때에는 로그함수로 변환을 취해주는것이 도움이 된다.
<br></br>
> 로그함수 변환
```python
ts_log = np.log(ts2)
plt.plot(ts_log)
```
<br></br>
> 로그변환 후의 Augmented Dickey-Fuller Test를 수행
```python
augmented_dickey_fuller_test(ts_log)
```
p-value 가 0.42 로 무려 절반 이상 줄어듬을 볼 수 있다.

정성적으로도 시간 추이에 따른 분산이 어느정도 일정해진 것을 확인할 수 있다.

하지만 시간 추이에 따라 계속해서 평균이 증가한다는 점 역시 고려해야한다.
<br></br>

#### Moving Average 제거 - 추세 (Trend) 상쇄하기

추세 (Trend) 라나 시간 추이에 따라 나타나는 평균값의 변화를 의미한다.

이러한 평균값의 변화량을 제거하여 Moving Average, 즉 rolling mean 을 구해서 `ts_log` 에서 빼주는 것이 도움이 된다.
<br></br>
> Moving Average 를 시각화를 통해 확인
```python
moving_avg = ts_log.rolling(window=12).mean()  # moving average구하기 
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
```
<br></br>
> rolling mean 을 구해서 `ts_log` 에서 빼주기
```python
ts_log_moving_avg = ts_log - moving_avg # 변화량 제거
ts_log_moving_avg.head(15)
```
Moving Average 계산 시 (windows size = 12인 경우) 앞의 11 개의 데이터는 Moving Average 가 계산되지 않으므로 `ts_log_moving_avg` 에 결측치가 발생한다. 

이러한 결측치는 이후 Dicky - Fuller Test  시 에러 발생을 유발하므로 제거가 필요하다.
<br></br>
> 결측치 제거
```python
ts_log_moving_avg.dropna(inplace=True)
ts_log_moving_avg.head(15)
```
<br></br>
> `ts_log_moving_avg` 의 정성적 분석
```python
plot_rolling_statistics(ts_log_moving_avg)
```
<br></br>
> `ts_log_moving_avg` 의 정량적 분석
```python
augmented_dickey_fuller_test(ts_log_moving_avg)
```
p-value 가 0.02 수준으로 95% 의 신뢰구간으로 시계열은 안정적이라고 할 수 있게 되었다.

하지만 Moving Average 를 계산하는 window = 12로 정확하게 지정해 주어야 한다는 점이 아직 남아있다.
<br></br>
> rolling mean 을 구해서 `ts_log` 에서 빼줄 때 `window = 6` 으로 설정
```python
moving_avg_6 = ts_log.rolling(window=6).mean()
ts_log_moving_avg_6 = ts_log - moving_avg_6
ts_log_moving_avg_6.dropna(inplace=True)
```
<br></br>
> `ts_log_moving_avg` 의 정성적 분석 
> (`window = 6` 일때)
```python
plot_rolling_statistics(ts_log_moving_avg_6)
```
<br></br>
> `ts_log_moving_avg` 의 정량적 분석
> (`window = 6` 일때)
```python
augmented_dickey_fuller_test(ts_log_moving_avg_6)
```
window = 6 으로 하면 정성적 분석에서는 큰 차이를 못 느끼지만 정량적 분석에서 p-value 는 0.18 수준이어서 아직도 안정적 시계열이라고 말할 수 없다.

해당 데이터셋은 월 단위로 발생하는 시계열이므로 12개월 단위로 주기성이 있기 때문에 window = 12가 적당하다는 것을 추측할 수도 있지만 moving average 를 고려할 때는 rolling mean 을 구하기 위한 window 크기를 결정하는 것이 매우 중요하다.
<br></br>

#### 차분 (Differencing) - 계절성 (Seasonality) 상쇄하기

계정성 (Seasonality) 란 추세에는 잡히지 않지만 시계열 데이터 안에 포함된 패턴이 파악되지 않은 주기적 변화는 예측에 방해가 되는 불안정성 요소이며, Moving Average 제거로는 상쇄되지 않는 효과이다.

이러한 계설정을 상쇄하기 위해서는 차분 (Differencing) 을 활용하면 된다.

차분이란 시계열을 한 스텝 앞으로 시프트한 시계열을 원래 시계열에서 빼 주는 것이다.

즉, 현새 스탭 값 - 직전 스텝 값이 되어 이번 스텝에서 발생한 변화량을 확인할 수 있다.
<br></br>
> 현재 스텝의 시계열 산출 및 시각화
```python
ts_log_moving_avg_shift = ts_log_moving_avg.shift()

plt.plot(ts_log_moving_avg, color='blue')
plt.plot(ts_log_moving_avg_shift, color='green')
```
<br></br>
> 이전 스텝의 시계열 산출 및 시각화
```python
ts_log_moving_avg_diff = ts_log_moving_avg - ts_log_moving_avg_shift
ts_log_moving_avg_diff.dropna(inplace=True)
plt.plot(ts_log_moving_avg_diff)
```
<br></br>
> 차분(현재 스탭 시계열 - 이전 스탭 시계열) 산출
```python
plot_rolling_statistics(ts_log_moving_avg_diff)
```
<br></br>
> 차분의 정량적 분석
```python
augmented_dickey_fuller_test(ts_log_moving_avg_diff)
```
Trend 를 제거하고 난 시계열에다가 1 차 차분 (1st order differencing) 을 적용하여 Seasonality 효과를 다소 상쇄한 결과, p-value 가 이전의 10 % 정도까지로 줄어드는 것을 확인할 수 있다.

데이터에 따라 2 차 차분 (2nd order differencing, 차분의 차분), 3차 차분 (3rd order differencing, 2차 차분의 차분) 을 적용하면 더욱 p-value 를 낮출 수도 있다.
<br></br>

### 시계열 분해 (Time Series Decomposition)

statsmodels 라이브러리 안에는 `seasonal_decompose` 메소드를 통해 시계열 안에 존재하는 trend, seasonality 를 직접 분리해 낼 수 있는 기능이있다.

이를 활용하면 moving average 제거, differencing 등을 거치지 않고도 훨씬 안정적인 시계열을 분리해 낼 수 있다.
<br></br>
> statsmodels 라이브러리 안에는 `seasonal_decompose` 메소드를 통해 시계열을 분리
```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.rcParams["figure.figsize"] = (11,6)
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
```
`Original` 시계열에서 `Trend` 와 `Seasonality` 를 제거하고 난 나머지를 `Residual` 이라고 한다.

즉, `Trend+Seasonality+Residual=Original` 이다. 
<br></br>
> `Residual`에 대해 안정성 여부 확인
```python
plt.rcParams["figure.figsize"] = (13,6)
plot_rolling_statistics(residual)
```
<br></br>
> `Residual` 의 안정성을 정량적으로 평가
```python
residual.dropna(inplace=True)
augmented_dickey_fuller_test(residual)
```
`Decomposing` 을 통해 얻어진 `Residual` 은 압도적으로 낮은 `p-value` 를 보여 준다.
<br></br>

## ARIMA  모델의 개념

### ARIMA 모델의 정의

ARIMA (Autoregressive Integrated Moving Average) 은 안정적인 시계열 데이터 예측 모델을 자동으로 만들어주는 모델이다.

`AR(Autoregressive)` + `I(Integrated)` + `MA(Moving Average)`가 합쳐져 구성되어 있다.
<br></br>
### AR (자기회귀, AutoRegressive)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-16-2_nh8iD9A.max-800x600.png)
<br></br>

자기회귀란 시계열 데이터 $Y = \lbrace Y_t: t ∈ T \rbrace$ 에서 $Y_t$ 가 이전 p 개의 데이터 $Y_{t−1}, Y_{t−2},...,Y_{t−p}$ 의 가중합으로 수렴한다고 보는 것이며, 가중치의 크기가 1 보다 작은 $Y_{t−1}, Y_{t−2},...,Y_{t−p}$ 의 가중합으로 수렴하는 자기회귀 모델과 안정적 시계열은 통계적으로 동치이다.

자기회귀는 일반적인 시계열에서 추세와 계절성을 제거한 잔차에 해당하는 부분을 모델링한다.

예를들어 주식값이 항상 일정한 균형 수준을 유지할 것이라고 예측하는 관점이 바로 자기회귀이다.
<br></br>

### MA (이동평균, Moving Average)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-16-3.max-800x600.png)
<br></br>

이동평균 (Moving Average) 란 $Y_t$ 가 이전 q 개의 예측오차값 $e_{t-1}, e_{t-2}, ... , e_{t-q}$ 의 가중합으로 수렴한다고 보는 모델이다.

이동평균은 일반적인 시계열에서 추세에 해당하는 부분을 모델링하며, 예측오차값이 $e_{t-1}$ 이 + 라면 모델 예측보다 관측값이 더 높다는 의미이다. 즉, 다음 $Y_y$ 의 예측시에는 예측치를 올려잡게된다.

예를들면, 주식값은 항상 최근의 증감 패턴이 지속될 것이라고 예측하는 관점이다.
<br></br>


### I (차분누적, Integration)

차분누적 (Integragion) 은  $Y_t$ 의 이전 데이터와 d 차 차분의 누적의 합이라고 보는 모델이다.

예를 들어 d = 1 일 때,  $Y_t$ 는  $Y_{t-1}$ 과 $\varDelta Y_{t-1}$ 의 합으로 보는 것이다.

차분누적 I 는 일반적인 시계열에서 계절성에 해당하는 부분을 모델링한다.

ARIMA 는 위 3 가지 모델, 자기회귀, 이동평균, 차분누적을 모두 한꺼번에 고려하는 모델이다.
<br></br>

### ARIMA 모델의 모수 p, q, d

ARIMA 를 활용해서 시계열 예측 모델을 성공적으로 만들기 위해서는 ARIMA 의 모수 (parameter) 를 데이터에 맞게 설정할 필요가 있다.

ARIMA 의 모수는 3가지가 있는데, 자기회귀 모형 (AR) 의 시차를 의미하는 `p`, 차분 (diffdrence) 횟수를 의미하는 `d`, 이동평균 모형 (MA) 의 시차를 의미하는 `q` 가 그것이다.

이들 중 `p` 와 `q` 에 대해서는 통상적으로 `p + q` < 2, `p * q` = 0 인 값들을 사용하는데, 이는 `p` 나 `q` 중 하나의 값이 0 이라는 뜻이다.

실제로 대부분의 시계열 데이터는 자기회귀 모형이나 이동평균 모형 중 하나의 경향만을 강하게 띠기 때문이다.

`ARIMA(p,d,q)` 모델의 모수 `p,d,q` 를 결정하는 방법은 `ACF(Autocorrelation Function)` 와 `PACF(Partial Autocorrelation Function)` 을 통해 결정할 수 있다. 

`AutoCorrelation` 은 자기 상관계수와 같은 의미이며,  `ACF` 는 시차 (lag) 에 따른 관측치들 사이의 관련성을 측정하는 함수, `PACF` 는 다른 관측치의 영향력을 배제하고 두 시차의 관측치 간 관련성을 측정하는 함수이다.
<br></br>
> `statsmodels`에서 제공하는 `ACF`와 `PACF` 플로팅 기능을 사용
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
plt.show()
```
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-16-4.max-800x600.png)
<br></br>

위 그림은 ACF 를 통해 MA 모델의 시차 q 를 결정 하고, PACF 를 통해 AR 모델의 시차 p 를 결정 할 수 있음을 통계학적으로 설명하는 것이다.

이를 통해 출력 결과를 보면 `PACF` 그래프를 볼 때 `p`가 2 이상인 구간에서 `PACF` 는 거의 0 에 가까워지고 있기 때문에 `p=1` 이 매우 적합한 것으로 보인다.

`PACF`가 0 이라는 의미는 현재 데이터와 `p` 시점 떨어진 이전의 데이터는 상관도가 0, 즉 아무 상관 없는 데이터이기 때문에 고려할 필요가 없다는 의미이다.

반면 `ACF` 는 점차적으로 감소하고 있어서 `AR(1)` 모델에 유사한 형태를 보이고 있으며, 따라서 `q` 에 대해서는 적합한 값이 없어 보인다.

`MA` 를 고려할 필요가 없다면 `q=0`으로 둘 수 있으며, `q` 를 바꿔가며 확인해봐도 된다.
<br></br>
> 파라미터 `d` 를 구하기 위해 차분을 구해보고 안정된 시계열인지 확인
> (1차 차분 구하기)
```python
# 1차 차분 구하기
diff_1 = ts_log.diff(periods=1).iloc[1:]
diff_1.plot(title='Difference 1st')

augmented_dickey_fuller_test(diff_1)
```
<br></br>
> > 파라미터 `d` 를 구하기 위해 차분을 구해보고 안정된 시계열인지 확인
> (2차 차분 산출)
```python
# 2차 차분 구하기
diff_2 = diff_1.diff(periods=1).iloc[1:]
diff_2.plot(title='Difference 2nd')

augmented_dickey_fuller_test(diff_2)
```
1 차 차분을 구했을 때 약간 애매한 수준의 안정화 상태를 보였으며, 2차 차분을 구했을 때는 확실히 안정화 상태를 보였다. 

따라서 `d=1` 로 설정할 수 있으며, `d` 값도 바꿔가면서 최적의 값을 찾아봐도 된다.
<br></br>

### 학습 데이터 분리

> 학습 데이터와 테스트 데이터 분리
> (9 : 1 로 분리)
```python
train_data, test_data = ts_log[:int(len(ts_log)*0.9)], ts_log[int(len(ts_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(ts_log, c='r', label='training dataset')  # train_data를 적용하면 그래프가 끊어져 보이므로 자연스러운 연출을 위해 ts_log를 선택
plt.plot(test_data, c='b', label='test dataset')
plt.legend()
```
<br></br>
> 분리된 데이터셋의 형태 확인
```python
print(ts_log[:2])
print(train_data.shape)
print(test_data.shape)
```
<br></br>

## ARIMA 모델 훈련과 추론

> ARIMA 모델 훈련
```python
from statsmodels.tsa.arima_model import ARIMA

# Build Model
model = ARIMA(train_data, order=(1, 1, 0))  
fitted_m = model.fit(disp=-1)  
print(fitted_m.summary())
```
<br></br>
> ARIMA 모델의 훈련과정 시각화
```python
fitted_m.plot_predict()
```
<br></br>
> forecast() 메소드를 이용해 테스트 데이터 구간의 데이터를 예측
```python
# Forecast : 결과가 fc에 담깁니다. 
fc, se, conf = fitted_m.forecast(len(test_data), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)   # 예측결과
lower_series = pd.Series(conf[:, 0], index=test_data.index)  # 예측결과의 하한 바운드
upper_series = pd.Series(conf[:, 1], index=test_data.index)  # 예측결과의 상한 바운드

# Plot
plt.figure(figsize=(9,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, c='b', label='actual price')
plt.plot(fc_series, c='r',label='predicted price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.legend()
plt.show()
```
<br></br>
> 최종적인 모델의 오차율 계산
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse = mean_squared_error(np.exp(test_data), np.exp(fc))
print('MSE: ', mse)

mae = mean_absolute_error(np.exp(test_data), np.exp(fc))
print('MAE: ', mae)

rmse = math.sqrt(mean_squared_error(np.exp(test_data), np.exp(fc)))
print('RMSE: ', rmse)

mape = np.mean(np.abs(np.exp(fc) - np.exp(test_data))/np.abs(np.exp(test_data)))
print('MAPE: {:.2f}%'.format(mape*100))
```
오차율 계선을 위해 로그 변환된 시계열을 사용해 왔던 것을 모두 지수 변환하여 원본의 스케일로 계산해야 하며, `np.exp()` 를 통해 전부 원본 스케일로 돌린 후 `MSE`, `MAE`, `RMSE`, `MAPE` 를 계산하였다.

최종 모델의 메트릭으로 활용하기 적당한 MAPE 기준으로 14% 의 오차율을 보이며, 이는 최적의 모수를 찾게된다면 성능을 향상할 수 있는 여지가 있다.

+ 참고 : [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

