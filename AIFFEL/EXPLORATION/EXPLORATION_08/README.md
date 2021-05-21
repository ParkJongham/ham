# 08. 아이유 팬이 좋아할 만한 다른 아티스트 찾기

## 실습 목표

1.  추천시스템의 개념과 목적을 이해한다.  
2. Implicit 라이브러리를 활용하여 Matrix Factorization(이하 MF) 기반의 추천 모델을 만들어 본다.  
3. 음악 감상 기록을 활용하여 비슷한 아티스트를 찾고 아티스트를 추천해 본다.  
4. 추천 시스템에서 자주 사용되는 데이터 구조인 CSR Matrix을 익힌다  
5. 유저의 행위 데이터 중 Explicit data와 Implicit data의 차이점을 익힌다.  
6. 새로운 데이터셋으로 직접 추천 모델을 만들어 본다.
<br></br>

## 추천 시스템이란게 뭔가요?

유투브, 여러 스트리밍 음원 서비스 등 많은 유저들이 어떤 콘텐츠를 즐기고 있는지에 대한 엄청난 양의 데이터를 축적하고 있다.

추천 시스템이란 이렇게 축적된 사용자의 데이터를 통해 해당 사용자가 좋아할만한 제품 혹은 서비스, 콘텐츠 등을 향유할 수 있도록 제시해주는 것이다.

콘텐츠 추천 알고리즘은 크게 협업 필터링 (Collaborative Filtering) 방식과 콘텐츠 기반 필터링 (Contents - Based Filtering), 2 가지로 나눠진다.

협업 필터링은 다수의 사용자가 아이템을 구매한 이력 정보만으로 사용자간 유사성 및 아이템 간 유사성을 파악하지만, 콘텐츠 기반 필터링은 아이템의 고유의 정보를 바탕으로 아이템 간 유사성을 파악한다.

즉, 협업 필터링에서는 아이템과 사용자 간의 행동 또는 관계에만 주목할 뿐 아이템 자체의 고유한 속성에 주목하지않는다.

반면에 콘텐츠 기반 필터링은 아이템 자체의 속성에 주목한다. 하지만 협업 필터링과는 반대로 사용자와 아이템 간의 관련성에 주목하지 않으므로, 사용자의 특성에 따른 개인화된 추천을 제공하기 어렵다는 단점이 있다.

다시말해 다수의 사용자의 판단을 기반으로 추천을 하는 시스템의 핵심 근간은 협업 필터링이다. 하지만 여러 제약조건이 있다.

첫 번째로는 시스템이 충분한 정보를 모으지 못한 사용자나 아이템에 대한 추론을 할 수 없는 상태를 의미하는 콜드 스타트 (Cold Start) 문제가 있다.

두 번째로는 계산량이 너무 많아 추천의 효율이 떨어진다. 

세 번째로는 롱테일의 꼬리 부분, 즉 사용자의 관심이 저조한 항목의 정보가 부족하여 추천에서 배제되는 상황이 바로 제약조건이다.

+ 참고 : 
	+ [치열한 음원 큐레이션 서비스 경쟁](https://blog.naver.com/businessinsight/222191549425)
	+ [콘텐츠 추천 알고리즘의 진화](http://www.kocca.kr/insight/vol05/vol05_04.pdf)
<br></br>

추천시스템은 아이템은 매우 많지만 유저의 취향을 다양할 때 유저가 소비할만한 아이템을 예측하는 모델이다.

이를 위해서는 필연적으로 유저가 소비하는 아이템에 대한 매우 방대한 양의 데이터는 정확한 협업 필터링, 즉, 추천시스템을 위한 필수 조건이다.
<br></br>
> Last.fm 에서 제공하는 유저가 소비한 음원 정보 데이터 다운
> [데이터](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html)
```bash
1) 작업디렉토리 생성 
$ mkdir -p ~/aiffel/recommendata_iu/data 

2) wget으로 데이터 다운로드 (주의) 오래걸립니다. 
$ wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz 

3) 다운받은 데이터를 작업디렉토리로 옮기고, 작업디렉토리로 이동합니다. 
$ mv lastfm-dataset-360K.tar.gz ~/aiffel/recommendata_iu/data & cd ~/aiffel/recommendata_iu/data 

4) gzip으로 압축된 압축을 해제하면 tar 파일이 하나 나옵니다. 
$ gunzip lastfm-dataset-360K.tar.gz 

5) tar 압축을 다시 해제하면 우리가 사용할 최종 데이터 파일이 나옵니다. 
$ tar -xvf lastfm-dataset-360K.tar 

6) 필요 라이브러리 설치 
$ pip install implicit

7) 위 4 ~ 5 번 과정을 z 옵션을 통해 한꺼번에 처리
$ tar -xvzf lastfm-dataset-360K.tar.gz
```
<br></br>

## 데이터 탐색하기와 전처리

### 데이터 준비

위에서 다운받은 데이터를 확인 및 전처리를 진행해보자.
<br></br>
> 다운받은 데이터의 형식 확인
```bash
$ more ~/aiffel/recommendata_iu/data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv
```
다운받은 데이터는 `tsv` 데이터로써 `csv` 파일과 거의 동일하지만 구분자가 콤마가 아닌 `tab` 으로 된 파일 양식이며, 판다스의 `read_csv` 를 통해서도 열어볼 수 있다.
<br></br>
> 판다스를 통해 데이터 열어 상위 10 개 데이터 확인 컬럼명 지정
```python
import pandas as pd
import os

fname = os.getenv('HOME') + '/aiffel/recommendata_iu/data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv'
col_names = ['user_id', 'artist_MBID', 'artist', 'play']   # 임의로 지정한 컬럼명
data = pd.read_csv(fname, sep='\t', names= col_names)      # sep='\t'로 주어야 tsv를 열 수 있습니다.  
data.head(10)
```
데이터는 4 개의 컬럼으로 구성되어 있으며, 컬럼은 각 각 User ID, Artist MBID, Artist Name, Play횟수 를 의미한다. 원본 데이터는 컬럼 명이 지정되어 있지 않아 컬럼명을 추가해준다.
<br></br>
> 불필요한 컬럼 삭제
```python
# 사용하는 컬럼만 남겨줍니다.
using_cols = ['user_id', 'artist', 'play']
data = data[using_cols]
data.head(10)
```
<br></br>
> 검색의 편의를 위해 아티스트명을 모두 소문자로 변경
```python
data['artist'] = data['artist'].str.lower() # 검색을 쉽게하기 위해 아티스트 문자열을 소문자로 바꿔줍시다.
data.head(10)
```
<br></br>
> 첫 번째 유저가 어떤 아티스트의 노래를 듣는지 확인
```python
condition = (data['user_id']== data.loc[0, 'user_id'])
data.loc[condition]
```
<br></br>

### 데이터 탐색

추천 모델을 만들기 전에 데이터의 정보를 확인 해야한다. 이렇게 데이터에서 목적에 맞는 정보를 찾아내거나 데이터를 추가적으로 가공하여 목적에 맞는 새로운 데이터를 만들어내는 것은 데이터 탐색이라고 한다.

먼저 다음 3 가지 항목을 확인 해보자.

-   유저수, 아티스트수, 인기 많은 아티스트

-   유저들이 몇 명의 아티스트를 듣고 있는지에 대한 통계

-   유저 play 횟수 중앙값에 대한 통계
<br></br>
> 총 유저의 수를 확인
> (`pandas.DataFrame.nunique()`은 특정 컬럼에 포함된 유니크한 데이터의 개수를 알아보는데 유용하다.)
```python
# 유저 수
data['user_id'].nunique()
```
<br></br>
> 총 아티스트의 수 확인
```python
# 아티스트 수
data['artist'].nunique()
```
<br></br>
> 인기가 많은 아티스트의 수
```python
# 유저별 몇 명의 아티스트를 듣고 있는지에 대한 통계
user_count = data.groupby('user_id')['artist'].count()
user_count.describe()
```
유저 아이디를 통해 많은 유저가 들은 음악의 아티스트를 찾는다.
<br></br>
> 유저 별 플레이 횟수 중앙값에 대한 통계
```python
# 유저별 play횟수 중앙값에 대한 통계
user_median = data.groupby('user_id')['play'].median()
user_median.describe()
```
평균적으로 유저가 얼마나 많은 플레이를 했는지에 대한 통계량을 산출한다. 평균 대신 중앙값을 사용하는 이유는 전체 데이터의 분포를 확인하지 않았기에 특이점의 영향을 받지 않는 평균인 중앙값을 사용한 것이다.
<br></br>

### 모델 검증을 위한 사용자 초기 정보 세팅

유투브 뮤직 등 여러 추천 시스템은 새로운 유입 사용자를 위해 사용자의 취향과 유사한 아티스트 정보를 5개 이상 입력 받는 과정을 거치게 하는 경우가 많다.

이는 사용자가 서비스를 사용하면서 축적되는 데이터가 없을 때 추천시스템을 적용할 수 있게 하기 위해서이다.
<br></br>
> 좋아하는 아티스트를 기존 데이터에 추가
```python
# 본인이 좋아하시는 아티스트 데이터로 바꿔서 추가하셔도 됩니다! 단, 이름은 꼭 데이터셋에 있는 것과 동일하게 맞춰주세요. 
my_favorite = ['black eyed peas' , 'maroon5' ,'jason mraz' ,'coldplay' ,'beyoncé']

# 'zimin'이라는 user_id가 위 아티스트의 노래를 30회씩 들었다고 가정하겠습니다.
my_playlist = pd.DataFrame({'user_id': ['zimin']*5, 'artist': my_favorite, 'play':[30]*5})

if not data.isin({'user_id':['zimin']})['user_id'].any():  # user_id에 'zimin'이라는 데이터가 없다면
    data = data.append(my_playlist)                           # 위에 임의로 만든 my_favorite 데이터를 추가해 줍니다. 

data.tail(10)       # 잘 추가되었는지 확인해 봅시다.
```
<br></br>

### 모델에 활용하기 위한 전처리

모델에 데이터를 활용하기 위해서는 숫자를 이용해야한다.

이렇게 데이터 각각에 번호를 부여하는 과정을 인덱싱 (Indexing) 이라고 한다.
<br></br>
> 사용할 데이터에 인덱싱 작업을 수행
> (`pandas.DataFrame.unique()`은 특정 컬럼에 포함된 유니크한 데이터만 모아 주기때문에 indexing 작업을 위해 매우 유용하다.)
```python
# 고유한 유저, 아티스트를 찾아내는 코드
user_unique = data['user_id'].unique()
artist_unique = data['artist'].unique()

# 유저, 아티스트 indexing 하는 코드 idx는 index의 약자입니다.
user_to_idx = {v:k for k,v in enumerate(user_unique)}
artist_to_idx = {v:k for k,v in enumerate(artist_unique)}
```
<br></br>
> 인덱싱 결과 확인
```python
# 인덱싱이 잘 되었는지 확인해 봅니다. 
print(user_to_idx['zimin'])    # 358869명의 유저 중 마지막으로 추가된 유저이니 358868이 나와야 합니다. 
print(artist_to_idx['black eyed peas'])
```
<br></br>
> 유저 아이디와 아티스트 컬럼에 해당하는 데이터를 모두 인덱싱
```python
# indexing을 통해 데이터 컬럼 내 값을 바꾸는 코드
# dictionary 자료형의 get 함수는 https://wikidocs.net/16 을 참고하세요.

# user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다. 
# 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다. 
temp_user_data = data['user_id'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(data):   # 모든 row가 정상적으로 인덱싱되었다면
    print('user_id column indexing OK!!')
    data['user_id'] = temp_user_data   # data['user_id']을 인덱싱된 Series로 교체해 줍니다. 
else:
    print('user_id column indexing Fail!!')

# artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다. 
temp_artist_data = data['artist'].map(artist_to_idx.get).dropna()
if len(temp_artist_data) == len(data):
    print('artist column indexing OK!!')
    data['artist'] = temp_artist_data
else:
    print('artist column indexing Fail!!')

data
```
<br></br>

## 사용자의 명시적 / 암묵적 평가

흔히 사용자의 취향, 콘텐츠에 대한 사용자의 평가를 알아야 사용자들의 선호도를 알 수 있다.

이러한 선호도를 명시적 (explicit) 으로 표현한 것이 바로 좋아요 혹은 별점 이다.

하지만 우리는 이런 명시적 데이터를 가지고 있지 않기때문에 얼마나 재생을 많이 했는지 암묵적 (implicit) 인 평가를 통해 선호도를 파악 해보자.


일반적으로 암묵적 피드백은 노이즈에 해당하는 부정확한 정보가 많다. 암묵적 피드백은 수치로 신뢰도, 즉 선호도를 판단하는데 만약 특정 콘텐츠가 좋았음에도 한 번만 재생하고, 싫었지만 자리를 비우는 등 여러 이유로 여러번 재생을 할 경우 잘 못 판단할 수 있기 때문이다.

+ 참고 : [암묵적 평가 와 명시적 평가 비교 : Explicit vs. Implicit Feedback Datasets](https://orill.tistory.com/entry/Explicit-vs-Implicit-Feedback-Datasets?category=1066301)
<br></br>
> 데이터셋에서 1 회만 플레이한 데이터의 비율 확인
```python
# 1회만 play한 데이터의 비율을 보는 코드
only_one = data[data['play']<2]
one, all_data = len(only_one), len(data)
print(f'{one},{all_data}')
print(f'Ratio of only_one over all data is {one/all_data:.2%}')  # f-format에 대한 설명은 https://bit.ly/2DTLqYU
```
암묵적 데이터 해석을 위해 다음과 같은 규칙을 적용한다.

1.  한 번이라도 들었으면 선호한다고 판단한다.
2.  많이 재생한 아티스트에 대해 가중치를 주어서 더 확실히 좋아한다고 판단한다.

이 규칙은 사용자의 주관이나 도메인 지식에 따라 바뀔 수 있다.
<br></br>
## Matrix Factorization (MF)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-3v2-2_ekCv9hW.max-800x600.png)
<br></br>
위 그림은 추천시스템 모델 MF (Matrix Factorization) 의 개요로 m 명의 사용자들이 n 명의 아티스트에 대해 평가한 데이터를 포함한 (m, n) 사이즈의 평가행렬 (Rating Matrix) 을 만들어야 한다.

Rating Maxtrix R 을 보면 행렬 중 일부 데이터는 채워져 있지만 일부는 비어있다.

이렇게 비어 있는 부분을 포함한 완벽한 정보를 얻을 수 있다면 완벽한 추천 시스템이 되겠지만 실제로는 쉽지 않은 일이다.

이렇게 추천시스템의 `협업 필터링 (Collaborative Filtering)` 이란 결국은 이런 평가행렬을 전제로 하는 것이다.

추천시스템의 모델은 다양한데, 그 중 `Matrix Factorization(MF, 행렬분해)` 모델을 사용하고자 한다.

MF 모델은 위 그림와 같이 (m, n) 사이즈의 행렬 R 을 (m, k) 사이즈의 행렬 P 와 (k, n) 사이즈의 행렬 Q 로 분해한다면 R 이란 그저 P 와 Q 의 행렬곱으로 표현 가능할 수 있다는 아이디어로 구현된 모델이다.

대체로 k 는 m 이나 n 보다 훨씬 작은 값이기 때문에 계산량 측면으로도 훨씬 유리하다. 단순한 아이디어임에도 성능과 확장성 (Scalability) 이 뛰어나 널리 사용된다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-3v2-3.max-800x600.png)
<br></br>
위 그림은 MF 모델의 개요를 영화 추천 시스템에 대입하여 나타낸 것이다. 즉, m = 4, n = 5, k = 2 인 MF 모델이다.

MF 모델은 큰 평가행렬 R 을 두 개의 Feature Matrix P 와 Q 로 분해한다. 이때 Feature 는 (m, k) 사이즈의 Feature Matrix P 는 k 차원의 벡터를 사용자 수만큼 모아놓은 행렬을 의미한다.

그림의 첫 번째 벡터 $P_0=(1, 0.1)$ 은 바로 빨간 모자를 쓴 첫 번째 사용자의 특성(Feature) 벡터가 된다.

Q 행렬의 첫번째 벡터 $Q_0 = (0.9, -0.2)$ 는 해리포터 영화의 특성 벡터 가 된다.

이 두개의 벡터를 내적하여 얻어지는 0.88 이 $R_{0, 0}$ 으로 정의되는 사용자의 영화 선호도로 보는 모델이다.

추천시스템 모델의 목표는 모든 유저와 아이템에 대해 k - dimension 의 벡터를 잘 만드는 것이며, 벡터를 잘 만드는 기준은 유저 i 의 벡터 ($U_i$) 와 아이템 j 의 벡터 ($I_j$) 를 내적했을 때 유저 i 가 아이템 j 에 대해 평가한 수치 ($M_ij$)와 비슷한지이다.

이를 수식으로 표현하면 $U_i \cdot I_j = M_{ij}$ 와 같다.

우리가 추천시스템에 적용할 모델은 MF 모델의 변형 모델로 Collaborative Filtering for Implicit Feedback Datasets 논문에서 소개한 모델을 적용하고자 한다.

모델의 목표는 앞서 입력한 선호 아티스트의 벡터와 사용자의 벡터를 곱했을 떄 1 에 가깝게 하는 것이다.

이때 해당 아티스트를 몇 번을 듣던지에 관계없이 두 벡터를 곱했을 때 1 에 가까워져야 한다.

이를 통해 다른 아티스트와의 사용자 벡터를 곱하여 수치를 예상할 수 있다.

+ 참고 : 
	+ [논문 : Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
	+ [# Recommender System — Matrix Factorization](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b)
<br></br>

## CSR (Compressed Sparse Row) Matrix

유저와 아이템 간의 평가행렬은 양이 많을 수록 엄청난 컴퓨팅 자원을 필요로 하게된다.

유저는 36 만 명이고 아티스트는 29 만 명 일때, 행렬로 표현하고 행렬의 각 원소에 정수 한 개 (1 byte) 가 들어간다면 36 만 * 29 만 * 1 byte ≈≈  97 GB가 필요하게 된다.

평가행렬의 용량이 커지는 이유는 유저 수 X 아티스트 수만큼의 정보 안에는 유저가 들어보지 않은 아티스트에 대한 정보까지 모두 행렬에 포함되어 계산되기 때문이다.

이때 사용자들은 모든 아티스트를 알지 못하거나 플레이해 본 적이 없을 것이다. 따라서 총 아티스트는 29 만 명이 넘기 때문에 평가행렬 내의 대부분의 공간은 0 으로 채워진다. 이러한 행렬을 Sparse Matrix 라고 한다.

이런 메모리 낭비를 최소화하기 위해서는 유저가 들어본 아티스트에 대해서만 정보만을 저장하면서 전체 행렬 형태를 유추할 수 있는 데이터 구조가 필요하다.

이때 좋은 대안이 되는 것이 CSR (Compressed Sparse Row Matrix) 이다.

CSR Matrix 는 Sparse 한 matrix 에서 0 이 아닌 유효한 데이터로 채워지는 데이터의 값과 좌표 정보만으로 구성하여 메모리 사용량을 최소화하면서도 Sparse한 matrix 와 동일한 행렬을 표현할 수 있도록 하는 데이터 구조이다.

+ 참고 :
	+ [Scipy sparse matrix handling](https://lovit.github.io/nlp/machine%20learning/2018/04/09/sparse_mtarix_handling/#csr-matrix)  
	+ [StackOverflow csr_matrix 설명](https://stackoverflow.com/a/62118005)
<br></br>

이제 CSR Matrix 에 맞게 데이터를 변형해보자. CSR Matrix 를 만드는 방법은 다양하다.

+ 참고 : [CSR Maxtrix 생성 방법](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
<br></br>
> CSR Matrix 생성 구조
```python
csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
where  data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k]., M,N은 matrix의 shape
```
<br></br>
> CSR Matrix 생성
```python
# 실습 위에 설명보고 이해해서 만들어보기
from scipy.sparse import csr_matrix

num_user = data['user_id'].nunique()
num_artist = data['artist'].nunique()

csr_data = csr_matrix((data.play, (data.user_id, data.artist)), shape= (num_user, num_artist))
csr_data
```
<br></br>
## MF 모델 학습하기

MF 모델을 `implicit` 패키지를 사용하여 학습해보자.

`implicit` 패키지는 이전 스텝에서 설명한 암묵적(implicit) dataset 을 사용하는 다양한 모델을 굉장히 빠르게 학습할 수 있는 패키지이다. 

이 패키지에 구현된 `als(AlternatingLeastSquares) 모델` 을 사용한다. `Matrix Factorization` 에서 쪼개진 두 Feature Matrix 를 한꺼번에 훈련하는 것은 잘 수렴하지 않기 때문에, 한쪽을 고정시키고 다른 쪽을 학습하는 방식을 번갈아 수행하는 Alternating Least Squares 방식이 효과적이다.

+ 참고 : [implicit 패키지](https://github.com/benfred/implicit)
<br></br>
> `implicit` 패키지의 AlternatingLeastSquares 가져오기
```python
from implicit.als import AlternatingLeastSquares
import os
import numpy as np

# implicit 라이브러리에서 권장하고 있는 부분입니다. 학습 내용과는 무관합니다.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS']='1'
```
AlternatingLeastSquares 클래스의 __init__ 파라미터를 살펴보면

1. factors : 유저와 아이템의 벡터를 몇 차원으로 할 것인지를 의미
2. regularization : 과적합을 방지하기 위해 정규화 값을 얼마나 사용할 것인지
3. use_gpu : GPU를 사용할 것인지
4. iterations : epochs와 같은 의미로 데이터를 몇 번 반복해서 학습할 것인지를 의미한다.
<br></br>
> 과적합 방지를 위해 적당한 값을 찾기 및 모델 선언
```python
# Implicit AlternatingLeastSquares 모델의 선언
als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False, iterations=15, dtype=np.float32)
```
> 모델 입력을 위해 데이터 변형
```python
# als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
csr_data_transpose = csr_data.T
csr_data_transpose
```
<br></br>
> 모델 훈련
```python
# 모델 훈련
als_model.fit(csr_data_transpose)
```
<br></br>
> 모델이 유저의 벡터와 아티스트의 벡터를 어떻게 만들고 있는지
```python
zimin, black_eyed_peas = user_to_idx['zimin'], artist_to_idx['black eyed peas']
zimin_vector, black_eyed_peas_vector = als_model.user_factors[zimin], als_model.item_factors[black_eyed_peas]
```
<br></br>
> 유저와 아티스트 벡터 값 확인
```python
print(zimin_vector)

print(black_eyed_peas_vector)
```
<br></br>
> 유저와 아티스트 벡터의 곱 확인
```python
# zimin과 black_eyed_peas를 내적하는 코드
np.dot(zimin_vector, black_eyed_peas_vector)
```
0.49 라는 낮은 수치가 나왔다. 이때 수치를 항샹시키기 위해서는 factors 를 늘리거나 iterations를 늘리면 된다.

하지만 1 이 나왔다고 해도 과적합을 고려해야하므로 검증이 필요하다.
<br></br>
> 유저가 queen 에 대한 선호도를 예측
```python
queen = artist_to_idx['queen']
queen_vector = als_model.item_factors[queen]
np.dot(zimin_vector, queen_vector)
```
결과를 해당 유저만이 알고 있다. 이는 좋은 모델을 위해서는 데이터에 대한 이해도가 높아야 하는데 추천시스템에서 사용하는 데이터는 사람에 대한 데이터이기 때문이다.
<br></br>
## 비슷한 아티스트 찾기 + 유저에게 추천하기

### 비슷한 아티스트 찾기

> 비슷한 아티스트 찾기
> (`AlternatingLeastSquares` 클래스에 구현되어 있는 `similar_items` 메서드 활용 )
```python
favorite_artist = 'coldplay'
artist_id = artist_to_idx[favorite_artist]
similar_artist = als_model.similar_items(artist_id, N=15)
similar_artist
```
(아티스트의 id, 유사도) Tuple 로 반환한다.
<br></br>
> 아티스트의 id 를 아티스트의 이름로 매핑
```python
#artist_to_idx 를 뒤집어, index로부터 artist 이름을 얻는 dict를 생성합니다. 
idx_to_artist = {v:k for k,v in artist_to_idx.items()}
[idx_to_artist[i[0]] for i in similar_artist]
```
<br></br>
> 위 과정을 반복하기 위한 함수 생성
```python
def get_similar_artist(artist_name: str):
    artist_id = artist_to_idx[artist_name]
    similar_artist = als_model.similar_items(artist_id)
    similar_artist = [idx_to_artist[i[0]] for i in similar_artist]
    return similar_artist
```
<br></br>
> 유사한 아티스트 찾아보기
```python
get_similar_artist('2pac')
```
마니아들은 특정 장르의 아티스트들에게로 선호도가 집중되고, 다른 장르의 아티스트들과는 선호도가 낮게 나타난다. 

마니아들의 존재로 인해 같은 장르의 아티스트들의 벡터들도 더 가까워져서 get_similar_artist 시 장르별 특성이 두드러지게된다.
<br></br>
> 다른 아티스트의 유사 아티스트 찾아보기
```python
get_similar_artist('lady gaga')
```
여자 아티스트들이 추천됨을 확인할 수 있다.
<br></br>

### 유저에게 아티스트 추천하기

> 유저에게 좋아할만한 아티스트를 추천받기
> (`AlternatingLeastSquares` 클래스에 구현되어 있는 `recommend` 메서드 활용)
> (`filter_already_liked_items` 는 유저가 이미 평가한 아이템은 제외하는 Argument 이다.)
```python
user = user_to_idx['zimin']
# recommend에서는 user*item CSR Matrix를 받습니다.
artist_recommended = als_model.recommend(user, csr_data, N=20, filter_already_liked_items=True)
artist_recommended
```
<br></br>
> 추천받은 아티스트를 통해 유저에게 아티스트 추천하기
```python
[idx_to_artist[i[0]] for i in artist_recommended]
```
<br></br>
> 추천에 기여한 정도를 확인
> (`AlternatingLeastSquares` 클래스에 구현된 `explain` 메소드 활용)
```python
rihanna = artist_to_idx['rihanna']
explain = als_model.explain(user, csr_data, itemid=rihanna)
```
`AlternatingLeastSquares` 클래스에 구현된 `explain` 메소드를 사용하면 제가 기록을 남긴 데이터 중 이 추천에 기여한 정도를 확인할 수 있다.

`explain` 메소드는 추천한 콘텐츠의 점수에 기여한 다른 콘텐츠와 기여토를 반환한다. (두 점수의 합이 최종 점수가 된다.)
<br></br>
> 어떤 아티스트 들이 추천에 기여하고 있는지 확인
```python
[(idx_to_artist[i[0]], i[1]) for i in explain[1]]
```
<br></br>
모델이 추천한 것들 중 실제로 선호하는지 계산하여 모델의 객관적인 지표를 만들 수 있다.

+ 참고 : [추천시스템 평가 방법](https://danthetech.netlify.app/DataScience/evaluation-metrics-for-recommendation-system)

