# 02. Iris 의 세 가지 품종, 분류해볼 수 있겠어요?


### 학습 전제

-   scikit-learn을 활용해서 머신러닝을 시도해본 적이 없다.
-   scikit-learn에 내장되어 있는 분류 모델을 활용해본 적이 없다.
-   지도학습의 분류 실습을 해 본 적이 없다.
-   머신러닝 모델을 학습시켜보고, 그 성능을 평가해본 적이 없다.

### 학습 목표

-   scikit-learn에 내장되어 있는 예제 데이터셋의 종류를 알고 활용할 수 있다.
-   scikit-learn에 내장되어 있는 분류 모델들을 학습시키고 예측해 볼 수 있다.
-   모델의 성능을 평가하는 지표의 종류에 대해 이해하고, 활용 및 확인해 볼 수 있다.
-   Decision Tree, XGBoost, RandomForest, 로지스틱 회귀 모델을 활용해서 간단하게 학습 및 예측해 볼 수 있다.
-   데이터셋을 사용해서 스스로 분류 기초 실습을 진행할 수 있다.

<br></br>
## How to make?

일반적으로 딥러닝 기술은 

1. __데이터 준비__
2. __딥러닝 네트워크 설계__
3. __학습__ 
4. __테스트 (평가)__

의 과정을 따르게 된다.

## 붓꽃 분류 문제

> 사이킷런 (scikit-learn) / matplotlib 패키지 설치
```python
pip install scikit-learn  
pip install matplotlib
```

### 사용할 데이터

**사이킷런 (scikit-learn)** 은 가장 많이 사용되는 머신러닝 라이브러리이며 많은 데이터가 내장되어 있다.

최근 TensorFlow, PyTorch 등 딥러닝에 특화된 라이브러리들이 주로 사용되지만, 머신러닝의 다양한 알고리즘과 프레임워크를 편리하게 제공하는 Scikit-learn 역시 많이 사용된다.

[Scikit-learn 의 내장 데이터셋](https://scikit-learn.org/stable/datasets.html)

사이킷런에서는 Toy datasets 과 Real world datasets 2가지를 제공한다.

+ Toy datasets : boston, iris, diabetes, digits, linnerud, wine, breast cancer의 7가지 데이터셋을 제공

+ Real world datasets : olivetti faces, 20newsgroups, covtype, california housing 등 총 9가지 데이터셋을 제공

## 데이터 준비, 그리고 자세히 살펴보기는 기본!

> 데이터셋 가져오기
> `sklearn` 라이브러리의 `datasets` 패키지 안에 있다.
```python
from sklearn.datasets import load_iris

iris = load_iris()

print(type(dir(iris))) 
# dir()는 객체가 어떤 변수와 메서드를 가지고 있는지 나열함
```
<br></br>
> `iris` 에 담겨있는 정보 확인
```python
iris.keys()
```
`data`, `target`, `frame`, `target_names`, `DESCR`, `feature_names`, `filename` 까지 총 6개의 정보가 담겨 있는 것을 확인할 수 있다.
<br></br>
> `iris_data` 변수에 저장 후, 데이터의 크기 확인
```python
iris_data = iris.data

print(iris_data.shape) 
#shape는 배열의 형상정보를 출력
```
<br></br>
> 하나의 데이터를 확인
```python
iris_data[0]
```
순서대로 `sepal length`, `sepal width`, `petal length`, `petal width` 를 나타냄을 확인 가능하다.
<br></br>

붓꽃의 세 가지 품종 분류를 위한 아이디어로 꽃잎과 꽃받침의 길이 정보를 통해서 품종을 구분하고자 한다.

따라서 앞으로 만들 모델은 꽃잎, 꽃받침의 길이 정보가 입력되었을 때 해당하는 품종을 출력하도록 구현해야 한다.

이렇게 출력해야하는 정답을 라벨 (label) 또는 타겟 (target) 라고 한다.
<br></br>
> Iris 데이터에서 라벨 정보 확인 
> (`target` 메서드 사용)
```python
iris_label = iris.target
print(iris_label.shape)
iris_label
```
라벨에 해당하는 데이터를 `iris_label` 변수에 저장하였다.
<br></br>
> 라벨의 이름 확인 
> (`target_names` 메서드 사용)
```python
iris.target_names
```
`setosa`, `versicolor`, `virginica` 순서로 `0`이라면 `setosa`, `1`이라면 `versicolor`, `2`라면 `virginica`를 나타낸다.
<br></br>
> 다른 변수들에 대한 설명 확인
> (`DESCR` 메서드 사용)
```python
print(iris.DESCR)
```
<br></br>
> 데이터셋 파일이 저장된 경로 확인
> (`filename` 메서드 사용)
```python
iris.filename
```
<br></br>

## 머신러닝 모델을 학습시키기 위한 문제지와 정답지 준비

`pandas` : 판다스라는 라이브러리는 2차원 배열 데이터 및 표 데이터를 다룰 수 있는 라이브러리로 데이터 분석 및 대형 데이터의 여러 통계량에 최적화 되어 있다.

`irsi` 데이터는 행과 열로 구성된 2차원 데이터이므로 판다스를 통해 다룬다.

> 판다스 라이브러리 불러오기
```python
pip install pandas

import pandas as pd

print(pd.__version__)
```
굉장히 자주 사용되는 라이브러리로 `pd` 라는 약어로 많이 사용한다.
<br></br>
> `iris` 데이터셋을 판다스의 `DataFrame` 자료형으로 변환
```python
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df
```
`DataFrame` 을 만들면서 `data`에는 `iris_data`를 넣어주고, 각 컬럼에는 `feature_names`로 이름 지정.
<br></br>
> 정답에 해당하는 라벨 데이터를 새로운 컬럼에 추가
```python
iris_df["label"] = iris.target
iris_df
```
<br></br>

`iris.feature_names` 에 해당하는 4가지 특징들은 모델이 풀어야하는 문제지와 같으며, 라벨 데이터는 해당 문제들의 정답지와 같다.

+ 문제지 : 머신러닝 모델에게 입력되는 데이터. **feature**라고 부르기도 한다. 변수 이름으로는 `X`를 많이 사용

+ 정답지 : 머신러닝 모델이 맞추어야 하는 데이터. **label**, 또는 **target**이라고 부르기도 한다. 변수 이름으로는 `y`를 많이 사용

머신러닝 모델 학습을 위해서는 **학습에 사용하는 training dataset**과 **모델의 성능을 평가하는 데 사용하는 test dataset**으로 데이터셋을 나누는 작업이 필요하다.

> 데이터셋을 학습용 데이터와 테스트용 데이터로 분리
> (Scikit-learn 의 `train_test_split` 매서드를 사용)
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))
```
첫 번째 파라미터인 `iris_data`는 문제지, 즉 feature 를 의미하며, 품종 분류를 위한 입력 데이터이다.

두 번째 파라미터인 `iris_label`은 모델이 맞추어야 하는 정답값, 즉 label 이다.

이렇게 학습용 데이터와 테스트용 데이터를 생성하고, 각 데이터에서 4개의 필요한 특징 데이터만 있는 `X`, 정답 데이터만 있는 `y` 를 얻을 수 있다.

`X` 데이터셋을 머신러닝 모델에 입력하고, 그에 따라 모델이 내뱉는 품종 예측 결과를 정답인 `y`와 비교하며 점차 정답을 맞추어나가도록 학습을 진행할 예정이며, 세 번째 인자인 `test_size`로는 test dataset의 크기를 조절할 수 있다.

0.2는 전체의 20%를 테스트 데이터로 사용하겠다는 것을 나타낸다.

마지막 인자인 `random_state`는 `train`데이터와 `test`데이터를 분리(split)하는데 적용되는 랜덤성을 결정하며, `random_seed` 로도 사용된다. 어떠한 값을 입력해도 상관없다.
<br></br>
> 생성된 `X_train, y_test` 데이터셋 확인
```python
X_train.shape, y_train.shape

X_test.shape, y_test.shape
```
<br></br>
> 정답 데이터, `y` 확인
```python
y_train, y_test
```
처음 확인했던 `label` 과 달리 무작위로 섞여 있는 것을 확인할 수 있다.
<br></br>

## 머신러닝 모델 학습 시키기

머신러닝은 크게 **지도학습 (Supervised Learning)**, **비지도 학습 (Unsupervised Learning)**이라는 두 가지로 구분된다. 또한 지도학습은 **분류(Classification)**와 **회귀(Regression)** 로 나눌 수 있다.

+ **지도학습 (Supervised Learning)** : 지도받을 수 있는, 즉 **정답이 있는** 문제에 대해 학습하는 것
	+ **분류(Classification)** : 입력받은 데이터를 특정 카테고리 중 하나로 분류해내는 문제
		+ 환자의 나이, 병력, 혈당 등을 입력받아 암의 양성/음성을 판정하는 문제 등

	+ **회귀(Regression)** : 입력받은 데이터에 따라 특정 필드의 수치를 맞추는 문제
		+ 택시를 탄 시각, 내린 시각, 출발지, 도착지, 거리 등을 입력받아 택시 요금을 맞추는 문제 등

+ **비지도 학습 (Unsupervised Learning)** : **정답이 없는** 문제를 학습하는 것

붓꽃 문제는
-   첫 번째, 머신러닝 중 정답이 있고 그 정답을 맞추기 위해 학습하는  **지도 학습(Supervised Learning)**이며,
-   지도학습 중에서는 특정 카테고리 중 주어진 데이터가 어떤 카테고리에 해당하는지를 맞추는  **분류(Classification)**  문제

에 해당한다.

분류 문제에 사용되는 모델은 다양하지만 Decision Tree  모델을 사용하기로 한다.

+ [Decision Tree 모델](https://ratsgo.github.io/machine%20learning/2017/03/26/tree/) : 
	+ 의사결정나무, 데이터를 분석하여 이들 사이에 존재하는 패턴을 예측 가능한 규칙들의 조합으로 나타낸다.
	+ 결정경계가 데이터 축에 수직이어 특정 데이터에만 작동할 가능성이 높다는 단점이 있다. 
	+ 이를 극복하기 위한 모델은 '랜덤포레스트'이며, 같은 데이터에 대해 의사결정나무를 여러개 만들어 그 결과를 종합해 예측 성능을 높이는 기법이다.

Decision Tree는 `sklearn.tree` 패키지 안에 `DecisionTreeClassifier` 라는 이름으로 내장되어 있다.
<br></br>
> `decision_tree` 라는 변수에 모델을 저장
 ```python
 from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)
 ```
 <br></br>
 > 모델 학습
 > (`fit` 매서드 사용)
 ```python
 decision_tree.fit(X_train, y_train)
```
training dataset 으로 모델을 학습시킨다는 것은, 달리 말하면 training dataset에 맞게 모델을 fitting, 즉 맞추는 것이며, training dataset에 있는 데이터들을 통해 어떠한 패턴을 파악하고, 그 패턴에 맞게 예측을 할 수 있도록 학습한다.

모델은 training dataset에 존재하지 않는 데이터에 대해서는 정확한 정답 카테고리가 무엇인지 알지 못합니다.  
다만 training dataset을 통해 학습한 패턴으로 새로운 데이터가 어떤 카테고리에 속할지 예측할 뿐이다.

때문에 새로운 데이터에 대해 잘 예측하기 위해서는 training dataset 을 어떻게 구성하느냐가 가장 큰 관건이 된다.
<br></br>

## 머신러닝 모델 평가하기

> `test` 데이터를 통핸 예측
```python
y_pred = decision_tree.predict(X_test)
y_pred
```
`X_test` 데이터에는 정답인 label이 없고 feature 데이터만 존재한다.

학습이 완료된 모델에 `X_test` 데이터로 `predict`를 실행하면 모델이 예측한 `y_pred`을 얻게된다.
<br></br>

> 모델의 예측 값과 실제 정답을 `y_test` 와 비교
```python
y_test
```
이렇게 답지를 눈으로 비교하는 것보다 예측한 결과에 대한 수치를 조금 더 편리하게 확인할 수 있는 방법으로 scikit-learn에서 성능 평가에 대한 함수들이 모여있는 `sklearn.metrics` 패키지를 이용한다.
<br></br>

> 모델 성능을 평가하는 척도인 정확도 (Accuracy) 를 확인
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy
```
0. 9 라는 수치로 출력되며, 확률값을 의미한다. 즉, 전체 개수 중 맞은 것의 개수를 의미한다.
<br></br>
$정확도 = \frac{예측\ 결과가\ 정답인\ 데이터의\ 개수}{예측한\ 전체\ 데이터의\ 개수}$
<br></br>

## 다른 모델도 해보고 싶다면? 코드 한줄만 바꾸면 돼!

다른 모델을 사용하고 싶다면 scikit-learn 에서 제공하는 모델을 바꿔주기만 하면 된다.

먼저 Decision Tree 모델을 학습시키고 예측하는 과정을 정리하면 다음과 같다.
<br></br>
> Decision Tree 모델 학습 및 예측
```python
# (1) 필요한 모듈 import
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

# (3) train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))
```
다른 모델로 바꿀 때는 `(4) 모델 학습 및 예측` 부분에서 모델만 바꿔주면 된다.
<br></br>

Decision Tree 모델 외에 RandomForest 모델을 사용해 보자.

[참고 : RandomForest](https://medium.com/@deepvalidation/title-3b0e263605de)

먼저, RandomForest 모델은 Decision Tree 모델을 여러개 합쳐놓음으로써 Decision Tree의 단점을 극복한 모델이며, 이러한 여러 기법을 합쳐 사용하는 것을 **앙상블(Ensemble)** 기법이라고 한다. 단일 모델을 여러 개 사용하는 방법을 취함으로써 모델 한 개만 사용할 때의 단점을 집단지성으로 극복하는 개념이다.

RandomForest는 `sklearn.ensemble` 패키지 내에 들어있다.
<br></br>
> RandomForest 모델을 통한 학습 및 예측
```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(iris_data, 
                                                    iris_label, 
                                                    test_size=0.2, 
                                                    random_state=25)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
```
<br></br>


### 다른 Scikit-learn 내장 분류모델

1. #### [Support Vector Machine (SVM)](https://excelsior-cjh.tistory.com/66?category=918734)

> SVM 을 사용한 모델 학습 및 예측
```python
from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))
```
<br></br>

2. #### [Stochastic Gradient Descent Classifier (SGDClassifier)](https://scikit-learn.org/stable/modules/sgd.html)

> SGD Classifier 을 사용한 모델 학습 및 예측
```python
from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))
```
<br></br>

3. #### [Logistic Regression](http://hleecaster.com/ml-logistic-regression-concept/)

> Logistic Regression 을 통한 모델 학습 및 예측
```python
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)

from sklearn.metrics import classification_report
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))
```
<br></br>


## 내 모델은 얼마나 똑똑한가? 다양하게 평가해보기. 

정화도라는 척도를 통해 모델의 성능이 얼마나 뛰어난지 평가했었다.

하지만 모델을 평가하는 척도는 여러가지가 있다.

### 정확도에는 함정이 있다.

정확도라는 모델 평가 측도에는 치명적인 단점이 존재한다.

먼저 손글씨 데이터인 MNIST 데이터셋으로 확인해 보자.

> 손글씨 데이터 가져오기
```python
from sklearn.datasets import load_digits

digits = load_digits()
digits.keys()
```
<br></br>
> `data` 확인
```python
digits_data = digits.data
digits_data.shape
```
<br></br>
> 샘플을 통한 `data` 확인
```python
digits_data[0]
```
숫자로 이루어진 배열(array)이 출력되었다. 이 배열은 숫자 이미지를 나타내는 픽셀값을 의미한다. 해당 배열을 시각화를 통해 이미지로 확인 해 보자.
<br></br>
> 출력된 배열을 시각화를 통해 이미지로 확인
> (`matplotlib.pyplot` 라이브러리 사용, 이미지를 현재 화면에 보여주기 위해 `%matplotlib inline` 코드 추가)
```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(digits.data[0].reshape(8, 8), cmap='gray')
plt.axis('off')
plt.show()
```
일렬로 펴진 64개 데이터를 (8, 8)로 `reshape`해주는 것을 잊어선 안된다.
<br></br>
> 여러 개의 이미지를 한 번에 확인
```python
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.data[i].reshape(8, 8), cmap='gray')
    plt.axis('off')
plt.show()
```
0~9 까지의 숫자 확인이 가능하다.
<br></br>
> `target` 데이터 확인
```python
digits_label = digits.target
print(digits_label.shape)
```
<br></br>

이제 정확도의 단점이자 함정을 실험하기 위해 바로, 숫자 10개를 모두 분류하는 것이 아니라, 해당 이미지 데이터가 **3인지 아닌지**를 맞추는 문제로 변형해서 풀어보자.
<br></br>
> `target`인 `digits_label`을 3 이라는 값으로 변경
```python
new_label = [3 if i == 3 else 0 for i in digits_label]
new_label[:20]
```
기존의 `label`인 `digits_label`에서 숫자가 3이라면 그대로 3을, 아니라면 0을 가지는 `new_label`을 생성한다.
<br></br>
> digits_data와 new_label로 Decision Tree 모델을 학습시키고, 정확도를 확인
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits_data, new_label, test_size = 0.2, random_state = 32)

decision_tree = DecisionTreeClassifier(random_state = 7)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)

a = accuracy_score(y_test, y_pred)
print(a)
```
결과를 보면 굉장히 높은 성능을 보이고 있음을 알 수 있다. 하지만 델이 전혀 학습하지 않고 **정답을 모두 0으로만 선택해도 정확도가 90%가량이 나오게 된다는 것**을 알 수 있다.
<br></br>
> 길이는 `y_pred`와 같으면서 `0`으로만 이루어진 리스트를 `fake_pred` 라는 변수로 저장해 보고, 이 리스트와 실제 정답인 `y_test`간의 정확도를 확인
```python
fake_pred = [0] * len(y_pred)

accuracy = accuracy_score(y_test, fake_pred)
accuracy
```
답을 0으로만 찍었을 뿐인데, 높은 정확도를 보임을 알 수 있다. 이러한 문제는 불균형한 데이터, unbalanced 데이터에서 많이 발생할 수 있다.

이를 통해 정확도만을 통해 모델의 성능을 평가하는 것은 너무나 큰 취약점이 있음을 알 수 있다.
<br></br>

### 정답과 오답에도 종류가 있다.

정확도는 물론 중요하고, 널리 사용되는 평가 척도이지만 얼마나 안 틀렸느냐도 중요한 경우가 있다.

예를 들어, 코로나 바이러스가 의심되는 환자를 진단하는 경우, 실제 코로나에 걸리지 않았지만 걸린 것으로 오진을 한다면 비교적 다행이다. 

하지만 실제 코로나에 걸렸는데 걸리지 않았다고 오진을 하는 경우는 환자에게 치명적인 상황이 된다.

이러한 정답과 오답을 구분하여 표현하는 방법을 **오차행렬 (confusion matrix)** 이라고 한다.

[참고 : **오차행렬 (confusion matrix)**](https://manisha-sirsat.blogspot.com/2019/04/confusion-matrix.html)

오차행렬은 **TP ( True Positive), FN(False Negative), FP(False positive), TN(True Negative)** 4가지로 구분되며, 오차행렬에서 나타나는 성능 지표는 **Sensitivity, Specificity(True Negative Rate), Precision, Accuracy, F1 score** 5가지가 있다.

코로나 진단의 예에서 오차행렬은 

-   TP(True Positive) : 실제 환자에게 양성판정 (참 양성)
-   FN(False Negative) : 실제 환자에게 음성판정 (거짓 음성)
-   FP(False Positive) : 건강한 사람에게 양성판정 (거짓 양성)
-   TN(True Negative) : 건강한 사람에게 음성판정 (참 음성)

![오차행렬](https://aiffelstaticprd.blob.core.windows.net/media/images/E-2-3.max-800x600_mMmzi4T.jpg)

이러한 TP, FN, FP, TN의 수치로 계산되는 성능 지표 중 대표적으로 쓰이는 것은 **정밀도(Precision), 재현율(Recall, Sensitivity), F1 스코어(f1 score)** 이다.

Recall은 위 그림에서 Sensitivity라고 표시된 지표와 같다.

Precision과 Recall, 그리고 F1 score, 그리고 원래 확인했던 정확도까지 수식은 각각 다음과 같다.

$$Precision = \frac{TP}{FT + TP}$$
$$Recall = \frac{TP}{FN + TP}$$
$$F1score = \frac{2}{\frac{1}{Recall}+\frac{1}{Precision}}$$
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Precision과 Recall의 분자는 둘 다 $TP$입니다. $TP$는 맞게 판단한 양성이므로, 이 값은 높을수록 좋다. 하지만 분모에는 각각 $FP$와 $FN$가 있으며 이 값은 낮을수록 좋다.

Precision은 분모에 있는 $FP$가 낮을수록 커진다. Precision이 높아지려면 False Positive, 즉 음성인데 양성으로 판단하는 경우가 적어야 한다.

Recall은 분모에 있는 $FN$이 낮을수록 커진다. Recall이 높아지려면 False Negative, 즉 양성인데 음성으로 판단하는 경우가 적어야 한다.

F1 score은 Recall과 Precision의 조화평균이다.

* $조화평균 = \frac{1}{\frac{1}{X_1}, \cdots, \frac{1}{X_n}}$

Scikit-learn 에서 오차행렬은 `sklearn.metrics` 패키지 내의 `confusion_matrix`로 확인할 수 있다.
<br></br>
> Scikit-learn 에서의 오차행렬 확인
```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
```
오차행렬과 같이 각각은 왼쪽 위부터 순서대로 $TP$, $FN$, $FP$, $TN$ 의 개수를 나타낸다. 

손글씨 문제에서의 `0`은 Positive 역할을, `3`은 Negative 역할을 한다. 
<br></br>

> 모든 숫자를 0 으로 예측한 `fake_pred` 의 오차행렬 확인
```python
confusion_matrix(y_test, fake_pred)
```
든 데이터를 **0**, 즉 Positive로 예측했고 Negative로 예측한 것은 없기 때문에 $FN$ 과 $TN$은 둘 다 0 이다.
<br></br>
> 손글씨 모델의 Precision, Recall, f1 score 확인
> (`sklearn.metrics`의 `classification_report` 매서드 사용)
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```
`0`은 개수가 333개로 많기 때문에 `precision`과 `recall`에서 모두 0.97, 0.96으로 어렵지 않게 높은 점수를 받았지만 `3`은 27개 뿐이기 때문에 모두 맞추기가 어려웠나봅니다. `precision`과 `recall`은 각각 0.58, 0.67 로 비교적 낮은 점수를 받았다.
<br></br>
>모든 숫자를 0 으로 예측한 `fake_pred` 의 Precision, Recall, f1 score 확인
> (`sklearn.metrics`의 `classification_report` 매서드 사용)
```python
print(classification_report(y_test, fake_pred))
```
`0` 에 대한 precision과 recall은 0.93, 1 로 매우 높지만 `3` 에 대한 precision 과 recall 은 둘 다 0 이다. 즉, 하나도 맞추지 못했다는 것이다.

이를 통해 모델의 성능은 정확도만으로 평가하면 안되며, 특히, **label이 불균형하게 분포되어있는 데이터**를 다룰 때에는 더 조심해야 한다.


# PROJECT

## 데이터가 달라도 문제 없어요!

사이킷 런에서 제공하는 새로운 데이터셋을 활용해 머신러닝 모델을 만들고 분류를 해보자.

-   데이터셋 소개 :  [사이킷런 toy datasets](https://scikit-learn.org/stable/datasets/index.html#toy-**datasets**)


사용할 데이터 셋 : 
-   `load_digits`  : [손글씨 이미지 데이터](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)
    
-   `load_wine`  : [와인 데이터](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)
    
-   `load_breast_cancer`  : [유방암 데이터](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)

