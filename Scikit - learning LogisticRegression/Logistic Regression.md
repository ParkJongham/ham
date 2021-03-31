# Logistic Regression

[DcisionTree 를 활용한 IRIS 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20LogisticRegression/LogisticRegression_IRIS_Classification.ipynb)
[DcisionTree 를 활용한 와인 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20LogisticRegression/LogisticsRegression_Wine_Classification.ipynb)
[DcisionTree 를 활용한 유방암 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20LogisticRegression/LogisticsRegression_breast_cance_Classification.ipynb)
[DcisionTree 를 활용한 손글씨 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20LogisticRegression/LogisticsRegression_Digits_Classification.ipynb)

회귀를 통해 데이터가 어떤 범주에 속할 확률을 0~1 사이 확률 값으로 예측하고 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류하는 알고리즘.

- 데이터가 특정 범주에 속할 확률을 예측하는 방법

	1. 모든 속성 (feature) 들의 계수 (coefficient)와 절편 (intercept) 을 0으로 초기화 한다.
	2. 각 속성들의 값 (value) 에 계수를 곱하여 log_odds 를 구한다.
	3. log_odds 를 sigmoid 함수에 넣어 [0, 1] 범위의 확률을 구한다.

- log-odds : odds 에 log 를 취한 것.
	- $odds = \frac{사건이 발생할 확률}{사건이 발생하지 않을 확률}$

- 로그 손실 (Log Loss) : 모델의 적합성을 평가하기 위해 각 데이터 샘플의 손실 (모델의 잘못된 정도를 나타내는 지표) 를 계산한 다음 평균화한 값, 즉 손실함수 (Loss function) 으로 알 수 있는데 이를 로지스틱 회귀에서는 로그 손실이라 부른다. $$-\frac{1}{m} \sum_{i=1}^m [y^i log(h(z^i)) + (1 - y^i) log(1 - h(z^i))] \\ m : 데이터츼 총 개수 \\ y^i : 데이터 샘플 i 의 분류 \\ z^i : 데이터 샘플 i 의 log - odd \\ h(z^i) : 데이터 샘플 i 의 log - odd 의 sigmoid (데이터 샘플 i 가 분류에 속할 확률)$$

	- 로지스틱 함수를 구성하는 계수와 절편에 대해 Log Loss 를 최소화하는 값을 찾는 기법이며, 로지스틱 회귀는 특정 범주류 분류될 것인지 아닌지를 결정하기 떄문에 2진 분류이다. 따라서 로그 손실 역시 2개로 나뉜다.
		- 특정 분류에 속하는 경우 (y = 1 인 경우) : $loss_{y=1} = -log(h(z^i))$ 로 계산되며, 특정 사건이 발생할 확률에 로그를 씌운 것이다.

		- 특정 분류에 속하지 않을 경우 (y = 0 인 경우) : $loss_{y=0} = -log(1 - h(z^i))$ 로 계산되며, 특정 사건이 발생하지 않을 확률에 로그를 씌운 것이다.

- 경사하강법을 통해 모든 데이터에서 로그손실을 최소화하는 계수를 찾는다.

- 로지스틱 회귀에서 결과 값은 `분류 확률` 이며, 확률이 특정 수준 이상이면 해당 클래스에 속할지 말지 결정한다. 이 결정 기준을 `임계값 (classification threshold)` 이라 하며, 일반적인 임계값은 0.5 이다. (단, 필요에 따라 모델의 임계값 변경이 가능)
