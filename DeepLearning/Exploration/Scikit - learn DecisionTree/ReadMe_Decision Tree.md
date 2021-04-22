# Decision Tree

[DcisionTree 를 활용한 IRIS 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learn%20DecisionTree/DecisionTree_IRIS_Classification.ipynb)
[DcisionTree 를 활용한 와인 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learn%20DecisionTree/DecisionTree_Wine_Classification.ipynb)
[DcisionTree 를 활용한 유방암 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learn%20DecisionTree/DecisionTree_breast_cance_Classification.ipynb)
[DcisionTree 를 활용한 손글씨 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learn%20DecisionTree/DecisionTree_Digits_Classification.ipynb)

데이터를 분석하여 데이터 간 패턴을 예측 가능한 규칙들의 조합으로 나타내는 기법.

- 분류 및 회귀 문제에 사용

- 범주형 데이터 및 연속형 데이터 모두를 예측 가능
	- 범주형 데이터 : 데이터가 특정 terminal node 에 속한다는 사실을 확인 한 후 해당 terminal node 에서 가장 빈도가 높은 범주의 새로운 데이터를 분류

	- 회귀 데이터 : terminal node 의 종속변수 (y) 의 평균을 예측값으로 반환. 이 예측값의 종류는 terminal node 개수와 일치.

- 각 영역의 순도 (homogeneity) 가 증가, 불순도 (impurity) / 불확실성 (uncertainty) 이 최대한 감소하는 방향으로 학습. 

- 결정경계 (decision boundary) 가 데이터 축에 수직이어서 특정 데이터에만 작동할 가능성이 높다.

- 엔트로피, 지니계수, 오분류오차 3개의 지표로 성능을 평가
	
	- 엔트로피 (entropy) : 불확실성 감소를 나타내는 지표, 즉 순도 증가를 의미하며, 낮을수록 정보를 획득했다는 의미를 지닌다.
	- A 영역에 속한 모든 레코드가 동일한 범주에 속할 경우(=불확실성 최소 = 순도 최대) 엔트로피는 0.
	- 범주가 둘뿐이고 해당 개체의 수가 동일하게 반반씩 섞여 있을 경우(=불확실성 최대 = 순도 최소) 엔트로피는 1.
		- m 개의 레코드가 속하는 A 영역 (1개의 영역) 에 대한 엔트로피는 다음 식으로 정의.  (Pk=A영역에 속하는 레코드 가운데 k 범주에 속하는 레코드의 비율)$$Entropy(A)=-\sum _{ k=1 }^{ m }{ { p }_{ k }\log _{ 2 }{ { (p }_{ k }) }  }$$
	
		- 2개 이상의 영역에 대한 엔트로피는 다음 식으로 정의. (Ri=분할 전 레코드 가운데 분할 후 i 영역에 속하는 레코드의 비율) $$Entropy(A)=\sum _{ i=1 }^{ d }{ { R }_{ i } } \left( -\sum _{ k=1 }^{ m }{ { p }_{ k }\log _{ 2 }{ { (p }_{ k }) }  }  \right)$$

	- 지니계수 (Gini Index) : 엔트로피 외에 불순도 지표로 많이 사용. 다음 식과 같이 정의. $$G.I(A)=\sum _{ i=1 }^{ d }{ { \left( { R }_{ i }\left( 1-\sum _{ k=1 }^{ m }{ { p }_{ ik }^{ 2 } }  \right)  \right)  } }$$

	- 오분류오차 (misclassification error) : 엔트로피나 지니계수와 함께 불순도를 측정하는 지표. 하지만 미분이 불가능한 점으로 인해 거의 사용되지 않는다.

## 모델학습

1. 재귀적 분기 (recursive partitioning) :입력 변수 영역을 2개로 구분하는 과정
	- 설명변수를 기준으로 정렬한 후, 모든 분기점에서 엔트로피 / 지니계수를 구배 분기, 분기 전과 후를 비교하는 전체 과정을 반복
	- 개체가 n 개, 변수가 d 개 일 경우 분기의 경우의 수는 $d(n-1)$ 개가 된다.

2. 가지치기 (pruning) : 너무 자세하게 구분된 영역을 통합하는 과정
	- terminal node 의 순도가 100% 상태를 Full Tree 하고 하며, Full Tree 를 생성한 뒤 적절한 수준에서 terminal node 를 결합. 그렇지 않을 경우 과적합의 문제가 발생한다.
	- 의사결정 나무의 분기 수가 증가할 때 처음에는 새로운 데이터에 대한 오분류율이 감소하지만 일정 분기 이후에는 오분류율이 증가하는데 이때 가지치기를 수행해줘야 한다.
	- 비용함수 (cost fuction) 를 최소로 하는 분기를 찾아내도록 학습.
		- 비용함수 : $$CC(T)=Err(T)+\alpha \times L(T) \\ CC(T)=의사결정나무의 비용 복잡도(=오류가 적으면서 terminal node 수가 적은 단순한 모델일 수록 작은 값) \\ ERR(T)=검증데이터에 대한 오분류 \\  L(T)=terminal node의 수(구조의 복잡도) \\ Alpha=ERR(T)와 L(T)를 결합하는 가중치(사용자에 의해 부여됨, 보통 0.01~0.1의 값을 사용)$$
