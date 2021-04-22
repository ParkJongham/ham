# Support Vector Machine (SVM)

[DcisionTree 를 활용한 IRIS 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20SVM%20(Support%20Vector%20Machine)/SVM_IRIS_Classification.ipynb)
[DcisionTree 를 활용한 와인 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20SVM%20(Support%20Vector%20Machine)/SVM_Wine_Classification.ipynb)
[DcisionTree 를 활용한 유방암 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20SVM%20(Support%20Vector%20Machine)/SVM_breast_cance_Classification.ipynb)
[DcisionTree 를 활용한 손글씨 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20SVM%20(Support%20Vector%20Machine)/SVM_Digits_Classification.ipynb)

학습 데이터를 비선형 매핑 (Mapping) 을 통해 고차원으로 변환하고, 변환된 새로운 차원에서 초평면 (hyperplane) 을 최적으로 분리하는 선형분리를 찾어 최적의 의사 결정 영역 (Decision Boundary) 을 찾는 기법.

- 비선형 데이터의 경우 2차원에서는 각 각의 데이터를 분리할 수 없지만, 고차원으로 보내게되면 선형분리가 가능하다.

	- 매핑 (Mapping) : 저차원에서 고차원으로 보내는것. 반대로 고차원에서 저차원으로 보내는 것 역시 매핑이다.

	- 초평면 (hyperplane) : 비선형데이터를 고차원으로 매핑할 경우 선형데이터로 분리 될 수 있는 공간. 직교하는 2 다리를 위에서 보면 (2차원으로 보면) 교차하여 분리가 불가능하지만 옆에서 바라보면 다리와 다리 사이에 공간이 있어 이 2 다리를 따로 분리할 수 있는 공간이 있는데 이 공간은 초평면이라 한다.

- Random Forest 나 Decision Tree 등의 모델들 보다 과적합 우려가 적으며, 비선형 데이터를 모형화 할 수 있기 때문에 정확하다.

- 선형 데이터의 경우 SVM 의 적용 : 선형 데이터의 경우 이미 데이터들을 구분할 수 있는 직선이 존재한다. 하지만 이 경우 역시 SVM 은 선형 데이터를 고차원으로 보내 MMH (Maximum Marginal Hyperplane) 을 찾아 분리한다. 즉, 훈련 데이터 외의 데이터에 대한 분류 오류를 최소한으로 하는 직선을 찾는다.
