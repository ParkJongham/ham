# Random Forest

[DcisionTree 를 활용한 IRIS 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20RandomForest/RandomForest_IRIS_Classification.ipynb)
[DcisionTree 를 활용한 와인 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20RandomForest/RandomForest_Wine_Classification.ipynb)
[DcisionTree 를 활용한 유방암 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20RandomForest/RandomForest_breast_cance_Classification.ipynb)
[DcisionTree 를 활용한 손글씨 분류](https://github.com/ParkJongham/ham/blob/master/Scikit%20-%20learning%20RandomForest/RandomForest_Digits_Classification.ipynb)

많은 Decision Tree 가 모인 것. 즉, 같은 데이터에 Decision Tree 를 여러개 만들어 그 결과를 종합하여 예측하는 기법.

- 앙상블 (ensemble method) : 여러개의 의견을 통합하거나 여러가지 결과를 종합하는 방식

- 각 각의 Decision Tree 를 만들때 사용되는 요소을 무작위로 선정. 무작위로 선정된 일부 중 가장 알맞게 예측하는 요소가 Decision Tree 의 한 단계가 된다.

- 위 과정을 반복하여 원하는 수의 Decision Tree 를 생성한다. (Decision Tree 의 개수는 사용자가 직접 지정)

- 생성된 Random Forest 에 데이터를 주면 각 Decision Tree 에서 도출된 결과를 하나로 통합하여 예측

- Decision Tree 를 생성하는데 있어 모든 요소를 고려하지 않는 이유는 오히려 모든 요소를 고려하기 위한 방법.
	- Decision Tree 의 한 단계를 만드는데 모든 요소를 고려할 경우 모든 Decision Tree 가 같은 요소만을 가지고 있게 되며, 이는 오리혀 모든 요소를 고려하지 않는 경우.
	- 전교1 등한 명보다 전교5등 100명이 아는 것이 더 많은 것과 같이 집단지성을 모방한 아이디어이다.
