﻿# Face Recognition

이를 위해서는 다음 문제를 해결해야 하는데

1. Face detection : 사진을 통해 사진 속 모든 얼굴을 찾아낸다.

2. 얼굴의 위치 교정 (posing)및 투영(projection) : 찾아낸 사진의 각 얼굴에 초점을 맞췄을 때, 조명이나 각도가 달라도 사람이라는 것을 인식할 수 있어야 한다.

3. Face incoding : 다른 사람들과 다른 고유한 특징을 찾아낼 수 있어야 한다. (눈 크기, 얼굴 크기, 등)

4. 해당 특징을 통해 모든 사람들과 비교했을 때, 그 사람임을 특정할 수 있어야 한다.

위와 같은 각 단계를 분리하여 해결하고, 해결된 결과를 다음 단계로 보내주는 경로를 만들어야 한다.

이러한 경로는 pipeline 이라 한다.


## 1. Face detection :

dlib 패키지의 face detector 의 HOG (Histogram of Oriented Gradient) feature 를 통해 SVM (Support Vector Machine) 의 sliding window 로 얼굴을 찾아낸다.

1. 이미지에서 얼굴을 찾기 위해서는 이미지를 흑백으로 바꾼다.
	- 얼굴을 검출하는데 있어서 색상 정보는 필요없기 때문이다.

2. 이미지에서 각 필셀을 비교하여 얼마나 어두운지 명암 정도를 확인하고, 이미지가 어두워지는 방향을 나타내는 화살표 (gradients) 를 그린다. 
	- 이를 통해 밝은 부분에서 어두운 부분으로의 흐름을 파악한다.
	- 모든 픽셀에 대해 gradient 를 저장할 경우 너무 많으므로, 이미지를 16 x 16 픽셀의 정사각형으로 분리한다. 
	- 분리된 정사각형에서 gradient 들이 어느 방향을 가르키는지를 확인하고 이를 통해 얼굴을 추출한다.

3. 이미지에서 gradients 가 가르키는 방향으로써 이미지를 표현한 것을 HOG 라고하며, 많은 얼굴로 훈련된 HOG 이미지로부터 추출된 패턴과 가장 유사한 부분을 이미지에서 추출한다. 이를 통해 Face detection 이 가능하다.


## 2. Face posing and projection

Face detection 을 통해 이미지에서 얼굴을 분리해 내면 이미지의 인물이 바라보는 각도, 조명 등을 통해 잘못 인식하는 문제를 해결해야 한다. 

이를 위해 Facd landmark estimation 알고리즘을 통해 사진을 비틀어 눈, 코, 입이 항상 표준 위치에 올 수 있도록 해줘야 얼굴을 좀 더 쉽게 비교할 수 있다.

- Face lanmark estimation :
	- 얼굴에 68개의 랜드마크라 불리는 특정 포인트를 찾아내는 방법으로 얼굴의 눈. 코, 입, 눈썹 등의 위치를 찾아낸다.
	- 파악한 눈, 코, 입의 위치를 최대한 가운데로 올 수 있도록 이미지를 회전 (rotate) 하고 크기를 조절 (scale) 하고 비튼(shear)다. 이때 3d 변형 (3차원에서 변형 및 뒤틈) 을 사용하면 왜곡이 생기므로 사용하지 않는다.


## 3. Face incoding

 각 얼굴에서 기본 측정갑 (눈 크기, 귀 크기, 눈 사이 간격, 얼굴 길이 등) 을 추출하고, 임의의 얼굴을 같은 방법으로 기본 측정값을 추출하여 가장 가까운 값을 가지는 얼굴을 찾아낸다.

가장 신뢰할 수 있는 얼굴 측정 방법은 DCNN (Deep Convolution Neural Network) 를 통해 128개의 측정값을 생성하도록 훈련하는 것이다. 즉, 딥컨볼루션 신경망을 통해 얼굴의 특징을 128개로 임베딩 (embeding) 시키는 것이다.


## 4. 특정 사람으로 특정

테스트 이미지에 가장 근접한 측정값을 가진 사람을 데이터베이스에서 찾아내는 것이다. 이를 위한 기법으로는 SVM 을 통해 분류가 가능하다.