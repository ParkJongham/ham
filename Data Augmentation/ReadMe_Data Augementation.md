# Data Augmentation


[Data Augmentation 적용 코드](https://github.com/ParkJongham/ham/blob/master/Data%20Augmentation/Data%20Augmentation%20%EC%A0%81%EC%9A%A9.ipynb)

이미지 데이터 수집에는 많은 시간과 자원이 필요하며, 크롤링을 통해 이미지를 수집한다고 하더라도, 필요 목적에 적합한 이미지를 구분하게면 많은 이미지 데이터 셋을 구축하는데 난항을 겪을 수 있다.

Data Augmentation 은 말 그대로 데이터를 Agument, 즉 증강시켜주기 위한 방법이다.

즉, 1장의 이미지 데이터로 그 이상으로 증강시켜주는 기법이다.


## Data Augmentation 의 역할 : 

- 이미지 데이터는 각도, 조도 등에 따라 각기 다른 피처를 가지게되며 여러 경우의 수를 자기고 있어야 학습 후 실제 활용에도 좋은 성능을 보일 수 있다. 

- 데이터가 많으면 Overfitting 을 줄일 수 있다. 

- 테스트 이미지 시 다양한 노이즈가 있으면 이로 인해 성능이 좋지 못할 수 있는데, 이러한 노이즈를 학습 데이터에 노이즈를 삽입해 대응할 수 있도록 도와줄 수도 있다.


## Image Augmentation 방법 : 

여러가지 방법이 있지만 많이 사용되는 다음 방법에 대해 알아보자.

- Flipping : 
	- 이미지를 대칭으로 반전을 주는 방법. (상하 반전, 좌우 반전 모두 가능)
	- __물체 탐지 (detection), 세그멘테이션 (segmentation) 문제 등 정확한 답이 존재하는 문제에 적용할 때는 라벨도 같이 반전을 주어야 한다. (분류 문제에서는 문제가 없을 수 있다.)__
	- 숫자나 알파벳 문자를 인식하는 문제에서는 주의가 필요하다. (반전 시 다른 글자가 될 가능성이 있기 때문)
	- 일반적인 좌우반전 뿐만 아니라 확률에 따라 적용이 필요
		- 반전되지 않은 원본 데이터도 활용이 될 수 있기 때문이다.

- Gary scale : 
	- RGB 3가지 채널을 가진 이미지를 하나의 채널을 가지도록 해주는 방법
	- RGB 각 채널마다 가중치를 줘서 가중합 (weighted sum) 을 통해 사용하며, 가중치 합은 1이 된다.
	
- Saturation : 
	- RGB 이미지를 HSV  이미지로 변경하고 S (saturation) 채널에 오프셋 (offset) 을 적용하여 이미지를 선명하게 만들어 주는 방법
		- HSV : Hue (색조), Saturation (채도), Value (명도) 3가지 성분으로 색을 표현
	- HSV 이미지로 변경 후 다시 RGB 색상 모델로 변경
	
- Brightness : 
	- 밝기 조절을 해 주는 방법
	- RGB 는 (255, 255, 255) 값을 가지는데 각 값이 높을 수록 흰색에 가까우며, 0 에 가까울 수록 검은색에 가까움을 의미

- Rotation : 
	- 이미지의 각도를 변환하는 방법
	- 90도의 경우 직사각형 형태가 유지되기 때문에 이미지 크기만 조절해주면 바로 사용이 가능
	- 90도 단위로 돌리지 않을 경우 직사각형 형태에서 이미지로 채워지지 못하는 부분을 어떻게 처리해야할지 유의해야 한다.
	
- Center Crop : 
	- 이미지 중앙을 기준으로 확대하는 방법
	- 너무 작게 center crop 하게되면 원래의 라벨과 맞지 않을 수 있기 때문에 주의
	- 또한 확대하고자 하는 이미지의 특징을 유지하는 선에서 확대해야 한다. (동물 이미지 일 경우 털만 보이는 이미지를 만들거나 그래서는 안된다.)
		- 확대할 범위를 문제가 생기지 않는 범위에서 랜덤하게 조절해야 한다.
			- 파이썬 random 모듈을 사용하거나 텐서플로우의 랜덤 모듈을 사용
			- 텐서플로우의 `tf.random` 모듈의 `tf.random.uniform()` : 균일분포에서 임의의 난수를 생성하며, mean, std 를 통해 분포 조절이 가능
			- 텐서플로우의 `tf.random` 모듈의  `tf.random.normal()` : 정규분포에서 임의의 난수를 발생
	

이 외에도 Guassian noise, Contrast change, Sharpen, Affine transformation, Padding, Blurring 등이 있다.

* 참고 :  [Tensorflow의 data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)


# Imgaug

[imguag 를 통해 augmentation 기법 적용 코드](https://github.com/ParkJongham/ham/blob/master/Data%20Augmentation/imgaug%20%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20augmentation.ipynb)

Augumentation 만을 모아서 제공하는 전문 라이브러리 중 하나

- Imgaug 의 여러 augmentation 기법 : 
	- Affine() : 
		- 2D 변환의 한 종류인 아핀 변환 (Affine transform) 을 이미지에 적용하는 메소드
		- 이미지 스케일 조절, 평행이동, 회전(사진 회전) 등 적용가능
	
	- Crop() : 
		- 텐서플로우 API 와 같이 중심을 기준으로 확대하는 메소드

	- Sequential() : 
		-  여러가지의 aumentation 기법을 조합하여 한번에 사용하는 메소드
	
	- OneOf() : 
		- 여러 augmentation 기법들 중 한 가지 기법만 선택하여 적용하는 메소드 (여러개의 기법을 인자로 주었을 때 필요한 하나의 기법만 적용)
	
	- Sometimes() : 
		- OneOf() 메소드와 유사한 기능을 하는 메소드



## 사용 데이터 셋 : 

- [[mycat.jpg](https://aiffelstaticprd.blob.core.windows.net/media/documents/mycat.jpg)]
