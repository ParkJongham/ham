# 카메라 스티커 앱 만들기

## 사용 데이터

- 얼굴 이미지 & 다양한 스티커 이미지

## 사용 기법

- [Face Detection](https://github.com/ParkJongham/ham/blob/master/Face%20detection/face%20detection%2C%20face%20landmark.md)
- Face Landmark / Alignment (조정) / Key point Detection


## 모델 구성

[모델 작성 코드](https://github.com/ParkJongham/ham/tree/master/Camera_Sticker%20APP)

1. 얼굴이 포함된 이미지 준비 및 라이브러리 임포트
2. opencv 를 통해 이미지 불러오기
3. dlib HOG 를 통해 Face Detection
4. [이미지 피라미드](https://github.com/ParkJongham/ham/blob/master/Image%20Pyramid/Image%20Pyramids.md) (이미지 upsampling 을 통해 크기를 키우기)
	- HOG 의 2번째 파라미터 = 이미지 피라미드의 수
	- 얼굴을 크게 봄으로써 보다 정확한 검출이 가능
5. Face Landmark Localization (이목구비의 위치 추론)
	- Face Dection 의 결과물로 bounding box 로 Crop 된 이미지를 사용
	- Object keypoint estimation : 이목구비 위치 추론과 같이 객체 내부의 점을 찾는 기법으로 Top - down 방식을 이용
		Object keypoint estimation 의 2가지 방법 :
		- Top - down 방식 : bounding box 를 찾고 내부의 keypoint 를 예측
		- bottom - top 방식 : 이미지 전체의 keypoint 를 찾고 point 관계를 이용한 군집화를 통해 box 생성 
6. lansmark 모델 불러오기
	6-1. `landmark_predictor` 는 RGB 이미지와 `dlib.rectangle` 을 입력으로 받으며, `dlibfull_object_detection` 을 반환한다.
	6-2. `points` 는 `dlib.full_object_detection` 의 객체로, `parts()`  함수로 개별 위치에 접근 가능하다. 때문에 `list_points` 는 `tuple(x, y)` 68개로 이루어진 리스트로 구성된다. 이를 이미지에서 찾아진 얼굴 개수마다 반복하여 `llist_landmark` 에 68개의 랜드마크가 얼굴 개수만큼 저장된다.
7.  랜드마크를 영상에 출력
8. 스티커 적용
	8-1. 랜드마크를 기준으로 눈썹 위 얼굴 중앙에 스티커를 위치
	8-2. 얼굴 위치 및 카메라 거리 등에 따라 좌표가 달라지기 때문에 비율로 계산해줘야한다.
	8-3. 스티커의 위치를 비율로 계산 : 
		$x = x_{nose}$
		$y = y_{nose} - \frac {width}{2}$
	8-4. 스티커의 크기를 비율로 계산 : 
		$width = height = width_{bbox}$
9. 좌표 확인
10. 스티커 적용 
11. 원본 이미지에 스티커 적용을 위해 x, y 좌표 조정
	- 이미지 시작점은 top - left 좌표
12. 음수 좌표값에 대한 예외처리 및 오차범위 부분의 스티커 제거
13. top 의 `y` 좌표를 원본 이미지 경계값으로 수정
14. 원본 이미지에 스티커 적용
	- `sticker_area` 는 crop 된 이미지로, 실제 적용할 위치를 범위의 이미지로 지정
	- 스티커 이미지에서 `0` 이 아닌 색이있는 부분만을 사용하므로, `np.where` 를 통해 `img_sticker` 가 `0`인 부분은 `sticker_area`를 사용. 아닌부분은 반대로 `img_sticker` 를 사용
15. 스티커가 추가된 이미지 출력
16. bounding box 와 landmark 를 제거한 최종 이미지 출력
