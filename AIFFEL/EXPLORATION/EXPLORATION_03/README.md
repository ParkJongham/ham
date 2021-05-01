# 03. 카메라 스티커 앱 만들기 첫걸음


## 학습 목표

1.  얼굴인식 카메라의 흐름을 이해

2.  dlib 라이브러리 사용

3.  이미지 배열의 인덱싱 예외 처리

<br></br>
## 어떻게 만들까? 사진 준비하기

스티커앱을 만들기 위해서는 눈, 코, 입의 위치를 아는 것이 중요하다.

이렇게 얼굴의 눈, 코, 입의 위치를 찾아내는 기술을 KeyPoint Detection 의 종류인 랜드마크 (LandMark) 또는 조정 (alignment) 라고 한다.

대부분의 Face LandMark 는 눈, 코, 입, 턱을 포함한다. 이 랜드마크는 눈, 코, 입, 턱의 위치를 서로 떨어져 있는 정도를 데이터로부터 유추할 수 있다.

즉, 스티커앱을 만드는 순서는 다음과 같다.

1.  **얼굴이 포함된 사진을 준비** 한다.
2.  사진으로부터  **얼굴 영역**  **_face landmark_**  를 찾아낸다. (_landmark를 찾기 위해서는  **얼굴의 bounding box**를 먼저 찾아야한다._)
3.  찾아진 영역으로 부터 머리에 왕관 스티커를 붙여넣는다.
<br></br>

### 사진을 준비하자

얼굴이 포함된 사진을 준비해야한다.

> 작업 디렉토리 구성
```bash
$ mkdir -p ~/aiffel/camera_sticker/models
$ mkdir -p ~/aiffel/camera_sticker/images
```
`camera_sticker` 라는 작업 디렉토리 아래의 `images` 디렉토리에 사용할 인물 사진을 저장
<br></br>
> 스티커로 사용할 이미지 저장
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/original_images/king.png
$ wget https://aiffelstaticprd.blob.core.windows.net/media/original_images/hero.png
$ mv king.png hero.png ~/aiffel/camera_sticker/images
```
<br></br>
> 활용할 패키지 설치
```python
pip install opencv-python
pip install cmake
pip install dlib
```
<br></br>
> 활용할 패키지 가져오기
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
```
<br></br>
> 얼굴 이미지 읽어오기
```python
import os
my_image_path = os.getenv('HOME')+'/camera_sticker/images/image2.jpg'
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_bgr = cv2.resize(img_bgr, (640, 480))    # 640x360의 크기로 Resize
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
plt.imshow(img_bgr)
plt.show()
```
이미지 사이즈를 변경해 준다. 640 x 360 의 VGA 크기(16:9) 로 고정했지만 이미지의 가로 세로 비율에 맞게 변경하면 된다. 

(종횡비 (가로 세로 비율) 가 4 : 3이라면 640 x 480 으로 변경)
<br></br>

출력된 이미지를 보면 푸르딩딩한 것을 볼 수 있다.
이는 OpenCV 의 특징이다.

일반적으로 이미지는 RGB 순으로 빨강, 녹색, 파랑 순으로 색을 사용하지만 OpenCV 는 BGR 순, 즉 파랑, 녹색, 빨강 순으로 색을 사용한다.

이러한 특징 때문에 색상 보정을 통해 원본 이미지의 색상으로 변경해줘야한다.

+ 참고 : [이미지 다루기 - gramman 0.1 documentation](https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html)
<br></br>

> 색상 보정 처리
```python
# plt.imshow 이전에 RGB 이미지로 바꾸는 것을 잊지마세요. 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```
<br></br>

## 얼굴 검출 Face Detection

이제 사진에서 얼굴을 찾아야 한다. 

이렇게 특정 물체 (얼굴, 사람, 차 등) 를 감지하는 것을 Object Detection 이라고 하는데 이를 위해 공개되어 있는 패키지는 `dlib` 가 있다.

dlib 의 face detector 는 HOG (Histogram of Oriented Gradient) feature 를 사용해서  SVM (Support Vector Machine) 의 sliding window 로 얼굴을 찾는다.

+ 참고 : 
	- [딥러닝(Deep Learning)을 사용한 최신 얼굴 인식(Face Recognition)](https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-4-63ed781eee3c)  (한국어 번역본)
    -   [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)  (영어 원본)
<br></br>

HOG 는 이미지의 그래디언트 (Gradient) 를 특징으로 사용한다.

얼굴을 찾기 위해서 색상 데이터는 무의미하며, 픽셀간의 밝기 정도를 통해 찾기 때문이다.

즉, 특정 픽셀을 둘러싸고 있는 픽셀과 비교해서 얼마나 어두운지를 알아야하며, 어두워지는 방향을 화살표로 나타내는데 이 화살표를 그래디언트라고 한다.

이미지에서 밝은 부분으로부터 어두운 부분으로의 흐름을 알기아야하기 때문에 그래디언트를 특징으로 사용한다.

또한 단일 픽셀의 그래디언트를 사용하지 않고 16 x 16 의 정사각형을 이용한다.

모든 단일 픽셀에 그래디언트를 저장하게되면 너무 많고 자세하여 패턴파악이 어렵다. 때문에 범위를 통해 보다 빠르고 쉽게 패턴을 파악하기 위해 16 x 16 의 정사각형을 이용한다.
<br></br>

> dlib 를 활용한 hog detector 선언
```python
import dlib
detector_hog = dlib.get_frontal_face_detector()   #- detector 선언
```
<br></br>
> detector 를 이용해 얼굴의 bounding box 추출
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)   #- (image, num of img pyramid)
```
dlib 은 rgb 이미지를 입력으로 받기 때문에 `cvtColor()` 를 이용해서 opencv 의 bgr 이미지를 rgb로 변환해줘야 한다.

detector_hog의 두 번째 파라미터는 이미지 피라미드의 수이며, 이미지를 upsampling 방법을 통해 크기를 키우는 것을 이미지 피라미드라고 한다.

+ 참고 : [이미지 피라미드](https://opencv-python.readthedocs.io/en/latest/doc/14.imagePyramid/imagePyramid.html)

이미지피라미드에서 얼굴을 재검출하면 작게 촬영된 얼굴을 볼 수 있다. 따라서 보다 정확한 검출이 가능하다.
<br></br>
> 얼굴을 화면에 출력
```python
print(dlib_rects)   # 찾은 얼굴영역 좌표

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```
dlib detector 는  `dlib.rectangles`  타입의 객체를 반환합니다.

`dlib.rectangles`  는  `dlib.rectangle`  객체의 배열 형태로 이루어져 있다.

`dlib.rectangle` 객체는 `left(), top(), right(), bottom(), height(), width()` 등의 멤버 함수를 포함하고 있다.

+ 참고 : [dlib docs](http://dlib.net/python/index.html#dlib.rectangles)
<br></br>

## 얼굴 랜드마크 Face LandMark

이제 얼굴을 찾았으니 스티커를 적용하기 위해서는 눈, 코, 입의 위치를 찾아야 한다.

이렇게 눈, 코, 입 즉 이목구비를 찾는 기술을 Face LandMark localization 라고한다.

Face LandMark 는 얼굴을 감지한 결과물인 bounding box 로 잘라낸 crop 된 얼굴 이미지를 이용한다.

<br></br>
### Object KeyPoint Estimation 알고리즘

객체 내부의 점을 찾는 기술을 Object KeyPoint Estimation 이라고 한다. 이 KeyPoint 를 찾는 알고리즘은 2가지로 나뉜다.

1) top-down : bounding box를 찾고 box 내부의 keypoint를 예측

2) bottom-up : 이미지 전체의 keypoint를 먼저 찾고 point 관계를 이용해 군집화 해서 box 생성

여기서는 1 번에 해당하는 알고리즘을 통해 학습한다.

<br></br>
### Dlib landmark localization

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-8-8.png)
<br></br>

위 그림은 Crop 된 얼굴 이미지에서 68 개의 이목구비 위치를 의미한다.

해당 위치에 맞에 이목구비를 찾으면 해당 값을 이용하여 얼마나 떨어진 곳에 스티커를 위치할지 정할 수 있다.

이 점의 개수는 데이터셋과 논문마다 다르다.

AFLW 데이터셋은 21개를 사용하고 ibug 300w 데이터셋은 68개를 사용한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/E-8-9.max-800x600.png)
<br></br>
+ 참고 : [AFLW dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)
<br></br>
Dlib 는 regression tree 의 앙상블 모델을 사용한 300 - W 데이터셋으로 학습한 pretrained model 을 제공한다.

+ 참고 : 
	+ [300-W 데이터셋](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
	+ [논문 : One Millisecond Face Alignment with an Ensemble of Regression Trees](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf)
<br></br>

> Dlib 모델 사용 다운로드
```bash
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
$ mv shape_predictor_68_face_landmarks.dat.bz2 ~/aiffel/camera_sticker/models 
$ cd ~/aiffel/camera_sticker && bzip2 -d ./models/shape_predictor_68_face_landmarks.dat.bz2
```
공개된 weight file 을 다운받아 root 디렉토리의 models 디렉토리에 저장한다.
<br></br>
> Dlib 모델 불러오기
```python
import os
model_path = os.getenv('HOME')+'/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
```
<br></br>
> landmark 를 
```python
list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))
```
`landmark_predictor` 는 `RGB 이미지` 와 `dlib.rectangle` 을 입력 받아 `dlib.full_object_detection` 를 반환한다.

`points` 는 `dlib.full_object_detection` 의 객체이기 때문에 `parts()` 함수로 개별 위치에 접근할 수 있다. 

조금 더 직관적인 (x, y) 형태로 접근할 수 있도록 변환해 주었다. 따라서 `list_points` 는 tuple (x, y) 68개로 이루어진 리스트가 된다. 

이미지에서 찾아진 얼굴 개수마다 반복하면 `list_landmark`에 68개의 랜드마크가 얼굴 개수만큼 저장된다.
<br></br>
> 랜드마크를 영상에 출력
```python
for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```
<br></br>

## 스티커 적용하기

이제 왕관 스티커를 적용해보자.

왕관 스티커는 이마에 적용되어야 할 것이다. 이를 랜드마크를 기준으로 눈 썹 위, 얼굴 중앙에 위치해야 한다.

즉, 코를 기준으로 `x` 이상에 위치시키거나, 눈썹 위 `n` 픽셀 위에 스티커를 구현해도 된다.

얼굴의 위치나 카메라의 거리에 따라 픽셀 `x` 가 다르므로 비율로써 계산하게된다.

계산해야할 비율은 2가지, 스티커의 위치와 스티커의 크기이다.

스티커의 위치는 $\begin{aligned} x &= x_{nose} \\ y &= y_{nose}-\frac{width}{2} \end{aligned}$ 로 구하며 스티커의 크기는 $width=height=width_{bbox}$ 로 구한다.
<br></br>
> 좌표 확인
```python
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[30]) # nose center index : 30
    x = landmark[30][0]
    y = landmark[30][1] - dlib_rect.width()//2
    w = dlib_rect.width()
    h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))
```
코의 중심점이 출력된다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/E-8-12.png)
<br></br>
> 스티커 이미지를 읽어와 좌표에 적용
```python
import os
sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/king.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w,h))
print (img_sticker.shape)
```
> 스티커를 위에서 계산한 크기로 resize
```python
refined_x = x - w // 2  # left
refined_y = y - h       # top
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```
이미지 시작점은 top - left 좌표이다.

원본 이미지에 스티커 이미지를 추가하기 위해서 `x`, `y` 좌표를 조정한다.
<br></br>

이때 `y` 축 좌표값이 음수가 되는데 OpenCV 데이터는 넘파이의 어레이 형태를 데이터로 사용한다.

넘파이 어레이는 음수 인덱스에 접근이 불가능하기 때문에 예외처리가 필요하다. 따라서 원본 이미지에 적용된 스티커가 이미지 범위를 벗어나므로 이 벗어난 부분은 제거해줘야한다.

<br></br>
> `-y` 크기만큼 스티커를 crop
```python
img_sticker = img_sticker[-refined_y:]
print (img_sticker.shape)
```
<br></br>
> top 의 `y` 좌표를 원본 이미지 경계값으로 수정
```python
refined_y = 0
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```
<br></br>
> 원본 이미지에 스티커를 적용
```python
sticker_area = img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
```
`sticker_area`는 원본이미지에서 스티커를 적용할 위치를 crop한 이미지이다.

스티커 이미지에서 사용할 부분은 `0` 이 아닌 색이 있는 부분을 사용하며, `np.where` 를 통해 `img_sticker` 가 `0` 인 부분은 `sticker_area` 를 사용하고 0이 아닌 부분을 `img_sticker` 를 사용한다.
<br></br>
> 결과 이미지 출력
```python
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>
> bounding box, landmark 를 제거한 최종 이미지 출력
> (`img_show` 대신, 지금까지 아껴 두었던 `img_rgb`를 활용)
```python
sticker_area = img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>
