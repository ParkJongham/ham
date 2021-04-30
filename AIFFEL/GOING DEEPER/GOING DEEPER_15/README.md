# 15. 내 거친 생각과 불안한 눈빛과 그걸 지켜보는 카메라

머신러닝 서비스에서 가장 중요한 부분 중 하나는 바로 데이터를 어떻게 효율적으로 모으는지 이다.


## 학습 목표

1.  데이터를 직접 모아보기
2.  딥러닝이 아닌 머신러닝 방법을 사용하기
3.  Keypoints Regressor 제작하기
4.  라벨링 툴 다뤄보기


## 사전 준비

> 작업 디렉토리 구성
```
$ mkdir -p ~/aiffel/coarse_to_fine/images
```
<br></br>
> 모델 파일 다운 및 압축 해제
```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P ~/aiffel/coarse_to_fine/models/ 
$ cd ~/aiffel/coarse_to_fine/models/ && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
<br></br>

## 카메라 앱에 당황한 표정 효과를 추가하려면?

카메라 앱에 당환할 때 나오는 표정을 재미있게 표현하려면 어떻게 해야할까?

먼저 눈의 위치를 찾아야 한다. 이는 랜드마크를 이용해 눈을 찾는 방법이 있다. 하지만 랜드마크는 시선의 방향 즉, 눈이 바라보는 방향을 포함하고 있지 않다.

이렇게 시선을 어떻게 파악할 수 있을까?

간단하게는 눈동자의 위치를 파악해보는 것이다.


### 눈동자를 찾는 방법

대부분의 오픈 소스 라이브러리는 눈동자를 검출해 주진 않는다. 따라서 눈을 검출하고, 검출된 눈의 정보를 활용하여 눈동자를 검출 해야한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/11_Hxu6WiE.max-800x600.png)
<br></br>

눈동자 검출을 위한 데이터셋은 아래와 같이 찾을 수 있다.

눈동자 검출 데이터셋 : [Labeled pupils in the wild: A dataset for studying pupil detection in unconstrained environments](https://arxiv.org/pdf/1511.05768.pdf)

![](https://aiffelstaticprd.blob.core.windows.net/media/images/12_hawFj2h.max-800x600.png)
<br></br>

하지만 위 데이터셋은 AR 기기를 위한 10 cm 이내 근거리 촬영 환경이다.

이렇듯 목적에 맞는 데이터셋을 찾기는 매우 힘들다.

딥러닝 기반 서비스를 구현하기 위해서는 엄청나게 많은 양의 데이터가 필요하다.

많은 자원이 필요하며, 이는 꾀나 큰 부담이 따르므로 쉽지 않다.

이러한 문제는 기존 머신러닝 방법을 적절히 이용하는 방법과 잘 가공된 어노테이션 (annotation) 도구를 만듬으로서 해결이 가능하다.

이를 위한 학습은 다음과 같다.

-   기존 머신러닝 방법을 적용한 눈동자 검출 모듈을 만드는 방법에 대해 설명합니다.
-   딥러닝 방법을 이용한 눈동자 검출 모듈을 제작하고
-   앞에서 만들어진 데이터와 함께 높은 품질의 라벨을 얻을 수 있는 라벨링 툴(labeling tool) 대해 설명합니다.
-   마지막으로 앞의 과정을 어떻게 효율적인 순서로 진행할 수 있는지 논의하겠습니다.
<br></br>

## 대량의 Coarse Point Label 모아보기 (01). 도메인의 특징을 파악하자

딥러닝을 적용하지 않고 머신러닝을 적용한다는 말은 handcraft feature (사람이 정의한 특징) 를 사용한다는 것이다.

이를 위해서는 모데인 지식이 매우 중요하게 작용한다.

예를 들면 혈관성 치매를 진단하는 보조 솔루션을 만든다고 했을 때, MRI 를 데이터로 활용하게 된다.

MRI 는 T1w, T2w, FLAIR 등 다양한 촬영 방식 (protocol) 이 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/08_K2sCbM7.max-800x600.png)

위 그림은 FLAIR 한 MRI 이미지이다. 이를 보면 정상과 문제가 있는 것을 구분할 수 없다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/09_KAfySZT.png)
<br></br>

빨간색 영역은 뇌척수액 부분으로 주변보다 흰 값으로 나타나게 된다. 뇌혈관에 문제가 생기게 되면 피 등의 액체가 백질에 스며들게 되고 뇌조직에 문제가 생길 수 있다. 백질이 보다 밝은 값으로 나타난다고 해서 WMH (White Matter Hyperintensity) 라고 한다.

WMH 가 있다고 반드시 문제가 생기는 것은 아니지만 문제가 생긴 환자 중에서 많은 사례가 WMH를 가지고 있다.

그래서 뇌의학에서는 WMH 를 찾는 것을 중요하게 여기며, 이를 솔루션에서 자동으로 찾아야하는데 라벨이 매우 부족하다. 

세그멘테이션은 라벨링이 어렵기 때문에 초기에는 딥러닝을 사용하지 않는 방법을 선택해야 했다.

생각해보면 초기 모델 (Baseline) 을 만드는 것은 아주 쉽다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/10_RJffSnT.max-800x600.png)

WHM 은 이름처럼 하얗게 표시되는 부분으로 높은 픽셀 값을 가지고 있다. 즉, 0 ~ 255 범위 중 200 이상의 값을 가지는 픽셀만 찾아내면 간단한 초기 모델을 구현할 수 있다.

이렇게 초기 모델을 만들고 이를 바탕으로 딥러닝 모델을 학습시키면 좋은 성과를 얻을 수 있다.

이러한 방법이 ML Coarse - to - Fine 이라고 한다.
간단히 말해 근사적인 (Coarse) 데이터셋을 만들고, 이를 이용해 초기 모델을 만든 후 딥러닝 모델을 학습시키는 것이다.

하지만 이를 위해서는 WHM 과 같은 해결하고자 하는 문제에 대한 도메인 지식이 필요하며, 딥러닝이 아닌 방법을 적용할 수 있는 능력이 필요하다.

<br></br>
## 대량의 Coarse Point Label 모아보기 (02). 가우시안 블러


### 가우시안 블러

눈동자는 눈에서 가장 검은색에 가까운, 즉 어두운 부분이다. 따라서 위에서 WHM 의 특징으로 200 이상의 밝은 값을 사용했다면 눈동자는 일정 수준 이상의 어두운 부분을 찾아내면 되지 않을까?

![](https://aiffelstaticprd.blob.core.windows.net/media/images/06_gSLf6kG.max-800x600.png)
<br></br>

이를 위한 방법으로는 랜드마크를 이용해서 눈을 Crop 하고 눈에서 가장 어두운 부분의 중심을 찾는다.

간혹 눈동자에 빛이 반사되어 밝게 보이는 경우가 있는데 이러한 노이즈 (noise) 를 없애줘야한다.

노이즈를 제거하는 방법은 가우시안 블러 (Gaussian Blur) 가 있다.

+ 참고 : [Gaussian Blur](https://en.wikipedia.org/wiki/Gaussian_blur)
<br></br>

가우시안 블러는 Low - pass filter,  저역 통과 필터로 고주파 신호를 감쇄시키는 필터이다.

현재 픽셀값과 주변 픽셀값의 가중 평균을 사용해 현재 픽셀을 대체하는 방식으로 가중치 행렬을 이용하여 구현할 수 있다.

이렇게 가우시안 블러를 적용한 후 흑백을 반전시켜 가장 높은 값을 갖는 픽셀을 고르면 눈의 위치를 찾을 수 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/07_6oCf3cO.max-800x600.png)
<br></br>

'가장 높은 값'의 기준은 여러 방향으로 정의할 수 있습니다. 2D 이미지 상에서 바로 $(x, y)$ 위치를 추정하는 방법 $argmax_{x,y}(image)$ 이 있을 수 있고 위 그림에서 오른쪽과 아래 부분에 보이듯이 1차원으로 누적하는 방법이 있다.

2차원 이미지에서 눈 중심 부분 근처의 픽셀은 거의 255로 최대값을 나타낸다.

이러한 이유로 눈동자를 특정하기가 어려운데, 가우시안 블러를 이용하면 눈동자 중심을 평균으로 하는 가우시안 분포를 볼 수 있다. 물론 최대값 255로 truncate 되어 있고 눈동자만 밝은 것이 아니기 때문에 mixture density처럼 나타난다.

+ 참고 : [수식없이 이해하는 Gaussian Mixture Model](https://3months.tistory.com/154)

이때 1차원으로 누적해요 표현하면 255 로 truncated 되는 문제와 주변 노이즈에 대체할 수 있다.

<br></br>
## 대량의 Coarse Point Label 모아보기 (03). 구현 : 눈 이미지 얻기

> 사용할 라이브러리 임포트
```python
import matplotlib.pylab as plt
import tensorflow as tf
import os
from os.path import join
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import math
import dlib
```
<br></br>
> 사용 이미지 다운
```
$ cd ~/aiffel/coarse_to_fine/images && wget https://blog.kakaocdn.net/dn/bJuO9x/btqDnxmhE18/FHYdlDahOwxtXaXBlNZD2K/img.jpg
```
<br></br>
> 이미지 불러오기
```python
import os
img_path = os.getenv('HOME')+'/aiffel/coarse_to_fine/images/image.png'
img = cv2.imread(img_path)
print (img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>
> 얼굴과 랜드마크 검출
```python
img_bgr = img.copy()

detector_hog = dlib.get_frontal_face_detector() # detector 선언
dlib_model_path = os.getenv('HOME')+'/aiffel/coarse_to_fine/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(dlib_model_path)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1) # (image, num of img pyramid)

list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()
    cv2.rectangle(img_rgb, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_rgb, point, 2, (255, 255, 0), -1) # yellow

plt.imshow(img_rgb)
plt.show()
```
<br></br>
> 랜드마크를 이용한 눈 위치만 크롭
```python
def eye_crop(bgr_img, landmark):
    # dlib eye landmark: 36~41 (6), 42~47 (6)
    np_left_eye_points = np.array(landmark[36:42])
    np_right_eye_points = np.array(landmark[42:48])

    np_left_tl = np_left_eye_points.min(axis=0)
    np_left_br = np_left_eye_points.max(axis=0)
    np_right_tl = np_right_eye_points.min(axis=0)
    np_right_br = np_right_eye_points.max(axis=0)

    list_left_tl = np_left_tl.tolist()
    list_left_br = np_left_br.tolist()
    list_right_tl = np_right_tl.tolist()
    list_right_br = np_right_br.tolist()
    
    left_eye_size = np_left_br - np_left_tl
    right_eye_size = np_right_br - np_right_tl
    
    ### if eye size is small
    if left_eye_size[1] < 5:
        margin = 1
    else:
        margin = 6
    
    img_left_eye = bgr_img[np_left_tl[1]-margin:np_left_br[1]+margin, np_left_tl[0]-margin//2:np_left_br[0]+margin//2]
    img_right_eye = bgr_img[np_right_tl[1]-margin:np_right_br[1]+margin, np_right_tl[0]-margin//2:np_right_br[0]+margin//2]

    return [img_left_eye, img_right_eye]
```
`dlib` 의 랜드마크 자료형은 68개의 점을 가지고 있다.

+ 참고 : [Facial landmarks with dlib, OpenCV, and Python - PyImageSearch](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
<br></br>
> 적당한 `margin` 값 설정
```python
img_left_eye, img_right_eye = eye_crop(img_bgr, list_landmarks[0])

print (img_left_eye.shape)
plt.imshow(cv2.cvtColor(img_right_eye, cv2.COLOR_BGR2RGB))
plt.show()
```
랜드마크 만으론 눈 검출이 어려울 수 있기에 `margin` 값 설정
<br></br>

## 대량의 Coarse Point Label 모아보기 (04). 구현 : 눈동자 찾기

눈 중심을 찾는 함수를 구현해보자.

먼저 눈 이미지를 Low - pass fiter 를 통해 smoothing 를 한다. `bilateraFilter` 를 이용하자.

+ 참고 : [OpenCV 영상 화질 향상 기법 #5 - Bilateral Filter](http://egloos.zum.com/eyes33/v/6092269)
<br></br>

그 후 1 차원 값으로 누적시킨 후 y 축을 기준으로 최대값을 찾아  `center_y` 좌표를 구한다. y 축은 x 축에 비해 상대적으로 변화가 적기 때문이다.

x 축은 1차원 max point 를 기준으로 mean shift 를 수행한다. 양 끝단에 수렴하는 예외를 처리한 후 결과를 출력한다.

> 눈 중심을 찾는 함수 구현
```python
def findCenterPoint(gray_eye, str_direction='left'):
    if gray_eye is None:
        return [0, 0]

    # smoothing
    filtered_eye = cv2.bilateralFilter(gray_eye, 7, 75, 75)
    filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)
    filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)

    # 2D images -> 1D signals
    row_sum = 255 - np.sum(filtered_eye, axis=0)//gray_eye.shape[0]
    col_sum = 255 - np.sum(filtered_eye, axis=1)//gray_eye.shape[1]

    # normalization & stabilization
    def vector_normalization(vector):
        vector = vector.astype(np.float32)
        vector = (vector-vector.min())/(vector.max()-vector.min()+1e-6)*255
        vector = vector.astype(np.uint8)
        vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],))
        vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],))            
        return vector
    row_sum = vector_normalization(row_sum)
    col_sum = vector_normalization(col_sum)

    def findOptimalCenter(gray_eye, vector, str_axis='x'):
        axis = 1 if str_axis == 'x' else 0
        center_from_start = np.argmax(vector)
        center_from_end = gray_eye.shape[axis]-1 - np.argmax(np.flip(vector,axis=0))
        return (center_from_end + center_from_start) // 2

    center_x = findOptimalCenter(gray_eye, row_sum, 'x')
    center_y = findOptimalCenter(gray_eye, col_sum, 'y')

    inv_eye = (255 - filtered_eye).astype(np.float32)
    inv_eye = (255*(inv_eye - inv_eye.min())/(inv_eye.max()-inv_eye.min())).astype(np.uint8)

    resized_inv_eye = cv2.resize(inv_eye, (inv_eye.shape[1]//3, inv_eye.shape[0]//3))
    init_point = np.unravel_index(np.argmax(resized_inv_eye),resized_inv_eye.shape)

    x_candidate = init_point[1]*3 + 1
    for idx in range(10):
        temp_sum = row_sum[x_candidate-2:x_candidate+3].sum()
        if temp_sum == 0:
            break
        normalized_row_sum_part = row_sum[x_candidate-2:x_candidate+3].astype(np.float32)//temp_sum
        moving_factor = normalized_row_sum_part[3:5].sum() - normalized_row_sum_part[0:2].sum()
        if moving_factor > 0.0:
            x_candidate += 1
        elif moving_factor < 0.0:
            x_candidate -= 1
    
    center_x = x_candidate

    if center_x >= gray_eye.shape[1]-2 or center_x <= 2:
        center_x = -1
    elif center_y >= gray_eye.shape[0]-1 or center_y <= 1:
        center_y = -1
    
    return [center_x, center_y]
```
<br></br>
> 오른쪽, 왼쪽 두 눈 이미지에 대해 위 함수를 수행
```python
def detectPupil(bgr_img, landmark):
    if landmark is None:
        return

    img_eyes = []
    img_eyes = eye_crop(bgr_img, landmark)

    gray_left_eye = cv2.cvtColor(img_eyes[0], cv2.COLOR_BGR2GRAY)
    gray_right_eye = cv2.cvtColor(img_eyes[1], cv2.COLOR_BGR2GRAY)

    if gray_left_eye is None or gray_right_eye is None:
        return 

    left_center_x, left_center_y = findCenterPoint(gray_left_eye,'left')
    right_center_x, right_center_y = findCenterPoint(gray_right_eye,'right')

    return [left_center_x, left_center_y, right_center_x, right_center_y, gray_left_eye.shape, gray_right_eye.shape]
```
<br></br>
> 두 눈 중심 좌표 출력
```python
left_center_x, left_center_y, right_center_x, right_center_y, le_shape, re_shape = detectPupil(img_bgr, list_landmarks[0])
print ((left_center_x, left_center_y), (right_center_x, right_center_y), le_shape, re_shape)
```
<br></br>
> 오른쪽 눈을 이미지로 출력
```python
show = img_right_eye.copy()
    
show = cv2.circle(show, (right_center_x, right_center_y), 3, (0,255,255), -1)

plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>
> 왼쪽 눈을 이미지로 출력
```python
show = img_left_eye.copy()
    
show = cv2.circle(show, (left_center_x, left_center_y), 3, (0,255,255), -1)

plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
plt.show()
```
찾은 눈동자가 눈동자의 중심을 찾은 것은 아니기 때문에 다른 방법을 적용해보자.

+ 참고 : [[기계 학습] Mean Shift 클러스터링](https://bab2min.tistory.com/637)

Mean Shift 는 가우시안 커널로 밀도를 추정한 그래프에서 각 각 주변의 밀도를 계산하여 밀도가 높은 쪽으로 이동시키는 것이다. 즉, 3개의 봉우리가 있다면 3개 군집을 이루게 된다.

이를 눈동자에 해당하는 값을 밀도가 높은 부분으로 가정하면 눈동자를 찾을 수 있다.

머신러닝 방법들은 도메인 지식이 풍부할 경우 쉽고 빠르게 구현이 가능하지만, 일정 수준이상의 성능을 만족하기 어렵다.

때문에 대량의 Coarse 한 라벨을 수집한 뒤 딥러닝 모델을 통해 성능을 개선해 나간다.
<br></br>


## 키포인트 검출 딥러닝 모델 만들기

이제 앞서 도메인 지식을 통해 만든 머신 러닝을 딥러닝 모델로 만들어야 한다.

즉, 직접 설계한 특성들을 네트워크가 자동으로 추출할 수 있어야 한다.

일반적으로 이미지 분류 모델은 CNN 에 입력하고, 최종적으로 해당 클래스의 인덱스를 찾아낸다. 즉, 최종 출력은 단일 인덱스 혹은  one - hot - vector 이 된다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-8-l-cuga1.max-800x600.png)
<br></br>

하지만 아래와 같이 구조를 변경해줄 필요가 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/gc-8-l-cuga2.png)
<br></br>
 
 먼저 출력의 개수가 소프트맥스 함수를 통과해 1개의 클래스 인덱스를 출력하는 것에세 눈동자 중심 위치인 x, y 좌표를 출력하는 것으로 변경되어야 한다.

일반적으로 분류문제에서 출력 값은 불연속 정수형 값이된다. 하지만 x, y 좌표 값은 연속형 실수 값이어야한다. 즉, 분류 문제에서 회귀문제로 변경되어야 한다.

소프트맥스 - 크로스 앤트로피 (SoftMax - Cross - Entropy) 방법은 이미지 분류에서 사용되는 가장 대표적인 손실함수이다.

하지만 회귀에서는 여러 개의 값이 연속형 실수로 출력되기 때문에 MSE 와 같은 손실함수를 고려해야 한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-8-l-cuga3.max-800x600.png)
<br></br>

구조를 변경해 주었다면 모델 학습이 남아있다. 실제로 추구하는 목표에 맞는 태스크의 경우 데이터가 부족하다.

이런 경우 미리 학습된 가중치를 통해 fine tunning 하는 방법을 고려해 볼 수 있다.

가져온 가중치가 학습된 (pretrained) 태스크가 목표에 맞는 태스크과 동일한 도메인이면 가장 좋지만 아니어도 효과는 있다.

때문에 보통은 이미지넷이나 COCO 데이터셋으로 학습한 모델 가중치를 사용해 fine tunning 을 진행한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-8-l-cuga4.max-800x600.png)
<br></br>

코드로 표현하기 위한 절차는 다음과 같다.

1.  기본 ResNet 모델을 구현하고
2.  ImageNet 데이터셋으로 pretraining 을 열심히 한 후,
3.  ResNet의 fully connected layer 를 수정하고 회귀 손실 함수를 구현해서
4.  눈동자 위치를 학습시킵니다.
<br></br>

텐서플로우는 `tensorflow_hub` 를 통해 이미 학습된 이미지넷 모델을 가져올 수 있다.

따라서 1 ~ 3 번의 작업을 쉽게 수행할 수 있다.
<br></br>
> 필요 라이브러리 임포트
```python
pip install tensorflow_hub 

import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import LearningRateScheduler
```
`hub` 에는 VGG 네트워크 및 다양한 모델을 제공하고 있다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/gc-8-l-cuga5.max-800x600.png)
+ 참고 : [TensorFlow Hub](https://tfhub.dev/ml-kit/collections/image-classification/1)
<br></br>

모델은 ResNet 50 을 사용하고자 한다. 이 모델에서 사용된 데이터셋은 이미지넷이다.

이렇게 pretrained model 을 사용할 때는 Fine Tunning 시 모델 입력에 주의해야한다.

입력 크기의 차이가 많이 날 경우 pooling 후 크기가 유효하지 않을 수 있기 때문이다. 이 경우 모델이 작동하지 않는다.

ResNet 50 모델은 입력 이미지의 크기가 h x w = 224 x 224 이어야 한다.
<br></br>
> `tensorflow-hub` 에서 ResNet 모델 불러오기
> (모델의 특성 추출기 부분을 백본으로 사용)
```python
''' tf hub feature_extractor '''
feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(80,120,3))
```
<br></br>
> 좌표 학습을 위해 Dense 레이어 추가
```python
num_classes = 6

feature_extractor_layer.trainable = False
model = tf.keras.Sequential([
    feature_extractor_layer,
    #layers.Dense(1024, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid'),
])
```
<br></br>
> 모델 구성 출력
```python
model.summary()
```
<br></br>

## 라벨링 툴 (01). 소개

이제 모델을 생성했고, fine label 을 통해 모델 성능을 향상시켜야 한다.

이 fine label 은 어떻게 얻을까?

fine label 을 얻기 위해서 어노테이션 툴 (annotation tool) 혹은 라벨링 퉁 (labeling tool) 을 사용한다.

이미지 분류, 물체 감치, 시멘틱 세그멘테이션과 같은 컴퓨터 비전 태스크 공개된 어노테이션 도구가 많다.

OpenCV 의 CAT 은 이미지 검출에 사용되는 어노테이션 툴로, 무료로 베포하고 있다.

+ [OpenCV/cvat](https://github.com/opencv/cvat)

Imglab 어노테이션 툴은 COCO 데이터셋 형태로 저장이 가능하며, 키포인트 라벨링 (keypoints labeling) 도 가능하다.

- [COCO Dataset](https://cocodataset.org/#home)
- [NaturalIntelligence/imglab](https://github.com/NaturalIntelligence/imglab)


이 외에 

- labelimg ([https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg))
- labelme ([https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)) 

와 같은 어노테이션 툴도 있다.

이러한 어노테이션 툴은 QT 등 GUI 프레임워크를 이용하는 방법과 웹 기반으로 나눠지며, 단축키와 라벨 저장 포맷 등 편의성을 향상시키는 기능을 적용한다.

우리의 목적인 눈동자 위치 검출에 사용할 수 있는 도구는 imglab 의 키포인트 도구이다. 하지만 COCO 데이터셋에서 라벨링한 스타일을 가져가야하프로 17 개의 키포인트를 정해야하는 규칙이 있다.

이렇듯 목적에 맞는 라벨링 도구를 찾기란 쉽지않아 직접 제작하기도 한다.

-   [Linewalks - Data tech company for healthcare system](https://blog.linewalks.com/archives/6240)
<br></br>

하지만 프로그래밍 언어에 익숙하지 않다면 직접 제작하기란 쉽지않다. 따라서 간단하고 효율적인 라벨링 툴을 제작해야한다.

1.  기존에 만들어진 라벨이 잘 예측하는지 True / False 분류
2.  해당 태스크의 어노테이션 툴 (우리의 경우 키포인트 어노테이션)

위 2가지 목적에 맞게 OpenCV 를 이용해서 눈동자 검출이라는 목적에 맞는 라벨링 툴을 만들어 보자.

<br></br>
## 라벨링 툴 (02). 직접 제작하기

> 터미널 명령을 통해 어노테이션 툴 제작하기
> (디렉토리 안에 들어있는 많은 이미지들에 대해 True / False Binary Classification 을 위한 라벨링을 해주는 어노테이션 툴 제작)
```
# 우리가 제작하고자 하는 라벨링 툴의 사용법 
$ cd ~/aiffel/coarse_to_fine && python my_labeler_1st.py [imgpath or dir] [mask path or dir]
```
<br></br>
> 어노테이션 툴에 필요한 단축키 매핑
```
esc :  program  off  
n :  next  image  
p :  previous  image  
f :  true  tag  &  next  image  
d :  false  tag  &  next  image  
s :  save  v :  current  label  show
```
원본 이미지와 키포인트 위치에 라벨링 되어있는 정답을 모두 이미지 형태로 저장
<br></br>
> 답이 맞는지 알기 위해 `img_path` 와 `mask_path`에서 각 이미지를 읽어오기
```python
import os 
from os.path import join 
from glob import glob 
import cv2 
import numpy as numpy 
import argparse 
import numpy as np 
import json from pprint 
import pprint 

args = argparse.ArgumentParser() 

# hyperparameters 
args.add_argument('img_path', type=str, nargs='?', default=None) 

args.add_argument('mask_path', type=str, nargs='?', default=None) 

config = args.parse_args()
```
<br></br>
> 읽은 이미지를 적잘한 `blend_mask()` 함수를 통해 화면에 출력할 이미지로 만들기
```python
def blend_mask(img_orig, img_mask, alpha=0.3):
    '''
    alpha : alpha blending ratio. 0 ~ 1
    '''
    imgBlack = np.zeros(img_mask.shape, dtype=np.uint8)
    mask = (img_mask / img_mask.max()) * 255
    mask = mask.astype(np.uint8)

    if len(np.unique(mask)) > 2:
        # multi channel mask
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_white = cv2.merge((mask,mask,mask))
        mask_color = np.where(mask_white != 0, mask_color, 0)
    else:
        # 1 channel mask
        mask_color = cv2.merge((imgBlack, mask, mask))

    img_show = cv2.addWeighted(img_orig, 0.9, mask_color, alpha, 0.0)
    return img_show
```
`alpha` 값은 알파 블렌딩 (alpha blending) 기법의 블렌딩 상수이다.

- 참고:  [알파블렌딩(Alpha-blending)이란 무엇일까?](https://blog.daum.net/trts1004/12109286)

알파 플렌딩의 수식 : $pixelvalue = alpha * input + (1 - alpha) * mask$
<br></br>
> 이미지, 예상 라벨의 경로상 문제가 없는지 체크
```python
def check_dir():
    flg_mask = True
    if config.mask_path is None \
        or len(config.mask_path) == 0 \
        or config.mask_path == '':
        print ('[*] mask file not exist')
        flg_mask = False

    if config.img_path is None \
        or len(config.img_path) == 0 \
        or config.img_path == '' \
        or os.path.isdir(config.img_path):
        root = os.path.realpath('./')
        if os.path.isdir(config.img_path):
            root = os.path.realpath(config.img_path)
        img_list = sorted(glob(join(root, '*.png')))
        img_list.extend(sorted(glob(join(root, '*.jpg'))))
        config.img_path = img_list[0]

    img_dir = os.path.dirname(os.path.realpath(config.img_path))
    mask_dir = os.path.dirname(os.path.realpath(config.mask_path)) if flg_mask else None
    mask_dir = os.path.realpath(config.mask_path) if flg_mask and os.path.isdir(config.mask_path) else mask_dir

    return img_dir, mask_dir, flg_mask
```
`img_path`  가 디렉토리로 입력되는 경우(`os.path.isdir(config.img_path)`), 디렉토리 내에 있는 이미지 전체 인덱스를 찾고 첫 번째 이미지를 읽는다.

마스크로는,  `mask_path`  디렉토리에서 읽어진 이미지와 같은 이름을 갖는 라벨 이미지를 가지고 올 예정이다.
<br></br>
> 다음 이미지로 넘어가는 함수 구현
```python
def move(pos, idx, img_list):
    if pos == 1:
        idx += 1
        if idx == len(img_list):
            idx = 0
    elif pos == -1:
        idx -= 1
        if idx == -1:
            idx = len(img_list) - 1
    return idx
메인이 되는 함수를 만들겠습니다. img_list 의
```
`pos` 변수를 이ㅣ용해 순서 (`idx`) 를 하나씩 조절하고, 리스트 크기 이상이 되면 순서를 다시 0 으로 조정
<br></br>
> 메인이 되는 함수 구현
```python
def blend_view():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 500, 500)

    img_dir, mask_dir, flg_mask = check_dir()

    fname, ext = os.path.splitext(config.img_path)
    img_list = [os.path.basename(x) for x in sorted(glob(join(img_dir,'*%s'%ext)))]

    dict_label = {}
    dict_label['img_dir'] = img_dir
    dict_label['mask_dir'] = img_dir
    dict_label['labels'] = []

    json_path = os.getenv('HOME')+'/aiffel/coarse_to_fine/annotation.json'
    json_file = open(json_path, 'w', encoding='utf-8')

    idx = img_list.index(os.path.basename(config.img_path))
    while True:
        start = cv2.getTickCount()
        fname = img_list[idx]
        mname = fname
        orig = cv2.imread(join(img_dir, fname), 1)

        img_show = orig
        if flg_mask:
            mask = cv2.imread(join(mask_dir, mname), 0) 
            img_show = blend_mask(orig, mask)

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        print (f'[INFO] ({idx+1}/{len(img_list)}) {fname}... time: {time:.3f}ms')

        cv2.imshow('show', img_show)

        key = cv2.waitKey(0)
        if key == 27:   # Esc to Stop and Save Json result.
            return -1
        if key == ord('n'):
            idx = move(1, idx, img_list)
        elif key == ord('p'):
            idx = move(-1, idx, img_list)
        elif key == ord('f'):
            dict_label['labels'].append({'name':fname, 'class':1})
            idx = move(1, idx, img_list)
            print (f'[INFO] {fname}, class: true')
        elif key == ord('d'):
            dict_label['labels'].append({'name':fname, 'class':0})
            idx = move(1, idx, img_list)
            print (f'[INFO] {fname}, class: False')
        elif key == ord('v'):
            print ()
            pprint (dict_label)
            print ()
        elif key == ord('s'):
            json.dump(dict_label, json_file, indent=2)
            print (f'[INFO] < {json_path} > saved!')
    json_file.close()

if __name__ == '__main__':
    blend_view()
```
`img_list` 의 이미지들을 하나씩 읽으면서 `json_file`에 라벨을 하나씩 입력한다.

현재 이미지 순서를 알 수 있도록 출력하고 `p` 와 `f` 를 입력할 때 `dict_label` 에 정답을 입력하고 `s` 를 누르면 json 파일 형태로 저장한다.
<br></br>

[제작한 라벨링 툴](https://aiffelstaticprd.blob.core.windows.net/media/documents/my_labeler_1st.py)
<br></br>

제작한 라벨링 툴의 `img_path`는 이미지가 담겨 있는 디렉토리를 임의로 부여하면 된다.

 아래 예시는 `~/aiffel/coarse_to_fine/images` 디렉토리를 사용한 경우이다.

> 예시
```
# 라벨링 소스코드 다운로드 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/my_labeler_1st.py -P ~/aiffel/coarse_to_fine 

# 라벨링 수행 
$ cd ~/aiffel/coarse_to_fine && python my_labeler_1st.py ./images
```
정의한 단축키 정보를 기억 하자. `f` 나 `d` 를 이용해서 라벨을 부여하고, `s` 를 눌러 저장한 후 `esc` 를 눌러 파일에 저장한 후 프로그램을 종료한다.

위 소스코드에서 지정한 대로, 라벨링 파일은 `~/aiffel/coarse_to_fine/annotation.json` 으로 저장된다.
<br></br>

## Human - in - the - loop & Active Learning

이제 학습 시스템을 어떻게 효율적으로 만들어야하는지, 라벨링을 어떤 시점에서 시작해야하는지 알아보자.

먼저 전처리 후 키포인트 검출 모델을 만들어 눈동차 위치를 예측하였다.

<br></br>
### 1. Coarse 데이터셋 만들기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/01_9FpqmuA.max-800x600.png)

데이터가 없을 경우 Mean Shift 를 통해 Coarse 한 예측 결과를 만들었다.

이 예측 결과를 딥러닝 모델 학습의 데이터로 사용한다.

<br></br>
### 2. Fine Dataset 만들기

![](https://aiffelstaticprd.blob.core.windows.net/media/images/02_OPcJWpe.max-800x600.png)

Coarse dataset 은 라벨의 정확도가 낮기 때문에 fine 한 라벨을 얻을 수 있도록 개선이 필요하다.

얻은 라벨 정보를 이용하여 이미지 분류 모델을 만들 수 있는데 이때 CAM (Class Activation Map) 을 추가하여 어디가 잘못되었는지 조금 더 쉽게 확인할 수 있다.

- 참고: [Class Activation Map(Learning Deep Features for Discriminative Localization)](https://kangbk0120.github.io/articles/2018-02/cam)

CAM 은 CNN 을 통과한 logit 을 GAP 해서 채널 수 별로 embedding 한다. embedding layer 에 클래스 개수만큼 fully connected layer 를 통과시키며, 이 때 FC layer 의 weight 과 logit 의 가중합을 계산하면 CAM 을 얻을 수 있다.

만들어진 이미지 분류기는 fine label 을 만들때 사용한다.

예측 결과가 좋은 라벨은 fine label 로 간주하여 학습시키고 예측 결과가 나쁜 라벨 중 CAM 결과도 좋지 않다면 라벨링을 했을 때 효과적인 데이터셋을 만들 수 있을 것이다.
<br></br>

### 3. Active Learning

위 방법이 액티브 러닝 (active learning) 의 시작이다.

액티브 러닝이란 어떤 데이터를 선정할지 고민하고 모델이 사람에게 피드백을 주는 학습 방법이다.

이렇게 액티브 러닝을 통해 얻은 후보군을 직접 라벨링 한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/03_rhUKcXQ.max-800x600.png)
<br></br>

앞으로는 라벨링 툴을 사용해서 양질의 라벨링을 만들어낸다. 이렇게 만들어진 데이터셋은 Fine Dataset 이라고 한다.

Fine Dataset 으로 모델을 학습시키면 모델 개선의 반복문이 만들어진다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/04_FD7oeB4.max-800x600.png)
<br></br>

학습 후 다시 예측을 하면서 모델 성능을 개선해 나간다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/05_5gBiwDe.max-800x600.png)
<br></br>


