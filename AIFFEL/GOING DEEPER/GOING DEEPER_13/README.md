# 13. 내가 만든 카메라앱, 무엇이 문제일까?


## 학습목표

1.  **Detection, Landmark, Angle, Tracker**의 필요성 공감하기
2.  컴퓨터비전 프로그램의 성능 이해하기
3.  스티커앱 성능 분석하기
4.  동영상 처리
5.  칼만 필터로 동영상 성능 개선하기


## 사전 준비

> 작업을 위한 디렉토리 구성
```bash
$ mkdir -p ~/aiffel/video_sticker_app/models
```
<br></br>
> 필요 라이브러리 설치
```bash
$ sudo apt install libgtk2.0-dev pkg-config
```
```python
pip install cmake
pip install dlib
pip install opencv-contrib-python
```
<br></br>
> 사용할 모델 파일 다운 및 압축 해제
```bash
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P ~/aiffel/video_sticker_app/models/
$ bzip2 -d ~/aiffel/video_sticker_app/models/shape_predictor_68_fa
```
<br></br>

## 스티커앱의 원리

스티커앱은 사진에서 얼굴을 찾고, 눈, 코, 입의 랜드마크를 검출한다.

랜드마크를 기준으로 스티커를 합성할 위치를 정하게 되며, 해당 위치에 스티커를 적용하면 된다.

+ 참고 : [Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
<br></br>

## 동영상에 스티커 붙이기 (01). 동영상이란?

> 샘플로 사용할 동영상 다운
> 동영상 자료 : [video_sticker_app.zip](https://aiffelstaticprd.blob.core.windows.net/media/documents/video_sticker_app.zip)
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/video_sticker_app.zip -P ~/aiffel/video_sticker_app
$ cd ~/aiffel/video_sticker_app && unzip video_sticker_app.zip
```
<br></br>


### 동영상이란?

동영상은 여러개의 이미지가 일정한 시간 간격으로 이어진 이미지 시퀀스 데이터이다.

즉, 프레임이 마다 각각의 이미지로 구성되어 있으며, 동영상에 스티커를 적용하기 위해서는 매 프레임에 포함되어 있는 각각의 이미지에 스티커를 적용하고 이를 동일 간격으로 이어 붙이면된다.

<br></br>
### 동영상에서의 용어

+ `Frame` : 동영상에서 특정 시간대의 이미지 1장
+ `FPS` : Frame Per Second, 초당 프레임 수

FPS 는 초당 프레임 수이다. 즉, 60 프레임이라면 초당 60개의 프레임 (이미지) 가 표시된다는 것이다.

사람은 최소 15 fps 를 동영상으로 인식한다고 한다.

대부분의 동영상은 30 fps 로 제작이되며, 빠른 순간을 잡아야하는 경우에는 60 fps 이상을 사용한다고 한다.

fps 의 값이 높을수록 부드럽다고 느끼지만 실제로 60 fps 이상은 사람의 눈으로 체감하기 힘들다고 한다.

<br></br>
## 동영상에 스티커 붙이기 (02). 동영상 처리 방법

### 동영상 확인하기

`moviepy` 패키지를 이용하여 동영상을 다룰 수 있다.

> `moviepy` 패키지 설치
```python
pip install moviepy
```
<br></br>
> 사용할 라이브러리 가져오기
```python
from moviepy.editor import VideoFileClip
from moviepy.editor import ipython_display
```
`VideoFileClip` 은 비디오 파일을 읽어올 때 사용되는 클래스이며, `ipython_display()` 는 동영상을 주피터 노트북에 렌더링할 수 있게 도와주는 함수이다.
<br></br>
> 동영상 불러오기
```python
import os
video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/video2.mp4'
clip = VideoFileClip(video_path)
clip = clip.resize(width=640)
clip.ipython_display(fps=30, loop=True, autoplay=True, rd_kwargs=dict(logger=None))
```
비디오 파일의 크기가 `HD(1280x720)` 이기 때문에 랩탑에서 보기 쉽도록 크기를 줄였다.

`ipython_display()` 함수는 비디오 클립을 출력하는 기능을 하며, `loop` 과 `autoplay` 는 각각 반복 재생과 자동 재생 옵션이다.
<br></br>

### 동영상 처리하기

`moviepy` 를 사용하면 이미 저장된 동영상을 쉽게 출력할 수 있다.

하지만 실시간으로 동영상을 처리하는 것은 어떨까?

이렇게 실시간으로 동영상을 처리하기 위해서는

1. 동영상을 읽고
2. 프레임 별 이미지를 넘파이 형태로 추출하고
3. 얼굴 검출과 같은 이미지 처리를 수행하고
4. 다시 동영상으로 조합

위와 같은 과정을 실시간으로 처리해야한다.

동영상을 읽어오는 것은 쉽지만 이후 과정을 수행해야하지 때문에 실시간으로 출력하는 것은 어려운 작업이다.

이러한 문제로 동영상을 다룰 때는 주피터 노트북보다는 코드 에디어톼 터미널을 주로 이용한다.

<br></br>
> OpenCV 를 통해 동영상 다루기
```python
import cv2

vc = cv2.VideoCapture('./images/video2.mp4')

vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print (vlen) # video length

for i in range(vlen):
    ret, img = vc.read()
    if ret == False:
        break

    cv2.imshow('show', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
```
위 코드를 터미널에서 아래와 같이 실행한다.
```bash
$ cd ~/aiffel/video_sticker_app && python videocheck.py
```
새로운 창이 띄워지며 동영상이 재생될 것이다. `esc` 를 누르면 종료된다.
<br></br>

위 코드들을 하나씩 살펴보자.

> 동영상 읽어오기
```python
vc = cv2.VideoCapture('./images/video2.mp4')
```
동영상을 읽어오는데 오디오 정보는 포함되지 않는다.
<br></br>
> 동영상이 가진 정보를 읽어오기
```python
vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
```
`vc`의 `get()` 함수를 통해 동영상이 가진 정보를 읽어오며, `FRAME_COUNT` 은 비디오 전체 프레임 개수를 의미한다.
<br></br>
> img 를 읽어오기
```python
ret, img = vc.read()
```
`vc` 객체를 통해 `read()` 함수로 `img` 를 읽어온다. `ret` 는 `read()` 함수에 이미지가 반환될 경우 `True`, 반대의 경우 `False` 를 받는다.
<br></br>
> `ret` 값이 `False` 일때 프로그램을 중단
```python
if ret == False:
    break
```
<br></br>
> 동영상 출력
```python
cv2.imshow('show', img)
key = cv2.waitKey(1) 
if key == 27:
    break
```
`imshow()` 함수를 사용해서 동영상 출력한다.

`waitKey()` 함수 파라미터에 wait time 값을 적절히 넣으면 루프를 돌면서 이미지가 연속적으로 화면에 출력된다. 

이미지를 연속으로 재생시키면 동영상으로 볼 수 있게된다. 이때 wait time 값은 ms 단위이다. 

즉, 30 fps 값을 가진다고 했을 때 wait time 을 33 으로 입력하면 비슷한 속도로 출력된다. 이미지 처리 시간을 고려한다면 조금 더 줄여야 할 수도 있다.

`waitKey()` 함수는 키보드가 입력될 때 키보드 값을 반환합니다. `if key == 27` 는 '27 번 키 값을 가지고 있는 키보드 버튼이 입력될 때' 를 의미하며, 27 번 키는 ESC 이다.
<br></br>

## 동영상에 스티커 붙이기 (03). 이미지 시퀀스에 스티커앱 적용하기

이제 동영상에 스티커앱을 적용해보자.

> 파일 다운로드 및 압축해제, 저장경로 설정
> [스티커 이미지](https://aiffelstaticprd.blob.core.windows.net/media/documents/king.zip)
```bash
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/king.zip 
$ unzip king.zip -d ~/aiffel/video_sticker_app/images
```
스티커 이미지 `king.png`가 `images`폴더 내부에 있어한다.
<br></br>
> 이미지 한장을 처리하는 스티커 앱 함수 생성
```python
import dlib
import cv2

def img2sticker_orig(img_orig, img_sticker, detector_hog, landmark_predictor):
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # detector
    dlib_rects = detector_hog(img_rgb, 0)
    if len(dlib_rects) < 1:
        return img_orig

    # landmark
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)

    # head coord
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[30][0] # nose
        y = landmark[30][1] - dlib_rect.width()//2
        w = dlib_rect.width()
        h = dlib_rect.width()
        break

    # sticker
    img_sticker = cv2.resize(img_sticker, (w,h), interpolation=cv2.INTER_NEAREST)

    refined_x = x - w // 2
    refined_y = y - h

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:]
        refined_y = 0

    img_bgr = img_orig.copy()
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)

    return img_bgr
```
`addsticker.py` 파일로 압축 파일에 이미 포함되어 있다.

스티커가 머리 위로 이미지 경계를 벗어나는 상황을 고려해 `refined_y` 가 0 보다 작을 때 `img_sticker[-refined_y:]` 만 표시되게 한다.
<br></br>
> 동영상에 스티커를 적용
```python
detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

vc = cv2.VideoCapture('./images/video2.mp4')
img_sticker = cv2.imread('./images/king.png')

vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print (vlen) # 비디오 프레임의 총 개수
```
기술자 `vc` 에서 `img` 를 읽으며, 읽은 `img` 를 `img2sticker_orig()` 에 입력한다.

이를 통해 스티커가 적용된 이미지를 출력하고, `imshow` 함수로 `img_result` 를 화면에 렌더링한다.
<br></br>
> `img2sticker_orig()` 함수의 시간을 측정하는 기능 추가
```python
for i in range(vlen):
    ret, img = vc.read()
    if ret == False:
        break

    ## 추가된 부분
    start = cv2.getTickCount()
    img_result = img2sticker_orig(img, img_sticker.copy(), detector_hog, landmark_predictor)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('[INFO] time: %.2fms'%time)

    cv2.imshow('show', img_result)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
```
OpenCV에서는 `getTickCount()` 와 `getTickFrequency()` 를 사용해서 시간을 측정한다.

초 단위로 나오기 때문에 일반적으로 이미지 한 장을 처리할 때 1,000 을 곱해 ms 단위로 속도를 관찰한다.

단위를 ms 로 하는 이유는 fps 가 프레임당 ms 단위를 가지기 때문이다.

위 코드를 `addsticker.py` 에 저장한다.
<br></br>
> 1초로 잘라 스티커가 적용된 영상을 확인
```bash
cd ~/aiffel/video_sticker_app && python addsticker.py
```
<br></br>

## 동영상에 스티커 붙이기 (04). 동영상 저장하기

OpenCV 로 처리한 동영상은 어떻게 저장할까?

> OpenCV 로 동영상을 저장하기
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter('./images/result.mp4', fourcc, 30, (1280, 720))
```
위 코드를 마지막 부분에 추가 후 파이썬 파일을 실행하면 동영상을 저장할 수 있다.

`VideoWriter` 로 동영상 핸들러를 정의하며, 저장할 파일 이름, 코덱 정보, fps, 동영상 크기를 파라미터로 입력한다.

코드의 "fourcc"는 "four character code"의 약자로, 코덱의 이름을 명기하는 데 사용한다. 이는 컴퓨터 os 에 따라 지원하는 코덱 (`avc1`, `mp4v`, `mpeg`, `x264` 등) 이 다르기 때문에 알맞은 코덱은 선택하여 사용해야한다.

우분투에서는 `mp4v` 를 사용한다.
<br></br>
> os 별 어떤 코덱을 지원하는지 알아내기
```python
fourcc = int(vc.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "%c%c%c%c"%(fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255)
print ("CAP_PROP_FOURCC: ", fourcc_str)
```
`vc.get()`을 사용해서 알아낼 수 있다. 이때 얻어지는 값은 정수형이기 때문에 비트연산을 이용해서 `char` 형태로 변경한다.

`checkfourcc.py`에 입력 동영상의 `fourcc`를 알아내는 코드 예제를 작성해두었다.
<br></br>
> 원본 동영상에 스티커를 붙여 합성한 영상을 `result.mp4` 파일로 최종 저장하는 코드
```python
detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

vc = cv2.VideoCapture('./images/video2.mp4')
img_sticker = cv2.imread('./images/king.png')

vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
print (vlen) # 비디오 프레임의 총 개수

# writer 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter('./images/result.mp4', fourcc, 30, (1280, 720))

for i in range(vlen):
    ret, img = vc.read()
    if ret == False:
        break

    start = cv2.getTickCount()
    img_result = img2sticker_orig(img, img_sticker.copy(), detector_hog, landmark_predictor)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('[INFO] time: %.2fms'%time)

        # 매 프레임 마다 저장합니다.    
    vw.write(cv2.resize(img_result, (1280,720)))

    cv2.imshow('show', img_result)
    key = cv2.waitKey(1)
    if key == 27:
        break

vw.release()
cv2.destroyAllWindows()
```
위 코드를 `savevideo.py`에 저장 후 터미널에서 다음과 같이 실행한다.

```bash
$ cd ~/aiffel/video_sticker_app && python savevideo.py
```
<br></br>

## 동영상에서 문제점 찾기

### 너무 느린 속도

앞에서 개별 프레임을 처리하는 시간을 측정하는 코드를 실행해보면 한 장을 이미지를 처리하는데 걸리는 시간이 100 ms 정도가 나온다.

물론 이 속도는 컴퓨터 자원에 따라 달라지겠지만 이 속도가 느리면 출력 결과도 만족스럽지 않다.

우리가 만든 프로그램은 10 fps 의 속도를 가지므로 동영상보다는 이미지를 조금 빠르게 넘기는 것에 가까울 것이다.

보다 동영답게 하기 위해서는 프레임 처리 시간을 33 ms 이하로 줄여야 할 것이다. 이를 위해서는 프로그램 내의 병목현상이 있는 함수를 파악하고, 이를 개선해야한다.

<br></br>
### 스티커가 안정적이지 못하고 떨리는 현상이 발생

동영상에서 스티커 합성이 자연스럽지 못한 것을 보로 수 있는데 2 가지 원인이 있다.

1.  얼굴 검출기의 바운딩 박스 크기가 매 프레임마다 변경되기 때문에

2.  바운딩 박스의 x, y 위치를 측정할 때 에러가 존재 (가우시안 노이즈 형태)

1 번의 발생 이유는 알고리즘 문제이며, 2번의 발생 이유는 학습 데이터의 라벨 정확도가 떨어지기 때문이다.

<br></br>
### 고개를 돌리면 자연스럽지 하다.

고개를 좌우로 돌리거나 얼굴의 각도가 바뀌면 이에따라 스티커도 함께 바껴야 자연스럽다.

하지만 부자연스러운 이유는 바운딩 박스가 얼굴의 각도를 반영하기 못하기 때문에 발생한다.

특히 3 차원 공간을 2 차원으로 투명할 때 고개를 좌우로 돌리며 갸우뚱하는 경우 원근 (perspective) 변환을 적용해야한다.

이 변환을 위해 3 차원에서 `yaw`, `pitch`, `roll` 각도를 알고 있어야 한다.

<br></br>
### 얼굴을 잘 못 찾는다.

카메라에서 조금만 멀어져도 얼굴을 찾지 못하는 경우가 발생할 수 있는데 이는 얼굴을 검출하는 Object Detection 의 성능이 낮기 때문이다. 

HOG 알고리즘 기반으로 학습된 박스는 일정 크기 이상의 박스를 출력하는데, 화면에서 멀어지면 얼굴 크기가 작아지므로 학습된 박스 크기와 다르게 나올 가능성이 높아진다.

<br></br>
## 더 빠른 스티커앱 (01). 실행시간 분석하기

`img2sticker_orig()` 함수 내부의 알고리즘 속도를 분석해 속도가 느린 문제를 분석해보자.

`img2sticker_orig()` 함수는 5 단계로 나눠진다.

1.  전처리 (Preprocess)
2.  얼굴 검출 (Detection)
3.  랜드마크 검출 (Landmark)
4.  좌표변환 (Coord)
5.  스티커 합성 (Sticker)

이제 위 다섯 단계에서 각 각 걸리는 시간을 `img2sticker_orig()` 함수를 수정함으로써 측정해보자.

> `img2sticker_orig()` 함수에 각 단계별 소요시간 측정 기능을 추가
```python
def img2sticker_orig(img_orig, img_sticker, detector_hog, landmark_predictor):
    # preprocess
    start = cv2.getTickCount()
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    preprocess_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    # detector
    start = cv2.getTickCount()
    dlib_rects = detector_hog(img_rgb, 0)
    if len(dlib_rects) < 1:
        return img_orig
    detection_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    # landmark
    start = cv2.getTickCount()
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)
    landmark_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    # head coord
    start = cv2.getTickCount()
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[30][0] # nose
        y = landmark[30][1] - dlib_rect.width()//2
        w = dlib_rect.width()
        h = dlib_rect.width()
        # x,y,w,h = [ele*2 for ele in [x,y,w,h]]
        break
    coord_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    # sticker
    start = cv2.getTickCount()
    img_sticker = cv2.resize(img_sticker, (w,h), interpolation=cv2.INTER_NEAREST)

    refined_x = x - w // 2
    refined_y = y - h

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:]
        refined_y = 0

    img_bgr = img_orig.copy()
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)
    sticker_time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

    print (f'p:{preprocess_time:.1f}ms, d:{detection_time:.1f}ms, l:{landmark_time:.1f}ms, c:{coord_time:.1f}ms, s:{sticker_time:.1f}ms')

    return img_bgr
```
각 단계마다 시간을 측정하고 마지막에 결과를 출력해준다.

`addsticker.py`를 카피해서 만든 `addstiker_timecheck.py` 안에 `img2sticker_orig()` 함수 부분만 위 내용을 수정하여 저장하였다.
<br></br>

> 위에서 생성한 함수를 터미널에서 실행
```bash
$ cd ~/aiffel/video_sticker_app && python addsticker_timecheck.py
```
실행 시간 대부분이 `d`, _detection_에서 소모되는 것을 알 수 있으며, 검출 시간에서 가장 큰 시가을 소요하는 것을 알 수 있다.

따라서 검출 시간을 줄이는 것을 목표로 해야한다.
<br></br>

## 더 빠른 스티커앱 (02). 얼굴 검출기 이해하기 - 속도, 안정성

`dlib` 얼굴 검출기(face detector)는 **HOG(histogram of oriented gradient) 기반 알고리즘** 을 사용한다.

+ 참고 : [기계 학습(Machine Learning, 머신 러닝)은 즐겁다! Part 4](https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-4-63ed781eee3c)
<br></br>

`dlib`은 HOG 특징 공간에서 **슬라이딩 윈도우(sliding window)** 기법을 통해 얼굴을 검출한다.

![](https://pyimagesearch.com/wp-content/uploads/2015/03/sliding-window-animated-adrian.gif)

+ 참고 : [Sliding Windows for Object Detection with Python and OpenCV - PyImageSearch](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
<br></br>

HOG 특성 맵에서 입력 이미지 크기(HD : 1280 x 720) 만큼 슬라이딩 윈도우를 수행하기 때문에 프로그램은 $$O(1280 * 720 * (bbox~size) * 피라미드~개수) = O(n^3)$$ 의 시간을 소요한다.

이렇게 느려진 속도를 개선하는 가장 간단한 방법은 사용되는 이미지의 크기를 줄이거나 피라미드 수를 줄이는 방법이 있다.

> `img2sticker_orig()` 함수를 수정
```python
def img2sticker_orig(img_orig, img_sticker, detector_hog, landmark_predictor):
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # detector
    # detector*와 landmark*에 입력되는 img_rgb 이미지를 VGA 크기로 1/4만큼 감소시킵니다.
    img_rgb_vga = cv2.resize(img_rgb, (640, 360))
    dlib_rects = detector_hog(img_rgb_vga, 0)
    if len(dlib_rects) < 1:
        return img_orig

    # landmark
    # 줄인만큼 스티커 위치를 다시 2배로 복원해야 합니다.
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb_vga, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)

    # head coord
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        x = landmark[30][0] # nose
        y = landmark[30][1] - dlib_rect.width()//2
        w = dlib_rect.width()
        h = dlib_rect.width()
        # 줄인 만큼 스티커 위치를 다시 2배로 복원해야 합니다.
        x,y,w,h = [ele*2 for ele in [x,y,w,h]]
        break

    # sticker
    img_sticker = cv2.resize(img_sticker, (w,h), interpolation=cv2.INTER_NEAREST)

    refined_x = x - w // 2
    refined_y = y - h

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:]
        refined_y = 0

    img_bgr = img_orig.copy()
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)

    return img_bgr
```
`addsticker.py`를 카피해서 만든 `addstiker_modified.py` 안에 `img2sticker_orig()` 함수 부분만 위와 같이 수정하여 저장했다.
<br></br>
> 수정한 함수 실행
```bash
$ cd ~/aiffel/video_sticker_app && python addsticker_modified.py
```
이론적으로 가로, 세로 크기의 절반씩 전체 1/4 계산량이 감소하기 때문에 소모 시간도 25% 정도 되어야 한다.
<br></br>

## 동영상 안정화 (stabilizer) 설계 방법 (01). 신호와 신호처리

스티커가 떨리는 현상을 개선하려면 먼저 신호와 노이즈에 대해 알아야 한다.


### 신호 (Signal) 와 노이즈 (Noise)

신호란 시공간에 따라 변화하는 물리량을 나타내는 함수를 의미하며, 주로 시간과 관련된 물리량을 나타낸다.

신호는 파형을 이루고 있고다고 알려져 있지만 실제로 신호를 측정해볼 때 깔끔한 파형의 형태가 잘 나오지 않는다. 이는 신호에 노이즈가 섞이기 때문이다.

노이즈가 섞이는 이유는 크게 2가지 이다.

1.  신호를 출력하는 모델의 노이즈
2.  신호를 측정할 때 생기는 노이즈

신호를 출력할는 모델의 노이즈는 사람이 달리기를 할 때 항상 일정한 속도로 뛰지 못하기 때문에 조금 빨리 뛰었다가 늦게 뛰었다가 하기 때문이다.

신호를 측정할 때 생기는 노이즈는 측정 카메라가 사람 다리를 측정할 때와 머리를 측정할 때 속도가 달라지는 경우와 같다.

이렇게 측정시 생기는 오차를 측정 오차라고 한다. 이러한 측정오차로 인해 매번 같은 위치의 얼굴을 찾아내기 힘든 한계가 존재한다.

이론상 신호 이미지 : ![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-7-L-11.png)

실제 측정된 신호 이미지 : 
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-7-L-12.max-800x600.png)
<br></br>
### 신호처리 (Signal Processing)

노이즈는 같은 시간 동안 신호의 크기가 급격하게 변화한다. 이런 현상을 주파수가 높다 (High Frequency) 라고 표현한다. 

이렇게 일정 수준이상의 높은 주파수는 제거하고 낮은 주파수만 통과하는 것, 즉 원본 신호를 원하는 형태의 신호로 만드는 방법을 신호처리 (Signal Processing) 라고 한다.

높은 주파수를 제거하고  낮은 주파수만을 통과하는 하는 기술은 LPF (Low Pass Filter) 라고 한다.

LPF 를 구현하기 위해서는 이전 5개 측정값의 평균을 이용해서 구현할 수 있다. 하지만 이렇게 이전 5개 측정값의 평균을 이용하기 때문에 이전값에 큰 영향을 받아 Delay 가 생긴다. 즉, 현재의 실제 값과 차이가 반드시 존재하게 된다.

+ 참고 : [LPF의 이해 (유투브)](https://youtu.be/JezOJN1yhTA)
<br></br>

## 동영상 안정화 (Stabilizer) 설계 방법 (02). 칼만 필터

칼만 필터 (Kalman Filter) 란 시스템 모델과 측정 모델을 만들고 데이터 입력을 바탕으로 각 모델을 예측하는 알고리즘을 의미한다.

예측된 모델을 바탕으로 현재의 실제 값을 추정할 수 있고, 다음 시점의 모델 출력을 예측할 수 있다.

이때 시스템 모델과 측정 모델은 모두 선형이며, 가우시안 분포를 따르는 경우를 가정한다.

칼만 필터에서는 A, Q, H, R 의 4의 행렬을 사용하며, A : 시스템 모델, Q : 시스템 오차(노이즈), H : 측정 모델, R : 측정 오차(노이즈) 의 의미를 가진다.

또한 칼만 필터는 예측 단계와 추정 단계 2 가지 단계로 나눌 수 있다.

마지막으로 칼만 게인 (Kalman Gain) 은 측정값과 추정값 중 추정값에 영향을 미치는 가중치를 의미한다.

+ 칼만 필터의 단계를 정리한 플로우 차트 (Flow Chart) : 
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-7-L-13.max-800x600.png)

+ 참고 : [Kalman filter 소개](https://medium.com/@celinachild/kalman-filter-%EC%86%8C%EA%B0%9C-395c2016b4d6)
<br></br>

코드에서 얼굴 검출 과 얼굴 랜드마크는 프레임마다 가우시안 오차를 갖는 측정 시스템이며, 라벨링을 할 때 사람도 항상 같은 위치에서 찍을 수 없고 자연 상태의 측정값은 대체로 정규 분포를 따른다.

따라서 얼굴 랜드마크에 칼만 필터를 적용하면 비교적 안정적인 스티커 결과를 얻을 수 있다.
<br></br>

## 칼만 필터 적용하기

얼굴 랜드마크에 칼만 필터를 적용해보자.
칼만 필터를 적용하기 위해서는 좌표에서 객체의 위치와 속도에 대한 모델을 먼저 만들어야 한다.
<br></br>
> 사용할 라이브러리 가져오기
```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
```
<br></br>
> 초기 위치 0에서 시작해서 시간 단위 `dt` 로 `idx` 만큼 이동 했을 때 오차
```python
def geet_pos_vel(idx, dt, init_pos=0, init_vel=20):
    w = np.random.normal(0, 1)                # w: system noise.    v = np.random.normal(0, 2)                # v: measurement noise.    vel_true = init_vl + w                   # nominal velocity = 80 [m/s].
    pos_true = init_pos + sum([vel_true*dt for i in range(idx)])
    z_pos_meas = pos_true + v                 # z_pos_meas: measured position (observable)
    
    return z_pos_meas, vel_true, pos_true     #, v, w
```
필연적으로 $(w \sim N(0,1))$ 만큼 조금씩 오차가 발생한다.
<br></br>
> 1초단위로 10번 측정
```python
for i in range(0,10):
    print (get_pos_vel(i, 1))
```
<br></br>
> 칼만 필터를 구현하는 `kalman_filter()` 함수 생성
```python
def kalman_filter(z, x, P):
# Kalman Filter Algorithm
    # 예측 단계
    xp = A @ x
    Pp = A @ P @ A.T + Q

    # 추정 단계
    K = Pp @ H.T @ inv(H @ Pp @ H.T + R)
    x = xp + K @ (z - H @ xp)
    P = Pp - K @ H @ Pp
    return x, P
```
코드에서 사용된 `@` 연산자는 Matrix Multiplication 산자이다. 

+ 참고 : [Matrix Multiplication](https://alysivji.github.io/python-matrix-multiplication-operator.html)
<br></br>
> 칼만 필터 시스템 행렬 등 초기값 지정
```python
# time param
time_end = 5
dt= 0.05
```
<br></br>
> 칼만 필터 시스템 모델 A 를 `pos + vel * dt` 형태의 연산이 가능하도록 설계
> (칼만 필터 입력으로 위치와 속도을 사용합니다 (`x = [pos, vel]`))
```python
# init matrix
A = np.array([[1, dt], [0, 1]]) # pos * 1 + vel * dt = 예측 위치
H = np.array([[1, 0]])
Q = np.array([[1, 0], [0, 1]])
R = np.array([[200]])
```
오차 행렬을 일정 기준에 따라 초기화해 준다.

+ 참고 : [오차 행렬 초기화에 대한 참고자료](https://www.researchgate.net/post/how_to_initialize_the_error_covariance_matrix_and_process_noise_covariance_matrix_How_are_they_different_and_in_what_way_they_impact_the_filter/561e6af65f7f71e2648b4615/citation/download)
<br></br>
> 추정을 위한 초기화 오차행렬 초기화
```python
# Initialization for estimation.
x_0 = np.array([0, 20])  # position and velocity
P_0 = 1 * np.eye(2)
```
<br></br>

### 결과

결과를 저장할 곳을 만들어 두자. 시간 축으로 `time_end * dt` 개의 공간이 필요하다.
<br></br>
> 결과를 저장할 변수 생성
```python
time = np.arange(0, time_end, dt)
n_samples = len(time)
pos_meas_save = np.zeros(n_samples)
vel_true_save = np.zeros(n_samples)
pos_esti_save = np.zeros(n_samples)
vel_esti_save = np.zeros(n_samples)
```
<br></br>
> 저장할 공간에 실제 위치와 칼만 필터 결과를 대입
```python
pos_true = 0
x, P = None, None
for i in range(n_samples):
    z, vel_true, pos_true = get_pos_vel(i, dt)
    if i == 0:
        x, P = x_0, P_0
    else:
        x, P = kalman_filter(z, x, P)

    pos_meas_save[i] = z
    vel_true_save[i] = vel_true
    pos_esti_save[i] = x[0]
    vel_esti_save[i] = x[1]
```
<br></br>
> 결과 시각화
```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time, pos_meas_save, 'r*--', label='Measurements', markersize=10)
plt.plot(time, pos_esti_save, 'bo-', label='Estimation (KF)')
plt.legend(loc='upper left')
plt.title('Position: Meas. v.s. Esti. (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')

plt.subplot(1, 2, 2)
plt.plot(time, vel_true_save, 'g*--', label='True', markersize=10)
plt.plot(time, vel_esti_save, 'bo-', label='Estimation (KF)')
plt.legend(loc='lower right')
plt.title('Velocity: True v.s. Esti. (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Velocity [m/s]')
```
좌측 그래프는 시간과 위치에 해당하는 그래프이며, 우측 그래프는 시간과 속도에 해당하는 그래프이다.

빨간색으로 표현된 실제 위치는 오차가 존재하기 때문에 노이즈 역시 존재한다. 하지만 칼만 필터가 적용된 파란색으로 표현된 위치는 안정적인 것을 확인할 수 있다.
<br></br>

이렇게 칼만 필터를 2차원으로 확장하여 동영상에 동일하게 적용할 수 있으며, 이를 통해 스티커앱의 동작을 개선해 볼 수 있다.


