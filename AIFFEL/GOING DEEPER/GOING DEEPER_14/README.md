# 14. 카메라 스티커앱을 개선하자!

카메라 스티커 앱을 동영상 처리할 수 있도록 개선해보자.

동영상 처리를 위해서는 피사체의 움직임에 의해 발생하는 예외상황, 처리속도 문제, 화면떨림, 안전성 문제 등에 대응이 필요하다.


## 학습목표

1.  동영상 쉽게 다룰 수 있는 방법 익히기
2.  스티커앱 성능 분석하기 + 간단한 개선 방법
3.  원하는 스펙을 수치로 지정하기
4.  칼만 필터로 동영상 성능 개선하기


## 사전준비

> 작업 디렉토리 구성
```
$ mkdir -p ~/aiffel/video_sticker_app/models 
$ mkdir -p ~/aiffel/video_sticker_app/images
```
<br></br>
> 사용할 데이터 다운
```
$ cd ~/aiffel/video_sticker_app 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/new_video_sticker_app.zip 
$ unzip new_video_sticker_app.zip
```
* [사용할 데이터](https://aiffelstaticprd.blob.core.windows.net/media/documents/new_video_sticker_app.zip) 
<br></br>

## 프로젝트 (01). Moviepy 로 비디오 처리하기

동영상을 다룰 수 있는 방법으로 파이썬 기반의 `moviepy` 라이브러리가 있다.

`moviepy` 가 동영상을 다룰 때 가장 중요한 처리 속도 측면에서 적합한지 확인해보자.

> 라이브러리 임포트
```python
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.editor import ipython_display
import cv2
import numpy as np
import os
```
<br></br>

샘플로 제공된 `video.mp4` 를 `moviepy` 로 읽어서 `width = 640` 으로 축소하여 화면에서 플레이해보고 플레이한 내용을 `mvresult.mp4` 라는 파일로 저장해보자.

또한 원본과 저장된 두 영상의 화면크기나 파일 용량을 비교해보자.

>`moviepy` 를 통해 노트북 상에서 비디오를 읽고 쓰는 프로그램 작성
```python
# 읽기
video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/video2.mp4'
clip = VideoFileClip(video_path)
clip = clip.resize(width=640)
clip.ipython_display(fps=30, loop=True, autoplay=True, rd_kwargs=dict(logger=None))

# 쓰기
result_video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/mvpyresult.mp4'
clip.write_videofile(result_video_path)
```
<br></br>
> `moviepy` 로 읽은 동영상을 넘파이 형태로 변환하고 영상 밝기를 50% 어둡게 만든 후 저장
```python
# 읽기
video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/video2.mp4'
clip = VideoFileClip(video_path)
clip = clip.resize(width=640)
clip.ipython_display(fps=30, loop=True, autoplay=True, rd_kwargs=dict(logger=None))

# clip 에서 numpy 로 데이터 추출
vlen = int(clip.duration*clip.fps)
video_container = np.zeros((vlen, clip.size[1], clip.size[0], 3), dtype=np.uint8)
for i in range(vlen):
    img = clip.get_frame(i/clip.fps)
    video_container[i] = (img * 0.5).astype(np.uint8)

# 새 clip 만들기
dur = vlen / clip.fps
outclip = VideoClip(lambda t: video_container[int(round(t*clip.fps))], duration=dur)

# 쓰기
result_video_path2 = os.getenv('HOME')+'/aiffel/video_sticker_app/images/mvpyresult2.mp4'
outclip.write_videofile(result_video_path2, fps=30)
```
<br></br>
> 영상을 읽고 쓰는 시간을 측정
> (OpenCV 를 사용할 때와의 차이 측정)
```python
# CASE 1 : moviepy 사용
start = cv2.getTickCount()
clip = VideoFileClip(video_path)
clip = clip.resize(width=640)

vlen = int(clip.duration*clip.fps)
video_container = np.zeros((vlen, clip.size[1], clip.size[0], 3), dtype=np.uint8)

for i in range(vlen):
    img = clip.get_frame(i/clip.fps)
    video_container[i] = (img * 0.5).astype(np.uint8)

dur = vlen / clip.fps
outclip = VideoClip(lambda t: video_container[int(round(t*clip.fps))], duration=dur)

mvpy_video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/mvpyresult.mp4'
outclip.write_videofile(mvpy_video_path, fps=30)

time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
print (f'[INFO] moviepy time : {time:.2f}ms')

# CASE 2 : OpenCV 사용
start = cv2.getTickCount()
vc = cv2.VideoCapture(video_path)

cv_video_path = os.getenv('HOME')+'/aiffel/video_sticker_app/images/cvresult.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(cv_video_path, fourcc, 30, (640,360))

vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(vlen):
    ret, img = vc.read()
    if ret == False: break
    
    img_result = cv2.resize(img, (640, 360)) * 0.5
    vw.write(img_result.astype(np.uint8))
    
time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
print (f'[INFO] cv time : {time:.2f}ms')
```
<br></br>

## 프로젝트 (02). 어디까지 만들고 싶은지 정의하기

### 1. 실시간 카메라 스티커앱을 만들어보자.

익스플로레이션의 카메라 스티커앱 만들기에서 `img2sticker_orig` 라는 이름의 메소드를 구현하여 `addsticker.py` 에 저장하였다.

해당 메소드를 복하사여 `img2sticker` 라는 이름의 메소드를 만들고 이를 `newaddsticker.py` 에 저장하자.

이 메소드를 보완하여 구현하는 방식으로 진행한다.

그렇다면 동영상 입력은 어떻게 받을 수 있을까?
쉽게는 노트북에 달려있는 웹캠이나 스마트폰의 카메라를 통해 입력받을 수 있다.

### 1. 웹캠을 입력으로 사용하는 경우

`cv2.VideoCapture(0)`  을 이용하면 웹캠 입력을 받을 수 있다.

> 웹캠을 이용한 실시간 스티커앱 구현
```python
import numpy as np
import cv2
import dlib

from newaddsticker import img2sticker

detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def main():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 640, 360)

    vc = cv2.VideoCapture(0)   # 연결된 영상 장치의 인덱스, 하나만 있는 경우 0을 사용
    img_sticker = cv2.imread('./images/king.png')

    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print (vlen) # 웹캠은 video length 가 0 입니다.

    # 정해진 길이가 없기 때문에 while 을 주로 사용합니다.
    # for i in range(vlen):
    while True:
        ret, img = vc.read()
        if ret == False:
            break
        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전

        # 스티커 메소드를 사용
        img_result = img2sticker(img, img_sticker.copy(), detector_hog, landmark_predictor)   

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)

        cv2.imshow('show', img_result)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
```
`img2sticker` 를 활용하여 웹캠 기반 실시간 스티커앱을 만들어 보았다.

여기서 파라미터 `0` 은 시스템에 연결된 영상 입력장치의 인덱스이다. 대부분의 경우 웹캠은 1 개 이므로 `0` 을 사용하면 된다.
<br></br>

위에서 구현한 웹캠 기반 실시간 스티커앱을 `webcam_sticker.py` 에 저장하여 터미널로 실행해 보자.

```
$ cd ~/aiffel/video_sticker_app && python webcam_sticker.py
```

얼굴을 인식하고 왕관을 씌워주는 동영상 기반 스티커앱이 작동하는 것을 볼 수있다.

<br></br>
### 2. 스마트폰 영상의 스트리밍 입력을 사용하는 경우

스마트폰 영상의 스트리밍을 입력으로 사용할 때 동영상 스트리밍을 위한 다양한 프로토콜 중, OpenCV 로 처리할 수 있는 `RTMP` 를 활용할 수 있다.

`RTMP` 는 다양한 동영상 스트리밍 서비스에 사용되는 프로토콜 중 하나이다.

* 참고 : [`RTMP`](https://juyoung-1008.tistory.com/30)
<br></br>

#### 1. 스마트폰에 `RTMP` 스트리밍 어플리케이션 설치

RTMP 스트리밍을 위해서는 별도의 서버에서 `rtmp://` url 을 오픈해줘야한다. 어플 Broadcast Me 
는 자동으로 url 을 생성해주기 때문에 편리한다.

하지만 다른 어플을 사용해도 상관없다.

#### 2. 어플리케이션에서 동영상을 송출할 RTMP url 을 지정

Brpadcast Me 어플을 실행하고, 우측의 환경설정에 들어가면 Sever URL 항목이 있다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-7-P-stream02.max-800x600.png)

해당 항목을 클릭하면 rtmp url 을 지정할 수 있는데, 이 어플은 자동으로 생성해준다. 해당 url 은 생성될 때마다 달라진다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-7-P-stream03.max-800x600.png)

생성된 url 을 복사해 보관해둔다.

#### 3. 어플리케이션에서 영상 촬영 시작

#### 4. 카메라 스티커앱 구동

`cv2.VideoCapture()` 를 활용하면 촬영중인 동영상 스트리밍을 입력으로 받을 수 있다.

하지만 웹캠과는 달리 파라미터를 `0` 으로 줄 수 없다.
스트리밍의 경우에는 위에서 생성되었던 rtmp ual 을 파라미터로 사용한다.

`webcam_sticker.py`에서 `cv2.VideoCapture()` 부분을 아래와 같이 바꾸기만 하면 된다.

> 스트리밍 기반 실시간 스티커앱 구현
```python
import numpy as np
import cv2
import dlib

from newaddsticker import img2sticker

detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def main():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 640, 360)

    vc = cv2.VideoCapture('rtmp://rtmp.streamaxia.com/streamaxia/1d7A28') # XXXXXX 부분은 본인 어플리케이션에서 확인한 코드로 대체해 주세요.
    img_sticker = cv2.imread('./images/king.png')

    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print (vlen) # 웹캠은 video length 가 0 입니다.

    # 정해진 길이가 없기 때문에 while 을 주로 사용합니다.
    # for i in range(vlen):
    while True:
        ret, img = vc.read()
        if ret == False:
            break
        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전

        # 스티커 메소드를 사용
        img_result = img2sticker(img, img_sticker.copy(), detector_hog, landmark_predictor)   

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)

        cv2.imshow('show', img_result)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
```
<br></br>
> 스트리밍 기반 스티커앱 구동
```
$ cd ~/aiffel/video_sticker_app && python webcam_sticker.py
```
<br></br>

이제 스티커앱을 실행하고 카메라를 고정한 후 거리를 바꿔보자.

아주 가까이 왔었다가 아주 멀어졌을 때 어느 정도 거리에서 얼굴을 인식하지 못하는지 확인해보자.

일반적으로 15 ~ 130cm 범위 사이에서 얼굴 인식이 가능하다.


거리를 확인하였으면 이제 고개를 상하좌우로 움직여보고 어느 각도까지 정삭적으로 작동되는지 확인해보자.

이를 통해 yaw, pitch, roll 각도의 개념을 이해하자.

+ 참고 : 
	- yaw : y축 기준 회전 → 높이 축
	- picth : x축 기준 회전 → 좌우 축
	- roll : z축 기준 회전 → 거리 축

일반적인 허용 범위는
 -   yaw : -45 ~ 45도
-   pitch : -20 ~ 30도
-   roll : -45 ~ 45도

이다.

이제 만들고자 하는 스티커 앱의 스펙을 정해야한다.

(예시)

-   거리 : 25cm ~ 1m → 너무 가까우면 스티커의 의미가 없음, 셀카봉을 들었을 때의 유효거리
-   인원 수 : 4명 → 4인 가족 기준
-   허용 각도 : pitch : -20 ~ 30도, yaw : -45 ~ 45도, roll : -45 ~ 45도 → 화면을 바라볼 수 있는 각도
-   안정성 : 위 조건을 만족하면서 FPPI (false positive per image) 기준 < 0.003, MR (miss rate) < 1 300장당 1번 에러 = 10초=30*10에 1번 에러

<br></br>
## 프로젝트 (03). 스티커 Out Bound 예외처리 하기

이전 웹캠 스티커앱을 실행하며 발생하는 예외상황을 찾아 보완해보자.

특히 서서히 영상에서 좌우 경계 밖으로 나카며 코드의 행동을 확인해보고, 예외 상황을 기록하자.

그리고 문제가 발생한 부분의 코드를 확인해보자.

좌우 경계 밖으로 나갈 때 발생하는 예외 상황을 코드로 해석해보자면 왼쪽 경계를 벗어나 detection 이 되는 경우 `refind_x` 값이 음수가 된다.

따라서 `img_bgr[..., refined_x:...]` 에서 numpy array의 음수 index에 접근하게 되므로 예외가 발생한다.

이 경우 `newaddsticker.py`의 `img2sticker` 메소드에서 다음과 같이 수정해줘야한다.

<br></br>
> 경계를 벗어나는 detection 예외처리
```python
### (이전 생략) ###

# sticker
img_sticker = cv2.resize(img_sticker, (w,h), interpolation=cv2.INTER_NEAREST)

refined_x = x - w // 2
refined_y = y - h

if refined_y < 0:
    img_sticker = img_sticker[-refined_y:]
    refined_y = 0

###
# TODO : x 축 예외처리 코드 추가
###

img_bgr = img_orig.copy()
sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)

return img_bgr
```
<br></br>

> Out bound 오류 (경계 밖으로 대상이 나가서 생기는 오류) 처리
> (`newaddsticker.py` 파일을 수정)
```python
if refined_x < 0: img_sticker = img_sticker[:, -refined_x:] refined_x = 0  elif refined_x + img_sticker.shape[1] >= img_orig.shape[1]: img_sticker = img_sticker[:, :-(img_sticker.shape[1]+refined_x-img_orig.shape[1])]
```
<br></br>

다른 예외에 어떤 것들이 있는지 정의해보자. 정해진 것은 없으며 원했던 부분이 제대로 구현되지 않았던 부분도 좋다.

<br></br>
## 프로젝트 (04). 스티커앱 분석 - 거리, 인원 수, 각도, 시계열 안정성

#### 1. 멀어진 경우 왜 스티커앱이 동작하지 않았을까?

카메라에서 멀어질 경우 스티커 앱이 작동하지 않은 원이은 detection 문제일까? 아니면 landmark 문제? 아니면 blending 문제일까?

바로 dlib detection 문제이다. 멀어지게되면 `detector_hog` 단계에서 bbox 가 출력되지 않는다.

> 멀어진 경우 스티커앱 동작 수정
```python
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # detector
    img_rgb_vga = cv2.resize(img_rgb, (640, 360))
    dlib_rects = detector_hog(img_rgb_vga, 0)
    if len(dlib_rects) < 1:
        return img_orig
```
<br></br>

#### 2. detector_hog 성능 문제를 해결하기

`detector_hog` 성능 문제를 해결하는데 간단한 방법은 이미지 피라미드를 조절하는 것이다.

> 이미지 피라미드를 활용한 `detector_hog` 성능 문제 해결
```python
def img2sticker(img_orig, img_sticker, detector_hog, landmark_predictor):
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # detector
    img_rgb_vga = cv2.resize(img_rgb, (640, 360))
    dlib_rects = detector_hog(img_rgb_vga, 1) # <- 이미지 피라미드 수 변경
    if len(dlib_rects) < 1:
        return img_orig

    # landmark
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img_rgb_vga, dlib_rect)
        list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        list_landmarks.append(list_points)
```
수정 후 `webcam_sticker.py` 를 다시 실행해보자.
<br></br>
#### 3. 위에서 새롭게 시도한 방법의 문제점은 무엇일까?

문제를 해결하였지만 또 다른 문제가 발생했다. 바로 속도가 느려진다는 것이다. 기존 30ms / frame 에서 120ms / frame 으로 약 4배 느려져 실시간 구동이 사실상 불가능 하다.

<br></br>
#### 4. 실행시간을 만족할 수 있는 방법을 찾아보자.

첫 번째 방법으로는 hog 디텍터를 딥러닝 기반 디텍터로 변경하는 방법이 있다. hog 학습 단계에서 다양한 각도에 대한 hog 특징을 모두 추출해 일반화 하기 어렵기 때문에 딥러닝 기반 검출기의 성능이 훨씬 뛰어나다.

두 번째 방법으로는 딥러닝 기반 detection 방법을 찾는 것이다.

+ 참고 : [How does the OpenCV deep learning face detector work?](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
<br></br>
opencv 는 intel cpu 를 사용할 때 dnn 모듈이 가속화를 지원한다. 

따라서 mobilenet 과 같은 작은 백본 모델을 사용하려고 ssd 를 사용할 때 충분한 시간과 성능을 얻을 수 있다.
<br></br>
#### 5. 인원 수, 각도 등 각 문제에 대해 1 - 4 번을 반복한다.

각도 문제에 대해서는 아래를 참고하자.

+ 참고 : [Facial Landmark Detection](https://www.learnopencv.com/facial-landmark-detection/)
