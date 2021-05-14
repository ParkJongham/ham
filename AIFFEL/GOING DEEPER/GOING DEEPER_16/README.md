# 16. 불안한 시선 이펙트 추가하기

## 학습 목표

1.  공개 데이터 사용해서 라벨 직접 모아보기
2.  색상 값을 이용한 검출 방법
3.  라벨링 툴 만들기 - point selection
4.  째려보는 효과 구현하기


## 학습 준비

> 작업 디렉토리 설정
```bash
$ mkdir -p ~/aiffel/coarse_to_fine/data
```
<br></br>

> 사용할 데이터 다운
```bash
$ cd ~/aiffel/coarse_to_fine 
$ wget https://aiffelstaticprd.blob.core.windows.net/media/documents/coarse_to_fine_pjt.zip 
$ unzip coarse_to_fine_pjt.zip
```
+ [데이터 다운](https://aiffelstaticprd.blob.core.windows.net/media/documents/coarse_to_fine_pjt.zip)
<br></br>

## 위치 측정을 위한 라벨링 툴 만들기 (01). OpenCV 사용

눈동자 위치를 선택할 수 있는 도구를 만들어보자.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/1_hZAciQ9.png)
<br></br>
앞서 눈동자 및 Key Point 를 검출하는 툴을 만들어보았었다. 하지만 이상한 곳을 측정했었으며, 이 경우 눈동자의 위치를 새로 지정해 fine 라벨로 만들어야한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/2_fA179vj.png)
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/3_Aje9mFT.png)
<br></br>
정확한 곳을 지정하기 위해서는 위 그림의 포인터 같은 것을 이용하여 지정할 수 있다.

OpenCV 에서 마우스 이벤트를 callback 함수 형태로 지원하기에 이를 활용한다.

callback 함수란 특정 이벤트에 의해 호출되어지는 함수이며, 다른 함수의 인자로 이용된다. OpenCV 에서 마우스 이벤트를 확인하고 callback 를 호출하는 함수는 `cv2.setMouseCallback` 이 있으며, 마우스 왼쪽 버튼이 눌러졌을 경우, 즉 클릭을 통해 정확한 곳을 지정한 경우 `cv2.EVENT_LBUTTONDOWN` `flag` 가  `on` 된다.

+ 참고 :
	+ [콜백 함수(Callback)의 정확한 의미는 무엇일까?](https://satisfactoryplace.tistory.com/18)
	+ [Mouse로 그리기 - gramman 0.1 documentation](https://opencv-python.readthedocs.io/en/latest/doc/04.drawWithMouse/drawWithMouse.html)
<br></br>

## 위치 측정을 위한 라벨링 툴 만들기 (02). 툴 만들기

OpenCV 의 마우스 이벤트를 활용하여 라벨링 툴을 만들어보자.

> keypoint_using_mouse.py 생성
```python
import os from os.path 
import join from glob 
import glob import cv2 
import argparse 
import numpy as np 
import json from pprint 
import pprint args = 

argparse.ArgumentParser() 

# hyperparameters 
args.add_argument('img_path', type=str, nargs='?', default=None) 

config = args.parse_args() 

flg_button = False
```
위 코드를 `keypoint_using_mouse.py` 로 저장한다.
해당 코드는 학습 준비 단계에서 다운받은 코드 중에 포함되어 있다.

마우스 이벤트를 활용한 라벨링 툴은 주로 `cv2` 패키지를 이용하며, 존 라벨을 읽지 않고 새로 위치를 정하기 때문에 `img_path` 만 불러오면된다.

`flg_button` 은 마우스 이벤트가 발생할 때 사용할 불리언 (boolean) 타입 전역변수이다.
<br></br>

> `img_path`, `move()` 함수 선언
```python
def check_dir():
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

    return img_dir

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
```
`img_path()` 함수는 먼저 `img_path` 가 유효한지 체크하며 `img_path` 로 디렉토리가 입력될 경우 해당 디렉토리 내의 첫 번째 이미지를 `img_path` 에 입력하고 경로를 반환한다.

`move()` 함수는 이미지 간 이동을 위한 함수이다.

코드를 복사해 `keypoint_using_mouse.py`에 붙여넣어 준다.
<br></br>

> mouse callback 함수 정의
```python
# Mouse callback function
def select_point(event, x,y, flags, param):
    global flg_button, gparam
    img = gparam['img']

    if event == cv2.EVENT_LBUTTONDOWN:
        flg_button = True

    if event == cv2.EVENT_LBUTTONUP and flg_button == True:
        flg_button = False
        print (f'({x}, {y}), size:{img.shape}')
        gparam['point'] = [x,y]
```
전역변수인 `gparam` 에는 `img`와 `point` 정보를 저장하며 마우스 왼쪽 버튼이 눌러졌다 떨어질 때 `gparam` 의 `point` 에 `x, y` 를 리스트로 저장한다.

코드를 복사해 `keypoint_using_mouse.py`에 붙여넣어 준다.
<br></br>

> `main` 함수인 `blend_view()` 함수 구현
```python
def blend_view():
    global gparam
    gparam = {}
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 500, 500)

    img_dir = check_dir()

    fname, ext = os.path.splitext(config.img_path)
    img_list = [os.path.basename(x) for x in sorted(glob(join(img_dir,'*%s'%ext)))]

    dict_label = {}
    dict_label['img_dir'] = img_dir
    dict_label['labels'] = {}

    json_path = os.getenv('HOME')+'/aiffel/coarse_to_fine/eye_annotation.json'
    json_file = open(json_path, 'w', encoding='utf-8')

    idx = img_list.index(os.path.basename(config.img_path))
    pfname = img_list[idx]
    orig = None
    local_point = [] # 저장할 point list
    while True:
        start = cv2.getTickCount()
        fname = img_list[idx]
                # 파일의 변경이 없거나 이미지가 없을 때, point 를 초기화함
        if pfname != fname or orig is None:
            orig = cv2.imread(join(img_dir, fname), 1)
            gparam['point'] = []
            pfname = fname
                # 저장할 point(local point) 와 새로 지정한 gparam['point'] 가 변경된 경우,
                # local_point 를 업데이트
        if local_point != gparam['point']:
            orig = cv2.imread(join(img_dir, fname), 1)
            local_point = gparam['point']

        img_show = orig
        gparam['img'] = img_show
        cv2.setMouseCallback('show', select_point) # mouse event

        if len(local_point) == 2:
            img_show = cv2.circle(img_show, tuple(local_point),
                                  2, (0,255,0), -1)
            dict_label['labels'][fname] = local_point # label 로 저장

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000

        if img_show.shape[0] > 300:
            cv2.putText(img_show, '%s'%fname, (5,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255))

        print (f'[INFO] ({idx+1}/{len(img_list)}) {fname}... time: {time:.3f}ms', end='\r')

        cv2.imshow('show', img_show)

        key = cv2.waitKey(1)
        if key == 27:
            return -1
        if key == ord('n'):
            idx = move(1, idx, img_list)
        elif key == ord('p'):
            idx = move(-1, idx, img_list)
        elif key == ord('v'):
            print ()
            pprint (dict_label)
            print ()
        elif key == ord('s'):
            json.dump(dict_label, json_file, indent=2)
            print (f'[INFO] < {json_path} > saved!')

if __name__ == '__main__':
    blend_view()
```
코드를 복사해 `keypoint_using_mouse.py`에 붙여넣어 준다.

`main` 함수인 `blend_view()` 앞서 만들어본 라벨링 툴 구조와 유사하게 구현하였으며, 마우스 이벤트를 사용하기 위해 무한루프를 사용해서  `gparam`을 입력 받을 수 있으며, 이미지 변경이 없다면  `gparam['point']`  를 초기화하지 않고, 이미지 변경이 없더라도 callback 함수에서  `gparam`  변경이 일어나는 경우는 수정한다는 점이 차이점이다.
<br></br>

> `keypoint_using_mouse.py` 프로그램을 실행시켜 마우스로 정확한 위치를 지정
```python
$ cd ~/aiffel/coarse_to_fine
$ wget https://aiffelstaticprd.blob.core.windows.net/media/original_images/1_hZAciQ9.png -O ./data/eye.png
$ python keypoint_using_mouse.py ./data/eye.png
```
위 눈 이미지를 다운받아 `eye.png` 이름으로 현재 프로젝트 폴더의 하위 디렉토리인 `data` 에 저장한 후, 아래 코드를 터미널에서 실행하였다.

눈동자 지점을 클릭한 후 `s` 를 눌러 저장하면 `esc` 를 눌러 프로그램을 종료할 때 `~/aiffel/coarse_to_fine/eye_annotation.json` 에 레이블이 저장된다.

이러한 레이블을 모아 학습 시킬 수 있다.
<br></br>

## 데이터를 모아보자.

이제 라벨링을 위한 초기 데이터를 수집해야한다. 데이터를 수집하는 방법은 여러가지가 있지만 공개된 데이터를 적극적으로 활용하는 것이 시간 및 비용 대비 효율이 좋다.

눈동자 위치를 라벨링하기 위해 우리가 찾아야할 데이터의 조건은 첫 째,눈이 crop 되어 있고 눈동자 위치를 라벨로 가지고 있는 데이터, 둘 째, 얼굴 랜드마크 (face landmark) 를 가지고 있는 데이터, 셋 째,얼굴 이미지를 가지고 있는 데이터 순서로 데이터셋을 찾는 것이 좋다.

첫 번째에 해당하면서 공개되어 있는 데이터는 BioID 가 있다.

+ [BioID Face Database | Dataset for Face Detection | facedb - BioID](https://www.bioid.com/facedb/)

BioID 의 경우 찾고자하는 데이터에는 부합하지만 384 x 286 크기의 1,521 장의 gray image 이미지를 가지고 있으며, 총 23 명의 사람으로 구성되어 있다. 이는 데이터셋의 규모가 너무 작으며, 충분한 양의 데이터를 확보하기 위해 둘, 셋 째 조건에 부합하는 데이터를 수집해야 할 것이다.
<br></br>

## 랜드마크를 제공하는 데이터셋을 찾아보자.

랜드마크를 제공하는 데이터셋은 어떻게 찾을 수 있을까? 일반적인 이미지 데이터세 `dlib`  패키지를 통해 얻을 수 있는 랜드마크를 제공하는 데이터셋이 있을까? 

이러한 고민은 어쩌면 당연한 것일지도 모른다. 하지만 의외로 간단히 찾을 수 있다.

`dlib` 패키지의 얼굴 랜드마크를 사용하고 있다면, `dlib` 패키지를 구현하기 위해 사용된 랜드마크는 어떤 데이터셋으로 학습되었는지를 고민해 보면된다.

"dlib face landmark dataset" 을 검색해보면 쉽게 찾을 수 있다.

+ [dlib face landmark dataset](http://dlib.net/face_landmark_detection.py.html)
<br></br>

확인해보면 `iBUG 300-W` 라는 데이터셋으로 학습했다고 하니 해당 데이터셋을 이용하면된다.

하지만 매번 이렇게 쉽게 데이터셋을 구할 수 없다. 따라서 데이터를 구하기 어려울 때 사용 목적에 맞는 데이터를 수집하는 방법을 어느정도 생각해봐야할 필요가 있다.

조건에 부합하는 데이터셋을 수집하기 위해 LFW 데이터셋을 사용해보자.

LFW 데이터셋은 안면 인식 (Face Recognition) 에 관련된 데이터셋으로 얼굴이 포함되어있는 이미지만 존재하며, 얼굴의 랜드마크에 대한 정보는 존재하지 않는다.

이 데이터셋에 `dlib` 를 적용해 얼굴 위치와 랜드마크 위치를 찾고, 눈을 크롭하여 잘라낸 뒤 라벨링을 한다면 본래 목적에 맞는 데이터셋으로 탈바꿈 할 것이다.

+ [LFW 데이터셋 다운로드](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
<br></br>
> 데이터셋 다운로드
```bash
$ cd ~ && wget http://vis-www.cs.umass.edu/lfw/lfw.tgz 
$ tar -xvzf lfw.tgz
```
<br></br>

## Meas - Shift 를 이용한 눈동자 검출 방법 (01). 이론

![](https://aiffelstaticprd.blob.core.windows.net/media/images/6_ZBlO83G.max-800x600.png)
<br></br>

앞서 수학한 눈동자 검출 방법을 복기해보자.

눈동자 검출을 위한 방법의 대표적인 것은 눈동자는 주변 부분에 비해 어두운 색을 지니고 있다. 라는 가정을 기반으로 반전된 1 D 이미지에서 최대값을 찾는 방법이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/7_JjJNZgz.max-800x600.png)
<br></br>
이 과정에서 가우시안 블러를 활용해  눈동자에 존재하는 노이즈를 제거 해 준다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/8_gygk9eg.max-800x600.png)
<br></br>

하지만 가우시안 블러가 문제를 완벽히 해결해 줄 수 없다. 위 그림은 눈 근처에 머리카락이 나타나 눈 가장자리에 검정색이 큰 비중으로 등장하는 경우이다. 이 경우 눈동자보다 가장자리로 수렴할 확률이 높으며, 머리가 긴 사람들이게 흔히 나타날 수 있는 현상이다.

2 차원 블러 특성 이미지 (Feature Image) 에서 눈동자가 2 차원 정규분포로 나타나는 영역이 있음을 볼 수 있다. 하지만 1 차원 누적 그래프를 보면 x 축으로 2 개의 봉우리를 가지는 것을 관찰할 수 있으며, 최대값을 찾는 알고리즘을 왼쪽부터 시작할 경우 가장 왼족에서 만나는 255 에 수렴하게된다.

이러한 이유로 1 D 누적 그래프와 2 D 특성 이미지 모두를 활용해야한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/9_EivoVX7.max-800x600.png)
<br></br>

2 D 에서 최고점을 찾아가기 위한 방법은 다음과 같다.

1.  **이미지 중심점을 초기값으로 설정힌다.  
    **눈의 중심에 눈동자가 있을 확률이 높기 때문에 초기값으로 정하기에 아주 좋다.)

2.  **중심점을 기준으로 작은 box를 설정힌다.  
    **box의 크기는 문제에 따라 적절한 값을 설정해야 힌다.  
    그림에서 회색박스를 생각하시면 된다.
    
3.  **box 내부의 pixel 값을 이용해서 '무게중심'을 찾는다.  
    **이 때 무게중심은 pixel intensity를 weight 로 사용할 수 있다.
    
4.  **찾은 무게중심을 새로운 box의 중심으로 설정힌다.  
    **이 단계에서 박스가 이동하게 됩니다. 이제 회색박스에서 초록색박스로 관심영역이 이동했다.
    
5.  **다시 초록색 박스를 기준으로 2-4를 반복한다.**

6.  **중심점이 수렴할 때 까지 2~5를 반복하면 수렴한 점의 위치로 눈동자를 찾을 수 있다.**

머신러닝에는 이와 같은 비슷한 알고리즘이 이미 존재한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/10_BHmd9aH.png)
<br></br>

바로 위 그림과 같은 Mean Shift 라는 알고리즘으로 현재 위치와 탐색 반경을 가질 때 평균의 위치를 이용해 반복적으로 움직이는 알고리즘이다.

Mean Shift 는 물체추적(object tracking), 영상 세그멘테이션 (segmentation), 데이터 클러스터링 (clustering), 경계를 보존하는 영상 스무딩 (smoothing) 등 다양하게 활용용되고 있다.

Mean Shift 알고리즘은 Local Optima 에만 수렴하기 때문에 Global Optima 를 찾을 방법은 존재하지 않으며,  초기값에 따라 수렴 위치가 달라지기 때문에 항상 일정한 성능을 보장하기 힘들다. 또한 탐색 윈도우 (탐색 반경) 의 크기를 정해줘야하는데 추적 대상에 따라 이를 적절히 바꿔줘야한다. 적절히 변경되지 않으면 성능에 큰 영향을 끼친다.

+ 참고 : [영상추적#1 - Mean Shift 추적](https://darkpgmr.tistory.com/64)
<br></br>

## Mean - Shift 를 이용한 눈동자 검출 방법 (02). 실습

Mean Shift 기법을 코드로 구현해 눈동자를 검출해보자.

<br></br>
> 앞서 구현한 눈동자 찾기 코드를 `eye_center_basic.py`에 저장
```python
import matplotlib.pylab as plt import tensorflow as tf import os from os.path import join from glob import glob from tqdm import tqdm import numpy as np import cv2 import math import dlib import argparse args = argparse.ArgumentParser() # hyperparameters args.add_argument('show_substep', type=bool, nargs='?', default=False) config = args.parse_args() img = cv2.imread('./images/image.png') print (img.shape) if config.show_substep: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) plt.show() img_bgr = img.copy() detector_hog = dlib.get_frontal_face_detector() # detector 선언 landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) dlib_rects = detector_hog(img_rgb, 1) # (image, num of img pyramid) list_landmarks = [] for dlib_rect in dlib_rects: points = landmark_predictor(img_rgb, dlib_rect) list_points = list(map(lambda p: (p.x, p.y), points.parts())) list_landmarks.append(list_points) for dlib_rect in dlib_rects: l = dlib_rect.left() t = dlib_rect.top() r = dlib_rect.right() b = dlib_rect.bottom() cv2.rectangle(img_rgb, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA) for landmark in list_landmarks: for idx, point in enumerate(list_points): cv2.circle(img_rgb, point, 2, (255, 255, 0), -1) # yellow  if config.show_substep: plt.imshow(img_rgb) plt.show() def eye_crop(bgr_img, landmark):  # dlib eye landmark: 36~41 (6), 42~47 (6) np_left_eye_points = np.array(landmark[36:42]) np_right_eye_points = np.array(landmark[42:48]) np_left_tl = np_left_eye_points.min(axis=0) np_left_br = np_left_eye_points.max(axis=0) np_right_tl = np_right_eye_points.min(axis=0) np_right_br = np_right_eye_points.max(axis=0) list_left_tl = np_left_tl.tolist() list_left_br = np_left_br.tolist() list_right_tl = np_right_tl.tolist() list_right_br = np_right_br.tolist() left_eye_size = np_left_br - np_left_tl right_eye_size = np_right_br - np_right_tl ### if eye size is small  if left_eye_size[1] < 5: margin = 1  else: margin = 6 img_left_eye = bgr_img[np_left_tl[1]-margin:np_left_br[1]+margin, np_left_tl[0]-margin//2:np_left_br[0]+margin//2] img_right_eye = bgr_img[np_right_tl[1]-margin:np_right_br[1]+margin, np_right_tl[0]-margin//2:np_right_br[0]+margin//2] return [img_left_eye, img_right_eye] # 눈 이미지 crop img_left_eye, img_right_eye = eye_crop(img_bgr, list_landmarks[0]) print (img_left_eye.shape) # (26, 47, 3)  if config.show_substep: plt.imshow(cv2.cvtColor(img_right_eye, cv2.COLOR_BGR2RGB)) plt.show() # 눈 이미지에서 중심을 찾는 함수  def findCenterPoint(gray_eye, str_direction='left'):  if gray_eye is  None: return [0, 0] filtered_eye = cv2.bilateralFilter(gray_eye, 7, 75, 75) filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75) filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75) # 2D images -> 1D signals row_sum = 255 - np.sum(filtered_eye, axis=0)//gray_eye.shape[0] col_sum = 255 - np.sum(filtered_eye, axis=1)//gray_eye.shape[1] # normalization & stabilization  def vector_normalization(vector): vector = vector.astype(np.float32) vector = (vector-vector.min())/(vector.max()-vector.min()+1e-6)*255 vector = vector.astype(np.uint8) vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],)) vector = cv2.blur(vector, (5,1)).reshape((vector.shape[0],)) return vector row_sum = vector_normalization(row_sum) col_sum = vector_normalization(col_sum) def findOptimalCenter(gray_eye, vector, str_axis='x'): axis = 1  if str_axis == 'x'  else  0 center_from_start = np.argmax(vector) center_from_end = gray_eye.shape[axis]-1 - np.argmax(np.flip(vector,axis=0)) return (center_from_end + center_from_start) // 2 center_x = findOptimalCenter(gray_eye, row_sum, 'x') center_y = findOptimalCenter(gray_eye, col_sum, 'y') if center_x >= gray_eye.shape[1]-2  or center_x <= 2: center_x = -1  elif center_y >= gray_eye.shape[0]-1  or center_y <= 1: center_y = -1  return [center_x, center_y] # 눈동자 검출 wrapper 함수  def detectPupil(bgr_img, landmark):  if landmark is  None: return img_eyes = [] img_eyes = eye_crop(bgr_img, landmark) gray_left_eye = cv2.cvtColor(img_eyes[0], cv2.COLOR_BGR2GRAY) gray_right_eye = cv2.cvtColor(img_eyes[1], cv2.COLOR_BGR2GRAY) if gray_left_eye is  None  or gray_right_eye is  None: return left_center_x, left_center_y = findCenterPoint(gray_left_eye,'left') right_center_x, right_center_y = findCenterPoint(gray_right_eye,'right') return [left_center_x, left_center_y, right_center_x, right_center_y, gray_left_eye.shape, gray_right_eye.shape] # 눈동자 중심 좌표 출력 left_center_x, left_center_y, right_center_x, right_center_y, le_shape, re_shape = detectPupil(img_bgr, list_landmarks[0]) print ((left_center_x, left_center_y), (right_center_x, right_center_y), le_shape, re_shape) # 이미지 출력 show = img_right_eye.copy() show = cv2.circle(show, (right_center_x, right_center_y), 2, (0,255,255), -1) plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB)) plt.show()
```
`eye_center_basic.py` 코드를 베이스라인으로 한다.
<br></br>

> `eye_center_basic.py` 동작 확인
```bash
$ cd ~/aiffel/coarse_to_fine && python eye_center_basic.py True
```
`show_substep` argument 의 옵션을 True 로 주게 되면 매 스텝의 작동을 차례차례 확인해볼 수 있으며, 옵션을 False 로 주거나 생략 (기본옵션) 하면 최종 결과만 확인한다.
<br></br>

> mean shift 알고리즘을 적용
```basg
$ cd ~/aiffel/coarse_to_fine && cp eye_center_basic.py eye_center_meanshift.py
```
`eye_center_basic.py` 파일을 복사하여 `eye_center_meanshift.py`를 생성한다.
<br></br>

> `eye_center_meanshift.py` 의 `findCenterPoint` 를 수정
```python
def findCenterPoint(gray_eye, str_direction='left'):
    if gray_eye is None:
        return [0, 0]
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

    # x 축 center 를 찾는 알고리즘을 mean shift 로 대체합니다.
    # center_x = findOptimalCenter(gray_eye, row_sum, 'x')
    center_y = findOptimalCenter(gray_eye, col_sum, 'y')

    # 수정된 부분
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

# 눈동자 중심 좌표 출력
left_center_x, left_center_y, right_center_x, right_center_y, le_shape, re_shape = detectPupil(img_bgr, list_landmarks[0])
print ((left_center_x, left_center_y), (right_center_x, right_center_y), le_shape, re_shape)

# 이미지 출력
show = img_right_eye.copy()
show = cv2.circle(show, (right_center_x, right_center_y), 2, (0,255,255), -1)

plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
plt.show()
```
`eye_center_meanshift.py` 파일을 편집기로 열어 기존 함수 중 `findCenterPoint`를 수정한다.

눈 이미지를 low pass filter 를 이용해서 smoothing 을 한다. (bilateral filter 를 이용를 이용해도 된다.)

그리고 1 차원 값으로 누적시킨 후 `y` 축 기준으로 최대값을 찾아서 `y`축의 중심점 좌표를 먼저 얻어낸다. (y 축은 x 축에 비해 상대적으로 변화가 적기 때문에 간단하게 구현한다.)

`x` 축은 1 차원 최댓값 지점을 기준으로 mean shif t를 수행하며 양 끝단에 수렴하는 예외를 처리한 후 결과를 출력한다.
<br></br>

> 결과확인
```bash
$ cd ~/aiffel/coarse_to_fine && python eye_center_meanshift.py
```
완전히 눈동자의 중심은 아니지만 어느정도 눈동자 중심에 가까운 것을 확인할 수 있다.
<br></br>

## 키포이늩 검출 딥러닝 모델 만들기 (01). 데이터 확인

더 나은 성능을 위해 딥러닝 모델을 만들어보자.

데이터 학습을 위해 대량의 눈동자 위치 라벨이 필요하며, 앞서 만들었던 coarse dataset 또는 직접 어노테이션 한 라벨이 10,000 개 이상 있어야 성능을 확인할 수 있다.
<br></br>

> `prepare_eye_dataset.py`를 실행하여 데이터셋 가공
```bash
$ cd ~/aiffel/coarse_to_fine && python prepare_eye_dataset.py
```
데이터셋을 생성하는 코드 `prepare_eye_dataset.py`를 통해 LFW 데이터셋에 눈동자 검출 방법을 적용하여 데이터셋 생성하며, 생성된 데이터셋은 `~/lfw/data/train`, `~/lfw/data/valid` 에 저장된다.

> 필요라이브러리 임포트
```python
import tensorflow as tf
import numpy as np
import math
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import LearningRateScheduler
```
TensorFlow Hub 에서 제공하는 pretrained image feature embedding 을 활용하여 fine tunning 수행
<br></br>

> fine tuning
```python
import glob
import os

home_dir = os.getenv('HOME')+'/lfw'
list_image = sorted(glob.glob(home_dir+'/data/train/input/img/*.png'))
list_label = sorted(glob.glob(home_dir+'/data/train/label/mask/*.png'))
print (len(list_image), len(list_label))

# 32의 배수를 벗어나는 파일 경로들을 담은 list
list_image_out_of_range = list_image[len(list_image) - (len(list_image) % 32):]
list_label_out_of_range = list_label[len(list_label) - (len(list_label) % 32):]

# 해당 list가 존재한다면, 파일 삭제
if list_image_out_of_range:
    for path in list_image_out_of_range:
        os.remove(path)
if list_label_out_of_range:
    for path in list_label_out_of_range:
        os.remove(path)

IMAGE_SHAPE = (80, 120)
data_root = home_dir+'/data/train/input'
label_root = home_dir+'/data/train/label'

image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
label_generator = tf.keras.preprocessing.image.ImageDataGenerator()
image_data = image_generator.flow_from_directory(str(data_root), class_mode=None, target_size=IMAGE_SHAPE, batch_size=32)
label_data = label_generator.flow_from_directory(str(label_root), class_mode=None, target_size=IMAGE_SHAPE, batch_size=32)
```
가지고 있는 데이터를 케라스 `ImageDataGenerator` 형식으로 읽고, 라벨을 `image` 형태로 저장한다.

총 train 데이터셋이 23,712 쌍, val 데이터셋이 2,638 쌍 생성된다.

(경우에 따라서는 train 데이터셋의 갯수가 23,712 쌍과 다소 다르게 만들어질 수 있다. atch_size 32의 배수인 23,712 쌍과 같아지도록 이미지 데이터의 갯수를 맞춰며, 꼭 23,712 쌍이 아니더라도 32의 배수 조건만 만족하면 된다.)
<br></br>

> `image_generator`, `label generator`를 학습할 수 있는 입출력 형식으로 편집
```python
def user_generation(train_generator, label_generator):
    h, w = train_generator.target_size
    for images, labels in zip(train_generator, label_generator):
        images /= 255.
        images = images[..., ::-1] # rgb to bgr

        list_point_labels = []
        for img, label in zip(images, labels):

            eye_ls = np.where(label==1) # leftside
            eye_rs = np.where(label==2) # rightside
            eye_center = np.where(label==3)

            lx, ly = [eye_ls[1].mean(), eye_ls[0].mean()]
            rx, ry = [eye_rs[1].mean(), eye_rs[0].mean()]
            cx, cy = [eye_center[1].mean(), eye_center[0].mean()]

            if len(eye_ls[0])==0 or len(eye_ls[1])==0:
                lx, ly = [0, 0]
            if len(eye_rs[0])==0 or len(eye_rs[1])==0:
                rx, ry = [w, h]
            if len(eye_center[0])==0 or len(eye_center[1])==0:
                cx, cy = [0, 0]

            np_point_label = np.array([lx/w,ly/h,rx/w,ry/h,cx/w,cy/h], dtype=np.float32)

            list_point_labels.append(np_point_label)
        np_point_labels = np.array(list_point_labels)
        yield (images, np_point_labels)
```
`image_generator`, `label generator`를 학습할 수 있는 입출력 형식으로 편집하며, 텐서플로우의 제너레이터(generator) 형식을 사용하고 있기 때문에 출력 형식을 맞춰준다.

+ 참고 : [제너레이터](https://tensorflow.blog/%ED%9A%8C%EC%98%A4%EB%A6%AC%EB%B0%94%EB%9E%8C%EC%9D%84-%ED%83%84-%ED%8C%8C%EC%9D%B4%EC%8D%AC/%EC%A0%9C%EB%84%88%EB%A0%88%EC%9D%B4%ED%84%B0/)
<br></br>
학습 라벨을 만들 때 3개의 점을 `label` 이미지에 표시했으며, 눈의 왼쪽 끝점을 `1`의 값으로, 오른쪽 끝점은 `2`의 값으로, 가장 중요한 눈 중심(눈동자)는 `3`으로 인코딩하여 `np.where()` 함수로 이미지에서 좌표로 복원한다.

좌표 복원시 `eye_ls[1].mean()` 으로 평균값을 구한 이유는 눈 크기가 이미지, 사람마다 다르기 때문에 resize 는 반드시 필요하다. 이때 라벨의 이미지에 하나의 점으로 표현할 경우 resize 과정에서 소실될 수 있으며, 따라서 라벨 이미지를 만들 때 gaussian smoothing을 적용해서 변화에 유연하게 대응 할 수 있도록 해야한다.

이 방법을 통해 이후 augmentation 을 구현할 때도 추가적인 노력없이 바로 라벨을 사용할 수 있다.
<br></br>

> 제너레이터로 데이터 포인트 뽑아보기
```python
user_train_generator = user_generation(image_data, label_data)
for i in range(2):
    dd = next(user_train_generator)
    print (dd[0][0].shape, dd[1][0])
```
120 x 80의 정해진 크기로 이미지가 잘 출력되고 라벨 또한 0 ~ 1 값으로 정규화 (normalize) 되어 있는 것을 확인할 수 있다.
<br></br>

## 키포인트 검출 딥러닝 모델 만들기 (02). 모델 설계

학습을 위한 데이터가 없기 때문에 Pretrained  Model 을 적극 활용해야 한다. 모델로는 Tensorflow Hub 에서 ResNet 의 특성 추출기 부분을 백본으로 활용한다.
<br></br>

> 모델 구성
```python
''' tf hub feature_extractor ''' feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4" feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(80,120,3)) image_batch = next(image_data) feature_batch = feature_extractor_layer(image_batch) print(feature_batch.shape) num_classes = 6 feature_extractor_layer.trainable = False model = tf.keras.Sequential([ feature_extractor_layer, #layers.Dense(1024, activation='relu'),  #layers.Dropout(0.5), layers.Dense(num_classes, activation='sigmoid'), ]) model.summary()
```
`tf.keras.Sequential()`을 이용해서 백본 네트워크와 fully connected layer 를 쌓아서 모델을 완성했으며, 데이터 제너레이터를 만들 때 출력을 6 개((x, y) 좌표 2 개 * 점 3 개) 로 했기 때문에 `num_classes` 는 6 으로 설정하였다.
<br></br>

> `mae` 를 통해서 픽셀 위치가 평균적으로 얼마나 차이나는지 확인하면서 학습하도록 설정
```python
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='mse',
  metrics=['mae']
  )
```
점을 맞는 위치로 추정하는 position regression 문제이기 때문에 `loss`와 `metric` 을 각각 `mse` 와 `mae` 로 설정하고 `mae` 를 통해서 픽셀 위치가 평균적으로 얼마나 차이나는지 확인하면서 학습할 수 있다.
<br></br>

> 학습률 (learning rate) 을 조절하는 함수 생성
```python
def lr_step_decay(epoch):
      init_lr = 0.0005 #self.flag.initial_learning_rate
      lr_decay = 0.5 #self.flag.learning_rate_decay_factor
      epoch_per_decay = 2 #self.flag.epoch_per_decay
      lrate = init_lr * math.pow(lr_decay, math.floor((1+epoch)/epoch_per_decay))
      return lrate
```
지수적으로 감소하도록 학습률을 조절하게 하였다.
<br></br>

> 모델 학습
```python
steps_per_epoch = image_data.samples//image_data.batch_size
print (image_data.samples, image_data.batch_size, steps_per_epoch)
# 23712 32 741 -> 데이터를 batch_size(32) 의 배수로 맞춰 준비해 주세요. 

assert(image_data.samples % image_data.batch_size == 0)  # 데이터가 32의 배수가 되지 않으면 model.fit()에서 에러가 발생합니다.

learning_rate = LearningRateScheduler(lr_step_decay)

history = model.fit(user_train_generator, epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [learning_rate]
                    )
```
<br></br>

## 키포인트 검출 딥러닝 모델 만들기 (03). 평가

> 검증용 데이터 생성
```python
IMAGE_SHAPE = (80, 120)

home_dir = os.getenv('HOME')+'/lfw'

val_data_root = home_dir + '/data/val/input'
val_label_root = home_dir + '/data/val/label'

image_generator_val = tf.keras.preprocessing.image.ImageDataGenerator()
label_generator_val = tf.keras.preprocessing.image.ImageDataGenerator()
image_data_val = image_generator.flow_from_directory(str(val_data_root), class_mode=None, target_size=IMAGE_SHAPE, shuffle=False)
label_data_val = label_generator.flow_from_directory(str(val_label_root), class_mode=None, target_size=IMAGE_SHAPE, shuffle=False)
```
검증용 데이터는 섞어줄 (shuffle) 필요가 없기 때문에 `shuffle=False` 옵션을 추가했다.
<br></br>

> 모델 평가
```python
user_val_generator = user_generation(image_data_val, label_data_val)
mse, mae = model.evaluate_generator(user_val_generator, image_data_val.n // 32)
print(mse, mae)
```
제너레이터를 만들고 `evaluate_generator()` 로 평가를 수행한다.

평균 에러가 0.026 정도가 나오며, 찍은 점들은 120 픽셀을 기준으로 `120 * 0.026 = 3.12` 픽셀 정도 에러가 나는 것을 확인할 수 있다.
<br></br>

> 실제 이미지에 출력
```python
# img test
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(val_data_root+'/img/eye_000010_l.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```
<br></br>

> 배치 크기 조절
```python
np_inputs = np.expand_dims(cv2.resize(img, (120, 80)), axis=0)
preds = model.predict(np_inputs/255., 1)

repred = preds.reshape((1, 3, 2))
repred[:,:,0] *= 120
repred[:,:,1] *= 80
print (repred)
```
입력을 위해 이미지를 120 x 80 으로 resize 한 후, 배치(batch) 를 나타낼 수 있는 4 차원 텐서로 변경한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/13_Rgiy3Ps.png)
<br></br>

위 그림은 출력 결과이며, 이미지 1 장에 대해서 출력하기 때문에 배치 크기 (batch size) 를 1 로 만든다.

출력 결과는 1행부터 좌측, 우측, 중앙 좌표를 나타낸다.
<br></br>

> 결과를 이미지에 출력
```python
show = img.copy()
for pt in repred[0]:
    print (pt.round())
    show = cv2.circle(show, tuple((pt*0.5).astype(int)), 3, (0,255,255), -1)

plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
plt.show()
```
`pt` 값은 120 x 80 으로 뽑았는데 우리가 사용하는 데이터 크기는 60 x 40 이며, 이는 `pt` 에 `0.5` 를 곱해서 그림에 출력하기 때문이다.

`pt` 값을 뽑을때의 이미지 크기 기준 (120 X 80) 은 고정이지만, 사용하는 데이터의 크기는 매번 달라진다. 따라서 보정치 설정에 유의해야한다.
<br></br>
