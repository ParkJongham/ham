# 11. OCR 기술의 개요

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-6-L-00.max-800x600.png)
<br></br>

OCR  이란 이미지 속에서 문자 영역을 검출하고, 검출한 문자 영역에서 문자를 인식하는 기술을 의미하며, 문자 영역을 검출하는 Text Detection 과 검출된 문자 영역에서 문자를 인식하는 Text Recognition, 2가지 기법을 활용한다.

Text Detection 은 Object Detection 의 한 갈래로 볼 수 있지만, Segmentation 기법 및 문자가 가지는 특별한 특성을 함께 고려해 사용된다.

Text Recognition 은 이미지 분류의 MNIST 처럼 이미지 속의 문자를 구분해 내는 작업으로 볼 수도 있지만 문자 단위로 분리되어 있는 텍스트 이미지만 다루는 것이 아니기 때문에 독특한 모델 구조를 가지게된다.
<br></br>
## 학습 목표

-   Deep learning기반의 OCR을 이해합니다.
-   Text를 Detection하기 위한 딥러닝 기법을 배웁니다.
-   Text를 Recognize하기 위한 딥러닝 기법을 배웁니다.
<br></br>

## Before Deep Learning

현재 OCR 은 스캔, 자동차 번호판 인식 등 일상 생활 속에서 자연스럽게 사용하고 있다.

딥러닝이 OCR 에 활용되기 전에는 어떻게 이미지에서 문자를 찾았을까?

+ 참고 : [논문 : From Videos to URLs: A Multi-Browser Guide To Extract User’s Behavior with Optical Character Recognition](https://arxiv.org/pdf/1811.06193.pdf)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/GC-6-L-01.png)
<br></br>

위 논문에서는 브라우저에서 동작하는 OCR 을 통해 웹에서 유저의 행동을 관찰하는 방법을 제안하였으며, 온라인 마케팅, 광고 등에도 OCR 기술을 활용할 수 있으며, 모델의 구성은 그림과 같다. 

OCR 엔진으로 Tesseract OCR 을 사용하며, 총 5 단계로 나누어, 위 3 단계는 입력 이미지의 추출 및 전처리, 4 번째 단계는 OCR 처리, 마지막 5 번째 단계는 OCR 의 출력 텍스트의 후처리로 구성된다.

+ 참고 : [Tesseract ocr github](https://github.com/tesseract-ocr/tesseract)
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/images/ARCHIT1.max-800x600.png)
<br></br>

위 그림은 Tesseract OCR 을 통한 4단계의 과정을 나타낸다.

[Adaptive Thresholding] 단계에서 입력영상의 이진화 하여 흑백으로 변환한다.

[Connected Component Analysis] 단계에서 문자영역을 검출하며,

[Find Lines and Words] 에서 라인 또는 워드 단위를 추출한다.

마지막으로 [Recognize Word] 단계에서 Word 단위 이미지를 Text 로 변환하기 위해 문자를 하나씩 인식하고 다시 결합하는 과정을 거친다.

딥러닝 기반의 OCR도 위처럼 복잡해질 수 있지만 기본적인 기능을 하기 위해서 필요한 단계가 많으며, 딥러닝의 적용을 통해서 우리는 원하는 단위로 문자를 검출해내고 이를 한번에 인식하도록 Architecture를 단순화하고 빠른 인식을 이뤄낼 수 있으며, 검출과 인식을 동시에 하는 End - to - End OCR 모델도 연구되고 있다.
<br></br>

## Text Detection

이미지 속에서 문자를 검출하는 Text Detection 방법에 대해 알아보자.

간단하게는 Object Detection 이나 Segmentation 을 통해 문자를 검출 혹은 문자가 있는 영역을 분리할 수 있을 것이다.

하지만 문자는 문자가 하나 하나 모여 단어나 문장을 이룬다. 즉, 이미지에서 문자를 검출하기 위해서는 검출을 위한 최소 단위를 정해야한다.

최소 단위란 이미지 속에서 문장 단위 혹은 단어나 글자 단위로 위치를 찾겠다고 정의해주는 것을 의미한다. 이때 문장이나 단어의 길이를 고려해야한다.

만약 글자 단위로 인식할 경우 찾아낸 글자를 맥락에 맞게 묶어주는 과정이 필요하다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-6-L-02.max-800x600.png)

[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155v2.pdf)
<br></br>
위 그림은 논문 EAST 에서 Text Detection 의 다양한 기법을 정리한 것이다. 논문이 발표된 2017 년 당시에는 Text 의 바운딩박스를 구하는 방식이 주류를 이루었습니다. 위 그림을 보면 가로방향 (Horizontal) 으로만 텍스트 박스를 구하는 방식 및 기울어지거나 세로방향 등 다양한 방향 (Multi-oriented) 의 텍스트 박스를 구하는 방식이 다양하게 소개되고 있다.

(e) 의 경우 전체 파이프라인의 길이가 짧고 간결해서 빠르면서도 정확한 Text detection 성능을 보인다고 소개하고 있다.

또한 그림을 자세히 보면 단어 단위의 탐지와 글자 단위의 탐지 모두 활용되고 있음을 알 수있다.

단어 단위의 탐지는 Object detection 의 Regression 기반의 Detection 방법이다. Anchor 를 정의하고 단어의 유무, 그리고 Bounding box 의 크기를 추정해서 단어를 찾아낸다.

글자 단위의 탐지는 Bounding box regression 을 하는 대신 글자인 영역을 Segmentation 하는 방법으로 접근한다.
<br></br>
### 대표적인 Text Detection 기법들

1. Regression

![](https://aiffelstaticprd.blob.core.windows.net/media/images/architecture_of_textboxes.max-800x600.png)
- [TextBoxes: A Fast Text Detector with a Single Deep Neural Network](https://arxiv.org/pdf/1611.06779.pdf)
<br></br>

2017 년 발표된 TextBoxes 이전에는 글자 단위로 인식한 후 결합하는 방식을 취해왔지만, 해당 논문에서는 딥러닝 기반의 Detection 을 이용하여 단어 단위로 인식한다.

SSD 의 구조를 활용하여 빠르게 문자영역을 탐지해 낸다.

+ 참고 : [SSD: single shot multibox detector](https://arxiv.org/pdf/1512.02325.pdf)
<br></br>

일반적으로 단어들은 길이가 길다. 즉, 가로 사이즈가 길며, Aspect ratio가크다.

이러한 이유로 몇가지 변형이 필요한데, 기존의 SSD 에서 Regression 을 위한 Convolution Layer 에서 3 x 3 커널을 가진다.

긴 단어의 특성을 활용하기 위해 1 x 5 로 Convolution Filter 를 정의하여 사용한다.

Anchor box 또한 1, 2, 3, 5, 7 로 큰 aspect ratio 로 만들고 이를 vertical offset 을 적용하여 세로 방향으로 촘촘한 단어의 배열에 대응하도록 한다.
<br></br>

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/textbox_anchor.png)
<br></br>

위 그림을 보면 Grid cell 의 중앙을 기점으로 생성할 경우를 예로 든 것이 파란색 (aspect ratio : 1) 그리고 검은색 박스 (aspect ratio : 1) 이다. 

이를 수직방향으로 옮겨서 촘촘하게 만들어준 것이 빨간색과 녹색이며, 수직방향으로 Anchor box 의 중앙을 하나 더 둠으로써 세로로 촘촘하게 Anchor box 를 배치할 수 있게 된다.
<br></br>

2. Segmentation

![](https://aiffelstaticprd.blob.core.windows.net/media/images/segmentation_map.max-800x600.png)
- [PixelLink: Detecting Scene Text via Instance Segmentation](https://arxiv.org/pdf/1801.01315.pdf)
<br></br>

세그멘테이션을 통해 이미지의 영역을 분리할 수 있다. 이 방법을 문자 영역을 찾기 위해 적용한다면 배경과 글자가 있는 영역으로 분리해야 한다.

하지만 문자는 일반적으로 배열이 촘촘하기 때문에 글자 영역으로 찾아낸 후 이를 분리하는 작업 및 연결하는 작업을 더해 원하는 최소단위로 만들어줘야 한다.

논문 PixelLink 는 Text 영역을 찾아내는 segmentation 과 함께 어느 방향으로 연결되는지 같이 학습을 하여 Text 영역간의 분리 및 연결을 할 수 있는 정보를 추가적으로 활용한다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/architecture_pixellink.max-800x600.png)
<br></br>

PixelLink 는 U - Net 과 유사한 구조를 지니지만 Output 으로 9 가지 정보를 얻는다.

위 그림의 녹색 부분이 input 과 output 을 의미하는데, output 중 하나는 Text / non-text Prediction 을 위한 class segmentation map 으로 해당 영역이 Text 인지 Non-text 인지 예측값을 의미하는 2 개의 커널을 가진다.

나머지 8 가지는 글자의 Pixel 을 중심으로 인접한 8 개의 Pixel 에 대한 연결여부를 의미하는 16 개의 커널로 이루어진 Link Prediction map 이다.

위 그림을 보면 conv 1 x 1, 2 (16) 형태의 레이어가 U - Net 구조로 연결되어 인접 pixel 간 연결 구조가 지속적으로 유지되도록 하는 모델 구조임을 알 수 있다.

따라서 인접한 pixel 이 중심 pixel 과 단어단위로 연결된 pixel 인지 분리된 pixel 인지 구분하며, 문자영역이 단어 단위로 분리된 Instance segmentation이 가능해진다.
<br></br>

3. 최근의 방법들

TextBoxes 나 PixelLink 는 3년전에 공개된 논문들이며, 최근에는 더욱 다양한 방법으로 문자 영역을 찾아내고 있다.

3 - 1. CRAFT

![](https://aiffelstaticprd.blob.core.windows.net/media/images/craft_affinity_map.max-800x600.png)
-   [Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)
<br></br>

CRAFT 는 Character 단위로 문자의 위치를 찾아낸 뒤 이를 연결하는 방식을 Segmentation 기반으로 구현한 방법이다.

문자의 영역을 boundary 로 명확히 구분하지 않고 가우시안 분포를 따르는 원형의 score map 을 만들어서 배치시키는 방법으로 문자의 영역을 학습하기 때문에 문자 단위 라벨을 가진 데이터 셋이 많지 않다.

따라서 단어 단위의 정보만 있는 데이터셋에 대해 단어 영역에 Inference 를 한 후 얻어진 문자 단위의 위치를 다시 학습해 활용하는 Weakly supervised learning을 활용한다.
<br></br>

3-2. Pyramid Mask Text Detector

![](https://aiffelstaticprd.blob.core.windows.net/media/images/PMTD.max-800x600.png)
<br></br>

PMTD (Pyramid Mask Text Detector) 는 Mask-RCNN 의 구조를 활용하여 먼저 Text 영역을 Region proposal network 로 찾아낸다.

그리고 Box head 에서 더 정확하게 regression 및 classification 을 하고 Mask head 에서 Instance 의 Segmentation 을 하는 과정을 거친다.

PMTD 는 여기서 Mask 정보가 부정확한 경우를 반영하기 위해서 Soft-segmentation 을 활용하며, 이전의 Mask-RCNN 의 경우 단어 영역이 Box head 에 의해 빨간색으로 잡히면 우측 처럼 boundary 를 모두 Text 영역으로 잡지만, PMTD 는 단어의 사각형 배치 특성을 반영하여 피라미드 형태의 Score map 을 활용한다.
<br></br>

## Text Recognition

1. Unsegmented Data

![](https://aiffelstaticprd.blob.core.windows.net/media/images/BEN-RO1.max-800x600.jpg)

위 이미지에서 YOU 라는 글자는 각각 Y, O, U 영역으로 분리가 가능하다.

하지만 위 이미지 처럼 세그멘테이션이 되어 있지 않은 데이터를 Unsegmented data 라고 하며, 많은 데이터들이 Unsegmented data 에 해당한다.

Unsegmented Data 의 특징 중 하나는 segment 되어 있지 않은 하위데이터들끼리 시퀀스 (sequence) 를 이루고 있다는 점이다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/unsegmented_data_copy.max-800x600.png)
<br></br>

위 그림은 어노테이션이 제대로 되지 않은 음성 데이터이다. 이 데이터도 Unsegmented data 의 한 종류이다.
<br></br>

2. CNN 과 RNN 의 만남 RCNN

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/crnn.png)
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/crnn_structure.png)
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)

<br></br>
CRNN 은 CNN (Convolutional neural network) 과 RNN (Recurrent neural network) 을 같이 쓰는 방법이다.

문자이미지가 있을 때 여기서 정보를 추출하기 위해서는 Feature Extractor 가 필요하며, Feature Extractor 로 사용되는 CNN 을 기반으로 VGG, ResNet 과 같은 네트워크로부터 문자의 정보를 가지 특성을 얻어낼 수 있다.

추출된 특성을 Map - To - Sequence 를 통해 시퀀스 형태의 특성으로 변환하고 다양한 길이의 Input 을 처리할 수 있는 RNN 을 넣는 것이다.

RNN 이 특성으로부터 문자를 인식하기 위해서 문자 영역은 넓은 정보가 필요하기 때문에 LSTM 으로 구성한다.

또한 앞의 정보뿐만 아니라 뒤의 정보가 필요하기 때문에 이를 Bidirectional 로 구성해서 Bidirectional LSTM  을 사용한다. Bidirectional LSTM 으로 step 마다 나오는 결과는 Transcription Layer 에서 문자로 변환된다.
<br></br>

3. CTC

CRNN 은 Steop 마다 Fully Connected Layer 의 logit 를 Softmax 를 통해 어떤 문자일 확률이 높은지 구분한다.

하지만 이 결과를 그대로 문자로 변환하면 모델의 Output 은 24 개의 글자로 이루어진 Sequence 이지만 실제 결과는 이와 다르기 때문에 기대한 것과 다른 결과가 나오게 된다. 

즉, "HELLO" 라는 이미지가 들어오면 이것의 Output 이 "HHHEEELLLOOOOO…" 와 같이 24 자의 시퀀스를 보게 된다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/ctc.png)
- [논문 : CTC(Connectionist Temporal Classification)](http://www.cs.toronto.edu/~graves/icml_2006.pdf)
<br></br>

CTC 는 CRNN 에서 Unsegmented data 를 위해 활용되는 것으로 써, Input 과 Output 이 서로 다른 길이의 시퀀스 일 때 이를 Align 없이 활용하는 방법이다.

모델의 Output 에서 최종적으로 알고 싶어하는 라벨 시퀀스의 확률을 구할 수 있는 방법이 가장 중요하게 된다.

"HHHEEELLLOOOOO…" 를 "HELLO" 로 만들기 위해서는 중복되는 단어인 "HHHH…" 나 "EEE…", "LLL…" 들을 "H", "E", "L" 등으로 바꿔볼 수 있다.

하지만 이렇게 되면 "HELO" 가 된다. 즉, 중복되는 글자가 있을 때 이를 처리해 줘야하는데, Label Encode 에서 이렇게 같은 문자를 구분하기 위한 Blank 를 중복된 라벨 사이를 구분하기 위해 넣어줌으로 써 해결한다.
<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/wbHRk.png)
<br></br>

위 그림은 Blank token 을 '-' 로 대신하여 Output 을 만드는 Decoder 를 의미한다. 

Decode 후에 중복을 제거하고, 인식할 문자가 아닌 값을 지워주면 "HELLO" 라는 결과를 얻을 수 있다.

이렇게 최종적으로 예측한 라벨이 실제 라벨과 얼마나 일치하는지에 대한 정확도를 측정하는 방법으로 Edit distance 라는 방법이 있다. 

한국어로는 편집거리라고 하며 두 문자열 사이의 유사도를 판별하는 방법이다.

예측된 단어에서 삽입, 삭제, 변경을 통해 얼마나 적은 횟수의 편집으로 정답에 도달할 수 있는지 최소 거리를 측정한다.
<br></br>

4. TPS

![](https://aiffelstaticprd.blob.core.windows.net/media/images/spn.max-800x600.png)
- [논문 : Robust Scene Text Recognition With Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)
<br></br>

OCR 을 수행하다보면 책 보다 거리에 있는 글자를 읽어낼 때 더욱 어렵다는 것을 알 수 있는데, 이는 불규칙한 방향이나 휘어진 진행방향에 영향을 받기 때문이다.

<br></br>
![](https://aiffelstaticprd.blob.core.windows.net/media/images/thinplates-dy.max-800x600.png)

위 논문에서는 Thin Plate Spline Transformation을 적용하여 입력이미지를 단어영역에 맞게 변형시켜 인식이 잘 되도록 해준다.

Thin plate spline 은 control point 를 정의하고 해당 point 들이 특정 위치로 옮겨졌을 때, 축방향의 변화를 interpolation 하여 모든 위치의 변화를 추정한다.

이를 통해서 전체 이미지 pixel의 변화를 control point로 만들어낼 수 있다.

또한 Control point 20 개를 미리 정의하며, Spatial Transformer Network 를 통해서 Control point 가 얼마나 움직여야 하는지 예측하는 네트워크를 아래 그림과 같이 Recognition model 앞단에 붙여 입력이미지를 정방향으로 맞춰준다.

![](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-6-L-03.max-800x600.png)

+ 참고: [Spatial Transformer Network](https://papers.nips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf)란 인풋 이미지에 크기, 위치, 회전 등의 변환을 가해 추론을 더욱 용이하게 하는 transform matrix를 찾아 매핑해 주는 네트워크를 의미한다.
	
	+ 	[Spatial Transformation Network 란 무엇인가?](https://3months.tistory.com/197)
<br></br>

## Text Recognition + Attention

Attention 와 Transformer 는 딥러닝 분야에서 큰 변화를 가져온 기법들이며, OCR 분야에서도 적용된다.

1. Attention Sequence Prediction

![](https://aiffelstaticprd.blob.core.windows.net/media/original_images/rnn_decoder.png)
-   [Robust Scene Text Recognition With Automatic Rectification](https://arxiv.org/pdf/1603.03915.pdf)
-   [Focusing Attention: Towards Accurate Text Recognition in Natural Images](https://arxiv.org/pdf/1709.02054.pdf)
<br></br>

CTC 를 활용한 CRNN 의 경우, column 에 따라서 prediction 된 라벨의 중복된 것들을 제거해줌으로써 우리가 원하는 형태의 라벨로 만들어주었다.

Attention기반의 sequence prediction 은 문장의 길이를 고정하고 입력되는 Feature 에 대한 Attention 을 기반으로 해당 글자의 Label을 prediction 힌다.

RNN 으로 Character label 을 뽑아낸다고 생각하면 되는데 첫번째 글자에서 입력 feature 에 대한 Attention 을 기반으로 label 을 추정하고, 추정된 label 을 다시 입력으로 사용하여 다음 글자를 추정해내는 방식이다.

이 때 우리가 20 글자를 뽑겠다고 정하게 되면 "YOU" 같은 경우에는 3 글자를 채우고 빈자리가 문제가 되는데요, 이러한 경우를 위해 미리 정해둔 Token 을 사용한다.

Token 에는 처음에 사용되는 "start" token 그리고 끝에 사용되는 "end" token 이 있으며, 필요에 따라서 예외처리나 공백을 위한 token을 만들어서 사용하기도 한다.

+ 참고 : 
	+ [유투브 : Naver Clova OCR](https://youtu.be/NQeaLc2X8vk)
	+ [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/pdf/1904.01906.pdf)
<br></br>

Attention 기반의 Recognition 이 더욱 좋은 성능을 보이고 있는 것을 위 링크에 소개한 네이버 Clova 의 논문 'What Is Wrong With Scene Text Recognition Model Comparisons?' 에서 확인 할 수 있으며, ctc와 attention만이 아니라 TPS 등 Recognition에서 쓰이는 다양한 모듈들을 비교 평가를 하고 있다.
<br></br>

2. Transformer 와 함께!

![](https://aiffelstaticprd.blob.core.windows.net/media/images/transformer_rec.max-800x600.png)
-   [A Simple and Strong Convolutional-Attention Network for Irregular Text Recognition](https://arxiv.org/pdf/1904.01375.pdf)
-   [hulk89님의 논문리뷰](https://hulk89.github.io/machine%20learning/2019/05/15/A-Simple-and-Robust-Convolutional-Attention-Network-For-Irregular-Text-Recognition/)
<br></br>

Transformer도 Recognition 모델에 활용된다. 다양한 논문에서 시도되고 있지만 A Simple and Strong Convolutional - Attention Network for Irregular Text Recognition가 대표적인 논문이다.

논문에서는 Irregular text 를 잘 인식하기 위해서 2d space 에 대한 attention 을 활용하여 문자를 인식하기 위해 Transformer 를 활용하며, Transformer 는 Query, Key, Value 라는 개념을 통해서 Self-Attention 을 입력으로부터 만들어 낸다. 이를 통해 입력에서 중요한 특성에 대해 가중치를 부여한다.

Attention 의 핵심은 Decoder 의 현재 포지션에서 중요한 Encoder 의 State 에 가중치가 높게 매겨진다는 점이다.

위 그림은 Decoder 의 각 Step 에 따라 입력에 대한 Visual Attention 이 시각화된 모습을 보여준다.

