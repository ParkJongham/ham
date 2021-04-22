# 영화리뷰 감성분석

[모델 구현 코드](https://github.com/ParkJongham/ham/blob/master/IMDb%20Movie%20Review%20Sentimantal%20Analysis/IMDb%20%EC%98%81%ED%99%94%EB%A6%AC%EB%B7%B0%20%EA%B0%90%EC%84%B1%EB%B6%84%EC%84%9D.ipynb)

1. 사용 데이터셋 :

- IMDb Large Movie Dataset (Learning Word Vectors for Sentiment Analysis, 2011 논문에서 소개)
- keras 데이터셋안에 포함

2. 사용 기법 : 

- [1 - D CNN : 합성곱 연산을 통한 이미지 특징을 추출하는 신경망](https://github.com/ParkJongham/ham/blob/master/IMDb%20Movie%20Review%20Sentimantal%20Analysis/Sentimental%20Analysis%20Classification.md)

3. 모델 구성 : 

	1. 사용 라이브러리 임포트
	2. 데이터 다운로드 및 분리
	3. 텍스트 인코딩을 위한 함수 생성(get_encoded_sentences, get_decoded_sentences)
	4. 문장의 최대 길이 확인 및 데이터셋의 분포 확인
	5. 모델 설계
	6. train dataset 에서 validation dataset 으로 사용할 데이터 분리
	7. 모델학습
	8. 모델 평가 및 시각화
