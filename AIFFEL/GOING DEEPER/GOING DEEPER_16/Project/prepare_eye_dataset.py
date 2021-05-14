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

detector_hog = dlib.get_frontal_face_detector()  # detector 선언
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')


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

    img_left_eye = bgr_img[np_left_tl[1] - margin:np_left_br[1] + margin,
                   np_left_tl[0] - margin // 2:np_left_br[0] + margin // 2]
    img_right_eye = bgr_img[np_right_tl[1] - margin:np_right_br[1] + margin,
                    np_right_tl[0] - margin // 2:np_right_br[0] + margin // 2]

    return [img_left_eye, img_right_eye]



# 눈 이미지에서 중심을 찾는 함수
def findCenterPoint(gray_eye, str_direction='left'):
    if gray_eye is None:
        return [0, 0]
    filtered_eye = cv2.bilateralFilter(gray_eye, 7, 75, 75)
    filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)
    filtered_eye = cv2.bilateralFilter(filtered_eye, 7, 75, 75)

    # 2D images -> 1D signals
    row_sum = 255 - np.sum(filtered_eye, axis=0) // gray_eye.shape[0]
    col_sum = 255 - np.sum(filtered_eye, axis=1) // gray_eye.shape[1]

    # normalization & stabilization
    def vector_normalization(vector):
        vector = vector.astype(np.float32)
        vector = (vector - vector.min()) / (vector.max() - vector.min() + 1e-6) * 255
        vector = vector.astype(np.uint8)
        vector = cv2.blur(vector, (5, 1)).reshape((vector.shape[0],))
        vector = cv2.blur(vector, (5, 1)).reshape((vector.shape[0],))
        return vector

    row_sum = vector_normalization(row_sum)
    col_sum = vector_normalization(col_sum)

    def findOptimalCenter(gray_eye, vector, str_axis='x'):
        axis = 1 if str_axis == 'x' else 0
        center_from_start = np.argmax(vector)
        center_from_end = gray_eye.shape[axis] - 1 - np.argmax(np.flip(vector, axis=0))
        return (center_from_end + center_from_start) // 2


    # x 축 center 를 찾는 알고리즘을 mean shift 로 대체합니다.
    # center_x = findOptimalCenter(gray_eye, row_sum, 'x')
    center_y = findOptimalCenter(gray_eye, col_sum, 'y')

    # 수정된 부분
    inv_eye = (255 - filtered_eye).astype(np.float32)
    inv_eye = (255 * (inv_eye - inv_eye.min()) / (inv_eye.max() - inv_eye.min())).astype(np.uint8)

    resized_inv_eye = cv2.resize(inv_eye, (inv_eye.shape[1] // 3, inv_eye.shape[0] // 3))
    init_point = np.unravel_index(np.argmax(resized_inv_eye), resized_inv_eye.shape)

    x_candidate = init_point[1] * 3 + 1
    for idx in range(10):
        temp_sum = row_sum[x_candidate - 2:x_candidate + 3].sum()
        if temp_sum == 0:
            break
        normalized_row_sum_part = row_sum[x_candidate - 2:x_candidate + 3].astype(np.float32) // temp_sum
        moving_factor = normalized_row_sum_part[3:5].sum() - normalized_row_sum_part[0:2].sum()
        if moving_factor > 0.0:
            x_candidate += 1
        elif moving_factor < 0.0:
            x_candidate -= 1

    center_x = x_candidate

    if center_x >= gray_eye.shape[1] - 2 or center_x <= 2:
        center_x = -1
    elif center_y >= gray_eye.shape[0] - 1 or center_y <= 1:
        center_y = -1

    return [center_x, center_y]

# 눈동자 검출 wrapper 함수
def detectPupil(bgr_img, landmark):
    if landmark is None:
        return

    img_eyes = []
    img_eyes = eye_crop(bgr_img, landmark)

    gray_left_eye = cv2.cvtColor(img_eyes[0], cv2.COLOR_BGR2GRAY)
    gray_right_eye = cv2.cvtColor(img_eyes[1], cv2.COLOR_BGR2GRAY)

    if gray_left_eye is None or gray_right_eye is None:
        return

    left_center_x, left_center_y = findCenterPoint(gray_left_eye, 'left')
    right_center_x, right_center_y = findCenterPoint(gray_right_eye, 'right')

    return [left_center_x, left_center_y, right_center_x, right_center_y, gray_left_eye.shape, gray_right_eye.shape]

#########################################

data_source_dir = os.getenv('HOME') + '/lfw/'
data_train_input_dir = os.getenv('HOME') + '/lfw/data/train/input/img/'
data_train_label_dir = os.getenv('HOME') + '/lfw/data/train/label/mask/'
data_val_input_dir = os.getenv('HOME') + '/lfw/data/val/input/img/'
data_val_label_dir = os.getenv('HOME') + '/lfw/data/val/label/mask/'

os.makedirs(data_train_input_dir, exist_ok=True)
os.makedirs(data_train_label_dir, exist_ok=True)
os.makedirs(data_val_input_dir, exist_ok=True)
os.makedirs(data_val_label_dir, exist_ok=True)


def search(dirname):
    files = []
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                files.extend(search(full_filename))
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.jpg' or ext == '.png':
                    files.append(full_filename)
    except PermissionError:
        pass

    return files

files = search(data_source_dir)

for idx, img_file in enumerate(files):
    try:
        img = cv2.imread(img_file)
        img_bgr = img.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if idx % 10 == 0:  # for validation
            input_dir = data_val_input_dir
            label_dir = data_val_label_dir
        else:    # for train
            input_dir = data_train_input_dir
            label_dir = data_train_label_dir

        left_eye_img_path = input_dir + ('eye_%06d_l.png' % idx)
        right_eye_img_path = input_dir + ('eye_%06d_r.png' % idx)
        left_eye_label_path = label_dir + ('eye_%06d_l.png' % idx)
        right_eye_label_path = label_dir + ('eye_%06d_r.png' % idx)

        dlib_rects = detector_hog(img_rgb, 1)  # (image, num of img pyramid)

        list_landmarks = []
        for dlib_rect in dlib_rects:
            points = landmark_predictor(img_rgb, dlib_rect)
            list_points = list(map(lambda p: (p.x, p.y), points.parts()))
            list_landmarks.append(list_points)

        landmark = list_landmarks[0]

        # 눈 이미지 crop
        img_left_eye, img_right_eye = eye_crop(img_bgr, landmark)

        # 눈동자 중심 좌표 출력
        left_center_x, left_center_y, right_center_x, right_center_y, le_shape, re_shape = detectPupil(img_bgr, landmark)

        left_left_x = landmark[36][0]
        left_left_y = landmark[36][1]
        left_right_x = landmark[39][0]
        left_right_y = landmark[39][1]
        right_left_x = landmark[42][0]
        right_left_y = landmark[42][1]
        right_right_x = landmark[45][0]
        right_right_y = landmark[45][1]

        # left eye 이미지 출력
        show_left = img_left_eye.copy()
        cv2.imwrite(left_eye_img_path, show_left)

        img_label_left = cv2.circle(np.zeros_like(show_left), (left_left_x, left_left_y), 3, (1), -1)  # Left label = 1
        img_label_left = cv2.circle(img_label_left, (left_right_x, left_right_y), 3, (2), -1)  # Right label = 2
        img_label_left = cv2.circle(img_label_left, (left_center_x, left_center_y), 3, (3), -1)  # Right label = 3
        cv2.imwrite(left_eye_label_path, img_label_left)

        # right eye 이미지 출력
        show_right = img_right_eye.copy()
        cv2.imwrite(right_eye_img_path, show_right)

        img_label_right = cv2.circle(np.zeros_like(show_right), (right_left_x, right_left_y), 3, (1), -1)   # Left label = 1
        img_label_right = cv2.circle(img_label_right, (right_right_x, right_right_y), 3, (2), -1)  # Right label = 2
        img_label_right = cv2.circle(img_label_right, (right_center_x, right_center_y), 3, (3), -1)  # Right label = 3
        cv2.imwrite(right_eye_label_path, img_label_right)
    except:
        continue   # dlib, cv2 exception 발생시 skip

    if idx % 1000 == 0:
        print('%d image OK..' % idx)

print('Finished!!')




