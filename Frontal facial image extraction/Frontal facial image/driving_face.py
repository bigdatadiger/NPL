# 可以通过人脸检测，判断68个特征点是否都能检测到，再判断嘴巴是否闭着，

import sys
sys.path.append('D:/anaconda3/envs/faceExtraction/Lib/site-packages')

# 眨眼检测原理：计算眼睛长宽比 Eye Aspect Ratio，EAR.当人眼睁开时，EAR在某个值上下波动，当人眼闭合时，EAR迅速下降，理论上会接近于零，
# 当时人脸检测模型还没有这么精确。所以我们认为当EAR低于某个阈值时，眼睛处于闭合状态。

# 张口检测原理：类似眨眼检测，计算Mouth Aspect Ratio,MAR.当MAR大于设定的阈值时，认为张开了嘴巴。

from imutils import face_utils
import numpy as np
import dlib
import cv2

def eye_aspect_ratio(eye):
    # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    # (|m2-m9|+|m4-m7|)/(2|m0-m6|)
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def liveness_detection(videoFile):
    vs = cv2.VideoCapture(videoFile)

    EAR_THRESH = 0.2
    MAR_THRESH = 0.5

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # 从视频文件读取帧
    vs = cv2.VideoCapture(videoFile)

    # 示意开始
    print("[INFO] starting video stream thread...")

    while True:
        rval, frame = vs.read()
        candidate_images = []

        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            # 特征点数量检测
            for rect in rects:
                shape = predictor(gray,rect)
                # 判断是否为68个
                if shape.num_parts == 68:
                    print()

            # 只能处理一张人脸
            if len(rects) == 1:
                shape = predictor(gray, rects[0])  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                if ear < EAR_THRESH:
                    print(1)
                    continue
                elif mar > MAR_THRESH:
                    print(2)
                    continue
                elif ear > EAR_THRESH and mar < MAR_THRESH:
                    candidate_images.append(frame)

        else:
            break
    # cv2.imwrite('best_frame.png', frame)
    # cv2.destroyAllWindows()
    vs.release()


liveness_detection(r"评书.mp4")