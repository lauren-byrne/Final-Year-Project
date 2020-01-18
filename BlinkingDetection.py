'''
Lauren Byrne C16452764 FYP - Prototype
Program that will detect a face from webcam input using Dlib HoG face detection.
Detected face will have a green bounding box drawn around it.
Using Dlib facial landmark detection, facial landmarks will be detected
and drawn on face using blue circles

----References----
Face landmarks detection - Opencv with Python - YouTube [Internet]. [cited 2020 Jan 15].
Available from: https://www.youtube.com/watch?v=MrRGVOhARYY&t=429s

Eye Blinking detection – Gaze controlled keyboard with Python and Opencv p.2 - YouTube [Internet].
[cited 2020 Jan 15]. Available from: https://www.youtube.com/watch?v=mMObcjHs59E&t=182s
Eye Blinking detection – Gaze controlled keyboard with Python and Opencv p.2 - YouTube [Internet]. [cited 2020 Jan 15].
Available from: https://www.youtube.com/watch?v=mMObcjHs59E&t=182s

Eye motion tracking - Opencv with Python - YouTube [Internet].
[cited 2019 Nov 6]. Available from: https://www.youtube.com/watch?v=kbdbZFT9NQI&t=1166s

'''

import cv2
import dlib
import csv
from math import hypot
import numpy as np


cap = cv2.VideoCapture(0)

#dlib face detector and facial landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_points = list()


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # using eye points for drawing
   # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 1)
   # ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 1)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio


def get_eye_shape(eye_points, facial_landmarks, black_image):
    # Track gaze
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    cv2.polylines(black_image, [eye_region], True, 255, 2)
    cv2.fillPoly(black_image, [eye_region], 255)

    return min_x, max_x, min_y, max_y, black_image


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print('frame', frame)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)




        print(landmarks)

        # mapping each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        RightEyeX = landmarks.part(46).x
        RightEyeY = landmarks.part(46).y

        #Track blinking
        blinking_ratio = (right_eye_ratio + left_eye_ratio)/2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))



        height = np.size(frame, 0)
        width = np.size(frame, 1)

        black_img = np.zeros((height, width), dtype="uint8")

        min_x_left, max_x_left, min_y_left, max_y_left, black_img = get_eye_shape([36, 37, 38, 39, 40, 41], landmarks, black_img)
        min_x_right, max_x_right, min_y_right, max_y_right, black_img = get_eye_shape([42, 43, 44, 45, 46, 47], landmarks, black_img)

        result_eye = cv2.bitwise_and(frame, frame, mask=black_img)

        eye = result_eye[min_y_left: max_y_left, min_x_left: max_x_left]
        eye = result_eye[min_y_right: max_y_right, min_x_right: max_x_right]
        eye = cv2.resize(eye, None, fx=5, fy=5)


        cv2.imshow('Eye', eye)

        #creating threshold for eye
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        T, B = cv2.threshold(gray_eye, 50, maxval=255, type=cv2.THRESH_BINARY)

        eye_height = np.size(gray_eye, 0)
        eye_width = np.size(gray_eye, 1)

        left_white_eye_L = gray_eye[0:eye_height, 0:int(eye_width/2)]
        right_white_eye_L = gray_eye[0:eye_height, int(eye_width/2):eye_width]

        left_white_eye_R = gray_eye[0:eye_height, 0:int(eye_width / 2)]
        right_white_eye_R = gray_eye[0:eye_height, int(eye_width / 2):eye_width]


        left_white_amount_L = cv2.countNonZero(left_white_eye_L)
        right_white_amount_L = cv2.countNonZero(right_white_eye_L)
        left_white_amount_R = cv2.countNonZero(left_white_eye_R)
        right_white_amount_R = cv2.countNonZero(right_white_eye_R)

        white_ratio_L = left_white_amount_L/right_white_amount_L
        white_ratio_R = left_white_amount_R/right_white_amount_R

        gaze_ratio = (white_ratio_L+white_ratio_R)/2

        cv2.putText(frame, str(gaze_ratio), (50, 150), font, 3, (255, 0, 0))


        if gaze_ratio < 0.83:
            cv2.putText(frame, 'LEFT', (100, 200), font, 3, (255, 0, 0))
        if 0.83 <= gaze_ratio < 0.94:
            cv2.putText(frame, 'CENTER', (100, 200), font, 3, (255, 0, 0))
        if gaze_ratio >= 0.94:
            cv2.putText(frame, 'RIGHT', (100, 200), font, 3, (255, 0, 0))



        cv2.imshow('Black Frame', black_img)
        cv2.imshow('Thresh Eye', B)

        cv2.imshow('result', result_eye)

    cv2.imshow('frame', frame)

    # save on pressing 'y'
    if cv2.waitKey(1) & 0xFF == ord('y'):
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # print('x :', x)
            # print('y :', y)
            face_points.append(x)
            face_points.append(y)
        if gaze_ratio < 0.85:
            cv2.putText(frame, 'LEFT', (100, 200), font, 3, (255, 0, 0))
            face_points.append(0)
        if 0.85 <= gaze_ratio < 0.94:
            cv2.putText(frame, 'CENTER', (100, 200), font, 3, (255, 0, 0))
            face_points.append(1)
        if gaze_ratio >= 0.94:
            cv2.putText(frame, 'RIGHT', (100, 200), font, 3, (255, 0, 0))
            face_points.append(2)


        with open('test.csv', 'a', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(face_points)

        face_points = []

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
# print('landmarks:  ', face_points)'''


