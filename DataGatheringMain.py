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

Eye Gaze detection 2 – Gaze controlled keyboard with Python and Opencv p.4 - YouTube [Internet]. [cited 2020 Jan 21].
Available from: https://www.youtube.com/watch?v=-VVih_oJ3jc

Eye Gaze detection – Gaze controlled keyboard with Python and Opencv p.3 - YouTube [Internet]. [cited 2020 Jan 21].
Available from: https://www.youtube.com/watch?v=UCu6M3drlYg&t=691s

'''

import cv2
import dlib
import csv
import math
import numpy as np
import pytesseract
from scipy import stats
from imutils import face_utils
from imutils.object_detection import non_max_suppression
from gazeTracking import get_eye_shape, get_white_ratio

cap = cv2.VideoCapture(0)

# dlib face detector and facial landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cheek_control = 0
cheek_list = []
face_points = list()
capture = False
control_mouth_length = 0
numframes = 40
currentframes = 0
running = True
_, frame = cap.read()

# cap = cv2.VideoCapture('videos/Q3.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX


# function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# function to caculate the rate at which an eye will blink
def get_blinking_ratio(eye_points, facial_landmarks):

    # calculating the central vertical, and central horizontal lines in an eye
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # using eye points for drawing
    # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 1)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 1)

    # hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    # ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    # finding the length of the the horizontal and vertical lines
    hor_line_length = math.sqrt(((left_point[0] - right_point[0]) ** 2) + ((left_point[1] - right_point[1]) ** 2))
    ver_line_length = math.sqrt(((center_top[0] - center_bottom[0]) ** 2) + ((center_top[1] - center_bottom[1]) ** 2))

    ratio = hor_line_length / ver_line_length

    return ratio

def get_eyebrow_ratio(eyebrow_points, facial_landmarks):
    lower_left_point = (facial_landmarks.part(eyebrow_points[0]).x, facial_landmarks.part(eyebrow_points[0]).y)
    lower_right_point = (facial_landmarks.part(eyebrow_points[1]).x, facial_landmarks.part(eyebrow_points[1]).y)
    center = midpoint(facial_landmarks.part(eyebrow_points[2]), facial_landmarks.part(eyebrow_points[2]))
    upper_left_point = midpoint(facial_landmarks.part(eyebrow_points[3]), facial_landmarks.part(eyebrow_points[2]))
    upper_right_point = midpoint(facial_landmarks.part(eyebrow_points[4]), facial_landmarks.part(eyebrow_points[2]))

    dist1 = math.sqrt(((lower_left_point[0] - center[0]) ** 2) + ((lower_left_point[1] - center[1]) ** 2))
    dist2 = math.sqrt(((lower_right_point[0] - center[0]) ** 2) + ((lower_right_point[1] - center[1]) ** 2))
    ratio = dist1/dist2

    dist3 = math.sqrt(((upper_left_point[0] - center[0]) ** 2) + ((upper_left_point[1] - center[1]) ** 2))
    dist4 = math.sqrt(((upper_right_point[0] - center[0]) ** 2) + ((upper_right_point[1] - center[1]) ** 2))
    ratio2 = dist3/dist4

    return ratio, ratio2


while True:
    _, frame = cap.read()

    # frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # img = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

    height = np.size(frame, 0)
    width = np.size(frame, 1)

    # creating a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # using fact detection model
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # draw rectangle around detected face using points found in face detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # predict facial landmarks
        landmarks = predictor(gray, face)

        print(landmarks)

        # mapping each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # RightEyeX = landmarks.part(46).x
        # RightEyeY = landmarks.part(46).y

        # calling function to calculate blink using landmarks from left eye, and landmarks from right eye
        # returns the ratio of both of both left and right eye that is caused by a blink
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)



        # Track blinking
        blinking_ratio = (right_eye_ratio + left_eye_ratio) / 2
        # cv2.putText(frame, str(blinking_ratio), (50, 150), font, 3, (255, 0, 0))

        if blinking_ratio > 4.5:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))

        # creating new black frame that is same size and width as original frame
        black_img = np.zeros((height, width), dtype="uint8")

        # calling functions to retrieve the eye shape of both the right and left eye to appear on new black frame
        min_x_left, max_x_left, min_y_left, max_y_left, black_img = get_eye_shape([36, 37, 38, 39, 40, 41], landmarks,
                                                                                  black_img)
        min_x_right, max_x_right, min_y_right, max_y_right, black_img = get_eye_shape([42, 43, 44, 45, 46, 47],
                                                                                      landmarks, black_img)

        left_ratio = get_white_ratio(min_x_left, max_x_left, min_y_left, max_y_left, black_img, frame)
        right_ratio = get_white_ratio(min_x_right, max_x_right, min_y_right, max_y_right, black_img, frame)

        # cv2.imshow('Eye', eye)

        gaze_ratio = (left_ratio + right_ratio) / 2

        if gaze_ratio < 0.85:
            cv2.putText(frame, 'LEFT', (100, 200), font, 3, (255, 0, 0))
        if 0.85 <= gaze_ratio < 0.94:
            cv2.putText(frame, 'CENTER', (100, 200), font, 3, (255, 0, 0))
        if gaze_ratio >= 0.94:
            cv2.putText(frame, 'RIGHT', (100, 200), font, 3, (255, 0, 0))

        #  cv2.putText(frame, str(gaze_ratio), (50, 150), font, 3, (255, 0, 0))

        # cv2.imshow('Black Frame', black_img)
        #cv2.imshow('Thresh Eye', B)

        # cv2.imshow('masked result', masked_eye)

        landmarks2 = face_utils.shape_to_np(landmarks)

        crop = frame[landmarks2[29][1]:landmarks2[33][1], landmarks2[54][0]:landmarks2[12][0]]  # right cheeks
        crop2 = frame[landmarks2[29][1]:landmarks2[33][1], landmarks2[4][0]:landmarks2[48][0]]  # left cheek

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('right cheek', crop)
        # cv2.imshow('left cheek', crop2)

        cheek_average = (np.mean(crop) + np.mean(crop2))

        cheek_list.append(cheek_average)
        cheek_array = mat = np.array(cheek_list)
        # print('cheek list: ', cheek_list)

        if cheek_control == 0:
            cheek_control = cheek_average
        # print('z_score: ', stats.zscore(cheek_list))

        cheek_difference = ((cheek_control - cheek_average) / cheek_control) * 100

       # contours, h = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       # for c in contours:
       #     if len(contours) != 0:
       #         area = cv2.contourArea(c)
       #         cv2.drawContours(eye, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)

        right_mouth1 = landmarks.part(54)
        right_mouth2 = landmarks.part(55)
        right_mouth3 = landmarks.part(65)
        right_mouth4 = landmarks.part(56)

        left_mouth1 = landmarks.part(50)
        left_mouth2 = landmarks.part(49)
        left_mouth3 = landmarks.part(61)
        left_mouth4 = landmarks.part(60)

        right_mouth_region = np.array([(landmarks.part(54).y),
                                       (landmarks.part(55).y),
                                       (landmarks.part(65).y),
                                       (landmarks.part(56).y)],
                                      np.int32)

        left_mouth_region = np.array([(landmarks.part(50).y),
                                      (landmarks.part(49).y),
                                      (landmarks.part(61).y),
                                      (landmarks.part(60).y)],
                                     np.int32)

        if control_mouth_length == 0:
            control_mouth_length = math.sqrt(((landmarks.part(49).y - landmarks.part(55).y) ** 2) + (
                        (landmarks.part(55).x - landmarks.part(55).x) ** 2))

        current_mouth_length = math.sqrt(
            ((landmarks.part(49).y - landmarks.part(55).y) ** 2) + ((landmarks.part(55).x - landmarks.part(55).x) ** 2))

        mouth_distance_increase = ((current_mouth_length - control_mouth_length) / control_mouth_length) * 100.0
        #print('increase: ', mouth_distance_increase)

        eyebrow_ratio, upper_eyebrow_ratio = get_eyebrow_ratio([22, 23, 28, 21, 24], landmarks)

        #print('ratio: ', upper_eyebrow_ratio)

       # if eyebrow_ratio < 0.72:
        #    cv2.putText(frame, 'frown', (50, 150), font, 3, (255, 0, 0))
        #elif eyebrow_ratio > 0.765 or upper_eyebrow_ratio > 0.705:
         #   cv2.putText(frame, 'raised', (50, 150), font, 3, (255, 0, 0))
        #else:
         #   cv2.putText(frame, 'normal', (50, 150), font, 3, (255, 0, 0))

    cv2.imshow('frame', frame)

    # save on pressing 'y'
    if (cv2.waitKey(1) & 0xFF == ord('y')) or capture:
        # running = False
        if currentframes < numframes:
            capture = True
            currentframes = currentframes + 1
        if currentframes == numframes:
            capture = False
            currentframes = 0

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # print('x :', x)
            # print('y :', y)
            face_points.append(x)
            face_points.append(y)
        if gaze_ratio < 0.85:
            cv2.putText(frame, 'LEFT', (100, 200), font, 3, (255, 0, 0))
            face_points.append('left')
        if 0.85 <= gaze_ratio < 0.94:
            cv2.putText(frame, 'CENTER', (100, 200), font, 3, (255, 0, 0))
            face_points.append('center')
        if gaze_ratio >= 0.94:
            cv2.putText(frame, 'RIGHT', (100, 200), font, 3, (255, 0, 0))
            face_points.append('right')
        if blinking_ratio > 5.7:
            face_points.append('blink')
        else:
            face_points.append('no blink')
        face_points.append(cheek_difference)
        print('difference: ', cheek_difference)

        with open('test.csv', 'a', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(face_points)

        face_points = []

    # break program on press of 'Esc'
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
