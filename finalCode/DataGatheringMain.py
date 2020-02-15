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
import imutils
import numpy as np
from gazeTracking import get_eye_shape, get_white_ratio
from blushTracking import get_blush_change
from blinkTracking import get_blinking_ratio
from eyebrowTracking import get_eyebrow_ratio
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import pointMaths
cap = cv2.VideoCapture(0)

# dlib face detector and facial landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


cheek_control = 0
cheek_list = []
face_points = list()
capture = False
control_mouth_length = 0
numframes = 40
currentframes = 0
running = True
_, frame = cap.read()
font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    _, frame = cap.read()


    # creating a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    #frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    # image
    faces = detector(gray)

    # loop over the face detections
    for face in faces:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(face)
        faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(frame, gray, face)

    gray2 = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
    faces2 = detector(gray2)

    for face in faces2:
        # predict facial landmarks
        landmarks = predictor(gray2, face)

        # mapping each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #cv2.circle(faceAligned, (x, y), 1, (255, 0, 0), -1)

        height = np.size(faceAligned, 0)
        width = np.size(faceAligned, 1)



        # calling function to calculate blink using landmarks from left eye, and landmarks from right eye
        # returns the ratio of both of both left and right eye that is caused by a blink
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        # Track blinking
        blinking_ratio = (right_eye_ratio + left_eye_ratio) / 2
        #cv2.putText(faceAligned, str(blinking_ratio), (50, 150), font, 3, (255, 0, 0))

        #if blinking_ratio > 4.5:
         #   cv2.putText(faceAligned, "BLINKING", (20, 60), font, 2, (255, 0, 0))

        # creating new black frame that is same size and width as original frame
        black_img = np.zeros((height, width), dtype="uint8")

        # calling functions to retrieve the eye shape of both the right and left eye to appear on new black frame
        min_x_left, max_x_left, min_y_left, max_y_left, black_img = get_eye_shape([36, 37, 38, 39, 40, 41], landmarks, black_img)
        min_x_right, max_x_right, min_y_right, max_y_right, black_img = get_eye_shape([42, 43, 44, 45, 46, 47], landmarks, black_img)

        left_ratio = get_white_ratio(min_x_left, max_x_left, min_y_left, max_y_left, black_img, faceAligned)
        right_ratio = get_white_ratio(min_x_right, max_x_right, min_y_right, max_y_right, black_img, faceAligned)

        # cv2.imshow('Eye', eye)

        gaze_ratio = (left_ratio + right_ratio) / 2

        if gaze_ratio < 0.92:
            cv2.putText(faceAligned, 'LEFT', (20, 50), font, 1, (0, 0, 255))
        elif gaze_ratio > 0.98:
         cv2.putText(faceAligned, 'RIGHT', (20, 50), font, 1, (0, 0, 255))
        else:
            cv2.putText(faceAligned, 'CENTER', (20, 50), font, 1, (0, 0, 255))

        #cv2.putText(frame, str(gaze_ratio), (50, 450), font, 3, (255, 0, 0))

        cheek_average = get_blush_change(faceAligned, cheek_list, [29, 1, 33, 1, 54, 0, 12, 4, 48], landmarks)

        if cheek_control == 0:
            cheek_control = cheek_average

        cheek_difference = ((cheek_control - cheek_average) / cheek_control) * 100

       # print('average: ', cheek_average, '\n control: ', cheek_control)

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
            control_mouth_length = math.sqrt(((landmarks.part(49).y - landmarks.part(55).y) ** 2) + ((landmarks.part(55).x - landmarks.part(55).x) ** 2))

        current_mouth_length = math.sqrt(((landmarks.part(49).y - landmarks.part(55).y) ** 2) + ((landmarks.part(55).x - landmarks.part(55).x) ** 2))

        mouth_distance_increase = ((current_mouth_length - control_mouth_length) / control_mouth_length) * 100.0
        #print('increase: ', mouth_distance_increase)

        eyebrow_ratio, upper_eyebrow_ratio = get_eyebrow_ratio([22, 23, 28, 21, 24], landmarks)

        p1 = (landmarks.part(18).x, landmarks.part(18).y)
        p2 = (landmarks.part(22).x, landmarks.part(22).y)
        p3 = (landmarks.part(23).x, landmarks.part(23).y)

        dist1 = pointMaths.distance(p1, p2)
        dist2 = pointMaths.distance(p2, p3)

        ratiodist = dist2/dist1

        print('ratio: ', ratiodist)
        #cv2.putText(faceAligned, str(ratiodist), (100, 200), font, 3, (255, 0, 0))

        #if ratiodist > 0.208:
         #   cv2.putText(faceAligned, 'frown', (50, 150), font, 3, (255, 0, 0))
        #else:
         #   cv2.putText(faceAligned, 'normal', (50, 150), font, 3, (255, 0, 0))

        #if eyebrow_ratio < 0.725:
         #   cv2.putText(faceAligned, 'frown', (20, 50), font, 2, (255, 0, 0))
        #elif eyebrow_ratio > 0.77 or upper_eyebrow_ratio > 0.705:
         #   cv2.putText(faceAligned, 'raised', (20, 50), font, 2, (255, 0, 0))
        #else:
         #   cv2.putText(faceAligned, 'normal', (20, 50), font, 2, (255, 0, 0))

    cv2.imshow('frame', frame)

    cv2.imshow("Aligned", faceAligned)

    # save on pressing 'y'
    if (cv2.waitKey(1) & 0xFF == ord('y')) or capture:
        if currentframes < numframes:
            capture = True
            currentframes = currentframes + 1
        if currentframes == numframes:
            capture = False
            currentframes = 0

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
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
