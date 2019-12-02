import cv2
import numpy as np
import dlib
import csv
import pandas as pd
from imutils import face_utils
import sys


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_points = list()
i = 0


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


cap = cv2.VideoCapture(0)

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
        #landmarks_array = face_utils.shape_to_np(landmarks)

        print(landmarks)

        # mapping each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        RightEyeX = landmarks.part(46).x
        RightEyeY = landmarks.part(46).y

        # setting points for eye
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # using eye points for drawing
        hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 1)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 1)

    cv2.imshow('frame', frame)

    start = 0
    maxFrames = 5

    if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
        #cv2.imwrite('pic'+str(start)+'.jpg', frame)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # print('x :', x)
            # print('y :', y)
            face_points.append(x)
            face_points.append(y)

        with open('test.csv', 'a', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(face_points)

        start += 1
        face_points = []



    #c = cv2.waitKey(1)
    if cv2.waitKey(10) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

    '''if c == 27:
        cap.release()
        cv2.destroyAllWindows()
        break'''

# print('landmarks:  ', face_points)


