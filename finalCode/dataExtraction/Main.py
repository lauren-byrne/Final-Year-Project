import os, fnmatch
import cv2
import dlib
import csv
import imutils
import numpy as np
from gazeTracking import get_eye_shape, get_white_ratio
from blushTracking import get_blush_change
from blinkTracking import get_blinking_ratio
from eyebrowTracking import get_eyebrow_ratio
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb


videos = []
cheek_control = 0
video_num = 0
face_points = list()
eyebrow_start = 0
font = cv2.FONT_HERSHEY_SIMPLEX

startpath = 'Answers\\volunteer1\\lies\\'
listOfFiles = os.listdir(startpath)

# print(listOfFiles)
pattern = "*.mp4"
names = []
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        names.append(entry)
# print(names) #this list has all the .mp4 files

video_index = 0
cap = cv2.VideoCapture('Answers\\volunteer1\\lies\\' + names[0])

# dlib face detector and facial landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# facial normalisation
fa = FaceAligner(predictor, desiredFaceWidth=256)


while True:

    ret, frame = cap.read()

    # frame = cv2.resize(frame, (0, 0), fx=1.1, fy=1.1)
    if frame is None:
        print("end of video " + str(video_index) + " .. next one now")
        video_index += 1
        if video_index >= len(names):
            break
        cap = cv2.VideoCapture('Answers\\volunteer1\\lies\\' + names[video_index])
        ret, frame = cap.read()

    video_num = video_index + 1

    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.9)

    # creating a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    frame = imutils.resize(frame, width=800)
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
            cv2.circle(faceAligned, (x, y), 1, (255, 0, 0), -1)

        height = np.size(faceAligned, 0)
        width = np.size(faceAligned, 1)

        # calling function to calculate blink using landmarks from left eye, and landmarks from right eye
        # returns the ratio of both of both left and right eye that is caused by a blink
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        # Track blinking
        blinking_ratio = (right_eye_ratio + left_eye_ratio) / 2

        # creating new black frame that is same size and width as original frame
        black_img = np.zeros((height, width), dtype="uint8")

        # calling functions to retrieve the eye shape of both the right and left eye to appear on new black frame
        min_x_left, max_x_left, min_y_left, max_y_left, black_img = get_eye_shape([36, 37, 38, 39, 40, 41], landmarks,
                                                                                  black_img)
        min_x_right, max_x_right, min_y_right, max_y_right, black_img = get_eye_shape([42, 43, 44, 45, 46, 47],
                                                                                      landmarks, black_img)

        left_ratio = get_white_ratio(min_x_left, max_x_left, min_y_left, max_y_left, black_img, faceAligned)
        right_ratio = get_white_ratio(min_x_right, max_x_right, min_y_right, max_y_right, black_img, faceAligned)

        gaze_ratio = (left_ratio + right_ratio) / 2

        if gaze_ratio < 0.92:
            cv2.putText(faceAligned, 'LEFT', (80, 50), font, 1, (0, 0, 255))
        elif gaze_ratio > 0.98:
            cv2.putText(faceAligned, 'RIGHT', (80, 50), font, 1, (0, 0, 255))
        else:
            cv2.putText(faceAligned, 'CENTER', (80, 50), font, 1, (0, 0, 255))

        cv2.putText(faceAligned, str(gaze_ratio), (50, 100), font, 1, (255, 255, 0))

        cheek_average = get_blush_change(faceAligned, [29, 1, 33, 1, 54, 0, 12, 4, 48], landmarks)

        if cheek_control == 0:
            cheek_control = cheek_average

        cheek_difference = ((cheek_control - cheek_average) / cheek_control) * 100

        eyebrow_ratio, upper_eyebrow_ratio = get_eyebrow_ratio([22, 23, 28, 21, 24], landmarks)

        if eyebrow_start == 0:
            eyebrow_start = eyebrow_ratio

        if eyebrow_ratio < eyebrow_start * 0.955:
            cv2.putText(faceAligned, 'frown', (20, 50), font, 2, (255, 0, 0))
        elif eyebrow_ratio > eyebrow_start * 1.03:
            cv2.putText(faceAligned, 'raised', (20, 50), font, 2, (255, 0, 0))
        else:
            cv2.putText(faceAligned, 'normal', (20, 50), font, 2, (255, 0, 0))

    cv2.imshow('frame', frame)
    cv2.imshow("Aligned", faceAligned)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        face_points.append(x)
        face_points.append(y)
    if gaze_ratio < 0.915:
        face_points.append('left')
    elif gaze_ratio > 0.98:
        face_points.append('right')
    else:
        face_points.append('center')
    if blinking_ratio > 5.7:
        face_points.append('blink')
    else:
        face_points.append('no blink')
    face_points.append(cheek_difference)
    if eyebrow_ratio < eyebrow_start * 0.955:
        face_points.append('low')
    elif eyebrow_ratio > eyebrow_start * 1.03:
        face_points.append('high')
    else:
        face_points.append('normal')
    face_points.append('lie')
    face_points.append(video_num)

    with open('data.csv', 'a', newline='') as myFile:
        wr = csv.writer(myFile)
        wr.writerow(face_points)

    face_points = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
