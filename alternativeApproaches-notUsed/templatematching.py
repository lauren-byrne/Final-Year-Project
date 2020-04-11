import cv2
import numpy as np

cap = cv2.VideoCapture('dadtest2.mp4')
count = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
switch = False
number = 1

out = cv2.VideoWriter('Answers\\Answer.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
while True:

    if switch:
        out = cv2.VideoWriter('Answers\\Answer2.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        switch = False

    _, frame = cap.read()
    # img_rgb = cv2.imread('lastWaldo.jpg')
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('patch2.png', 0)

    template = cv2.resize(template, (0, 0), fx=0.4, fy=0.4)

    # saves the width and height of the template into 'w' and 'h'
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.93
    # finding the values where it exceeds the threshold
    #loc = np.where(res >= threshold)
    #for pt in zip(*loc[::-1]):
    #    # draw rectangle on places where it exceeds threshold
    #    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    if (res >= threshold).any():
        out.write(frame)
        switch = True

    cv2.imshow('frame', frame)
    cv2.imshow('patch', template)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
