import cv2
import numpy as np


def get_cropped_text_image(img):

    img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([172, 100, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow('frame',img)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=2)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    edged = cv2.Canny(dilation, 30, 200)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if len(contours) != 0:
            area = cv2.contourArea(c)

            edged = cv2.drawContours(edged, c, -1, 255, 1)
            # I = cv2.drawContours(img, c, contourIdx=-1, color=(0, 0, 255), thickness=5)

            contours2 = sorted(contours[0], key=cv2.contourArea, reverse=True)
            I = cv2.drawContours(img, contours2, contourIdx=-1, color=(0, 0, 255), thickness=5)

            x, y, w, h = cv2.boundingRect(contours[0])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            new_img = img[y:y + h, x:x + w]

    # cv2.imshow('cropped', new_img)
    # cv2.waitKey(0)
    return new_img
