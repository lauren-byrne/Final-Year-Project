import cv2
import numpy as np


def get_cropped_image2(img):
    frame_changed = False

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(g, 1, 10, 120)

    edges = cv2.Canny(gray, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('closed', closed)

    contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:

        if cv2.contourArea(cont) > 5000:

            arc_len = cv2.arcLength(cont, True)

            approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

            if len(approx) == 4:
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
                # new_img = img[approx[0, 0, 1]:approx[2, 0, 1], approx[0, 0, 0]:approx[2, 0, 0]]
                x, y, w, h = cv2.boundingRect(approx)
                new_img = img[y:y + h, x:x + w]
                gray2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

                gray2 = cv2.bilateralFilter(gray2, 1, 10, 120)

                edges2 = cv2.Canny(gray2, 10, 250)

                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

                closed2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel2)

                contours, hierarchy = cv2.findContours(closed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print(contours)
                for c in contours:

                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

                    # check to see if contours have been found
                    if len(approx) == 3:
                        cv2.drawContours(new_img, [approx], -1, (255, 0, 0), 2)
                        frame_changed = True

    return frame_changed
