import cv2


#function to isolate the laptop screen from original image in order for detection of triangle occurence
def event_detection(img):
    frame_changed = False

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(g, 1, 10, 120)
    edges = cv2.Canny(gray, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        if cv2.contourArea(cont) > 5000:
            arc_len = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                new_img = img[y:y + h, x:x + w]

                grayCropped = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                grayCropped = cv2.bilateralFilter(grayCropped, 1, 10, 120)

                edges2 = cv2.Canny(grayCropped, 10, 250)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel2)

                contours, hierarchy = cv2.findContours(closed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

                    # check to see if contours have been found
                    if len(approx) == 3:
                        frame_changed = True

    return frame_changed
