import cv2
from EastTextDetection import get_text

cap = cv2.VideoCapture('mom-test.mp4')
while True:
    _, frame = cap.read()

    # frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # img = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

    # creating a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    text = get_text(frame)

    print('text', text)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
