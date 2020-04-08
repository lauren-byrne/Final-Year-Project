import cv2
import dlib
from eyebrowTracking import get_eyebrow_ratio


def detection(faces, image):
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(image, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        eyebrow_ratio, upper_eyebrow_ratio = get_eyebrow_ratio([22, 23, 28, 21, 24], landmarks)

        return eyebrow_ratio


control = cv2.imread('normal.jpg')
test = cv2.imread('raised.jpg')

control = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces_control = detector(control)
faces_test = detector(test)

control_average = detection(faces_control, control)
test_average = detection(faces_test, test)

print(control_average*0.955)
print(test_average)

if test_average < control_average * 0.955:
    print('low')
elif test_average > control_average * 1.03:
    print('raised')
else:
    print('normal')


key = cv2.waitKey(0)
# 27 is ESC key



