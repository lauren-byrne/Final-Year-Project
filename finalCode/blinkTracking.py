import math
from pointMaths import midpoint

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