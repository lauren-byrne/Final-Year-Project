import math
from pointMaths import midpoint

def get_eyebrow_ratio(eyebrow_points, facial_landmarks):
    lower_left_point = (facial_landmarks.part(eyebrow_points[0]).x, facial_landmarks.part(eyebrow_points[0]).y)
    lower_right_point = (facial_landmarks.part(eyebrow_points[1]).x, facial_landmarks.part(eyebrow_points[1]).y)
    center = midpoint(facial_landmarks.part(eyebrow_points[2]), facial_landmarks.part(eyebrow_points[2]))
    upper_left_point = midpoint(facial_landmarks.part(eyebrow_points[3]), facial_landmarks.part(eyebrow_points[2]))
    upper_right_point = midpoint(facial_landmarks.part(eyebrow_points[4]), facial_landmarks.part(eyebrow_points[2]))

    dist1 = math.sqrt(((lower_left_point[0] - center[0]) ** 2) + ((lower_left_point[1] - center[1]) ** 2))
    dist2 = math.sqrt(((lower_right_point[0] - center[0]) ** 2) + ((lower_right_point[1] - center[1]) ** 2))
    ratio = dist1/dist2

    dist3 = math.sqrt(((upper_left_point[0] - center[0]) ** 2) + ((upper_left_point[1] - center[1]) ** 2))
    dist4 = math.sqrt(((upper_right_point[0] - center[0]) ** 2) + ((upper_right_point[1] - center[1]) ** 2))
    ratio2 = dist3/dist4

    return ratio, ratio2
