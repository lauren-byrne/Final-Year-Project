import pointMaths


#function to return the two ratio distance of the two eyebrows for current frame
def get_eyebrow_ratio(eyebrow_points, facial_landmarks):
    lower_left_point = (facial_landmarks.part(eyebrow_points[0]).x, facial_landmarks.part(eyebrow_points[0]).y)
    lower_right_point = (facial_landmarks.part(eyebrow_points[1]).x, facial_landmarks.part(eyebrow_points[1]).y)
    center = pointMaths.midpoint(facial_landmarks.part(eyebrow_points[2]), facial_landmarks.part(eyebrow_points[2]))
    upper_left_point = pointMaths.midpoint(facial_landmarks.part(eyebrow_points[3]), facial_landmarks.part(eyebrow_points[2]))
    upper_right_point = pointMaths.midpoint(facial_landmarks.part(eyebrow_points[4]), facial_landmarks.part(eyebrow_points[2]))

    dist1 = pointMaths.distance(lower_left_point, center)
    dist2 = pointMaths.distance(lower_right_point, center)
    dist3 = pointMaths.distance(upper_left_point, center)
    dist4 = pointMaths.distance(upper_right_point, center)

    ratio = dist1 / dist2
    ratio2 = dist3 / dist4

    return ratio, ratio2
