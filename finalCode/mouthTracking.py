import pointMaths

def trackMouth(mouth_points, facial_landmarks):
    left_point = (facial_landmarks.part(mouth_points[0]).x, facial_landmarks.part(mouth_points[0]).y)
    center = (facial_landmarks.part(mouth_points[1]).x, facial_landmarks.part(mouth_points[1]).y)
    right_point = (facial_landmarks.part(mouth_points[2]).x, facial_landmarks.part(mouth_points[2]).y)

    dist1 = pointMaths.distance(left_point, center)
    dist2 = pointMaths.distance(right_point, center)

    ratio = dist1 / dist2

    return dist1, dist2

    '''if control_mouth_length == 0:
        control_mouth_length = math.sqrt(
            ((landmarks.part(49).y - landmarks.part(55).y) ** 2) + ((landmarks.part(55).x - landmarks.part(55).x) ** 2))

    current_mouth_length = math.sqrt(
        ((landmarks.part(49).y - landmarks.part(55).y) ** 2) + ((landmarks.part(55).x - landmarks.part(55).x) ** 2))

    mouth_distance_increase = ((current_mouth_length - control_mouth_length) / control_mouth_length) * 100.0
    # print('increase: ', mouth_distance_increase)'''