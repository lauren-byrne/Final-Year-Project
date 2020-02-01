import cv2 as cv2
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480))

while cap.isOpened():
    ret, frame = cap.read()

    while cv2.waitKey(1) & 0xFF == ord('y'):
        out.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()