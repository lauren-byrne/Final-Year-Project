import glob
import os, fnmatch
import cv2


startpath='Answers'
listOfFiles = os.listdir(startpath)
frame_counter = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writeVideo = False
count = 1
videos = []




#print(listOfFiles)
pattern = "*.mp4"
names = []
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        names.append(entry)
#print(names) #this list has all the .mp4 files

video_index = 0
cap = cv2.VideoCapture('Answers\\'+names[0])

out = cv2.VideoWriter('AnswerFULL.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

print(names[0])

while True:

    ret, frame = cap.read()

    #frame = cv2.resize(frame, (0, 0), fx=1.1, fy=1.1)
    if frame is None:
        print("end of video " + str(video_index) + " .. next one now")
        video_index += 1
        if video_index >= len(names):
            break
        cap = cv2.VideoCapture('Answers\\'+names[video_index])
        ret, frame = cap.read()

    out.write(frame)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

