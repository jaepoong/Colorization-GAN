import cv2
import os

print(cv2.__version__)
filepath='./data/yourname.mp4'
video=cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open :",filepath)
    exit(0)

#불러온 비디오 파일의 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("fps :", fps)\

#프레임을 저장할 디렉토리를 생성
try:
    if not os.path.exists("./data/yourname"):
        os.makedirs("./data/yourname")
except OSError:
    print ('Error: Creating directory. ' + "./data/yourname")

count = 0

while(video.isOpened()):
    ret, image = video.read()
    if(int(video.get(1)) % 60 == 0): #앞서 불러온 fps 값을 사용하여 2초마다 추출
        cv2.imwrite("./data/yourname" + "/frame%d.jpg" % count, image)
        print('Saved frame number :', str(int(video.get(1))))
        count += 1
        
video.release()