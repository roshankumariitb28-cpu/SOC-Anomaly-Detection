import cv2
import numpy as np 

img = cv2.imread("airplane.jpg")

flags =False
ix=-1
iy=-1

def draw(event,x,y,flags,params):

    global flag,ix,iy

    if event == 1:
        flags=True
        ix=x
        iy=y
    
    elif event==0:
        if flags==True:
            cv2.rectangle(img,pt1=(ix,iy),pt2=(x,y),thickness=1,color=(0,0,0))

    elif event==4:
        flag = False
        cv2.rectangle(img,pt1=(ix,iy),pt2=(x,y),thickness=1,color=(0,0,0))

cv2.namedWindow(winname="window")
cv2.setMouseCallback("window",draw)

while True:

    cv2.imshow("window",img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()