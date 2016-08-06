import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(frame,(3,3),0)
    #    blur = cv2.bilateralFilter(gray,9,75,50)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#    flap = cv2.Laplacian(gray,cv2.CV_64F,ksize=5)
#    flap = cv2.convertScaleAbs(flap)

#    edges = cv2.compare(flap, 254, cv2.CMP_GT)
    edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(edges,1,0.01,10,25,60)

    if lines != None:
        print len(lines)
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('flap',edges)
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()