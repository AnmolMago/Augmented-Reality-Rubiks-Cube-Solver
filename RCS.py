import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
thres = 100
while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img,(7,7),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,60,180,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/90,thres)
    if lines != None:
        if len(lines) < 30:
            thres -= 3
        if len(lines) > 80:
            thres += 3
        print str(thres) + "|" + str(len(lines))
        for x in range(0, len(lines)):
            for rho, theta in lines[x]:
#                if (theta > np.pi/10 and theta < 2*np.pi/5) or (theta > 2*np.pi/3 and theta < 5*np.pi/6):
#                    continue
    #            thetanew = math.radians(round(math.degrees(theta), -1))
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#    lines = cv2.HoughLinesP(edges,1,np.pi/45,threshold=thres,minLineLength=25,maxLineGap=30)
#    if lines != None:
#        if len(lines) < 50:
#            thres -= 5
#        if len(lines) > 75:
#            thres += 5
#        for line in lines:
#            x1,y1,x2,y2 = line[0]
#            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    elif thres > 5:
        thres -= 5
    else:
        print "Failed to detect anything....!"

    cv2.imshow('flap',edges)
    cv2.imshow('frame',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()