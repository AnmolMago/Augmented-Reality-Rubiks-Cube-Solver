import cv2
import numpy as np
import math

def drawLine(img, rho, theta, color):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),color,2)

cap = cv2.VideoCapture(0)
thres = 100
def main():
    global thres
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(img,(7,7),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,35,111,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/45,thres)
        
        if lines is not None:

            linesByAngle = {}
            
            if len(lines) < 30:
                thres -= 2
            if len(lines) > 80:
                thres += 2

            for x in range(0, len(lines)):
                for rho, theta in lines[x]:
                    drawLine(img, rho, theta, (0,0,255))
                    angle = round(math.degrees(theta), -1)
                    if not angle in linesByAngle:
                        linesByAngle[angle] = [(rho,theta)]
                    else:
                        linesByAngle[angle].append((rho, theta))

            lineGroups = {}#{int, list}

            for approxAngle, values in linesByAngle.iteritems():
                # sort by similar rhos
                for i in range(0, len(values)):
                    found = False
                    for j in range(0, i):
                        if abs(values[i][0] - values[j][0]) < 50:
                            lineGroups[values[j][2]].append(values[i])
                            values[i] = (values[i][0], values[i][1], values[j][2])
                            found = True
                            break
                    if not found:
                        values[i] = (values[i][0], values[i][1], len(lineGroups))
                        lineGroups[len(lineGroups)] = [(values[i][0], values[i][1], len(lineGroups))]

            for index, lines in lineGroups.iteritems():
                # print green on avg of those rhos and thetas
                if len(lines) < 3:
                    continue
                rho = 0
                theta = 0
                for l in lines:
                    rho += l[0]
                    theta += l[1]
                rho /= len(lines)
                theta /= len(lines)
                drawLine(img, rho, theta, (0,255,0))

        elif thres > 2:
            thres -= 2
        else:
            print "Failed to detect anything....!"

        cv2.imshow('edges',edges)
        cv2.imshow('frame',img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()


#old code

#                if (theta > np.pi/10 and theta < 2*np.pi/5) or (theta > 2*np.pi/3 and theta < 5*np.pi/6):
#                    continue
#            thetanew = math.radians(round(math.degrees(theta), -1))

#    lines = cv2.HoughLinesP(edges,1,np.pi/45,threshold=thres,minLineLength=25,maxLineGap=30)
#    if lines != None:
#        if len(lines) < 50:
#            thres -= 5
#        if len(lines) > 75:
#            thres += 5
#        for line in lines:
#            x1,y1,x2,y2 = line[0]
#            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)