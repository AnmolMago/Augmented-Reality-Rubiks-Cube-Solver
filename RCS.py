import os
import cv2
import time
import math
import numpy as np

height, width = -1, -1
thres = 65

def standardizeLines(v):
    rho = v[0][0]
    theta = v[0][1]
    if rho < 0:
        rho *= -1
        theta -= math.pi
    return [rho, theta, -1]

def distanceBetweenLines(line1, line2):
    extreme_x = (0,1,2,width,width-1,width-2)
    extreme_y = (0,1,2,height,height-1,height-2)
    y1l, y1r, x1b, x1t = extreme_xy_values(line1)
    y2l, y2r, x2b, x2t = extreme_xy_values(line2)
    if x1b in extreme_x and x2b in extreme_x and x1t in extreme_x and x2t in extreme_x:
        return ( abs(y2l-y1l) + abs(y2r-y1r) )/2
    elif y1l in extreme_y and y2l in extreme_y and y1r in extreme_y and y2r in extreme_y:
        return ( abs(x2b-x1b) + abs(x2t-x1t) )/2
    else:
        return (math.sqrt( abs(x2b-x1b)**2 + abs(y2l-y1l) ) + math.sqrt( abs(x2t-x1t)**2 + abs(y2r-y1r) ))/2

def extreme_xy_values(line):
    a = math.cos(float(line[1]))
    b = math.sin(float(line[1]))
    x0 = a*float(line[0])
    y0 = b*float(line[0])
    if abs(b) < 10**-10:
        return 0, height, x0, x0
    if abs(a) < 10**-10:
        return y0, y0, 0, width
    slope = -a/b
    
    y_left = y0 - slope * x0
    y_right = y_left + slope * width
    y_left = min(max(y_left, 0), height)
    y_right = min(max(y_right, 0), height)

    x_bottom = (y_left-y0)/slope + x0
    x_top = (y_right-y0)/slope + x0
    x_bottom = min(max(x_bottom, 0), width)
    x_top = min(max(x_top, 0), width)
    # print str(y_left) + "|" + str(y_right) + "|" + str(x_bottom) + "|" + str(x_top) + "|" + str(a) + "|" + str(b) + "|" + str(slope) + "|"
    return int(y_left), int(y_right), int(x_bottom), int(x_top)

def are_lines_similar(line1, line2):

    if abs(line1[1] - line2[1]) > math.radians(5):
        return False
    extreme_x = (0,1,2,width,width-1,width-2)
    extreme_y = (0,1,2,height,height-1,height-2)
    y1l, y1r, x1b, x1t = extreme_xy_values(line1)
    y2l, y2r, x2b, x2t = extreme_xy_values(line2)
    s_thres = 30
    if abs(y2r-y1r+y2l-y1l) < s_thres and x1b in extreme_x and x2b in extreme_x and x1t in extreme_x and x2t in extreme_x:
        return True
    if abs(x2b-x1b+x2t-x1t) < s_thres and y1l in extreme_y and y2l in extreme_y and y1r in extreme_y and y2r in extreme_y:
        return True
    if abs(y2r-y1r+y2l-y1l) < s_thres and abs(x2b-x1b+x2t-x1t) < s_thres:
        return True
    # print str(math.degrees(abs(line1[1] - line2[1]))) + " | " + str(abs(y2r-y1r+y2l-y1l)) + " | " + str(abs(x2b-x1b+x2t-x1t))
    return False

def getIntersection(line1, line2):
    a = np.array([[math.cos(line1[1]), math.sin(line1[1])], [math.cos(line2[1]), math.sin(line2[1])]])
    b = np.array([line1[0], line2[0]])
    x = np.linalg.solve(a, b)
    return (x[0], x[1])

def getIntersections(lines):
    intersections = []
    intersections.append(getIntersection(lines[1], lines[2]))
    intersections.append(getIntersection(lines[0], lines[2]))
    intersections.append(getIntersection(lines[0], lines[3]))
    intersections.append(getIntersection(lines[1], lines[3])) 
    return intersections

def getPointAlongLine(ends, factor):
    x1, y1 = ends[0]
    x2, y2 = ends[1]

    x = x1 + (x2-x1) * factor 
    y = y1 + (y2-y1) * factor

    return x, y 

def getExpectedPoints(isecs):
    points = []
    prev, next = 0, 1
    while prev != 4:
        points.append(getPointAlongLine([isecs[prev], isecs[next]], float(1)/3))
        points.append(getPointAlongLine([isecs[prev], isecs[next]], float(2)/3))
        prev += 1
        next += 1
        if next == 4:
            next = 0
            
    points.append(getPointAlongLine([points[0], points[5]], float(1)/3))
    points.append(getPointAlongLine([points[0], points[5]], float(2)/3))

    points.append(getPointAlongLine([points[1], points[4]], float(1)/3))
    points.append(getPointAlongLine([points[1], points[4]], float(2)/3))
    return points

def drawLine(img, rho, theta, color):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 - 1000*b)
    x2 = int(x0 + 1000*b)
    y1 = int(y0 + 1000*a)
    y2 = int(y0 - 1000*a)
    # yl, yr, xb, xt = extreme_xy_values((rho,theta))
    # cv2.line(img,(0,int(yl)),(width,int(yl)),(0,0,0),2)
    # cv2.line(img,(0,int(yr)),(width,int(yr)),(0,0,0),2)
    # cv2.line(img,(int(xb),0),(int(xb),height),(255,255,255),2)
    # cv2.line(img,(int(xt),0),(int(xt),height),(255,255,255),2)
    cv2.line(img,(x1,y1),(x2,y2),color,2)

cap = cv2.VideoCapture(0)
def main():
    for f in os.listdir("."):
        if f.startswith("l4some") or f.startswith("lineGroup") or f.startswith("lparallel"):
            os.remove(f)
    global height, width
    timing = {}
    avgCount = {}
    def increaseTime(key, time):
        timing[key] = timing.get(key, 0) + time
        avgCount[key] = avgCount.get(key, 0) + 1
    def getAvgTime(key):
        if not key in timing:
            return -1
        return str(float(timing[key])/avgCount[key])
    thres = 40
    isBlacklist = True
    blacklist = []
    count = 0
    eth = 65
    while True:
        start = time.time()
        ret, frame = cap.read()
        img = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        if height == -1:
            height, width = img.shape[:2]
        black = np.zeros((height,width,3), np.uint8)
        blur = cv2.GaussianBlur(img,(7,7),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,eth, eth*3,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/45,thres)
        
        if lines is not None and len(lines) <= 100:
            lines = map(standardizeLines, lines)
            lineGroups = {}
            if not isBlacklist:
                if len(lines) < 35:
                    thres -= 2		
                if len(lines) > 50:
                    thres += 2

            for i in range(0,len(lines)):
                foundGroup = False
                # drawLine(img, lines[i][0], lines[i][1], (255,0,0))

                for j in range(0, i+1):
                    if (lines[j][0], lines[j][1]) in blacklist:
                        continue
                    if i != j and are_lines_similar(lines[i], lines[j]):
                        lines[i][2] = lines[j][2]
                        lineGroups[lines[j][2]].append(lines[i])
                        foundGroup = True
                        break

                if not foundGroup:
                    lines[i][2] = len(lineGroups)
                    lineGroups[len(lineGroups)] = [lines[i]]

            avgLines = []
            for index, lines in lineGroups.iteritems():
                rho = 0
                theta = 0
                for l in lines:
                    rho += l[0]
                    theta += l[1]

                rho /= len(lines)
                theta /= len(lines)
                drawLine(img, rho, theta, (0,255,0))
                similarExists = False
                for avgLine in avgLines:
                    if are_lines_similar(avgLine, [rho, theta]):
                        similarExists = True
                        break
                if not similarExists:
                    avgLines.append([rho, theta])
            
            avgTime = time.time()
            increaseTime("avg", avgTime-start)
            parallelPairs = []
            
            for i in range(0, len(avgLines)):
                for j in range(0, i+1):
                    angles = (avgLines[i][1],  avgLines[j][1])
                    angles = map(lambda a: a if a > 0 else (a + math.pi), angles)
                    angle_diff = abs(angles[0] - angles[1])
                    if i != j and angle_diff <= math.radians(5):
                        avgAngle = (angles[0] + angles[1])/2
                        dist = distanceBetweenLines(avgLines[i], avgLines[j])
                        if dist > 20: #Lines too close together do not count
                            parallelPairs.append((avgAngle, dist, [avgLines[i], avgLines[j]]))

            paraTime = time.time()
            increaseTime("parapair", paraTime-avgTime)
            index = 0
            intersectionPoints = []
            perpendicularFoursome = []

            for i in range(0, len(parallelPairs)):
                for j in range (0, i+1):
                    if i == j:
                        continue

                    l1 = parallelPairs[i]
                    l2 = parallelPairs[j]

                    if l1[1] < 75 or l2[1] < 75:
                        continue
                    unique = True
                    for line in l1[2]:
                        if line in l2[2]:
                            unique = False
                            break
                    if not unique:
                        continue
                    
                    angle_diff = abs(parallelPairs[i][0] - parallelPairs[j][0])
                    dist_diff = abs(parallelPairs[i][1] - parallelPairs[j][1])
                    if angle_diff >= math.radians(80) and angle_diff <= math.radians(100) and dist_diff <= 20:
                        arrLines = parallelPairs[i][2] + parallelPairs[j][2]
                        
                        for line in arrLines:
                            drawLine(img, line[0], line[1], (0,0,255))
                        
                        intersections = getIntersections(arrLines)
                        
                        for point in intersections:
                            # cv2.circle(img, tuple(map(lambda x: int(x), point)), 5, (255,0,0), 2)
                            if not point in intersectionPoints:
                                intersectionPoints.append(point)
                        
                        expectedPoints = getExpectedPoints(intersections)
                        perpendicularFoursome.append((arrLines, expectedPoints))
                        # for point in expectedPoints:
                        #     cv2.circle(img, tuple(map(lambda x: int(x), point)), 10, (0,255,0), 1)

                        index += 1

            perpTime = time.time()
            increaseTime("perpsome", perpTime-paraTime)
            topLines = None
            topScore = -1

            for obj in perpendicularFoursome:
                lines, expectedPoints = obj
                score = 0
                for ePoint in expectedPoints:
                    for iPoint in intersectionPoints:
                        if math.sqrt( (ePoint[0] - iPoint[0])**2 + (ePoint[1] - iPoint[1])**2 ) < 20:
                            score += 1
                            break
                if score > topScore:
                    topLines = lines
                    topScore = score

            print "topScore is " + str(topScore)

            guessTime = time.time()
            increaseTime("guess", guessTime-perpTime)
            if topScore >= 3:
                for line in topLines:
                    drawLine(img, line[0], line[1], (255,255,255))

        elif lines is not None and len(lines) > 100:
            thres += 5
        elif thres > 2:
            thres -= 2
        else:
            print "Failed to detect anything....!"

        # cv2.imshow('guess',edges)
        cv2.imshow('frame',img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    print "avg lines: " + getAvgTime("avg")
    print "parallelPairs: " + getAvgTime("parapair")
    print "perpendicularFoursome: " + getAvgTime("perpsome")
    print "guess: " + getAvgTime("guess")

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