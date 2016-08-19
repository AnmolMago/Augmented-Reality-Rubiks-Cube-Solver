import cv2
import math

thres = 65

def standardizeLines(v):
    rho = v[0][0]
    theta = v[0][1]
    if rho < 0:
        rho *= -1
        theta -= math.pi
    return [rho, theta, -1]

def distanceBetweenLines(line1, line2):
    height, width = img.shape[:2]
    extreme_x = (0,1,2,width,width-1,width-2)
    extreme_y = (0,1,2,height,height-1,height-2)
    y1l, y1r, x1b, x1t = extreme_xy_values(line1)
    y2l, y2r, x2b, x2t = extreme_xy_values(line2)
    s_thres = 30
    if x1b in extreme_x and x2b in extreme_x and x1t in extreme_x and x2t in extreme_x:
        return ( abs(y2l-y1l) + abs(y2r-y1r) )/2
    elif y1l in extreme_y and y2l in extreme_y and y1r in extreme_y and y2r in extreme_y:
        return ( abs(x2b-x1b) + abs(x2t-x1t) )/2
    else:
        return (math.sqrt( abs(x2b-x1b)**2 + abs(y2l-y1l) ) + math.sqrt( abs(x2t-x1t)**2 + abs(y2r-y1r) ))/2

def extreme_xy_values(line):
    height, width = img.shape[:2]
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
    # x_bottom = y_left/float(slope)
    x_bottom = (y_left-y0)/slope + x0
    x_top = (y_right-y0)/slope + x0
    x_bottom = min(max(x_bottom, 0), width)
    x_top = min(max(x_top, 0), width)
    # print str(y_left) + "|" + str(y_right) + "|" + str(x_bottom) + "|" + str(x_top) + "|" + str(a) + "|" + str(b) + "|" + str(slope) + "|"
    return int(y_left), int(y_right), int(x_bottom), int(x_top)

def are_lines_similar(line1, line2):
    #todo lines in shape of X return true, should be false
    height, width = img.shape[:2]
    extreme_x = (0,1,2,width,width-1,width-2)
    extreme_y = (0,1,2,height,height-1,height-2)
    y1l, y1r, x1b, x1t = extreme_xy_values(line1)
    y2l, y2r, x2b, x2t = extreme_xy_values(line2)
    s_thres = 30
    if abs(y2r-y1r) < s_thres and abs(y2l-y1l) < s_thres and x1b in extreme_x and x2b in extreme_x and x1t in extreme_x and x2t in extreme_x:
        return True
    if abs(x2b-x1b) < s_thres and abs(x2t-x1t) < s_thres and y1l in extreme_y and y2l in extreme_y and y1r in extreme_y and y2r in extreme_y:
        return True
    if abs(y2r-y1r) < s_thres and abs(y2l-y1l) < s_thres and abs(x2b-x1b) < s_thres and abs(x2t-x1t) < s_thres:
        return True
    return False

def getIntersection(line1, line2):
    r1 = line1[0]
    r2 = line2[0]

    t1 = line1[1]
    t2 = line2[1]

    # r1 = xcos(t1) + ysin(t1)
    # r2 = xcos(t2) + ysin(t2)

    # x = ( ysin(t1) - r1 ) / cos(t1) == ( ysin(t2) - r2 ) / cos(t2)

def getIntersections(lines):
    
    return points

def drawLine(img, rho, theta, color):
    height, width = img.shape[:2]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    yl, yr, xb, xt = extreme_xy_values((rho,theta))
    # cv2.line(img,(0,int(yl)),(width,int(yl)),(0,0,0),2)
    # cv2.line(img,(0,int(yr)),(width,int(yr)),(0,0,0),2)
    # cv2.line(img,(int(xb),0),(int(xb),height),(255,255,255),2)
    # cv2.line(img,(int(xt),0),(int(xt),height),(255,255,255),2)
    cv2.line(img,(x1,y1),(x2,y2),color,2)

frame = cv2.imread('0.jpg')
img = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
eth = 55
edges = cv2.Canny(gray,eth,eth*3,apertureSize = 3)
lines = cv2.HoughLines(edges,1,math.pi/180,thres)

if lines is not None:
    lines = map(standardizeLines, lines)
    lineGroups = {}

    for i in range(0, len(lines)):
        foundGroup = False

        for j in range(0, i):
            # drawLine(img, lines[i][0], lines[i][1], (255,0,0))
            if i != 0 and are_lines_similar(lines[i], lines[j]):
                # print "Lines: " + str(extreme_xy_values(lines[i])) + " and " + str(extreme_xy_values(lines[j])) + " are similar"
                lines[i][2] = lines[j][2]
                lineGroups[lines[j][2]].append(lines[i])
                foundGroup = True
                break

        if not foundGroup:
            lines[i][2] = len(lineGroups)
            lineGroups[len(lineGroups)] = [lines[i]]

    imgTemp = img.copy()

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
        avgLines.append([rho, theta])

    parallelPairs = []
    
    for i in range(0, len(avgLines)):
        for j in range(0, i):
            if i != j and abs(avgLines[i][1] - avgLines[j][1]) <= math.radians(3):
                avgAngle = (avgLines[i][1] + avgLines[j][1])/2
                dist = distanceBetweenLines(avgLines[i], avgLines[j])
                parallelPairs.append((avgAngle, dist, [avgLines[i], avgLines[j]]))

    index = 0
    #         dist_diff_j = distanceBetween(parallelPairs[avg_j][0], parallelPairs[avg_j][1])
    intersectionPoints = []
    perpendicularFoursome = []

    for i in range(0, len(parallelPairs)):
        for j in range (0, i):
            if i == j:
                continue

            l1 = parallelPairs[i]
            l2 = parallelPairs[j]
            imgNew = imgTemp.copy()
            angle_diff = abs(parallelPairs[i][0] - parallelPairs[j][0])
            dist_diff = abs(parallelPairs[i][1] - parallelPairs[j][1])
            if angle_diff >= math.radians(80) and angle_diff <= math.radians(100) and dist_diff <= 20:
                arrLines = parallelPairs[i][2] + parallelPairs[j][2]
                perpendicularFoursome.append(arrLines)
                for line in arrLines:
                    drawLine(img, line[0], line[1], (0,0,255))
                    drawLine(imgNew, line[0], line[1], (0,0,255))
                for point in getIntersections(arrLines):

                    cv2.circle(imgNew, point, 5, (255,0,0), 2)
                    if not point in intersectionPoints:
                        intersectionPoints.append(point)
                cv2.imwrite('lineGroup'+str(index)+'.jpg',imgNew)
                index += 1

    #delete duplicates from crossingPoints

    # for lines in perpendicularFoursome:
    #     expectedPoints = getExpectedPoints(arrLines)



# drawLine(img, 50, 0, (0,0,255))
# drawLine(img, 50, math.pi/2, (0,0,125))

# drawLine(img, 150, math.radians(45), (0,255,0))
# drawLine(img, 155, math.radians(50), (0,255,0))

# drawLine(img, 150, math.radians(85), (0,255,0))
# drawLine(img, 155, math.radians(90), (0,255,0))

# drawLine(img, 150, math.radians(125), (0,255,0))
# drawLine(img, 100, 0, (0,255,0))
# drawLine(img, 155, math.radians(145), (0,255,0))

# drawLine(img, -100, math.radians(150), (0,255,0))
# drawLine(img, -100, math.radians(165), (0,255,0))
# drawLine(img, 100, math.pi-math.radians(165), (0,255,0))

# drawLine(img, 100, 0, (0,255,0))
# drawLine(img, -110, math.pi, (0,255,0))

# drawLine(img, -110, math.pi-math.radians(10), (0,255,0))
# drawLine(img, 100, -math.radians(10), (0,255,0))

cv2.imwrite('hcanny.jpg',edges)
cv2.imwrite('hlines.jpg',img)