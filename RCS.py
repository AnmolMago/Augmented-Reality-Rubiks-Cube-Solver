import cv2
import numpy as np
import math


height = -1
width = -1

def standardizeLines(v):
    rho = v[0][0]
    theta = v[0][1]
    if rho < 0:
        rho *= -1
        theta -= math.pi
    return [rho, theta, -1]

def distanceBetween(line1, line2):
    extreme_x = (0,1,2,width,width-1,width-2)
    extreme_y = (0,1,2,height,height-1,height-2)
    y1l, y1r, x1b, x1t = extreme_xy_values(line1)
    y2l, y2r, x2b, x2t = extreme_xy_values(line2)
    s_thres = 30
    if x1b in extreme_x and x2b in extreme_x and x1t in extreme_x and x2t in extreme_x:
        return ( abs(x2b-x1b) + abs(x2t-x1t) )/2
    elif y1l in extreme_y and y2l in extreme_y and y1r in extreme_y and y2r in extreme_y:
        return ( abs(y2l-y1l) + abs(y2r-y1r) )/2
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
    # x_bottom = y_left/float(slope)
    x_bottom = (y_left-y0)/slope + x0
    x_top = (y_right-y0)/slope + x0
    x_bottom = min(max(x_bottom, 0), width)
    x_top = min(max(x_top, 0), width)
    # print str(y_left) + "|" + str(y_right) + "|" + str(x_bottom) + "|" + str(x_top) + "|" + str(a) + "|" + str(b) + "|" + str(slope) + "|"
    return int(y_left), int(y_right), int(x_bottom), int(x_top)

def are_lines_similar(line1, line2):
    # TODO BUG: lines in from of x will be classified as similar, should not!!!
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

def drawLine(img, rho, theta, color):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    # yl, yr, xb, xt = extreme_xy_values((rho,theta))
    # cv2.line(img,(0,int(yl)),(width,int(yl)),(0,0,0),2)
    # cv2.line(img,(0,int(yr)),(width,int(yr)),(0,0,0),2)
    # cv2.line(img,(int(xb),0),(int(xb),height),(255,255,255),2)
    # cv2.line(img,(int(xt),0),(int(xt),height),(255,255,255),2)
    cv2.line(img,(x1,y1),(x2,y2),color,2)

def main():
    global height, width
    cap = cv2.VideoCapture(0)
    thres = 40
    isBlacklist = True
    blacklist = []
    count = 0
    eth = 55
    while True:
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

            # print str(thres) + "|" + str(len(lines))

            for i in range(0, len(lines)):

                if isBlacklist:
                    blacklist.append((lines[i][0], lines[i][1]))
                    continue

                if (lines[i][0], lines[i][1]) in blacklist:
                    continue

                # drawLine(img, lines[i][0], lines[i][1], (255,0,0))
                foundGroup = False

                for j in range(0, i):
                    if (lines[j][0], lines[j][1]) in blacklist:
                        continue
                    if i != 0 and are_lines_similar(lines[i], lines[j]):
                        # print "Lines: " + str(extreme_xy_values(lines[i])) + " and " + str(extreme_xy_values(lines[j])) + " are similar"
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
                # drawLine(img, rho, theta, (0,255,0))
                avgLines.append([rho, theta])

            parallelPairs = {}
            perpendicularFoursome = []
            
            for i in range(0, len(avgLines)):
                for j in range(0, i):
                    if i != j and abs(avgLines[i][1] - avgLines[j][1]) <= 10:
                        avg = (avgLines[i][1] + avgLines[j][1])/2
                        parallelPairs[avg] = [avgLines[i], avgLines[j]]

            for avg_i in parallelPairs:
                for avg_j in parallelPairs:
                    angle_diff = abs(avg_i - avg_j)
                    dist_diff_i = distanceBetween(parallelPairs[avg_i][0], parallelPairs[avg_i][1])
                    dist_diff_j = distanceBetween(parallelPairs[avg_j][0], parallelPairs[avg_j][1])
                    dist_diff = abs(dist_diff_i - dist_diff_j)
                    if angle_diff >= math.radians(80) and angle_diff <= math.radians(100) and dist_diff <= 10:
                        arrLines = parallelPairs[avg_i] + parallelPairs[avg_j]
                        perpendicularFoursome.append(arrLines)
                        for line in arrLines:
                            drawLine(img, line[0], line[1], (0,0,255))

        elif lines is not None and len(lines) > 100:
            thres += 5
        elif thres > 2:
            thres -= 2
        else:
            print "Failed to detect anything....!"

        cv2.imshow('guess',edges)
        cv2.imshow('frame',img)
        count += 1
        if count >= 100 and isBlacklist:
            isBlacklist = False
            eth += 10
            print "Stopped recording blacklist!!"
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