#include "opencv2/opencv.hpp"
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

bool similarLength(Vec4i l1, Vec4i l2){
    double length1 = sqrt( pow(l1[2] - l1[0], 2) + pow(l1[3] - l1[1], 2));
    double length2 = sqrt( pow(l1[2] - l1[0], 2) + pow(l1[3] - l1[1], 2));
    return max(length1, length2)/min(length1, length2) < 1.3;
}

bool almostPerpendicular(Vec4i l1, Vec4i l2){
    double angle1 = atan2 (l1[3] - l1[1], l1[2] - l1[0]) * 180 / CV_PI;
    double angle2 = atan2 (l2[3] - l2[1], l2[2] - l2[0]) * 180 / CV_PI;
    return abs( max(angle1, angle2) - min(angle1, angle2) - 90) < 5;
}

bool almostTouching(Vec4i l1, Vec4i l2){
    return true;
}

int main(){
    
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    
    int linesWanted = 50, linesFound = 0, houghThold = 40;
    
    while (true) {
        Mat frame, fblur, fgrey, flap, fcany;
        cap >> frame;
        
        resize(frame, frame, Size(), 0.5, 0.5, INTER_AREA);
        GaussianBlur(frame, fblur, Size(5,5), 0, 0);
        cvtColor(fblur, fgrey, CV_BGR2GRAY);
        
        int thold = 50;
        Canny(fgrey, fcany, thold, thold * 3, 3 );
        vector<Vec4i> lines;
        if(linesFound > linesWanted){
            houghThold++;
        }else{
            houghThold = max(houghThold-1, 10);
        }
        HoughLinesP(fcany, lines, 1, 0.01, houghThold, 25, 25);
        for( size_t i = 0; i < lines.size(); i++ ){
            Vec4i l = lines[i];
            line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
        }
        vector<Vec4i> filtered;
        for( size_t i = 0; i < lines.size(); i++ ){
            Vec4i l1 = lines[i];
            for( size_t j = i+1; j < lines.size(); j++ ){
                Vec4i l2 = lines[i];
                if (similarLength(l1, l2) && almostPerpendicular(l1, l2) && almostTouching(l1, l2)){
                    if(find(filtered.begin(), filtered.end(), l1) == filtered.end())
                        filtered.push_back(l1);
                    if(find(filtered.begin(), filtered.end(), l2) == filtered.end())
                        filtered.push_back(l2);
                }
            }
        }
        for (size_t i = 0; i < filtered.size(); i++){
            Vec4i l = filtered[i];
            line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 1, CV_AA);
        }
        cout << "thold " << houghThold << " found " << linesFound << endl;
        linesFound = lines.size();
        imshow("AR RCS2", fcany);
        imshow("AR RCS", frame);
        if(waitKey(100) >= 0) break;
    }
    
}

/*
 Did not work:
    Using just black
    laplace is meh
 */

/*
 Check canny operators
 */