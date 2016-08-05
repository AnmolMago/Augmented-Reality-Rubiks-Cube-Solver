#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(){
    
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    
    while (true) {
        Mat frame, fblur, fgrey, flap, fcany;
        cap >> frame;
        
        resize(frame, frame, Size(), 0.5, 0.5, INTER_AREA);
        GaussianBlur(frame, fblur, Size(7,7), 0, 0);
        cvtColor(fblur, fgrey, CV_BGR2GRAY);
        
//        Laplacian(fgrey, flap, CV_16S, 3, 1, 0, BORDER_DEFAULT);
//        convertScaleAbs( flap, flap );
        
        int thold = 60;
        Canny(fgrey, fcany, thold, thold * 3, 3 );
        vector<Vec4i> lines;
        HoughLinesP(fcany, lines, 1, CV_PI/45, 40, 0, 50);
        for( size_t i = 0; i < lines.size(); i++ ){
            Vec4i l = lines[i];
            line(fblur, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        }
        cout << lines.size() << endl;
        imshow("AR RCS", fblur);
        imshow("AR RCS2", fcany);
        if(waitKey(10) >= 0) break;
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