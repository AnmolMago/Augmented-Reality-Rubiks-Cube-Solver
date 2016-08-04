#include "opencv2/opencv.hpp"

using namespace cv;

int main(){
    
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    
    while (true) {
        Mat frame, fblur, fgrey, flap, d_edges;
        cap >> frame;
        
        resize(frame, frame, Size(), 0.5, 0.5, INTER_AREA);
        GaussianBlur(frame, fblur, Size(3,3), 0, 0);
        cvtColor(fblur, fgrey, CV_BGR2GRAY);
        
        Laplacian(fgrey, flap, CV_16S, 3, 1, 0, BORDER_DEFAULT);
        convertScaleAbs( flap, flap );
        
        imshow("AR RCS", flap);
        if(waitKey(10) >= 0) break;
    }
    
}

/*
 Did not work:
    Using just black
    Canny
 */