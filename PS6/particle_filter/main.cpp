#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>
#include<opencv2/highgui.hpp>



#include "particle_filter.hpp"
//#include "helper_functions.h"

using namespace std;
using namespace cv;


void display_image( Mat image)
{
    cv::imshow("window", image);
       waitKey();
    
}
// helper function 

int binary_search1 (vector<int> count, int num)
{
    int l ,r;
    int result = 0;
    l = 0; r = (int)count.size() -1;
    while( l <=r)
    {
        int mid =(l+r)/2;
        if( count[mid] == num)
        {
             result = 1;
            break;
        }
           
        else if( count[mid] < num)
            l = mid+1;
        else
            r = mid-1;
    }
    return result;
}


int main(int argc, char *argv[]) {

    // declaration of varaibles 
    string filename = "pres_debate";
    int count = 0;
    string infix = "a";
    string model_size = "small";
    int present = 0;
    cv::Mat gray;
    string question ="2";
    
    

    cv::VideoCapture cap( filename+ ".avi");
    vector<int> save_frames = { 15, 50,140};                // frames to save 
    

    // return of can't be opened 
    if (!cap.isOpened())
    {
        std::cout << "!!! Failed to open file: " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >>frame;
    if(frame.empty())
    {
        cout<<"Error in reading the frame"<<endl;
         return 0;
        
    }

    // extract the model 

    int left_x, left_y, width, height, x_centre, y_centre;
    left_x = 320; width = 104;
    left_y = 175; height = 129;
    x_centre = ( left_x + width/2.0);
    y_centre = (left_y+ height/2.0);
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
   
    cv::Mat Model = gray(cv::Range( left_y, left_y+ height+1),cv::Range(left_x,left_x+width+1) ); 

    
    // initializing the particle filter
    ParticleFilter tracker;
    tracker.init(Model, frame , 500, 0, 0.1);
    //tracker.display_particles_debug(frame);
    count = 1;
    
    
    while ( 1)
    {
        cap>>frame;
        if(frame.empty())
            break;
        
        count+=1;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        tracker.update(Model, gray);

        tracker.visualise( Model, frame);
        cv::imshow("window", frame);
        waitKey(200);

        // if frames to save then save
        present = binary_search1(save_frames, count);
        if (present == 1)
        {
            imwrite("./output/question3/ps6"+ question+ filename+"-"+to_string(count)+".png",frame);
            cout<<count<<endl;
            
        }
        cout<<count<<endl;
            
        
        
    }
    cap.release();
    destroyAllWindows();
    
    
    
    
    
    
    
    
        
        



   return 0;
    
}
