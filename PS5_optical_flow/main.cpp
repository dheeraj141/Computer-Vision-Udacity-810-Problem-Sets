
#include "lucas_kanade.hpp"
#include "expand_and_reduce.hpp"


void display_image( Mat &img, int i )
{
    Mat temp =  img.clone();
    temp.convertTo(temp, CV_8U);
    string description =to_string(i) + "Display";
    imshow( description, temp );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    }



void laplacian(  Mat &image , int level)
{
    image.convertTo(image, CV_64F);
    Mat downSampleL1 = down_sample(image); // 124x156
    Mat downSampleL2 = down_sample(downSampleL1);  // 62x 78
    Mat downSampleL3 = down_sample(downSampleL2); // 31 x 39
    Mat upSampleL2 = up_sample(downSampleL3); // 62x78
    Mat upSampleL1 = up_sample(downSampleL2); // 124x 156
    Mat upSampleImage = up_sample(downSampleL1);  // 248x312
    Mat laplacian1 = downSampleL2 - upSampleL2;
    Mat laplacian2 = downSampleL1 - upSampleL1;
    Mat laplacian3 = image - upSampleImage;
    
    
}


int main( int argc, char** argv )
{
   Mat image1, image2, image3;
    vector<Mat> disp;
    image1 = imread( "./images/TestSeq/Shift0.png", IMREAD_GRAYSCALE ); // Read the file
    image2 = imread( "./images/TestSeq/ShiftR2.png", IMREAD_GRAYSCALE ); // Read the file
    //image1 = imread("./square1.jpeg", IMREAD_GRAYSCALE );
    if( image1.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    
    
    disp = Lucas_kanade(image1, image2);
    Mat flow = flow_map(image1, disp[0], disp[1], 8);
    display_image(flow, 4);
    
   
        
    
    
    
    
    return 0;
    
}
