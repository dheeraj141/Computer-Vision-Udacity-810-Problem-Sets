//
//  expand_and_reduce.hpp
//  opencv
//
//  Created by Dheeraj Kumar Ramchandani on 15/11/19.
//  Copyright © 2019 Dheeraj Kumar Ramchandani. All rights reserved.
//

#ifndef expand_and_reduce_hpp
#define expand_and_reduce_hpp

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
using namespace Eigen;



// function declaration


// function for downsampling the image
Mat down_sample( Mat &image);

Mat up_sample(Mat &image );

#endif /* expand_and_reduce_hpp */
