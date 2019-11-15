//
//  lucas_kanade.hpp
//  opencv
//
//  Created by Dheeraj Kumar Ramchandani on 15/11/19.
//  Copyright © 2019 Dheeraj Kumar Ramchandani. All rights reserved.
//

#ifndef lucas_kanade_hpp
#define lucas_kanade_hpp

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include "expand_and_reduce.hpp"
using namespace cv;
using namespace std;
using namespace Eigen;

// declaration of functions
// Single level lucas kanade
vector<Mat> Lucas_kanade(Mat &image1, Mat &image2);


// function to back_warp the image
Mat back_warp(Mat &image, Mat &x_disp, Mat &y_disp);

// iterative lucas kanade for greater motion

void iterative_lk( Mat & image1, Mat &image2);

// function for calculatinf the flow map using the displacement and granularity;
Mat flow_map( Mat &image,  Mat &x_disp, Mat &y_disp, int granularity );

#endif /* lucas_kanade_hpp */
