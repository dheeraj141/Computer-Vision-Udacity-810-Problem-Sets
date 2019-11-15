//
//  lucas_kanade.cpp
//  opencv
//
//  Created by Dheeraj Kumar Ramchandani on 15/11/19.
//  Copyright © 2019 Dheeraj Kumar Ramchandani. All rights reserved.
//

#include "lucas_kanade.hpp"

vector<Mat> Lucas_kanade(Mat &image1, Mat &image2)
{
    int i,j; int fs = 5;
    Mat grad_x, grad_y;
    vector<Mat> displacement;
    cv::Size s = image1.size();
    image1.convertTo(image1, CV_32FC1);
    image2.convertTo(image2, CV_32FC1);
    Mat It = image2- image1;
    
    
    Sobel(image1, grad_x, -1, 1, 0);
    Sobel(image1, grad_y, -1, 0, 1);
    Mat grad_x2 =  grad_x.mul(grad_x);
    Mat grad_y2 =  grad_y.mul(grad_y);
    Mat grad_xy =  grad_x.mul(grad_y);
    Mat IxIt = grad_x.mul(It);
    Mat IyIt = grad_y.mul(It);
    GaussianBlur(grad_x2, grad_x2, Size(fs,fs), 0);
    GaussianBlur(grad_y2, grad_y2, Size(fs,fs), 0);
    GaussianBlur(grad_xy, grad_xy, Size(fs,fs), 0);
    GaussianBlur(IxIt, IxIt, Size(fs,fs), 0);
    GaussianBlur(IyIt, IyIt, Size(fs,fs), 0);
    // make the matrix
    Mat x_disp(image1.size(), CV_32FC1, Scalar(0,0,255) );
    Mat y_disp(image1.size(), CV_32FC1, Scalar(0,0,255) );
    MatrixXf A(2,2); MatrixXf B(2,1);
    
    // now tracking the motion of only the feature points
   
    for (i = 0; i<s.height; i++)
    {
        for (j = 0; j<s.width; j++)
        {
            A<<grad_x2.at<float>(i,j), grad_xy.at<float>(i,j),grad_xy.at<float>(i,j),grad_y2.at<float>(i,j);
                      
            B<< -(IxIt.at<float>(i,j)), -(IyIt.at<float>(i,j));
            // solving the equation
            Vector2f x = A.colPivHouseholderQr().solve(B);
            // to eliminate nan values
            x_disp.at<float>(i,j) = (x[0] != x[0] ? 0: x[0]);
            y_disp.at<float>(i,j) = (x[1]!= x[1] ? 0: x[1]);
            
        }
    }
    displacement.push_back(x_disp);
    displacement.push_back(y_disp);
    return displacement;
    
    
    
}


Mat back_warp(Mat &image, Mat &x_disp, Mat &y_disp)
{
    int i, j;
    Mat warped_image;
    Mat map_x(x_disp.size(), CV_32FC1);
    Mat map_y(x_disp.size(), CV_32FC1);
    Size s = x_disp.size();
    for(i = 0; i <s.height; i++ )
    {
        for(j = 0; j<s.width; j++)
        {
            map_x.at<float>(i,j) = (float)(-6*x_disp.at<float>(i,j) + j);
            map_y.at<float>(i,j) = (float)(-6*y_disp.at<float>(i,j) + i);
            
        }
    }
    
    remap( image, warped_image, map_x, map_y, INTER_NEAREST );
    return warped_image;
}

Mat reduce( Mat &image , int level)
{
    Mat temp = image.clone();
    while(level >0)
    {
        temp = down_sample(temp);
        level--;
    }
    return temp;
}


Mat flow_map( Mat &image,  Mat &x_disp, Mat &y_disp, int gran)
{
     Mat flow = Mat::zeros(image.size(), CV_8UC1);
    Size s  =  flow.size();
    int y,x; float dx, dy;
    for (y = 0; y<s.height; y++)
    {
        for(x = 0; x<s.width; x++)
        {
            if ( x% gran == 0 && y%gran == 0)
            {
                dx = 20*x_disp.at<float>(y,x);
                dy = 20*y_disp.at<float>(y,x);
                if (dx >0 && dy >0)
                    arrowedLine(flow, Point(x,y), Point(x+dx, y+dy), 255, 1);
                
            }
        }
    }
    return flow;
}





void iterative_lk( Mat &image1, Mat &image2)
{
    int max_level = 6;
    Mat u,v;
    int current_level ;
    vector<Mat> disp;
    current_level = max_level;
    Mat left, right, wk;
    while(current_level >=0)
    {
        left =reduce(image1, current_level);
        right = reduce(image2, current_level);
        
        if (current_level == max_level )
        {
            u = Mat::zeros(left.size(), CV_32FC1);
            v = Mat::zeros(left.size(), CV_32FC1);
            
        }
        else
        {
            u= 2*up_sample(u);
            v = 2*up_sample(v);
        }
        if (left.size() != u.size())
        {
            resize(left, left, u.size());
            resize(right, right, u.size());
            
        }
       
        wk = back_warp(left, u, v);
        disp = Lucas_kanade(wk, right);
        u+=disp[0];
        v+=disp[1];
        current_level-=1;
        
        //flow_map(left, disp[0], disp[1], 8);
        
        
        
    }
    
    
    imwrite("./warped_ShiftR10.jpg", wk);
    imwrite("./ShiftR10.jpg", right);
    Mat diff = wk - right;
    imwrite("./diff_ShiftR10.jpg", diff);
    //display_image(diff, 5);
    Mat flow = flow_map(right, u, v, 8);
    imwrite("./flow_map_ShiftR10.jpg", flow);
    
    
    
    
}
