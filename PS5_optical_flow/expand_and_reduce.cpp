//
//  expand_and_reduce.cpp
//  opencv
//
//  Created by Dheeraj Kumar Ramchandani on 15/11/19.
//  Copyright © 2019 Dheeraj Kumar Ramchandani. All rights reserved.
//

#include "expand_and_reduce.hpp"

Mat down_sample( Mat &image)
{
    double a = 0.4;
    Mat w1 = (Mat_<double>(5,1) << (1.0/4 -a/2), (1.0/4), a, 1.0/4, 1.0/4-a/2);
    // transpose the matrix
    Mat w2  = w1.t();
    Mat w = w1*w2;
    cv::Size s = image.size();
           
    Mat output  = image.clone();
    image.convertTo(image, CV_64F);
    filter2D(image, output, -1, w);
           
    Mat M(s.height/2,s.width/2, CV_64F, Scalar(0,0,255));
    int i,j;
    for( i = 0; i < s.height/2; ++i)
    {
        for ( j = 0; j < s.width/2; ++j)
               M.at<double>(i,j) =output.at<double>(2*i, 2*j);
    }
    
    return M;
    
}

// function for upsampling the images

Mat up_sample(Mat & image)
{
    double a = 0.4;
    Mat w1 = (Mat_<float>(5,1) << (1.0/4 -a/2), (1.0/4), a, 1.0/4, 1.0/4-a/2);
    // transpose the matrix
    Mat w2  = w1.t();
    Mat w = w1*w2;
    cv::Size s = image.size();
    cout<<s.height<<" "<<s.width<<endl;
    image.convertTo(image, CV_32FC1);
    Mat M(2*s.height, 2*s.width, CV_32FC1, Scalar(0,0,255));
    int i,j, m,n;
    double sum;  int r,c;
    for( i = 0; i <  2*s.height; ++i)
    {
        for(j = 0; j<2*s.width; j++)
        {
            sum = 0;
            for(m = -2; m <=2; m++)
            {
                r = (i-m); if (r%2!= 0) continue;
                
                for (n = -2; n <=2; n++)
                {
                    double temp1,temp2;
                    c = (j - n); if(c%2!=0)continue;
                    if(r >=0 && r< 2*s.height && c>=0 && c< 2*s.width)
                    {
                        temp1 = image.at<float>(r/2,c/2);
                        temp2 = w.at<float>(m+2, n+2);
                        
                        sum+= (temp1*temp2);
                        
                    }
                    
                }
            }
            M.at<float>(i,j) = 4*sum;
        }
    }
    return M;
    
}

