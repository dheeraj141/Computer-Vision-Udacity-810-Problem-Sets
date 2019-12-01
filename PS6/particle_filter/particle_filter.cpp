
#include "particle_filter.hpp"


#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace std;


// helper functions
void display_vector( vector<int> vec)
{
    for( int i = 0; i<vec.size(); i++)
        cout<<vec[i]<<" ";
    cout<<endl;
}

void ParticleFilter::init(cv::Mat Model, cv::Mat Image, int number, int flag, double upd_m) {
    
    // wieghts and particles both are vectors so resizing to the number of particles choosen
    
    num_particles = number;                     // number of particles
    mw = Model.size().width;                    // model width and height ( object to track)
    mh =Model.size().height;
    sw  = Image.size().width;                   // search space ( which is image )
    sh = Image.size().height;
    debug_flag = flag;                          // debug  1 for normal deubbing and 2 for advanced  for normal use set it to 0
    alpha = upd_m;                              // for model updation   
    obj_to_track =Model;
    
   
    weights.resize(num_particles);
    particles.resize(num_particles);
    state.resize(2);

   // uniform distribution for the particles
    //std::uniform_real_distribution<double> dist_x(0.0,sw);    // for face uniform distribution is fine
    //std::uniform_real_distribution<double> dist_y(0.0,sh);
   normal_distribution<double> dist_x(571, 10);                 // for hand tracking normal distribution was used for initializing the particles
   normal_distribution<double> dist_y(439, 10);
   default_random_engine gen;

    // create particles and set their values
    for(int i=0; i<num_particles; ++i){
        Particle p;
        p.id = i;
        p.x = dist_x(gen); 
        p.y = dist_y(gen);
        p.weight = 1;

        particles[i] = p;
        weights[i] = p.weight;
    }
    is_initialized = true;
    estimate_state() ;
}

void ParticleFilter::prediction(double std_x, double std_y)
{
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
     default_random_engine gen;

     // updating the position of each particle using the normal distribution

    for(int i = 0; i<num_particles; i++)
    {
        particles[i].x +=dist_x( gen);
         particles[i].y +=dist_y( gen);
    }
    
    
}
// function to calculate MSE( Mean square error) between two images
// subtract the two images and sum the value.

double calculate_MSE ( cv::Mat image1, cv::Mat image2,int debug_flag)
{
    
    
    
    int mw, mh;
    mh = std::min( image1.size().height, image2.size().height);
    mw = std::min(image1.size().width, image2.size().width);
    image1.convertTo(image1, CV_32FC1);
    image2.convertTo(image2, CV_32FC1);
    
    
    
    
    double MSE = 0;
    for( int i = 0; i<mh;  i++)
    {
        for(int j = 0; j<mw; j++)
        {
            if( debug_flag == 2)
            {
                cout<<"individual pixel values"<<endl;
                cout<< image1.at<char>(i,j)<<" "<<image2.at<char>(i,j)<<endl;
                double x = image1.at<char>(i,j) - image2.at<char>(i,j);
                cout<<x<<endl;
            }
            if( debug_flag == 2)
            {
                cout<<image1.at<float>(i,j)<<" "<<image2.at<float>(i,j)<<endl;
            }
            MSE+= pow((image1.at<float>(i,j) - image2.at<float>(i,j)),2 );
        }
    }
    MSE/= double( mw*mh);
    return MSE;
}


double ParticleFilter::Mean_square_error( cv::Mat Model, cv::Mat image, int x_particle, int y_particle )
{
    cv::Mat temp = image.clone();
   // declaration of variables
    
    double sigma_MSE = 5.0;
    
    // extract the window for each particle 
    int minx,miny,max_x,max_y;
    minx = (x_particle - mw/2 ); miny =(y_particle - mh/2 );
    minx = (minx >=0)?minx:0;
    miny = (miny >=0)?miny:0;
    max_x = minx+ mw ;
    max_y = miny + mh;
    max_x = ( max_x < sw)?max_x: sw;
    max_y = ( max_y < sh) ? max_y : sh;
    minx  = (minx <=max_x) ?minx :max_x;
    miny = (miny <=max_y) ?miny :max_y;

    cv::Mat particle_window = image( cv::Range(miny, max_y ), cv::Range( minx, max_x));
    
    
    if( debug_flag  )
    {
        cv::rectangle(temp, cv::Point( minx,miny), cv::Point( max_x, max_y), cv::Scalar(0,255,0));
        cv::circle( temp, cv::Point(x_particle, y_particle),3,cv::Scalar(0,0,255), 2);
        cv::imshow("window", temp);
        cv::waitKey();
        
    
    }
    
    
    
    
    double MSE = 0;
    MSE+= calculate_MSE(particle_window, Model, debug_flag);
    
    
    if( debug_flag==2 )
    {
        cv::imshow("window", particle_window);
        cv::waitKey();
        cv::imshow("window", Model);
        cv::waitKey();
    }
  
        
    //}
    
    //MSE/=3;                       // this can be uncommented if the particle filter is not converging.
  
    MSE = MSE/( 2*sigma_MSE);
    MSE/=(25);
    if(debug_flag)
        cout<<MSE<<endl;
    return exp ( - MSE );
    
}

// sensor model or the observation function updating the weights of the particle

void ParticleFilter::observation( cv::Mat Model, cv::Mat image)
{
    // calculating the observation for each particle using the MSE
    double sum_weights = 0;
    double min_weight = pow(10, -5);
    double wt;
    for(int i = 0; i<num_particles; i++)
    {
       
        wt = Mean_square_error(Model, image, particles[i].x, particles[i].y);

        if ( wt == 0)
            wt = min_weight;
        sum_weights+=wt;
        particles[i].weight = wt;
        
    }
    // normalizing the weights

    for(int i = 0; i<num_particles; i++)
    {
        Particle *p = &particles[i];
        p->weight /= sum_weights;
        weights[i] = p->weight;
    }
    
    
    
    
}

// resampling the  particles based on their weights 
void ParticleFilter::resample( int image_width, int image_height )
{
    
    

    default_random_engine gen;
    
    discrete_distribution<int> distribution(weights.begin(), weights.end());
   
    if( debug_flag)
    {
        vector<int> new_weights( num_particles, 0);
        for( int i = 0; i<num_particles; i++)
            ++new_weights[distribution( gen)];
        
        display_vector(new_weights);
        
    }
    
    

    vector<Particle> resampled_particles;

    for (int i = 0; i < num_particles; i++){
        int index = distribution( gen);
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;

    // clipping if the particles position lies outside the image

    for(int i = 0; i<num_particles; i++)
    {
        if ( particles[i].x <0) particles[i].x= 0;
        else if(particles[i].x > image_width  ) particles[i].x = image_width-1;
        if (particles[i].y <0) particles[i].y= 0;
        else if(particles[i].y >image_height) particles[i].y = image_height-1;
    }
    
}
void ParticleFilter::update(cv::Mat Model,cv::Mat frame)
{

    double deviation = 10;
    prediction(deviation, deviation);
    observation(Model, frame);
    resample(frame.size().width, frame.size().height);
    estimate_state();
    if( alpha >0)
    {
        update_model(frame);
    }
    
    
}


void ParticleFilter::update_model(cv::Mat frame )
{
    int minx,miny, max_x, max_y;
    minx = int( state[0] - mw/2);
    miny = int( state[1] - mh/2);
    minx = (minx >=0)?minx:0;
    miny = (miny >=0)?miny:0;
    max_x = minx+ mw ;
    max_y = miny + mh;
    max_x = ( max_x < sw)?max_x: sw;
    max_y = ( max_y < sh) ? max_y : sh;

    // updated model only if they are of same size 
    cv::Mat best_model = frame( cv::Range( miny, max_y), cv::Range( minx, max_x));
    if ( best_model.size() == obj_to_track.size())
    {
        best_model.convertTo(best_model, CV_32FC1);
        obj_to_track.convertTo(obj_to_track, CV_32FC1);
        obj_to_track = alpha * best_model + ( 1- alpha)* obj_to_track;
    }
    
    
    
}

void ParticleFilter::visualise(cv::Mat Model, cv::Mat frame)
{
    display_particles(frame);
    draw_window(Model, frame);
    
}

//  helper function for debugging 
void ParticleFilter::display_particles (cv::Mat image)
{
    cv::Mat temp = image.clone();
    
    for(int i=0; i<num_particles; ++i)
    {
        cv::circle( image, cv::Point(particles[i].x, particles[i].y),1,cv::Scalar(0,0,255), 1);
        cv::circle( temp, cv::Point(particles[i].x, particles[i].y),1,cv::Scalar(0,0,255), 1);
        
    }
    
}

void ParticleFilter::estimate_state()
{
    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    int id = distribution( gen);
    state[0] = (particles[id].x);
    state[1] = ( particles[id].y);
    
    
    
}


void ParticleFilter::draw_window(cv::Mat Model,cv::Mat frame)
{
    
    cv::Point p1,p2;
    int best_idx;
    cv::Size s = Model.size();
    best_idx = (int)distance(weights.begin(),max_element(weights.begin(), weights.end()));
    p1 = cv::Point(particles[best_idx].x, particles[best_idx].y);
    p1.x-= (s.width/2);
    p1.y-=(s.height/2);
    p2  = cv::Point( p1.x+ s.width, p1.y+s.height );
    cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0));
}
