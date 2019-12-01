
#ifndef particle_filter_hpp
#define particle_filter_hpp

#include <iostream>
#include <vector>
#include<opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


struct Particle {

    int id;
    double x;
    double y;
    double weight;
};



class ParticleFilter {
    
    int num_particles;
    int mw,mh,sw,sh;                // model width , height and sw and sh are the search space ( image/frame  ) width and height
    int debug_flag;
    cv::Mat obj_to_track;
    double alpha;                   // for updating model
    std::vector<double> state;      
    
    
    // Flag, if filter is initialized
    bool is_initialized;
    
    // Vector of weights of all particles
    std::vector<double> weights;
    
public:
    
    // Set of current particles
    std::vector<Particle> particles;                // number of particles 

    // Constructor
    // @param M Number of particles
    ParticleFilter() : num_particles(0), is_initialized(false) {}

    // Destructor
    ~ParticleFilter() {}

    void init(cv::Mat Model, cv::Mat Image, int number, int flag, double );
   

   
    void prediction(double, double);
    // displaying the particles in the current image
    
    void observation( cv::Mat Model,cv::Mat image);
    
    void resample(int , int );
    void update(cv::Mat Model, cv::Mat image);
    void visualise( cv::Mat Model, cv::Mat frame);
    void draw_window(cv::Mat Model,cv::Mat frame);
    void display_particles( cv::Mat image);
    double Mean_square_error( cv::Mat Model, cv::Mat image, int x_particle, int y_particle );
    void estimate_state();
    void update_model(cv::Mat frame);
    
    
  
    
   
    const bool initialized() const {
        return is_initialized;
        
    }
};

#endif /* particle_filter_hpp */
