#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <omp.h>

#include "image_matrix.hpp"


void median_filter_images( const std::vector< image_matrix >& input_images_,
			   std::vector< image_matrix >& output_images_,
			   const int window_size_,
			   const int n_threads_,
			   const int mode_ ){
	int half_wind_size = window_size_/2;
  	if(mode_==0 || mode_==3){
  		//serial
  		double start = omp_get_wtime();
  		for(int i=0;i<input_images_.size();i++){
  			image_matrix image = input_images_[i];
  			int width = image.get_n_cols();
  			int height = image.get_n_rows();
  			
  			for(int x = 0; x < width; x++){
  				for(int y = 0; y < height; y++){
  					
  					std::vector<float> array;
  					int num_pixels=0;
  					for(int xx = x-half_wind_size; xx <= x+half_wind_size; xx++){
  						for(int yy = y - half_wind_size; yy <= y+half_wind_size; yy++){
  							if(xx<0 || xx>=width || yy<0 || yy>=height)
  								continue;
							array.push_back(image.get_pixel(yy,xx));
							num_pixels++;
  						}
  					}
  					std::sort(array.begin(), array.end());
  					
  					if(num_pixels%2==0){
  						int temp=(num_pixels-1)/2;
  						float tempp = array[temp] + array[temp+1];
  						tempp/=2;
  						output_images_[i].set_pixel(y,x,tempp);
  					}
  					output_images_[i].set_pixel(y,x,array[num_pixels/2]);
  				}
  			}
  		}
  		double end = omp_get_wtime();
  		if(mode_==3)
  			std::cout<<"Serial:  "<<end-start<<std::endl;
  	}

  	if(mode_==1 || mode_==3){
  		//parallel image level
  		omp_set_num_threads(n_threads_);
  		int i;
  		double start=omp_get_wtime();
  		#pragma omp parallel for schedule(static,1) shared(input_images_) shared(output_images_)  private(i)
  		for(i=0;i<input_images_.size();i++){
  			image_matrix image = input_images_[i];
  			int width = image.get_n_cols();
  			int height = image.get_n_rows();
  			
  			for(int x = 0; x < width; x++){
  				for(int y = 0; y < height; y++){
  					std::vector<float> array;
  					int num_pixels=0;
  					for(int xx = x-half_wind_size; xx <= x+half_wind_size; xx++){
  						for(int yy = y - half_wind_size; yy <= y+half_wind_size; yy++){
  							if(xx<0 || xx>=width || yy<0 || yy>=height)
  								continue;
							array.push_back(image.get_pixel(yy,xx));
							num_pixels++;
  						}
  					}
  					std::sort(array.begin(), array.end());
  					
  					if(num_pixels%2==0){
  						int temp=(num_pixels-1)/2;
  						float tempp = array[temp] + array[temp+1];
  						tempp/=2;
  						output_images_[i].set_pixel(y,x,tempp);
  					}
  					output_images_[i].set_pixel(y,x,array[num_pixels/2]);
  				}
  			}
  		}
  		double end = omp_get_wtime();
  		if(mode_==3)
  			std::cout<<"Image level parallelism: "<<end-start<<std::endl;
  	}
  	if(mode_==2 || mode_==3){
  		//parallel pixel level
  		double start = omp_get_wtime();
  		for(int i=0;i<input_images_.size();i++){
  			image_matrix image = input_images_[i];
  			int width = image.get_n_cols();
  			int height = image.get_n_rows();
  			int x;  	
  			omp_set_num_threads(n_threads_);		
  			#pragma omp parallel for schedule(static, n_threads_) shared(i) shared(output_images_) private(x)
  			for(x = 0; x < width; x++){
  				for(int y = 0; y < height; y++){
  					std::vector<float> array;
  					int num_pixels=0;
  					for(int xx = x-half_wind_size; xx <= x+half_wind_size; xx++){
  						for(int yy = y - half_wind_size; yy <= y+half_wind_size; yy++){
  							if(xx<0 || xx>=width || yy<0 || yy>=height)
  								continue;
							array.push_back(image.get_pixel(yy,xx));
							num_pixels++;
  						}
  					}
  					std::sort(array.begin(), array.end());
  					
  					if(num_pixels%2==0){
  						int temp=(num_pixels-1)/2;
  						float tempp = array[temp] + array[temp+1];
  						tempp/=2;
  						output_images_[i].set_pixel(y,x,tempp);
  					}
  					output_images_[i].set_pixel(y,x,array[num_pixels/2]);
  				}
  			}
  		}
  		double end=omp_get_wtime();
  		if(mode_==3)
  			std::cout<<"Pixel level parallelism " <<end-start<<std::endl;
  	}
}




bool read_input_image(const std::string& filename_, image_matrix& image_in_){
  bool ret = false;
  std::ifstream is(filename_.c_str());
  if( is.is_open()){
    int n_rows;
    int n_cols;

    is >> n_rows;
    is >> n_cols;

    image_in_.resize(n_rows, n_cols);

    for(int r=0; r<n_rows;r++){
      for(int c = 0; c < n_cols; c++){
        float value;
        is >> value;
        image_in_.set_pixel(r, c, value);
      }
    }
    is.close();
    ret = true;
  }
  return ret;
}



bool write_filtered_image(const std::string& filename_, const image_matrix& image_out_){
  bool ret = false;
  std::ofstream os(filename_.c_str());
  if(os.is_open()){
    int n_rows = image_out_.get_n_rows();
    int n_cols = image_out_.get_n_cols();

    os << n_rows << std::endl;
    os << n_cols << std::endl;

    for(int r = 0; r < n_rows; r++){
      for( int c = 0; c < n_cols; c++ ){
        os << image_out_.get_pixel(r, c) << " ";
      }
      os << std::endl;
    }
    os.close();
    ret = true;
  }
  return ret;
}



int main(int argc, char* argv[]){
  if( argc < 5 ){
    std::cerr << "Not enough arguments provided to " << argv[ 0 ] << ". Terminating." << std::endl;
    return 1;
  }

  // get input arguments
  int window_size = atoi( argv[ 1 ] );
  int n_threads = atoi( argv[ 2 ] );
  int mode = atoi( argv[ 3 ] );

  int input_images_count = argc - 4;
  std::vector< std::string > filenames;
  for( std::size_t f = 0; f < input_images_count; f++ ){
    filenames.push_back( argv[ 4 + f ] );
  }

  // input and filtered image matrices
  std::vector< image_matrix > input_images;
  std::vector< image_matrix > filtered_images;
  input_images.resize( input_images_count );
  filtered_images.resize( input_images_count );

  // read input matrices
  for(int i = 0; i < input_images_count; i++){
    read_input_image( filenames[ i ], input_images[ i ] );

    // resize output matrix
    int n_rows = input_images[ i ].get_n_rows();
    int n_cols = input_images[ i ].get_n_cols();

    filtered_images[ i ].resize( n_rows, n_cols );
  }

  // ***   start actual filtering   ***
  
  if(mode < 3){             // serial, parallel at image level or parallel at pixel level  
    // invoke appropriate filtering routine based on selected mode
    median_filter_images(input_images,filtered_images,window_size, n_threads,mode);
    
    // write filtered matrices to text files
    for(int i=0;i<filtered_images.size();i++){
    	write_filtered_image("OUT_"+filenames[i],filtered_images[i]);
    }
  }
  else if(mode == 3){       // benchmark mode
    double start;
    double time[ 3 ];

	median_filter_images(input_images, filtered_images, window_size, n_threads, mode);
  }
  else{
    std::cerr << "Invalid mode. Terminating" << std::endl;
    return 1;
  }

  return 0;
}
