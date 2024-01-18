#include <cmath>

#include "image.h"
#include "constants.h"
#include <opencv4/opencv2/opencv.hpp>

std::string SRC_FOLDER = "res/";
std::string TRG_FOLDER = "out/";

double get_luminance(int r, int g, int b) {
	// NTSC formula for pixel intensity;
	double ans = 0.299 * r + 0.587 * g + 0.114 * b;
	if (ans < 0) ans = 0;
	if (ans > 255) ans = 255;
	return ans;
}

uint8_t trunc_pixel(double pixel_value) {
	if (pixel_value > 254) pixel_value = 255;
	if (pixel_value < 0) pixel_value = 0;

	return (uint8_t)(pixel_value + 0.45);
}

// Euclidean distance
double dist(int x1, int y1, int x2, int y2){
	return sqrt((x1 - x2) * (x1 - x2)  +  (y1 - y2) * (y1 - y2));
}
double gaussian(double x, double sigma){
	double exponent = - x * x /(2 * sigma * sigma);
	double denom = 2 * CV_PI * sigma * sigma;
    return exp(exponent) / denom;	
}

cv::Mat subtract(cv::Mat m1, cv::Mat m2, int to_add){
    cv::Mat m = m2.clone();
    int w = m.cols;
    int h = m.rows;
    for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
            cv::Vec3b& res = m.at<cv::Vec3b>(r,c);
            cv::Vec3b& p1 = m1.at<cv::Vec3b>(r,c);
            cv::Vec3b& p2 = m2.at<cv::Vec3b>(r,c);
            
            res[0] = trunc_pixel(0.0 + to_add + p1[0] - p2[0]);
            res[1] = trunc_pixel(0.0 + to_add + p1[1] - p2[1]);
            res[2] = trunc_pixel(0.0 + to_add + p1[2] - p2[2]);    
        }
    }
    return m;
}

cv::Mat subtract_mod(cv::Mat m1, cv::Mat m2){
    cv::Mat m = m2.clone();
    int w = m.cols;
    int h = m.rows;
    for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
            cv::Vec3b& res = m.at<cv::Vec3b>(r,c);
            cv::Vec3b& p1 = m1.at<cv::Vec3b>(r,c);
            cv::Vec3b& p2 = m2.at<cv::Vec3b>(r,c);
            
            res[0] = trunc_pixel(std::abs(0.0 + p1[0] - p2[0]));
            res[1] = trunc_pixel(std::abs(0.0 + p1[1] - p2[1]));
            res[2] = trunc_pixel(std::abs(0.0 + p1[2] - p2[2]));    
        }
    }
    return m;
}

cv::Mat make_high_pass(cv::Mat image, int d, double sigma){
	cv::Mat blurred;
	// First, we compute a high-pass version H of the image using the same cutoff sigma-s
	cv::GaussianBlur(image, blurred, cv::Size(0,0), sigma, sigma);
	return subtract_mod(image,blurred);
}

// Odd diameter
double make_pixel_textureness(cv::Mat &image, cv::Mat &H, int x, int y, int d, int sigma_color, int sigma_space){
	double top_sum = 0;
	double bot_sum = 0;

	for(int a = 0; a < d; ++a){
		for(int b = 0; b < d; ++b){
			int x_neigh = x + a - d/2;
			int y_neigh = y + b - d/2;
			if(x_neigh < 0) x_neigh = 0;
			if(y_neigh < 0) y_neigh = 0;
			if(x_neigh >= image.cols) x_neigh -= x_neigh - image.cols + 1; 
			if(y_neigh >= image.rows) y_neigh -= y_neigh - image.rows + 1; 
			
			cv::Vec3b & h_neigh = H.at<cv::Vec3b>(y_neigh, x_neigh);
			cv::Vec3b & i_neigh = image.at<cv::Vec3b>(y_neigh, x_neigh);
			cv::Vec3b & i_here  = image.at<cv::Vec3b>(y, x);
			
			double p_color = gaussian( std::abs(i_neigh[0] - i_here[0]) ,sigma_color);
			double p_space = gaussian( dist(x,y, x_neigh, y_neigh) ,sigma_space);
			
			top_sum += p_color * p_space * h_neigh[0];
			bot_sum += p_color * p_space;
		}
	}
	
	double result = top_sum/bot_sum;
	// if( (x + y) % 231 == 0)
	// 	std::cout << "\t\t Pixel " << x <<" " << y << " :   " << result << "\n";
	return result;
}

cv::Mat make_textureness(cv::Mat image, int d, int sigma_color, int sigma_space){
	cv::Mat H = make_high_pass(image, d, 36.5);
	cv::Mat result = cv::Mat( image.rows, image.cols, CV_64F, 0.0);
	
	for (int r = 0; r < image.rows; ++r) {
		for (int c = 0; c < image.cols; ++c) {
			double t = make_pixel_textureness(image, H, c, r, d, sigma_color, sigma_space);
		
			double& p = result.at<double>(r,c);
			p = t;
		}
	}
	
	return result;
}

cv::Mat make_bilateral_filter(cv::Mat mat, double sigma_color, double sigma_space, int d){
    if( sigma_space == -1) { sigma_space = std::min(mat.rows, mat.cols) / 16.0; }
    std::cout << "Bilateral: C[" << sigma_color << "], S[" << sigma_space << "], D[" << d << "]\n";
    cv::Mat res;  cv::bilateralFilter(mat, res, d, sigma_color, sigma_space);
    return res;
}


void save_image(cv::Mat m , std::string filename){
    std::cout << "Saving matrix as <" << TRG_FOLDER + filename << "> with type " << m.type() << std::endl;
    cv::imwrite(TRG_FOLDER + filename, m);	
}

cv::Mat make_combine(cv::Mat background, cv::Mat detail, cv::Mat rho, int to_add){
	cv::Mat result = background.clone();
	
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {

			cv::Vec3b& p_res = result.at<cv::Vec3b>(r,c);	
			double p_backg = background.at<cv::Vec3b>(r,c)[0];	
			double p_detail = detail.at<cv::Vec3b>(r,c)[0];	
			double p_rho = rho.at<double>(r,c);	
			
			if( p_rho < 1 ) p_rho = 1; // Would cause halo effects
			p_res[0] = trunc_pixel( p_backg + p_rho * (p_detail + 0.0 + to_add));
			p_res[1] = p_res[2] = p_res[0];

			// if( (r + c) % 231 == 0)
			// 	std::cout << "\t\t Pixel " << r <<" " << c << " :   " <<  (int)p_backg[0]  << " + " <<  p_rho << " * " << (int) p_detail[0]  << "\n";
		}
	}
	
	return result;
}

cv::Mat make_rho(cv::Mat target_textureness, cv::Mat scaled_bkg_textureness, cv::Mat in_textureness){
	cv::Size sz = {target_textureness.cols, target_textureness.rows};
	cv::Mat result(sz, CV_64F, cv::Scalar(0, 0, 0));
		
	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			double & p = result.at<double>(r,c); 
			double & p_targ = target_textureness.at<double>(r,c); 
			double & p_bkg = scaled_bkg_textureness.at<double>(r,c); 
			double & p_det = in_textureness.at<double>(r,c); 


			p = (p_targ - p_bkg) / p_det;
			if( isnan(p) || isinf(p) ) p = 1;

			if( (r + c) % 231 == 0)
				std::cout << "\tRho sample " << r << " " << c << " :  " << p_targ << " - " << p_bkg << " / " << p_det << " = " << p << "\n";
		}
	}
	return result; 
}


Histogram::Histogram(cv::Mat& matrix) : m_lum_hist(256, 0) {
	m_n_pixels = matrix.rows * matrix.cols;

	
	for (int r = 0; r < matrix.rows; ++r) {
		for (int c = 0; c < matrix.cols; ++c) {
			cv::Vec3b& p = matrix.at<cv::Vec3b>(r, c);

			int lum = 0.5 + get_luminance(p[0], p[1], p[2]);
			m_lum_hist[lum] += 1;
		}
	}
	
}

Histogram::Histogram() : m_lum_hist(256, 0), m_n_pixels(0) {

}

int Histogram::GetNPixels() {
	return m_n_pixels;
}

int Histogram::GetMaxValue() {
	int max_val = 0;
	for (int i = 0; i < 255; ++i) {
		max_val = max_val > m_lum_hist[i] ? max_val : m_lum_hist[i];
	}
	return max_val;
}

std::vector<int> Histogram::GetLumHist() {
	return m_lum_hist;
}

std::vector<int> Histogram::GetCumulative() {
	std::vector<int> cum_hist = std::vector<int>(m_lum_hist);

	for (int i = 1; i < 256; ++i) {
			cum_hist[i] += cum_hist[i - 1];
	}
	return cum_hist;
}
std::vector<double> Histogram::GetRelCumulative() {
	std::vector<double> cum_hist = std::vector<double>(256, 0);

	cum_hist[0] = m_lum_hist[0] / (double)GetNPixels();
	for (int i = 1; i < 256; ++i) {
		cum_hist[i] += m_lum_hist[i] / (double)GetNPixels();
		cum_hist[i] += cum_hist[i - 1];
	}
	return cum_hist;
}


cv::Vec3b change_luminance(cv::Vec3b p, double new_luminance) {
	cv::Vec3b new_p;
	// TODO: convert to LAB, change L value, and change back;
	double lum = get_luminance(p[0], p[1], p[2]);
	new_p[0] = trunc_pixel((double)p[0] * new_luminance / lum);
	new_p[1] = trunc_pixel((double)p[1] * new_luminance / lum);
	new_p[2] = trunc_pixel((double)p[2] * new_luminance / lum );
	return new_p;
}



cv::Mat make_match_image_histogram(cv::Mat im1, cv::Mat & im2 ){
    Histogram hist = Histogram(im2);
    return make_match_histogram(im1, hist);
}

cv::Mat make_match_histogram(cv::Mat & mat, Histogram & new_hist ){
    cv::Mat result = mat.clone();
	Histogram cur_hist(mat);
    
    int h = mat.rows,  w = mat.cols;

	std::vector<double> cur_cum = cur_hist.GetRelCumulative();
	std::vector<double> new_cum = new_hist.GetRelCumulative();

	std::vector<int> new_shade(256, 0);

	int i_new = 0;
	// Try to match i_cur with some i_new
	for (int i_cur = 0; i_cur < 256; ++i_cur) {
		// Find closest new tone;
		while (i_new < 255) {
			double curr_dist = std::abs(cur_cum[i_cur] - new_cum[i_new]);
			double next_dist = std::abs(cur_cum[i_cur] - new_cum[i_new + 1]);
			if (next_dist <= curr_dist) {
				++i_new;
			} else break;
		}
		new_shade[i_cur] = i_new;
	}
        
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			cv::Vec3b& p = result.at<cv::Vec3b>(r, c);
			int lum = trunc_pixel(get_luminance(p[0], p[1], p[2]));
			p = change_luminance(p, new_shade[ lum ]);
			// p[1] = p[2] = p[0]; // TODO: FIX
		}
	}

    return result;
}