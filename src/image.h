#ifndef WXFPI_IMAGE_H
#define WXFPI_IMAGE_H

#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

uint8_t trunc_pixel(double pixel_value);
double dist(int x1, int y1, int x2, int y2);
double gaussian(double x, double sigma);

cv::Mat subtract(cv::Mat m1, cv::Mat m2, int to_add = 0);
cv::Mat make_high_pass(cv::Mat& image, int d, double sigma);
cv::Mat make_textureness(cv::Mat& image, int d, int sigma_color, int sigma_space);
cv::Mat convert_to_8uc3(cv::Mat& image);
cv::Mat convert_to_float(cv::Mat& image);

double get_luminance(int r, int g, int b);

class Histogram {
public:
	Histogram(cv::Mat& image);

	int GetNPixels();
	int GetMaxValue();
	std::vector<int> GetLumHist();
	std::vector<int> GetCumulative();
	std::vector<double> GetRelCumulative();
private:
	Histogram();

	std::vector<int> m_lum_hist;
	int m_n_pixels;
};

cv::Mat make_match_histogram(cv::Mat & mat, Histogram & new_hist );
cv::Mat make_match_image_histogram(cv::Mat im1, cv::Mat & im2 );
cv::Mat make_combine(cv::Mat& background, cv::Mat& detail, cv::Mat& rho, float to_add = 0);    
cv::Mat make_bilateral_filter(cv::Mat mat, double sigma_color, double sigma_space, int d = -1);
cv::Mat make_bilateral_filter_2(cv::Mat mat, double sigma_color, double sigma_space, int d = -1);
cv::Mat make_rho(cv::Mat target_textureness, cv::Mat scaled_bkg_textureness, cv::Mat in_textureness);
void save_image(cv::Mat m , std::string filename);


#endif /* WXFPI_IMAGE_H */