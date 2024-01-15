
#include "image.h"
#include "constants.h"
#include <opencv4/opencv2/imgproc.hpp>


Image::Image(std::string filename) : matrix(cv::imread(SRC_FOLDER + filename, cv::IMREAD_COLOR)) {
    this->filename = filename;
    if(matrix.empty()){
        std::cout << "Warning: empty image with name <" << filename << ">\n";
    }
    setSize();
}
Image::Image(cv::Mat m) :  w(m.cols), h(m.rows), matrix(m), filename("empty_filename.png")  {
    setSize();
}
Image::Image(const Image& other) : w(other.w), h(other.h), matrix(other.matrix.clone()), filename(other.filename) {
    setSize();
}
Image::Image(int h, int w): matrix(h, w, CV_8UC3, cv::Scalar(255,255,255)){
    setSize();
}

void Image::setSize(){
	cv::Size sz = matrix.size();
	this->w = sz.width;
	this->h = sz.height;
}

void Image::save() {
    std::cout << "Trying to save file as <" << TRG_FOLDER + filename << "> with type " << matrix.type() << "\n";
	cv::imwrite(TRG_FOLDER + filename, this->matrix);
}

int Image::getW(){ return w; }
int Image::getH(){ return h; }
cv::Mat& Image::getMatrix(){ return matrix;}

///////////////////////////////////////////////////

double get_luminance(int r, int g, int b) {
	// NTSC formula for pixel intensity;
	double ans = 0.299 * r + 0.587 * g + 0.114 * b;
	if (ans < 0) ans = 0;
	if (ans > 255) ans = 255;
	return ans;
}

uint8_t trunc_pixel(double pixel_value) {
	if (pixel_value > 255) pixel_value = 255;
	if (pixel_value < 0) pixel_value = 0;

	return (uint8_t)(pixel_value + 0.5);
}

Histogram::Histogram(Image& image) : m_lum_hist(256, 0) {
	m_n_pixels = image.getW() * image.getH();

	cv::Mat& matrix = image.getMatrix();
	
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


///////////////////////////////////////////////////

Image make_bilateral_filter(Image & image, int d, double sigma_color, double sigma_space){
    Image result(image);
    cv::bilateralFilter(image.getMatrix(), result.getMatrix(), d, sigma_color, sigma_space);
    return result;
}

Image make_subtract_image(Image & im1, Image & im2){
    Image result(im1);
    result.getMatrix() = 128 + im1.getMatrix() - im2.getMatrix();    
    return result;
}

Image make_match_image_histogram(Image & im1, Image & im2 ){
    Histogram hist = Histogram(im2);
    return make_match_histogram(im1, hist);
}

Image make_match_histogram(Image & im_src, Histogram & new_hist ){
    Image image(im_src);
	Histogram cur_hist(im_src);
    
    int h = im_src.getH(),  w = im_src.getW();

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
    
    cv::Mat & mat = image.getMatrix();
    
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			cv::Vec3b& p = mat.at<cv::Vec3b>(r, c);
			int lum = trunc_pixel(get_luminance(p[0], p[1], p[2]));
			p = change_luminance(p, new_shade[ lum ]);
		}
	}

    return image;
}

Image make_add_image(Image & im1, Image & im2){
    Image result(im1);
    cv::Mat & m = result.getMatrix();
    cv::Mat & m1 = im1.getMatrix();
    cv::Mat & m2 = im2.getMatrix();
    
    int h = im1.getH(),  w = im1.getW();
    
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
            cv::Vec3b& res = m.at<cv::Vec3b>(r,c);
            cv::Vec3b& p1 = m1.at<cv::Vec3b>(r,c);
            cv::Vec3b& p2 = m2.at<cv::Vec3b>(r,c);
            
            res[0] = trunc_pixel(0.0 + p1[0] + p2[0]);
            res[1] = trunc_pixel(0.0 + p1[1] + p2[1]);
            res[2] = trunc_pixel(0.0 + p1[2] + p2[2]);    
        }
    }
    return result;
}