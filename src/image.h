#ifndef WXFPI_IMAGE_H
#define WXFPI_IMAGE_H

#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

class Image;

class Image {
private:
	int w;
	int h;
	cv::Mat matrix;

    void setSize();

public:
    Image(std::string filename);
	Image(cv::Mat matrix);
	Image(const Image& other);
	Image(int h, int w);

    void save();

    int getW();
    int getH();
    cv::Mat& getMatrix();

    std::string filename;
};


double get_luminance(int r, int g, int b);

class Histogram {
public:
	Histogram(Image& image);

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


Image make_add_image(Image & im1, Image & im2);
Image make_bilateral_filter(Image & image, int d, double sigma_color, double sigma_space);
Image make_subtract_image(Image & im1, Image & im2);
Image make_match_image_histogram(Image & im1, Image & im2);
Image make_match_histogram(Image & im1, Histogram & h);

#endif /* WXFPI_IMAGE_H */