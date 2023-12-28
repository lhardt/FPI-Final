#ifndef WXFPI_IMAGE_H
#define WXFPI_IMAGE_H

#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

class Image;
class MotionField;

Image applyCannyTransform(Image &image);
MotionField createMotionField(Image & im1, Image & im2);


enum ImageType {
    IMG_COLOR
};

class MotionField {

};

class Image {
private:
	int w;
	int h;
	cv::Mat matrix;
    std::string filename;
    ImageType type;

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
    ImageType getType();
};

class ImageSequence {
private:
    int ref_frame;
    std::vector<Image> frames;

public:
    static ImageSequence createFromFileList(int ref_frame, std::vector<std::string> filenames);

    ImageSequence();

    void saveAll();

    int countFrames();
    int getRefFrameIndex();
    Image& getRefFrame();
    std::vector<Image>& getFrames();

    void addFrame(Image frame);
    void setRefFrameIndex(int i);
};

#endif /* WXFPI_IMAGE_H */