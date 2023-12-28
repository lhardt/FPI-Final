
#include "image.h"
#include "constants.h"

Image applyCannyTransform(Image & image){
    Image result(image);
    cv::Mat new_mat;
    cv::Canny(image.getMatrix(), result.getMatrix(), CANNY_T1, CANNY_T2);
    return result;
}

////////////////////////////////////////////

ImageSequence ImageSequence::createFromFileList(int ref_frame, std::vector<std::string> filenames){
    ImageSequence result;

    for(std::string name : filenames){
        result.addFrame(Image(name));
    }

    result.setRefFrameIndex(ref_frame);

    return result;
}

ImageSequence::ImageSequence(){

}

void ImageSequence::saveAll(){
    for(Image& img : frames){
        img.save();
    }
}

int ImageSequence::countFrames(){ return frames.size(); }
int ImageSequence::getRefFrameIndex(){ return ref_frame; };
Image& ImageSequence::getRefFrame(){ return frames[ref_frame]; }
std::vector<Image>& ImageSequence::getFrames(){ return frames; }

void ImageSequence::addFrame(Image frame){ frames.push_back(frame); }
void ImageSequence::setRefFrameIndex(int i){ ref_frame = i; }

////////////////////////////////////////////

Image::Image(std::string filename) : matrix(cv::imread(SRC_FOLDER + filename, cv::IMREAD_COLOR)), type(IMG_COLOR) {
    this->filename = filename;
    if(matrix.empty()){
        std::cout << "Warning: empty image with name <" << filename << ">\n";
    }
    setSize();
}
Image::Image(cv::Mat m) : matrix(m), type(IMG_COLOR)  {
    setSize();
}
Image::Image(const Image& other) : matrix(other.matrix), type(other.type), w(other.w), h(other.h), filename(other.filename) {
    setSize();
}
Image::Image(int h, int w): matrix(h, w, CV_8UC3, cv::Scalar(255,255,255)), type(IMG_COLOR){
    setSize();
}

void Image::setSize(){
	cv::Size sz = matrix.size();
	this->w = sz.width;
	this->h = sz.height;
}

void Image::save() {
    std::cout << "Trying to save file as <" << TRG_FOLDER + filename << ">\n";
	cv::imwrite(TRG_FOLDER + filename, this->matrix);
}

int Image::getW(){ return w; }
int Image::getH(){ return h; }
cv::Mat& Image::getMatrix(){ return matrix;}
ImageType Image::getType(){ return type; }