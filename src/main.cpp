#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "image.h"
#include "constants.h"


int main(){
    std::cout << "Hello, world!\n";

    cv::Mat im_input = cv::imread(SRC_FOLDER + "rock_input.png", cv::IMREAD_COLOR);
    cv::Mat im_model = cv::imread(SRC_FOLDER + "rock_model.png", cv::IMREAD_COLOR);
    save_image(im_input, "0-Input.png");
    save_image(im_model, "0-Model.png");
    
    // for (int r = 0; r < im_input.rows; ++r) {
	// 	for (int c = 0; c < im_input.cols; ++c) {
    //         cv::Vec3b & p = im_input.at<cv::Vec3b>(r,c);
    //         p = get_luminance(p[0], p[1], p[2]);
    //     }
    // }
    
    double sigma_color = 120;
    double sigma_space = std::min(im_input.rows, im_input.cols) / 16.0;
    
    cv::Mat in_bkg = make_bilateral_filter(im_input, sigma_color, -1, -1);
    cv::Mat md_bkg = make_bilateral_filter(im_model, sigma_color, -1, -1);

    save_image(in_bkg, "1-BKG_im.png");
    save_image(md_bkg, "1-BKG_md.png");

    cv::Mat in_detail =subtract(im_input, in_bkg);
    cv::Mat md_detail =subtract(im_model, md_bkg);
    
    save_image(in_detail, "2-DET_im.png");
    save_image(md_detail, "2-DET_md.png");

    cv::Mat scaled_background = make_match_image_histogram(in_bkg, md_bkg);
    cv::Mat scaled_detail = make_match_image_histogram(in_detail, md_detail);
    save_image(scaled_background, "4-SCA_bkg.png");
    save_image(scaled_detail, "4-SCA_frg.png");

    cv::Mat in_textureness = make_textureness(in_detail, 11, sigma_color, sigma_space);
    cv::Mat md_textureness = make_textureness(md_detail, 11, sigma_color, sigma_space);
    cv::Mat target_textureness = make_textureness(scaled_detail, 11, sigma_color, sigma_space);// make_match_image_histogram(in_textureness, md_textureness);
    save_image(in_textureness, "5-TX_im.png");
    save_image(in_textureness, "5-TX_scaled.png");
        
    cv::Mat scaled_bkg_textureness = make_textureness(scaled_background,11, sigma_color, sigma_space);
    save_image(scaled_bkg_textureness, "5-SCA_bkg_txt.png");
    
    
    cv::Mat rho = make_rho(target_textureness, scaled_bkg_textureness, in_textureness);

    cv::Mat final_result = make_combine(scaled_background, in_detail, rho);    
    save_image(final_result, "6-RES.png");
       
    return 0;
}