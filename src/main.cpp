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
    
    double sigma_color = 160;
    double sigma_space = std::min(im_input.rows, im_input.cols) / 16.0;
    int D = 15;
    
    cv::Mat in_bkg = make_bilateral_filter(im_input, sigma_color, sigma_space, -1);
    cv::Mat md_bkg = make_bilateral_filter(im_model, sigma_color, sigma_space, -1);

    save_image(in_bkg, "1-BKG_im.png");
    save_image(md_bkg, "1-BKG_md.png");

    cv::Mat in_detail =subtract(im_input, in_bkg);
    cv::Mat md_detail =subtract(im_model, md_bkg);
    
    save_image(in_detail, "2-DET_im.png");
    save_image(md_detail, "2-DET_md.png");

    cv::Mat reconstructed = in_bkg + in_detail;
    save_image(reconstructed, "3-Reconstructed.png");

    cv::Mat scaled_background = make_match_image_histogram(in_bkg, md_bkg);
    cv::Mat scaled_detail = make_match_image_histogram(in_detail, md_detail);
    cv::Mat scaled_image = make_match_image_histogram(im_input, im_model);
    save_image(scaled_background, "4-SCA_bkg.png");
    save_image(scaled_detail, "4-SCA_frg.png");
    save_image(scaled_detail, "4-SCA_img.png");
    
    // cv::Mat highpass = make_high_pass(in_detail);
    

    cv::Mat in_textureness = make_textureness(in_detail, D, sigma_color, 8*sigma_space);
    cv::Mat md_textureness = make_textureness(md_detail, D, sigma_color, 8*sigma_space);
    cv::Mat target_textureness = make_textureness(scaled_image, D, sigma_color, 8*sigma_space);
    save_image(in_textureness, "5-TX_im.png");
    save_image(in_textureness, "5-TX_scaled.png");
        
    cv::Mat scaled_bkg_textureness = make_textureness(scaled_background, D, sigma_color, 8* sigma_space);
    save_image(scaled_bkg_textureness, "5-SCA_bkg_txt.png");
    
    cv::Mat rho = make_rho(target_textureness, scaled_bkg_textureness, in_textureness);
    cv::Mat rho2 = rho.clone();
    for (int r = 0; r < rho.rows; ++r) 
		for (int c = 0; c < rho.cols; ++c) 
            rho2.at<double>(r,c) = 4;
    cv::Mat final_result = make_combine(scaled_background, in_detail, rho);    
    cv::Mat final_result_neutral = make_combine(scaled_background, in_detail, rho2);    
    save_image(final_result, "6-RES.png");
    save_image(final_result_neutral, "6-RES_neutral.png");
       
    return 0;
}