#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "image.h"
#include "constants.h"


int main(){
    std::cout << "Hello, world!\n";

    cv::Mat im_input_int = cv::imread(SRC_FOLDER + "rock_input.png", cv::IMREAD_COLOR);
    cv::Mat im_input = convert_to_float(im_input_int);
    save_image(im_input, "0-Input.png");    
    
    cv::Mat im_model_int = cv::imread(SRC_FOLDER + "rock_model.png", cv::IMREAD_COLOR);
    cv::Mat im_model = convert_to_float(im_model_int);
    save_image(im_model, "0-Model.png");
    
    double sigma_color_1 = 100;
    double sigma_space_1 = std::min(im_input.rows, im_input.cols) / 16.0;
    
    float SCALE = 4.0f;

    sigma_space_1 = sigma_space_1 / 4;
    int D = 31;

    double sigma_color_2 =  200;
    double sigma_space_2 =  2*sigma_space_1;
    
    cv::Mat in_bkg = make_bilateral_filter(im_input, sigma_color_1, sigma_space_1, -1);
    cv::Mat md_bkg = make_bilateral_filter(im_model, sigma_color_1, sigma_space_1, -1);

    save_image(in_bkg, "1-BKG_im.png");
    save_image(md_bkg, "1-BKG_md.png");

    cv::Mat in_detail = im_input - in_bkg + 128.0f;
    cv::Mat md_detail = im_model - md_bkg + 128.0f;
    
    save_image(in_detail, "2-DET_im.png");
    save_image(md_detail, "2-DET_md.png");

    cv::Mat highpass = make_high_pass(in_detail, D, sigma_space_1);
    save_image(highpass, "3-highpass.png");

    cv::Mat scaled_background = make_match_image_histogram(in_bkg, md_bkg);
    cv::Mat scaled_detail = make_match_image_histogram(in_detail, md_detail);
    cv::Mat scaled_image = make_match_image_histogram(im_input, im_model);

    save_image(scaled_background, "4-SCA_bkg.png");
    save_image(scaled_detail, "4-SCA_frg.png");
    save_image(scaled_image, "4-SCA_img.png");
    
    cv::Mat detail_textureness = make_textureness(in_detail, D, sigma_color_2, sigma_space_2);
    
    cv::Mat in_textureness = make_textureness(im_input, D, sigma_color_2, sigma_space_2);
    cv::Mat in_scaled_texture = SCALE * in_textureness;

    save_image(in_textureness, "5-Text_im.png");
    save_image(in_scaled_texture, "5-Text_in_scaled.png");

    cv::Mat md_textureness = make_textureness(im_model, D, sigma_color_2, sigma_space_2);
    cv::Mat md_scaled_texture = SCALE * md_textureness;

    save_image(md_textureness, "5-TX_md.png");
    save_image(md_scaled_texture, "5-TX_md_10.png");

    cv::Mat target_textureness = make_match_image_histogram(in_scaled_texture, md_scaled_texture);
    target_textureness = convert_to_float(target_textureness) / SCALE;
    save_image(target_textureness, "5-TX_target.png");
    
    
    // make_textureness(scaled_image, D, sigma_color_2, sigma_space_2);
        
    cv::Mat scaled_bkg_textureness = make_textureness(scaled_background, D, sigma_color_2, sigma_space_2);
    save_image(scaled_bkg_textureness, "5-SCA_bkg_txt.png");
    
    cv::Mat rho = make_rho(target_textureness, scaled_bkg_textureness, detail_textureness);
    cv::Mat rho2 = rho.clone();
    for (int r = 0; r < rho.rows; ++r) {      
        for (int c = 0; c < rho.cols; ++c){
            rho2.at<float>(r,c) = 4;        
        }  
    }
    cv::Mat final_result = make_combine(scaled_background, in_detail, rho, -128.0f);    
    cv::Mat final_result_neutral = make_combine(scaled_background, in_detail, rho2, -128.0f);    
    save_image(final_result, "6-RES.png");
    save_image(final_result_neutral, "6-RES_neutral.png");
       
    return 0;
}