#include <iostream>
#include "image.h"
#include <opencv4/opencv2/opencv.hpp>

int main(){
    std::cout << "Hello, world!\n";
    
    Image image_input("rock_input.png");
    Image image_model("rock_model.png");
    
    Image background_model = make_bilateral_filter(image_model, 100, 100, 100);  
    Image background_input = make_bilateral_filter(image_input, 100, 100, 100);  

    background_input.filename = "background_" + image_input.filename;
    background_model.filename = "background_" + image_model.filename;

    Image details_model = make_subtract_image(image_model, background_model);
    Image details_input = make_subtract_image(image_input, background_input);
    
    details_input.filename = "details_" + image_input.filename;
    details_model.filename = "details_" + image_model.filename;
    
    Image matched_background = make_match_image_histogram(background_input, background_model);
    Image final_test  = make_add_image(matched_background, details_input);
        
    matched_background.filename = "matchedbg_" + image_input.filename;
    final_test.filename = "final_" + image_input.filename;

 
    background_model.save();
    background_input.save();
    matched_background.save();
    details_model.save();
    details_input.save();
    final_test.save();
    
    
    return 0;
}