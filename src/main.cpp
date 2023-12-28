#include <iostream>
#include "image.h"


int main() {
	std::cout << "Hello, world!\n";

	std::vector<std::string> filenames = {"hanoi_input_1.png", "hanoi_input_2.png", "hanoi_input_3.png"};

	// Step 0: Load All Images;
	ImageSequence image_sequence = ImageSequence::createFromFileList(1, filenames);

	// Step 1: Canny Edge Detector;
	ImageSequence edges_sequence;
	for(Image& img : image_sequence.getFrames()){
		edges_sequence.addFrame(applyCannyTransform(img));
	}

	// Step 2: Save
	edges_sequence.saveAll();


	return 0;
}