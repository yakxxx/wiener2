#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
Mat padd_image(Mat I);
Mat with_noise(Mat image, int stddev);
Mat get_spectrum(Mat I);
Mat wiener2(Mat I, Mat image_spectrum, int noise_stddev);



int main(void) {
	Mat  I = imread("lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	                           //expand input image to optimal size
	Mat padded = padd_image(I);
	Mat noisy = with_noise(padded, 50);
	Mat spectrum = get_spectrum(padded);
	Mat enhanced = wiener2(noisy, spectrum, 50);

//	Mat spectrum = get_spectrum(I);
//
//	spectrum += Scalar::all(1);                    // switch to logarithmic scale
//	log(spectrum, spectrum);
//	normalize(spectrum, spectrum, 0, 1, CV_MINMAX);
//
	imshow("image 1", I);
	imshow("image 2", noisy);
	imshow("image 3", enhanced);
	waitKey();
}


Mat get_spectrum(Mat I){
	Mat padded = padd_image(I);
	Mat padded_f;
	padded.convertTo(padded_f, CV_32F);
	Mat planes[] = {Mat(padded_f.rows, padded_f.cols, CV_32FC1), Mat(padded_f.rows, padded_f.cols, CV_32FC1)};
	Mat complexI;
	dft(padded_f, complexI, DFT_COMPLEX_OUTPUT);

	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	return magI;
}

Mat with_noise(Mat image, int stddev){
	Mat noise(image.rows, image.cols, CV_8U);
	randn(noise,Scalar::all(0), Scalar::all(stddev));
	Mat noisy = image.clone();
	noisy += noise;
	return noisy;
}

Mat wiener2(Mat I, Mat image_spectrum, int noise_stddev){
	Mat padded = padd_image(I);
	Mat noise = Mat::zeros(I.rows, I.cols, CV_32F);
	randn(noise,Scalar::all(0), Scalar::all(noise_stddev));
	Mat noise_spectrum = get_spectrum(noise);

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);
	split(complexI, planes);	// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	Mat image_spectrum_2 = image_spectrum * image_spectrum;
	Mat noise_spectrum_2 = noise_spectrum * noise_spectrum;

	Mat factor = image_spectrum_2 / (image_spectrum_2 + noise_spectrum_2);
	multiply(planes[0],factor,planes[0]);
	multiply(planes[1],factor,planes[1]);


	merge(planes, 2, complexI);
	idft(complexI, complexI);
	split(complexI, planes);
	normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	return planes[0];
}

Mat padd_image(Mat I){
	Mat padded;
	int m = getOptimalDFTSize( I.rows );
	int n = getOptimalDFTSize( I.cols ); // on the border add zero pixels
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	return padded;
}
