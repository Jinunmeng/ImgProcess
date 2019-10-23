/***************************************************************************
                           BaseImgProcess.cpp
  ------------------------------------------------------------------------
  brief                :  brief
  date                 :  2019/10/15
  copyright            : (C) 2019 by Jinunmeng
  email                : jinunmeng@163.com
 ***************************************************************************/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


 // ����ͼ��
cv::Mat createMat()
{
	cv::Mat ima(500, 500, CV_8U, 50);
	return ima;
}

int main0()
{
	// ����240x320��СMat
	cv::Mat image1(240,320,CV_8U,cv::Scalar(100)); // Scalar��ֵ��ʼ��100
	cv::imshow("Image", image1);
	cv::waitKey(0); 

	// ���·����µ�ͼ��
	image1.create(200, 200, CV_8U);
	image1 = 200;
	cv::imshow("Image", image1);
	cv::waitKey(0);

	// ������ɫͼ��OpenCV��3ͨ��˳����BGR
	cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0, 0, 255));
	// ����
	// cv::Mat image2(cv::Size(320,240),CV_8UC3);
	// image2= cv::Scalar(0,0,255);
	cv::imshow("Image", image2);
	cv::waitKey(0);

	// ����ͼ��
	cv::Mat image3 = cv::imread("..\\..\\sample\\puppy.bmp");
	// ��Щͼ����ָ��ͬһ�����ݿ飨ǳ������
	cv::Mat image4(image3);
	image1 = image3;

	// ����Դͼ�������
	image3.copyTo(image2); // ���
	cv::Mat image5 = image3.clone(); //�����

	// ��תͼ��
	cv::flip(image3, image3, 1);

	// �����Щͼ���Ѿ��ܵ������Ӱ��
	cv::imshow("Image 3", image3);
	cv::imshow("Image 1", image1);
	cv::imshow("Image 2", image2);
	cv::imshow("Image 4", image4);
	cv::imshow("Image 5", image5);
	cv::waitKey(0); 

	// ��ȡ�Ҷ�ͼ��
	cv::Mat gray = createMat();
	cv::imshow("Image", gray);
	cv::waitKey(0); 

	// ���Ҷ�ͼ��
	image1 = cv::imread("..\\..\\sample\\puppy.bmp", IMREAD_GRAYSCALE);

	// convertTo����������ת��
	image1.convertTo(image2, CV_32F, 1 / 255.0, 0.0);
	cv::imshow("Image", image2);

	// Test cv::Matx
	// a 3x3 matrix of double-precision
	cv::Matx33d matrix(3.0, 2.0, 1.0,
		2.0, 1.0, 3.0,
		1.0, 2.0, 3.0);
	// a 3x1 matrix (a vector)
	cv::Matx31d vector(5.0, 1.0, 3.0);
	// multiplication
	cv::Matx31d result = matrix*vector;

	std::cout << result;

	cv::waitKey(0); // wait for a key pressed

    return 0;
}


void onMouse(int event, int x, int y, int flags, void* param) {

	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);

	switch (event) // �����¼�
	{
		// �������¼�
	case cv::EVENT_LBUTTONDOWN:
		// ��ʾ���������
		std::cout << "at (" << x << "," << y << ") value is: "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
		break;
	}
}

int main()
{
	cv::Mat image;
	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;

	image = cv::imread("..\\..\\sample\\puppy.bmp", cv::IMREAD_GRAYSCALE);
	if (image.empty()) { 
		std::cout << "Error reading image..." << std::endl;
		return 0;
	}

	std::cout << "This image is " << image.rows << " x "
		<< image.cols << std::endl;
	std::cout << "This image has "
		<< image.channels() << " channel(s)" << std::endl;

	// ��������
	cv::namedWindow("Original Image", WINDOW_NORMAL); 
	cv::imshow("Original Image", image); 

	// �������ص�����
	cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void*>(&image));

	cv::Mat result;
	cv::flip(image, result, 1); // ����ˮƽ��ת��0����ֱ��ת������ˮƽ��ֱ����ת
	cv::namedWindow("Output Image");
	cv::imshow("Output Image", result);

	cv::waitKey(0); 

	//cv::imwrite("output.bmp", result); // ������

	cv::namedWindow("Drawing on an Image");
	cv::circle(image,        // Ŀ��ͼ�� 
		cv::Point(155, 110), // ���������
		65,                  // �뾶
		Scalar(0, 255, 0),   // ������ɫ
		3);                  // ��ϸ

	cv::putText(image,           // destination image
		"This is a dog.",        // text
		cv::Point(40, 200),      // text position
		cv::FONT_HERSHEY_PLAIN,  // font type
		2.0,                     // font scale
		255,                     // text color (here white)
		2);                      // text thickness

	cv::imshow("Drawing on an Image", image);

	cv::waitKey(0); 
	
	return 0;
}