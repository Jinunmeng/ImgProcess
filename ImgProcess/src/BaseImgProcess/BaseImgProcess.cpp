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


 // 创建图像
cv::Mat createMat()
{
	cv::Mat ima(500, 500, CV_8U, 50);
	return ima;
}

int main0()
{
	// 创建240x320大小Mat
	cv::Mat image1(240,320,CV_8U,cv::Scalar(100)); // Scalar是值初始化100
	cv::imshow("Image", image1);
	cv::waitKey(0); 

	// 重新分配新的图像
	image1.create(200, 200, CV_8U);
	image1 = 200;
	cv::imshow("Image", image1);
	cv::waitKey(0);

	// 创建红色图，OpenCV中3通道顺序是BGR
	cv::Mat image2(240, 320, CV_8UC3, cv::Scalar(0, 0, 255));
	// 或者
	// cv::Mat image2(cv::Size(320,240),CV_8UC3);
	// image2= cv::Scalar(0,0,255);
	cv::imshow("Image", image2);
	cv::waitKey(0);

	// 读入图像
	cv::Mat image3 = cv::imread("..\\..\\sample\\puppy.bmp");
	// 这些图像都是指向同一个数据块（浅拷贝）
	cv::Mat image4(image3);
	image1 = image3;

	// 拷贝源图像（深拷贝）
	image3.copyTo(image2); // 深拷贝
	cv::Mat image5 = image3.clone(); //深拷贝）

	// 翻转图像
	cv::flip(image3, image3, 1);

	// 检查哪些图像已经受到处理的影响
	cv::imshow("Image 3", image3);
	cv::imshow("Image 1", image1);
	cv::imshow("Image 2", image2);
	cv::imshow("Image 4", image4);
	cv::imshow("Image 5", image5);
	cv::waitKey(0); 

	// 获取灰度图像
	cv::Mat gray = createMat();
	cv::imshow("Image", gray);
	cv::waitKey(0); 

	// 读灰度图像
	image1 = cv::imread("..\\..\\sample\\puppy.bmp", IMREAD_GRAYSCALE);

	// convertTo：数据类型转换
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

	switch (event) // 调度事件
	{
		// 鼠标左击事件
	case cv::EVENT_LBUTTONDOWN:
		// 显示鼠标点击坐标
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

	// 创建窗口
	cv::namedWindow("Original Image", WINDOW_NORMAL); 
	cv::imshow("Original Image", image); 

	// 设置鼠标回调函数
	cv::setMouseCallback("Original Image", onMouse, reinterpret_cast<void*>(&image));

	cv::Mat result;
	cv::flip(image, result, 1); // 正：水平翻转；0：垂直翻转；负：水平垂直都翻转
	cv::namedWindow("Output Image");
	cv::imshow("Output Image", result);

	cv::waitKey(0); 

	//cv::imwrite("output.bmp", result); // 保存结果

	cv::namedWindow("Drawing on an Image");
	cv::circle(image,        // 目标图像 
		cv::Point(155, 110), // 中心坐标点
		65,                  // 半径
		Scalar(0, 255, 0),   // 设置颜色
		3);                  // 粗细

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