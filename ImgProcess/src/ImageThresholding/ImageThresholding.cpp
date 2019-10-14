/***************************************************************************
                           ImageThresholding.cpp
  ------------------------------------------------------------------------
  brief                : 1、使用opencv中的OTSU接口进行二值化
						 2、使用自己写的otsu算法进行二值化
						 3、使用OpenCV的adaptiveThreshold方法进行二值化
			 OTSU算法步骤：
						 1、统计灰度级中每个像素在整幅图像中的个数
						 2、计算每个像素在整幅图像的概率分布
						 3、对灰度级进行遍历搜索，计算当前灰度值下前景背景类间概率
						 4、通过目标函数计算类间方差下对应的阈值（选择最大方差对应的阈值）
  date                 :  2019/10/14
  copyright            : (C) 2019 by Jinunmeng
  email                : jinunmeng@163.com
 ***************************************************************************/

#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

int OTSU(Mat srcImage);

int main()
{
	// 图像读取及判断
	cv::Mat srcImage = cv::imread("..\\..\\sample\\2121.jpg");
	if (!srcImage.data)
		return 1;
	// 灰度转换
	cv::Mat srcGray;
	cv::cvtColor(srcImage, srcGray, COLOR_BGR2GRAY);
	namedWindow("srcGray", WINDOW_NORMAL);
	cv::imshow("srcGray", srcGray);
	
	/** 
	 * 使用opencv中的OTSU接口进行二值化
	 */
	Mat result;
	threshold(srcGray, result, 0, 255, THRESH_BINARY | THRESH_OTSU);
	namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);

	/**
	* 使用自己写的otsu算法进行二值化
	*/
	int otsuThreshold = OTSU(srcGray); // 计算阈值
	cout << otsuThreshold << endl;
	//定义输出结果图像
	Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
	//利用得到的阈值进行二值化操作
	for (int i = 0; i < srcGray.rows; i++)
	{
		for (int j = 0; j < srcGray.cols; j++)
		{
			//cout << (int)srcGray.at<uchar>(i, j) << endl;
			//高像素阈值判断
			if (srcGray.at<uchar>(i, j) > otsuThreshold)
			{
				otsuResultImage.at<uchar>(i, j) = 255;
			}
			else
			{
				otsuResultImage.at<uchar>(i, j) = 0;
			}
			//cout <<(int)otsuResultImage.at<uchar>(i, j) << endl;
		}
	}
	namedWindow("otsuResultImage", WINDOW_NORMAL);
	imshow("otsuResultImage", otsuResultImage);

	/**
	* 使用OpenCV的adaptiveThreshold方法进行二值化
	*/
	cv::Mat dstImage;
	// 初始化自适应阈值参数
	int blockSize = 31;
	int constValue = 5;
	const int maxVal = 255;
	/* 自适应阈值算法
	0：ADAPTIVE_THRESH_MEAN_C
	1: ADAPTIVE_THRESH_GAUSSIAN_C
	阈值类型
	0: THRESH_BINARY
	1: THRESH_BINARY_INV */
	int adaptiveMethod = 1;
	int thresholdType = 0;
	//// 图像自适应阈值操作
	cv::adaptiveThreshold(srcGray, dstImage, maxVal, adaptiveMethod, thresholdType, blockSize, constValue);
	namedWindow("adaptiveThreshold", WINDOW_NORMAL);
	cv::imshow("adaptiveThreshold", dstImage);

	cv::waitKey(0);
	return 0;
}

//OTSU 函数实现
int OTSU(Mat srcImage)
{
	int nCols = srcImage.cols;
	int nRows = srcImage.rows;
	int threshold = 0;
	//init the parameters
	int nSumPix[256];
	float nProDis[256];
	for (int i = 0; i < 256; i++)
	{
		nSumPix[i] = 0;
		nProDis[i] = 0;
	}

	//统计灰度集中每个像素在整幅图像中的个数
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}

	//计算每个灰度级占图像中的概率分布
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols*nRows);
	}

	//遍历灰度级【0，255】，计算出最大类间方差下的阈值

	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		//初始化相关参数
		w0 = w1 = u0 = u1 = u0_temp = u1_temp = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//背景部分
			if (j <= i)
			{
				w0 += nProDis[j];
				u0_temp += j*nProDis[j];
			}
			//前景部分
			else
			{
				w1 += nProDis[j];
				u1_temp += j*nProDis[j];
			}
		}
		//计算两个分类的平均灰度
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		//依次找到最大类间方差下的阈值
		delta_temp = (float)(w0*w1*pow((u0 - u1), 2)); //前景与背景之间的方差(类间方差)
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return threshold;
}

