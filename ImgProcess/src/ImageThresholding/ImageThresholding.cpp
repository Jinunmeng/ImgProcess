/***************************************************************************
                           ImageThresholding.cpp
  ------------------------------------------------------------------------
  brief                : 1��ʹ��opencv�е�OTSU�ӿڽ��ж�ֵ��
						 2��ʹ���Լ�д��otsu�㷨���ж�ֵ��
						 3��ʹ��OpenCV��adaptiveThreshold�������ж�ֵ��
			 OTSU�㷨���裺
						 1��ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
						 2������ÿ������������ͼ��ĸ��ʷֲ�
						 3���ԻҶȼ����б������������㵱ǰ�Ҷ�ֵ��ǰ������������
						 4��ͨ��Ŀ�꺯��������䷽���¶�Ӧ����ֵ��ѡ����󷽲��Ӧ����ֵ��
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
	// ͼ���ȡ���ж�
	cv::Mat srcImage = cv::imread("..\\..\\sample\\2121.jpg");
	if (!srcImage.data)
		return 1;
	// �Ҷ�ת��
	cv::Mat srcGray;
	cv::cvtColor(srcImage, srcGray, COLOR_BGR2GRAY);
	namedWindow("srcGray", WINDOW_NORMAL);
	cv::imshow("srcGray", srcGray);
	
	/** 
	 * ʹ��opencv�е�OTSU�ӿڽ��ж�ֵ��
	 */
	Mat result;
	threshold(srcGray, result, 0, 255, THRESH_BINARY | THRESH_OTSU);
	namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);

	/**
	* ʹ���Լ�д��otsu�㷨���ж�ֵ��
	*/
	int otsuThreshold = OTSU(srcGray); // ������ֵ
	cout << otsuThreshold << endl;
	//����������ͼ��
	Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
	//���õõ�����ֵ���ж�ֵ������
	for (int i = 0; i < srcGray.rows; i++)
	{
		for (int j = 0; j < srcGray.cols; j++)
		{
			//cout << (int)srcGray.at<uchar>(i, j) << endl;
			//��������ֵ�ж�
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
	* ʹ��OpenCV��adaptiveThreshold�������ж�ֵ��
	*/
	cv::Mat dstImage;
	// ��ʼ������Ӧ��ֵ����
	int blockSize = 31;
	int constValue = 5;
	const int maxVal = 255;
	/* ����Ӧ��ֵ�㷨
	0��ADAPTIVE_THRESH_MEAN_C
	1: ADAPTIVE_THRESH_GAUSSIAN_C
	��ֵ����
	0: THRESH_BINARY
	1: THRESH_BINARY_INV */
	int adaptiveMethod = 1;
	int thresholdType = 0;
	//// ͼ������Ӧ��ֵ����
	cv::adaptiveThreshold(srcGray, dstImage, maxVal, adaptiveMethod, thresholdType, blockSize, constValue);
	namedWindow("adaptiveThreshold", WINDOW_NORMAL);
	cv::imshow("adaptiveThreshold", dstImage);

	cv::waitKey(0);
	return 0;
}

//OTSU ����ʵ��
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

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}

	//����ÿ���Ҷȼ�ռͼ���еĸ��ʷֲ�
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols*nRows);
	}

	//�����Ҷȼ���0��255��������������䷽���µ���ֵ

	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		//��ʼ����ز���
		w0 = w1 = u0 = u1 = u0_temp = u1_temp = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//��������
			if (j <= i)
			{
				w0 += nProDis[j];
				u0_temp += j*nProDis[j];
			}
			//ǰ������
			else
			{
				w1 += nProDis[j];
				u1_temp += j*nProDis[j];
			}
		}
		//�������������ƽ���Ҷ�
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		//�����ҵ������䷽���µ���ֵ
		delta_temp = (float)(w0*w1*pow((u0 - u1), 2)); //ǰ���뱳��֮��ķ���(��䷽��)
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return threshold;
}

