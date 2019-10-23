
/*
���ܣ��ù�����������6���˲���ʽ�Ĳ���ֵ��
��ʽ�˲�����ֵ�˲�����˹�˲�����ֵ�˲���˫���˲��������˲���
*/
#include <opencv2/core/core.hpp>                    
#include <opencv2/highgui/highgui.hpp>        
#include <opencv2/imgproc/imgproc.hpp>    
#include <iostream>       
using namespace std;
using namespace cv;

#define WINDOWNAME "���˲����������ڡ�"

//---------------��ȫ�ֱ����������֡�-------------------------
Mat g_srcIamge, g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5, g_dstImage6;
int g_nBoxFilterValue = 1;//��ʽ�˲��ں�ֵ
int g_nMeanBlurValue = 1;//��ֵ�˲��ں�ֵ
int g_nGaussianBlurValue = 1;//��˹�˲��ں�ֵ
int g_nMedianBlurValue = 1;//��ֵ�˲��ں�ֵ
int g_nBilateralFilterValue = 1;//˫���˲��ں�ֵ
int g_nGuidedFilterValue = 1;//�����˲��ں�ֵ
const int g_nMaxVal = 20; //Ԥ�軬�������ֵ 
						  //--------------��ȫ�ֺ����������֡�-------------------------
						  //�켣���ص�����
static void on_BoxFilter(int, void*);//��ʽ�˲���
static void on_MeanBlur(int, void*);//��ֵ�˲���
static void on_GaussianBlur(int, void*);//��˹�˲���
static void on_MedianBlur(int, void*);//��ֵ�˲���
static void on_BilateralFilter(int, void*);//˫���˲���
static void on_GuidedFilter(int, void*);//�����˲���
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps);//�����˲���

void ALTMRetinex(const Mat& src, Mat &dst, bool LocalAdaptation = false, bool ContrastCorrect = true);
Mat guidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps);

//----------------------------����������---------------------------
int main()
{
	//------------��1����ȡԴͼ�񲢼��ͼ���Ƿ��ȡ�ɹ�------------  
	g_srcIamge = imread("C:\\Users\\March\\Desktop\\���ǽ���\\03\\6-bayer.tif");
	const char* coutPath = "C:\\Users\\March\\Desktop\\���ǽ���\\03\\6-bayer-en.tif";
	if (!g_srcIamge.data)
	{
		cout << "��ȡͼƬ����������������ȷ·����\n";
		system("pause");
		return -1;
	}
	Mat dstMat;
	ALTMRetinex(g_srcIamge, dstMat);
	imwrite(coutPath, dstMat);

	//namedWindow("��Դͼ��", 1);//��������
	//imshow("��Դͼ��", g_srcIamge);//��ʾ����
	//------------��2����WINDOWNAME�����Ϸֱ𴴽��˲�6��������------------       
	//namedWindow(WINDOWNAME);//��������	
	//createTrackbar("�����˲�", WINDOWNAME, &g_nBoxFilterValue, g_nMaxVal, on_BoxFilter);//���������˲��켣��
	//on_BoxFilter(g_nBoxFilterValue, 0);	//�켣���Ļص�����
	//createTrackbar("��ֵ�˲�", WINDOWNAME, &g_nMeanBlurValue, g_nMaxVal, on_MeanBlur);//������ֵ�˲��켣��
	//on_MeanBlur(g_nMeanBlurValue, 0);
	//createTrackbar("��˹�˲�", WINDOWNAME, &g_nGaussianBlurValue, g_nMaxVal, on_GaussianBlur);//������˹�˲��켣��
	//on_GaussianBlur(g_nGaussianBlurValue, 0);
	//createTrackbar("��ֵ�˲�", WINDOWNAME, &g_nMedianBlurValue, g_nMaxVal, on_MedianBlur);//������ֵ�˲��켣��
	//on_MedianBlur(g_nMedianBlurValue, 0);
	//createTrackbar("˫���˲�", WINDOWNAME, &g_nBilateralFilterValue, g_nMaxVal, on_BilateralFilter);//����˫���˲��켣��
	//on_BilateralFilter(g_nBilateralFilterValue, 0);
	//createTrackbar("�����˲�", WINDOWNAME, &g_nGuidedFilterValue, g_nMaxVal, on_GuidedFilter);//���������˲��켣��
	//on_GuidedFilter(g_nGuidedFilterValue, 0);
	//------------��3���˳�����------------  
	//cout << "\t����'q'�����˳�����~��\n" << endl;

	waitKey(0);
	return 0;
}

//----------------------��on_BoxFilter()������------------------------
static void on_BoxFilter(int, void*)
{
	boxFilter(g_srcIamge, g_dstImage1, -1, Size(g_nBoxFilterValue * 2 + 1, g_nBoxFilterValue * 2 + 1));
	cout << "\n��ǰΪ����ʽ�˲�������Ч�������ں˴�СΪ��" << g_nBoxFilterValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage1);
}
//----------------------��on_MeanBlur()������------------------------
static void on_MeanBlur(int, void*)
{
	blur(g_srcIamge, g_dstImage2, Size(g_nMeanBlurValue * 2 + 1, g_nMeanBlurValue * 2 + 1), Point(-1, -1));
	cout << "\n��ǰΪ����ֵ�˲�������Ч�������ں˴�СΪ��" << g_nMeanBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage2);
}
//----------------------��on_GaussianBlur()������------------------------
static void on_GaussianBlur(int, void*)
{
	GaussianBlur(g_srcIamge, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	cout << "\n��ǰΪ����˹�˲�������Ч�������ں˴�СΪ��" << g_nGaussianBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage3);
}
//----------------------��on_MedianBlur()������------------------------
static void on_MedianBlur(int, void*)
{
	medianBlur(g_srcIamge, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	cout << "\n��ǰΪ����ֵ�˲�������Ч�������ں˴�СΪ��" << g_nMedianBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage4);
}
//----------------------��on_BilateralFilter()������------------------------
static void on_BilateralFilter(int, void*)
{
	bilateralFilter(g_srcIamge, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	cout << "\n��ǰΪ��˫���˲�������Ч�������ں˴�СΪ��" << g_nBilateralFilterValue << endl;
	imshow(WINDOWNAME, g_dstImage5);
}
//----------------------��on_GuidedFilter()������------------------------
static void on_GuidedFilter(int, void*)
{
	vector<Mat> vSrcImage, vResultImage;
	//��1����Դͼ�����ͨ�����룬����ÿ����ͨ�����е����˲�����
	split(g_srcIamge, vSrcImage);
	for (int i = 0; i < 3; i++)
	{
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//����ͨ��ת���ɸ���������
		Mat cloneImage = tempImage.clone();	//��tempImage����һ�ݵ�cloneImage
		Mat resultImage;
		guidedFilter(tempImage, cloneImage, resultImage, g_nGuidedFilterValue * 2 + 1, 0.01);//�Է�ͨ���ֱ���е����˲�
		vResultImage.push_back(resultImage);//����ͨ�������˲���Ľ����ŵ�vResultImage��
	}
	//��2������ͨ�������˲������ϲ�
	merge(vResultImage, g_dstImage6);
	cout << "\n��ǰ����Ϊ�������˲��������ں˴�СΪ��" << g_nGuidedFilterValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage6);
}

//-------------------��ʵ�ֵ����˲����������֡�-------------------------
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps)
{
	//------------��0��ת��Դͼ����Ϣ����������չΪ64λ�����ͣ��Ա��Ժ����˷�------------
	srcMat.convertTo(srcMat, CV_64FC1);
	guidedMat.convertTo(guidedMat, CV_64FC1);
	//--------------��1�����־�ֵ����----------------------------------
	Mat mean_p, mean_I, mean_Ip, mean_II;
	boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//���ɴ��˲�ͼ���ֵmean_p	
	boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//���ɵ���ͼ���ֵmean_I	
	boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//���ɻ���ؾ�ֵmean_Ip
	boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//���ɵ���ͼ������ؾ�ֵmean_II
																				 //--------------��2���������ϵ��������Ip��Э����cov��I�ķ���var------------------
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_II - mean_I.mul(mean_I);
	//---------------��3���������ϵ��a��b-------------------
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);
	//--------------��4������ϵ��a��b�ľ�ֵ-----------------
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
	boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
	//---------------��5�������������------------------
	dstImage = mean_a.mul(srcMat) + mean_b;
}



void ALTMRetinex(const Mat& src, Mat &dst, bool LocalAdaptation, bool ContrastCorrect)
{

	Mat temp, src_gray;

	src.convertTo(temp, CV_32FC3);
	//�Ҷ�ͼ
	cvtColor(temp, src_gray, COLOR_BGR2GRAY);

	double LwMax;
	//�õ����ֵ
	minMaxLoc(src_gray, NULL, &LwMax);

	Mat Lw_;
	const int num = src.rows * src.cols;
	//����ÿ������Ԫ�ؾ���ֵ����Ȼ����
	cv::log(src_gray + 1e-3f, Lw_);
	//������Ȼָ��
	float LwAver = exp(cv::sum(Lw_)[0] / num);

	Mat Lg;
	log(src_gray / LwAver + 1.f, Lg);
	//�������
	cv::divide(Lg, log(LwMax / LwAver + 1.f), Lg);

	//�ֲ�����Ӧ
	Mat Lout;
	if (LocalAdaptation)
	{
		int kernelSize = floor(std::max(3, std::max(src.rows / 100, src.cols / 100)));
		Mat Lp, kernel = cv::getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
		cv::dilate(Lg, Lp, kernel);
		Mat Hg = guidedFilter(Lg, Lp, 10, 0.01f);

		double eta = 36;
		double LgMax;
		cv::minMaxLoc(Lg, NULL, &LgMax);
		Mat alpha = 1.0f + Lg * (eta / LgMax);

		Mat Lg_;
		cv::log(Lg + 1e-3f, Lg_);
		float LgAver = exp(cv::sum(Lg_)[0] / num);
		float lambda = 10;
		float beta = lambda * LgAver;

		cv::log(Lg / Hg + beta, Lout);
		cv::multiply(alpha, Lout, Lout);
		cv::normalize(Lout, Lout, 0, 255, NORM_MINMAX);
	}
	else
	{
		cv::normalize(Lg, Lout, 0, 255, NORM_MINMAX);
	}

	Mat gain(src.rows, src.cols, CV_32F);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			float x = src_gray.at<float>(i, j);
			float y = Lout.at<float>(i, j);
			if (0 == x) gain.at<float>(i, j) = y;
			else gain.at<float>(i, j) = y / x;
		}
	}

	Mat bgr[3];
	cv::split(temp, bgr);
	// ��ƽ��ϵ������
	bgr[0] = bgr[0] * 1.4310;
	bgr[1] = bgr[1] * 0.9298;
	bgr[2] = bgr[2] * 1.0000;
	if (ContrastCorrect)
	{
		// У��ͼ��Աȶ�
		bgr[0] = (gain.mul(bgr[0] + src_gray) + bgr[0] - src_gray) *0.5f;
		bgr[1] = (gain.mul(bgr[1] + src_gray) + bgr[1] - src_gray) *0.5f;
		bgr[2] = (gain.mul(bgr[2] + src_gray) + bgr[2] - src_gray) *0.5f;
	}
	else
	{
		cv::multiply(bgr[0], gain, bgr[0]);
		cv::multiply(bgr[1], gain, bgr[1]);
		cv::multiply(bgr[2], gain, bgr[2]);
	}

	cv::merge(bgr, 3, dst);
	dst.convertTo(dst, CV_8UC3);
}

//�����˲���
Mat guidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps)
{
	/*
	�� GUIDEDFILTER   O(N) time implementation of guided filter.
	��
	��   - guidance image: I (should be a gray-scale/single channel image)
	��   - filtering input image: p (should be a gray-scale/single channel image)
	��   - local window radius: r
	��   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_32FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_32FC1);
	p = _p;

	//��Ϊopencv�Դ���boxFilter�����е�Size,����9x9,����˵�뾶Ϊ4
	r = 2 * r + 1;

	//mean_I = boxfilter(I, r) ./ N;
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));

	//mean_b = boxfilter(b, r) ./ N;
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
	cv::Mat q = mean_a.mul(I) + mean_b;

	return q;
}
