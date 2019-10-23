
/*
功能：用滚动条来控制6种滤波方式的参数值。
盒式滤波、均值滤波、高斯滤波、中值滤波、双边滤波、导向滤波。
*/
#include <opencv2/core/core.hpp>                    
#include <opencv2/highgui/highgui.hpp>        
#include <opencv2/imgproc/imgproc.hpp>    
#include <iostream>       
using namespace std;
using namespace cv;

#define WINDOWNAME "【滤波处理结果窗口】"

//---------------【全局变量声明部分】-------------------------
Mat g_srcIamge, g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5, g_dstImage6;
int g_nBoxFilterValue = 1;//盒式滤波内核值
int g_nMeanBlurValue = 1;//均值滤波内核值
int g_nGaussianBlurValue = 1;//高斯滤波内核值
int g_nMedianBlurValue = 1;//中值滤波内核值
int g_nBilateralFilterValue = 1;//双边滤波内核值
int g_nGuidedFilterValue = 1;//导向滤波内核值
const int g_nMaxVal = 20; //预设滑动条最大值 
						  //--------------【全局函数声明部分】-------------------------
						  //轨迹条回调函数
static void on_BoxFilter(int, void*);//盒式滤波器
static void on_MeanBlur(int, void*);//均值滤波器
static void on_GaussianBlur(int, void*);//高斯滤波器
static void on_MedianBlur(int, void*);//中值滤波器
static void on_BilateralFilter(int, void*);//双边滤波器
static void on_GuidedFilter(int, void*);//导向滤波器
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps);//导向滤波器

void ALTMRetinex(const Mat& src, Mat &dst, bool LocalAdaptation = false, bool ContrastCorrect = true);
Mat guidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps);

//----------------------------【主函数】---------------------------
int main()
{
	//------------【1】读取源图像并检查图像是否读取成功------------  
	g_srcIamge = imread("C:\\Users\\March\\Desktop\\卫星解析\\03\\6-bayer.tif");
	const char* coutPath = "C:\\Users\\March\\Desktop\\卫星解析\\03\\6-bayer-en.tif";
	if (!g_srcIamge.data)
	{
		cout << "读取图片错误，请重新输入正确路径！\n";
		system("pause");
		return -1;
	}
	Mat dstMat;
	ALTMRetinex(g_srcIamge, dstMat);
	imwrite(coutPath, dstMat);

	//namedWindow("【源图像】", 1);//创建窗口
	//imshow("【源图像】", g_srcIamge);//显示窗口
	//------------【2】在WINDOWNAME窗口上分别创建滤波6个滑动条------------       
	//namedWindow(WINDOWNAME);//创建窗口	
	//createTrackbar("方框滤波", WINDOWNAME, &g_nBoxFilterValue, g_nMaxVal, on_BoxFilter);//创建方框滤波轨迹条
	//on_BoxFilter(g_nBoxFilterValue, 0);	//轨迹条的回调函数
	//createTrackbar("均值滤波", WINDOWNAME, &g_nMeanBlurValue, g_nMaxVal, on_MeanBlur);//创建均值滤波轨迹条
	//on_MeanBlur(g_nMeanBlurValue, 0);
	//createTrackbar("高斯滤波", WINDOWNAME, &g_nGaussianBlurValue, g_nMaxVal, on_GaussianBlur);//创建高斯滤波轨迹条
	//on_GaussianBlur(g_nGaussianBlurValue, 0);
	//createTrackbar("中值滤波", WINDOWNAME, &g_nMedianBlurValue, g_nMaxVal, on_MedianBlur);//创建中值滤波轨迹条
	//on_MedianBlur(g_nMedianBlurValue, 0);
	//createTrackbar("双边滤波", WINDOWNAME, &g_nBilateralFilterValue, g_nMaxVal, on_BilateralFilter);//创建双边滤波轨迹条
	//on_BilateralFilter(g_nBilateralFilterValue, 0);
	//createTrackbar("导向滤波", WINDOWNAME, &g_nGuidedFilterValue, g_nMaxVal, on_GuidedFilter);//创建导向滤波轨迹条
	//on_GuidedFilter(g_nGuidedFilterValue, 0);
	//------------【3】退出程序------------  
	//cout << "\t按下'q'键，退出程序~！\n" << endl;

	waitKey(0);
	return 0;
}

//----------------------【on_BoxFilter()函数】------------------------
static void on_BoxFilter(int, void*)
{
	boxFilter(g_srcIamge, g_dstImage1, -1, Size(g_nBoxFilterValue * 2 + 1, g_nBoxFilterValue * 2 + 1));
	cout << "\n当前为【盒式滤波】处理效果，其内核大小为：" << g_nBoxFilterValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage1);
}
//----------------------【on_MeanBlur()函数】------------------------
static void on_MeanBlur(int, void*)
{
	blur(g_srcIamge, g_dstImage2, Size(g_nMeanBlurValue * 2 + 1, g_nMeanBlurValue * 2 + 1), Point(-1, -1));
	cout << "\n当前为【均值滤波】处理效果，其内核大小为：" << g_nMeanBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage2);
}
//----------------------【on_GaussianBlur()函数】------------------------
static void on_GaussianBlur(int, void*)
{
	GaussianBlur(g_srcIamge, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	cout << "\n当前为【高斯滤波】处理效果，其内核大小为：" << g_nGaussianBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage3);
}
//----------------------【on_MedianBlur()函数】------------------------
static void on_MedianBlur(int, void*)
{
	medianBlur(g_srcIamge, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	cout << "\n当前为【中值滤波】处理效果，其内核大小为：" << g_nMedianBlurValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage4);
}
//----------------------【on_BilateralFilter()函数】------------------------
static void on_BilateralFilter(int, void*)
{
	bilateralFilter(g_srcIamge, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	cout << "\n当前为【双边滤波】处理效果，其内核大小为：" << g_nBilateralFilterValue << endl;
	imshow(WINDOWNAME, g_dstImage5);
}
//----------------------【on_GuidedFilter()函数】------------------------
static void on_GuidedFilter(int, void*)
{
	vector<Mat> vSrcImage, vResultImage;
	//【1】对源图像进行通道分离，并对每个分通道进行导向滤波操作
	split(g_srcIamge, vSrcImage);
	for (int i = 0; i < 3; i++)
	{
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//将分通道转换成浮点型数据
		Mat cloneImage = tempImage.clone();	//将tempImage复制一份到cloneImage
		Mat resultImage;
		guidedFilter(tempImage, cloneImage, resultImage, g_nGuidedFilterValue * 2 + 1, 0.01);//对分通道分别进行导向滤波
		vResultImage.push_back(resultImage);//将分通道导向滤波后的结果存放到vResultImage中
	}
	//【2】将分通道导向滤波后结果合并
	merge(vResultImage, g_dstImage6);
	cout << "\n当前处理为【导向滤波】，其内核大小为：" << g_nGuidedFilterValue * 2 + 1 << endl;
	imshow(WINDOWNAME, g_dstImage6);
}

//-------------------【实现导向滤波器函数部分】-------------------------
void guidedFilter(Mat &srcMat, Mat &guidedMat, Mat &dstImage, int radius, double eps)
{
	//------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
	srcMat.convertTo(srcMat, CV_64FC1);
	guidedMat.convertTo(guidedMat, CV_64FC1);
	//--------------【1】各种均值计算----------------------------------
	Mat mean_p, mean_I, mean_Ip, mean_II;
	boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//生成待滤波图像均值mean_p	
	boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//生成导向图像均值mean_I	
	boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//生成互相关均值mean_Ip
	boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//生成导向图像自相关均值mean_II
																				 //--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_II - mean_I.mul(mean_I);
	//---------------【3】计算参数系数a、b-------------------
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);
	//--------------【4】计算系数a、b的均值-----------------
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
	boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
	//---------------【5】生成输出矩阵------------------
	dstImage = mean_a.mul(srcMat) + mean_b;
}



void ALTMRetinex(const Mat& src, Mat &dst, bool LocalAdaptation, bool ContrastCorrect)
{

	Mat temp, src_gray;

	src.convertTo(temp, CV_32FC3);
	//灰度图
	cvtColor(temp, src_gray, COLOR_BGR2GRAY);

	double LwMax;
	//得到最大值
	minMaxLoc(src_gray, NULL, &LwMax);

	Mat Lw_;
	const int num = src.rows * src.cols;
	//计算每个数组元素绝对值的自然对数
	cv::log(src_gray + 1e-3f, Lw_);
	//矩阵自然指数
	float LwAver = exp(cv::sum(Lw_)[0] / num);

	Mat Lg;
	log(src_gray / LwAver + 1.f, Lg);
	//矩阵除法
	cv::divide(Lg, log(LwMax / LwAver + 1.f), Lg);

	//局部自适应
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
	// 白平衡系数纠正
	bgr[0] = bgr[0] * 1.4310;
	bgr[1] = bgr[1] * 0.9298;
	bgr[2] = bgr[2] * 1.0000;
	if (ContrastCorrect)
	{
		// 校正图像对比度
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

//导向滤波器
Mat guidedFilter(cv::Mat& I, cv::Mat& p, int r, float eps)
{
	/*
	× GUIDEDFILTER   O(N) time implementation of guided filter.
	×
	×   - guidance image: I (should be a gray-scale/single channel image)
	×   - filtering input image: p (should be a gray-scale/single channel image)
	×   - local window radius: r
	×   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_32FC1);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_32FC1);
	p = _p;

	//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4
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
