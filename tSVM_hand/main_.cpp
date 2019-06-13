#include <stdio.h>  
#include <time.h>  
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h> 
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/ml/ml.hpp>  
#include <dirent.h>
#include <vector>
#include <string.h>
using namespace std; 
using namespace cv;
using namespace cv::ml;
void getFiles( string path, vector<string>& files); 
void get_1(Mat& trainingImages, vector<int>& trainingLabels); 
void get_0(Mat& trainingImages, vector<int>& trainingLabels); 
void get_2(Mat& trainingImages, vector<int>& trainingLabels); 
void getdata();
Size imageSize = Size(48,48);
void coumputeHog(const Mat& src, vector<float> & descriptors)
{
	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	myHog.compute(src.clone(),descriptors,Size(1,1),Size(0,0));

}


void train_hog()
{

	HOGDescriptor *myHog=new HOGDescriptor(cvSize(12,24),cvSize(8,8),cvSize(4,4),cvSize(4,4),9); 
	Ptr<SVM> mySVM = SVM::create();
	mySVM->setCoef0(0.0);
	mySVM->setType(SVM::C_SVC);
	mySVM->setKernel(SVM::LINEAR);
	mySVM->setDegree(1);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50000, 1e-6);
	mySVM->setGamma(0);
	mySVM->setP(0.1);
	mySVM->setC(0.01);
    string path1 = "/home/lk/projects/svm_hand/0";
    string path2 = "/home/lk/projects/svm_hand/2";
    string path3 = "/home/lk/projects/svm_hand/5";
	Mat sampleFeatureMat;//储存hog特征
	Mat sampleLabelMat;
	vector<int> label;
	vector<string> file1;
	vector<string> file2;
	vector<string> file3;
	getFiles(path1,file1);
	getFiles(path2,file2);
	getFiles(path3,file3);
	cout<<file1.size()<<"  "<<file2.size()<<" "<<file3.size()<<endl;
	int sum = file1.size()+file2.size()+file3.size();
	
	for (int i=0;i<file1.size();i++)
	{
	    vector<float> descriptors;
		Mat image = imread(file1[i]);
		if (image.empty())
			continue;
		Mat image__;
		cvtColor(image,image__,COLOR_BGR2GRAY);
		resize(image__,image__,imageSize);
		myHog->compute(image__,descriptors,Size(4,4));
		if (i==0)
		{
			int Des_dim = descriptors.size();
			sampleFeatureMat = Mat::zeros(sum,Des_dim,CV_32FC1);
			sampleLabelMat = Mat::zeros(sum,1,CV_32FC1);
		}
		for (int j=0;j<descriptors.size();j++)
		{
			sampleFeatureMat.at<float>(i,j) = descriptors[j];
		}
		label.push_back(0);
	}
	for (int i=0;i<file2.size();i++)
	{
	    vector<float> descriptors;
		Mat image = imread(file2[i]);
		if (image.empty())
			continue;
	    Mat image_;	
		cvtColor(image,image_,COLOR_BGR2GRAY);
		resize(image_,image_,imageSize);
		myHog->compute(image_,descriptors,Size(4,4));
		for (int j=0;j<descriptors.size();j++)
		{
			sampleFeatureMat.at<float>(i+file1.size(),j) = descriptors[j];
		}
		label.push_back(1);
	}
	for (int i=0;i<file3.size();i++)
	{
		vector<float> descriptors;
		Mat image = imread(file3[i]);
		if (image.empty())
			continue;
	    Mat image_;	
		cvtColor(image,image_,COLOR_BGR2GRAY);
		resize(image_,image_,imageSize);
		myHog->compute(image_,descriptors,Size(4,4));
		for (int j=0;j<descriptors.size();j++)
		{
			sampleFeatureMat.at<float>(i+file1.size()+file2.size(),j) = descriptors[j];
		}
		label.push_back(2);
	}
	cout<<"  "<<sampleFeatureMat.rows<<endl;
	Mat(label).copyTo(sampleLabelMat);
	mySVM->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat); //保存模型
	mySVM->save("svm2.xml");
	
	
	
	
	
	
	vector<string> file4;
	string path4 = "/home/lk/projects/svm_hand/test";
	getFiles(path4,file4);
	cout<<file4.size()<<endl;
	Ptr<SVM> svm = SVM::create();
	svm = mySVM->load("svm2.xml");
	for (int i=0;i<file4.size();i++)
	{
		Mat testimg = imread(file4[i]);
		Mat test_;
		cvtColor(testimg,test_,COLOR_BGR2GRAY);
		resize(test_,test_,imageSize);
		vector<float> descriptors;
		myHog->compute(test_,descriptors,Size(4,4));//计算HOG描述子，检测窗口移动步长(8,8)
		Mat testFeatureMat = Mat::zeros(1,descriptors.size(),CV_32FC1);//测试样本的特征向量矩阵
		for(int i=0; i<descriptors.size(); i++)
		{
			testFeatureMat.at<float>(0,i) = descriptors[i];
		}
		int result = svm->predict(testFeatureMat);//返回类标
        cout<<"result is: "<<result<<endl;
		getchar();
	}
	cout<<"over"<<endl;

}


void no_feature()
{
	Mat classes; 
	Mat trainingData; 
	Mat trainingImages; 
	vector<int> trainingLabels;
	get_0(trainingImages, trainingLabels); 
	get_1(trainingImages, trainingLabels); 
	get_2(trainingImages, trainingLabels); 
	Mat(trainingImages).copyTo(trainingData); 
	trainingData.convertTo(trainingData, CV_32FC1); 
	Mat(trainingLabels).copyTo(classes); //配置SVM训练器参数
	Ptr<SVM> mySVM = SVM::create();
	mySVM->setCoef0(0.0);
	mySVM->setType(SVM::C_SVC);
	mySVM->setKernel(SVM::LINEAR);
	mySVM->setDegree(3);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50000, 1e-6);
	mySVM->setGamma(0);
	mySVM->setP(0.1);
	mySVM->setC(0.01);

	mySVM->train(trainingData, ROW_SAMPLE, classes); //保存模型 
	mySVM->save("svm.xml"); 
	cout<<"训练好了！！！"<<endl;
    
	Ptr<SVM> svm = SVM::create();
    svm = mySVM->load("svm.xml");
	string path1 = "/home/lk/project/opencv_practice/practice/svm_digit/test";
	vector<string> strtr; 
	getFiles(path1,strtr);
	cout<<strtr.size()<<endl;
	int num_ = strtr.size();
	for (int i=0;i<num_;i++)
	{
		Mat sr = imread(strtr[i]);
		Mat sr_gray;
		//cvtColor(sr,sr_gray,COLOR_BGR2GRAY);
		cout<<strtr[i]<<endl;
		sr.convertTo(sr_gray,CV_32F);
		sr_gray= sr_gray.reshape(1, 1);
		int res = svm->predict(sr_gray);
		cout<<res<<endl;
		getchar();
	}
}


int main() 
{ //获取训练数据
	//getdata();//这个没用
	train_hog();//hog+svm手势识别
	//no_feature();//基于svm手势识别
	return 0; 
} 

void getdata()
{
	char ad[128]={0}; 
	int filename = 0,filenum=0;
	Mat img = imread("../digits.png"); 
	Mat gray; cvtColor(img, gray, CV_BGR2GRAY); 
	int b = 20; 
	int m = gray.rows / b; //原图为1000*2000 
	int n = gray.cols / b; //裁剪为5000个20*20的小图块 
	int i=0;
	int index=1;
	for (int i = 0; i < m; i++) 
	{ 
		int offsetRow = i*b; //行上的偏移量 
		if(i%5==0&&i!=0) 
		{ 
			filename++; 
			filenum=0; 
		} 
		for (int j = 0; j < n; j++) 
		{ 
			int offsetCol = j*b; //列上的偏移量
			//string ad = format("%6d",index++) + ".jpg";
			sprintf(ad, "/home/lk/project/opencv_practice/practice/svm_digit/train_image/1/%d.jpg",index++); //截取20*20的小块 
			Mat tmp; 
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp); 
			imwrite(ad,tmp); 
		} 
	}

}

void  getFiles(const string path, vector<string>& filename)//返回文件中的名称
{
	DIR *pDir;
	struct dirent* ptr;
	if(!(pDir = opendir(path.c_str())))
		return;
	while((ptr = readdir(pDir))!=0) {
		if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
			filename.push_back(path + "/" + ptr->d_name);
	}
	closedir(pDir);
	//return true;
}

void get_0(Mat& trainingImages, vector<int>& trainingLabels) 
{ 
	string filePath = "/home/lk/project/opencv_practice/practice/svm_digit/0";
	vector<string> files; 
	getFiles(filePath, files ); 
	int number = files.size();
	for (int i = 0;i < number;i++) 
	{ 
		Mat SrcImage=imread(files[i].c_str()); 
		SrcImage= SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage); 
		trainingLabels.push_back(0); 
	} 
}




void get_1(Mat& trainingImages, vector<int>& trainingLabels) 
{ 
	string filePath = "/home/lk/project/opencv_practice/practice/svm_digit/2";
	vector<string> files; 
	getFiles(filePath, files ); 
	int number = files.size();
	for (int i = 0;i < number;i++) 
	{ 
		Mat SrcImage=imread(files[i].c_str()); 
		SrcImage= SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage); 
		trainingLabels.push_back(1); 
	} 
} 
void get_2(Mat& trainingImages, vector<int>& trainingLabels) 
{ 
	string filePath = "/home/lk/project/opencv_practice/practice/svm_digit/5";
	vector<string> files; 
	getFiles(filePath, files ); 
	int number = files.size(); 
	for (int i = 0;i < number;i++)
	{
		Mat SrcImage=imread(files[i].c_str()); 
		SrcImage= SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(2); 
	} 
}
