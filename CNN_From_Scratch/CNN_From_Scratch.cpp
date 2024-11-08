#include <iostream>
#include <vector>
#include "raylib.h"
#include "Raygui.h"
#include <thread> //<===Big plans


/*This project is just an educational look into a CNN as a part of a one week project.
-A simple line classifier is implemented below with simple parameters which can be tweaked to see the changing performance of the architecture.
-In no way is this a general use CNN model and is custom made for very simple inputs and predictions.
-For Complex CNN Architectures, one can refer to my github projects using TensorFlow*/


/*What you can expect in the future:
*1-More Datasets
*2-Features like Dropout Layer,Batch Normalisation, Different loss functions, Xavier Initialisation
*3-Batch and Mini Batch Gradient Descent
*4-More Flexibility based on parameters
*/

/*Upcoming:
*MNIST IMPLEMENTATION
*/

/*SOME PRE TESTED PARAMETERS FOR LEARNING
* Train Number=10000-20000
* Dataset size=50
* Learning rate=0.1-0.15
* Hidden layers=2
* Neurons= 50-70
* Filter size=3
* */

#define stride 1

//Non Editable Parameters
#define screenwidth 1200
#define screenheight 800
#define outputnodes 2


//Editable Parameters
double momentum = 0;
int totalimages;
int totaltestimages;
int imagedimensions;
int filtersize;
int inputnodes;
int hiddenlayers;
int hiddennodes;



using namespace std;

//CNN Part

double CNNLearningRate;

double** filtererror;

double** inputlayer;
int inputlayersize;
double** filter_1;
double** featuremap_1;
double** inputlayer_error;

double** poolinglayer_1;
int poolinglayer_1_size;
double** filter_2;
double** featuremap_2;
double** poolinglayer_1_error;
double** poolinginput_1_error;

double** poolinglayer_2;
int poolinglayer_2_size;
double** filter_3;
double** featuremap_3;
double** poolinglayer_2_error;
double** poolinginput_2_error;

double** rotated_filter_2;
double** rotated_filter_3;

vector<int> condition;
vector<int> testcondition;
vector<vector<vector<double>>> images;
vector<vector<vector<double>>> test;


//MLP Part

float learningrate;
double* inputs;
double** hiddenvalues;
double* outputvalues;

double** hiddenerror;
double* outputerror;
double* inputerror;

double** inputweights;
double*** hiddenweights;
double** outputweights;

double** hiddenbiases;
double outputbiases[outputnodes];

double costs[outputnodes];
double totalcost = 0;


//Raylib Part
enum screenstate {
	datasetselection,
	menu,
	datasetcreation,
	mlpsettings,
	simulationoptions,
	trainmodel,
	simulatemodel,
	testmodel,
} screenstate;

bool size3 = false;
bool size5 = false;
bool size7 = false;
bool size9 = false;;
bool threadchecker = false;
float accuracy=0;
int testedimages = 0;
thread createData;
thread createTest;
thread trainModel;
thread testModel;

//General Purpose Functions
void specialCreateArray(double**** arr, int thickness, int width, int height)
{
	*arr = (double***)malloc(sizeof(double**) * thickness);
	for (int y = 0; y < thickness; y++)
	{
		(*arr)[y] = (double**)malloc(sizeof(double*) * width);
		for (int x = 0; x < width; x++)
		{
			(*arr)[y][x] = (double*)malloc(sizeof(double) * height);
		}
	}
}

void complexCreateArray(double*** arr, int width, int height)
{
	*arr = (double**)malloc(sizeof(double*) * width);
	for (int x = 0; x < width; x++)
	{
		(*arr)[x] = (double*)malloc(sizeof(double) * height);
	}
}

void createArray(double*** arr, int n)
{
	*arr = (double**)malloc(sizeof(double*) * n);
	for (int x = 0; x < n; x++)
	{
		(*arr)[x] = (double*)malloc(sizeof(double) * n);
	}
}

void deleteArray(double*** arr, int n)
{
	for (int x = 0; x < n; x++)
	{
		free((*arr)[x]);
	}
	free(*arr);
}

void printArray(double*** arr, int n)
{
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			cout << (*arr)[x][y] << " ";
		}
	}
	cout << "\n\n\n";
}

//MLP Part
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x)
{
	return (double)(x * (1 - x));
}

double Relu(double x)
{
	return (x > (double)0) ? x : (double)0;
}

void softmax()
{
	double denom = 0;
	for (int x = 0; x < outputnodes; x++)
	{
		denom += exp(outputvalues[x]);
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputvalues[x] = exp(outputvalues[x]) / (double)denom;
	}
}

void initialiseBiases()
{
	for (int x = 0; x < hiddenlayers; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			hiddenbiases[x][y] = 0.01;
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputbiases[x] = 0.01;
	}
}

void initialiseWeights()
{
	double limit = 0.1;//sqrt(6 / (inputnodes + outputnodes));		//Xavier Initialisation


	for (int x = 0; x < inputnodes; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			inputweights[x][y] = limit * (double)(rand() / (double)RAND_MAX);
			if (rand() % 2 == 0)
				inputweights[x][y] = -inputweights[x][y];
		}
	}


	for (int x = 0; x < hiddenlayers - 1; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			for (int z = 0; z < hiddennodes; z++)
			{
				hiddenweights[x][y][z] = limit * (double)(rand() / (double)RAND_MAX);
				if (rand() % 2 == 0)
					hiddenweights[x][y][z] = -hiddenweights[x][y][z];
			}
		}
	}


	for (int x = 0; x < hiddennodes; x++)
	{
		for (int y = 0; y < outputnodes; y++)
		{
			outputweights[x][y] = limit * (double)(rand() / (double)RAND_MAX);
			if (rand() % 2 == 0)
				outputweights[x][y] = -outputweights[x][y];
		}
	}
}

void MLPforwardPass()
{
	if (hiddenlayers == 0)
		return;

	for (int x = 0; x < sqrt(inputnodes); x++)
	{
		for (int y = 0; y < sqrt(inputnodes); y++)
		{
			inputs[x*(int)sqrt(inputnodes)+y] = featuremap_3[x][y];
		}
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		double z = 0;
		for (int y = 0; y < inputnodes; y++)
		{
			z += inputweights[y][x] * inputs[y];
		}
		hiddenvalues[0][x] = Relu(z + hiddenbiases[0][x]);

	}


	for (int x = 1; x < hiddenlayers; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			double r = 0;
			for (int z = 0; z < hiddennodes; z++)
			{
				r += hiddenvalues[x - 1][z] * hiddenweights[x - 1][z][y];
			}
			hiddenvalues[x][y] = Relu(r + hiddenbiases[x][y]);
		}
	}

	for (int x = 0; x < outputnodes; x++)
	{
		double r = 0;
		for (int y = 0; y < hiddennodes; y++)
		{
			r += hiddenvalues[hiddenlayers - 1][y] * outputweights[y][x];
		}
		outputvalues[x] = sigmoid(r + outputbiases[x]);

	}

	//softmax();
}

void MLPbackPropogation(int epoch)
{
	totalcost = 0;
	int max = condition[epoch];

	if (max == 1)
	{
		costs[0] = 1-outputvalues[0];
		costs[1] = -outputvalues[1];
	}
	else
	{
		costs[0] = - outputvalues[0];
		costs[1] = 1 - outputvalues[1];
	}

	for (int x = 0; x < outputnodes; x++)
	{
		outputerror[x] = costs[x] * dsigmoid(outputvalues[x]);
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		double error = 0;
		for (int y = 0; y < outputnodes; y++)
		{
			error += outputerror[y] * outputweights[x][y];
		}
		hiddenerror[hiddenlayers - 1][x] = error+momentum* hiddenerror[hiddenlayers - 1][x]; //dsigmoid(hiddenvalues[hiddenlayers - 1][x]) *
	}

	for (int x = hiddenlayers - 2; x >= 0; x--)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			double error = 0;
			for (int z = 0; z < hiddennodes; z++)
			{
				error += hiddenerror[x + 1][z] * hiddenweights[x][y][z];
			}
			hiddenerror[x][y] = error+momentum* hiddenerror[x][y];// dsigmoid(hiddenvalues[x][y]) * error;
		}
	}

	for (int x = 0; x < inputnodes; x++)
	{
		double error = 0;
		for (int y = 0; y < hiddennodes; y++)
		{
			error += hiddenerror[0][y] * inputweights[x][y];
		}
		inputerror[x] = error+momentum*inputerror[x];
	}


	//Changes Values of Weights and biases
	for (int x = 0; x < outputnodes; x++)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			outputweights[y][x] += outputerror[x] * hiddenvalues[hiddenlayers - 1][y] * learningrate;
		}
		outputbiases[x] += outputerror[x] * learningrate;
	}

	for (int x = hiddenlayers - 1; x >= 1; x--)
	{
		for (int y = 0; y < hiddennodes; y++)
		{
			for (int z = 0; z < hiddennodes; z++)
			{
				hiddenweights[x - 1][z][y] += hiddenerror[x][y] * hiddenvalues[x - 1][z] * learningrate;
			}
			hiddenbiases[x][y] += hiddenerror[x][y] * learningrate;
		}
	}

	for (int x = 0; x < hiddennodes; x++)
	{
		for (int y = 0; y < inputnodes; y++)
		{
			inputweights[y][x] += hiddenerror[0][x] * inputs[y] * learningrate;
		}
		hiddenbiases[0][x] += hiddenerror[0][x] * learningrate;
	}

	for (int x = 0; x < sqrt(inputnodes); x++)
	{
		for (int y = 0; y < sqrt(inputnodes); y++)
		{
			poolinglayer_2_error[x][y] = inputerror[x*(int)sqrt(inputnodes) + y];
		}
	}
}


//CNN Part
void arrayRelu(double*** arr, int n)
{
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			double temp = (*arr)[x][y];
			(*arr)[x][y] = (temp > 0) ? temp : 0;
		}
	}
}

void initiatefilter()
{
	
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			int randomcheck = rand() % 2;
			filter_1[x][y] = 0.1*rand() / (double)RAND_MAX;

			randomcheck = rand() % 2;
			filter_2[x][y] = 0.1*rand() /(double)RAND_MAX;

			randomcheck = rand() % 2;
			filter_3[x][y] = 0.1*rand() / (double)RAND_MAX;
		}
	}
}

void createDataset(int gate,int num)
{
	for (int x = 0; x < num; x++)
	{
		vector<vector<double>> image;
		if (rand() % 2 == 1)
		{
			for (int y = 0; y < imagedimensions; y++)
			{
				vector<double> temp;
				for (int z = 0; z < imagedimensions; z++)
				{
					if (z > imagedimensions / 3 && z < 2 * imagedimensions / 3)
					{
						temp.push_back(1);
					}
					else
					temp.push_back(0);
				}
				image.push_back(temp);
			}

			if(gate==1)
				condition.push_back(1);
			else
				testcondition.push_back(1);
		}
		else
		{
			for (int y = 0; y < imagedimensions; y++)
			{
				vector<double> temp;
				for (int z = 0; z < imagedimensions; z++)
				{
					if (y > imagedimensions / 3 && y < 2 * imagedimensions / 3)
					{
						temp.push_back(1);
					}
					else
						temp.push_back(0);
				}
				image.push_back(temp);
			}

			if(gate==1)
				condition.push_back(0);
			else
				testcondition.push_back(0);
		}

		if(gate==1)
			images.push_back(image);
		else
			test.push_back(image);

	}
}

void loadpixels(int epoch)
{
	for (int x = 0; x < imagedimensions; x++)
	{
		for (int y = 0; y < imagedimensions; y++)
		{
			inputlayer[x][y]=images[epoch][x][y];
		}
	}
}

void loadtestpixels(int epoch)
{
	for (int x = 0; x < imagedimensions; x++)
	{
		for (int y = 0; y < imagedimensions; y++)
		{
			inputlayer[x][y] = test[epoch][x][y];
		}
	}
}

void convolve(double***destination,double** layer,int n, double** filter,int m)
{
	for (int x = 0; x < n - m+1; x++)
	{
		for (int y = 0; y < n - m+1; y++)
		{
			double sum = 0;
			for (int a = 0; a < m; a++)
			{
				for (int b = 0; b < m; b++)
				{
					sum += layer[x + a][y + b] * filter[a][b];
				}
			}
			(*destination)[x][y] = sum;
		}
	}
}

void pool(double*** poolinglayer,double** layer,int n, int shrink)
{

	for (int x = 0; x < n; x+=shrink)
	{
		for (int y = 0; y < n; y+=shrink)
		{
			double max = -100000;
			for (int a = 0; a < shrink; a++)
			{
				for (int b = 0; b < shrink; b++)
				{
					if (layer[x + a][y + b] > max)
					{
						max = layer[x + a][y + b];
					}
				}
			}

			(*poolinglayer)[x/shrink][y/shrink] = max;
		}
	}
}

void pad(double*** paddedlayer,int size,double** arr,int n, int padding)
{
	deleteArray(paddedlayer,size);
	createArray(paddedlayer, n + 2*padding);

	for (int x = 0; x < n + 2*padding; x++)
	{
		for (int y = 0; y < n + 2*padding; y++)
		{
			(*paddedlayer)[x][y] = 0;
		}
	}

	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			(*paddedlayer)[x + padding][y + padding] = arr[x][y];
		}
	}
}

void rotate(double*** arr,double** reference, int n)
{
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			(*arr)[n - x - 1][n - y - 1] = reference[x][y];
		}
	}
}                 

void fit(double** referencearray,double*** mainarray,int n, double*** fitarray,int m)
{
	int poolingstride = n / m;

	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			(*mainarray)[x][y] = 0;
		}
	}

	for (int x = 0; x < n; x+=poolingstride)
	{
		for (int y = 0; y < n; y += poolingstride)
		{
			short int xindex = 0;
			short int yindex = 0;
			for (int a = 0; a< poolingstride; a++)
			{
				for (int b = 0; b < poolingstride; b++)
				{
					if (referencearray[x + a][y + b] > referencearray[x + xindex][y + yindex])
					{
						xindex = a;
						yindex = b;
					}
				}
			}
			(*mainarray)[x + xindex][y + yindex] = (*fitarray)[x / poolingstride][y / poolingstride];
		}
	}
}

void updateFilter(double*** filter, double** updatematrix,int n)
{
	for (int x = 0; x < n; x++)
	{
		for (int y = 0; y < n; y++)
		{
			(*filter)[x][y] += CNNLearningRate * updatematrix[x][y];
		}
	}
}

void forwardPass()
{

	convolve(&featuremap_1,inputlayer,inputlayersize,filter_1,filtersize);
	//cout << "CONVOLVED FEATURE MAP 1:" << endl;
	//printArray(&featuremap_1, 76);
	arrayRelu(&featuremap_1, inputlayersize-filtersize+1);
	//cout << "RELU FEATURE MAP 1:" << endl;
	//printArray(&featuremap_1, 76);
	
	pool(&poolinglayer_1,featuremap_1, inputlayersize - filtersize + 1, 2);
	//cout << "POOLED FEATURE MAP 1:" << endl;
	//printArray(&poolinglayer_1, 38);
	convolve(&featuremap_2,poolinglayer_1, poolinglayer_1_size, filter_2,filtersize);
	//cout << "CONVOLVED FEATURE MAP 2:" << endl;
	//printArray(&featuremap_2, 36);
	arrayRelu(&featuremap_2, poolinglayer_1_size-filtersize+1);
	//cout << "RELU FEATURE MAP 2:" << endl;
	//printArray(&featuremap_2, 36);
	
	pool(&poolinglayer_2,featuremap_2,poolinglayer_1_size-filtersize+1, 2);
	//cout << "POOLED FEATURE MAP 2:" << endl;
	//printArray(&poolinglayer_2, 18);
	convolve(&featuremap_3,poolinglayer_2, poolinglayer_2_size, filter_3,filtersize);
	//cout << "CONVOLVED FEATURE MAP 3:" << endl;
	//printArray(&featuremap_3, 16);
	arrayRelu(&featuremap_3,poolinglayer_2_size-filtersize+1 );
	//cout << "RELU FEATURE MAP 3:" << endl;
	//printArray(&featuremap_3, 16);
}

void backwardPass()
{
	double** paddedlayer;

	createArray(&paddedlayer, 1);

	convolve(&filtererror,poolinglayer_2,poolinglayer_2_size, poolinglayer_2_error,poolinglayer_2_size-filtersize+1);
	rotate(&rotated_filter_3,filter_3, filtersize);
	pad(&paddedlayer,1,poolinglayer_2_error, poolinglayer_2_size - filtersize + 1, filtersize - 1);
	convolve(&poolinginput_2_error,paddedlayer, poolinglayer_2_size +filtersize - 1, filter_3,filtersize);
	fit(featuremap_2,&poolinglayer_1_error,poolinglayer_1_size-filtersize+1, &poolinginput_2_error,poolinglayer_2_size);


	updateFilter(&filter_3, filtererror,filtersize);


	convolve(&filtererror,poolinglayer_1, poolinglayer_1_size, poolinglayer_1_error, poolinglayer_1_size-filtersize+1);
	rotate(&rotated_filter_2,filter_2, filtersize);
	pad(&paddedlayer,poolinglayer_2_size+filtersize-1,poolinglayer_1_error, poolinglayer_1_size-filtersize+1, filtersize - 1);
	convolve(&poolinginput_1_error,paddedlayer, poolinglayer_1_size+filtersize-1, filter_2, filtersize);
	fit(featuremap_1, &inputlayer_error, inputlayersize - filtersize + 1, &poolinginput_1_error, poolinglayer_1_size);


	updateFilter(&filter_2, filtererror,filtersize);


	convolve(&filtererror,inputlayer, inputlayersize, inputlayer_error, inputlayersize-filtersize+1);


	updateFilter(&filter_1, filtererror,filtersize);
	
}

void initializeArrays()
{
	createArray(&filter_1, filtersize);
	createArray(&filter_2, filtersize);
	createArray(&filter_3, filtersize);

	inputlayersize = imagedimensions;

	createArray(&inputlayer, inputlayersize);
	createArray(&featuremap_1, inputlayersize-filtersize+1);
	createArray(&inputlayer_error, inputlayersize-filtersize+1);

	poolinglayer_1_size = (imagedimensions - filtersize + 1)/2;

	createArray(&poolinglayer_1, poolinglayer_1_size);
	createArray(&featuremap_2, poolinglayer_1_size -filtersize+1);
	createArray(&poolinglayer_1_error, poolinglayer_1_size -filtersize+1);
	createArray(&poolinginput_1_error, poolinglayer_1_size);

	poolinglayer_2_size = (poolinglayer_1_size -filtersize+1)/2;

	createArray(&poolinglayer_2, poolinglayer_2_size);
	createArray(&featuremap_3, poolinglayer_2_size -filtersize+1);
	createArray(&poolinglayer_2_error, poolinglayer_2_size -filtersize+1);
	createArray(&poolinginput_2_error, poolinglayer_2_size);

	createArray(&filtererror, filtersize);
	createArray(&rotated_filter_2, filtersize);
	createArray(&rotated_filter_3, filtersize);

	inputnodes = pow(poolinglayer_2_size - filtersize + 1, 2);

	inputs = (double*)malloc(sizeof(double) * inputnodes);
	complexCreateArray(&hiddenvalues,hiddenlayers,hiddennodes);
	outputvalues = (double*)malloc(sizeof(double) * outputnodes);

	complexCreateArray(&hiddenerror,hiddenlayers,hiddennodes);
	outputerror = (double*)malloc(sizeof(double) * outputnodes);
	inputerror = (double*)malloc(sizeof(double) * inputnodes);

	complexCreateArray(&inputweights,inputnodes,hiddennodes);
	specialCreateArray(&hiddenweights,hiddenlayers - 1,hiddennodes,hiddennodes);
	complexCreateArray(&outputweights,hiddennodes,outputnodes);

	complexCreateArray(&hiddenbiases,hiddenlayers,hiddennodes);

}


//Model
void initializeModel()
{
	initiatefilter();

	initialiseWeights();
	initialiseBiases();
}

//Raylib
void checkfiltervalidity(int dims,bool* fil3,bool*fil5,bool*fil7,bool*fil9)
{
	if (dims % 2 == 1)
	{
		*fil3 = false;
		*fil5 = false;
		*fil7 = false;
		*fil9 = false;
		return;
	}

	for (int x = 3; x <= 9; x += 2)
	{
		int checker = (((dims - x + 1) / 2) - x + 1);
		if (checker % 2 == 0)
		{
			switch (x)
			{
			case 3:
				if (checker / 2 - 2 >= 1)
					*fil3 = true;
				else
					*fil3 = false;
				break;
			case 5:
				if (checker / 2 - 4 >= 1)
					*fil5 = true; 
				else
				*fil5 = false;
				break;
			case 7:
				if (checker / 2 - 6 >= 1)
					*fil7 = true; 
				else
				*fil7 = false;
				break;
			case 9:
				if (checker / 2 - 8 >= 1)
					*fil9 = true; 
				else
				*fil9 = false;
				break;
			}
		}
		else
		{
			switch (x)
			{
			case 3:*fil3 = false;
				break;
			case 5:*fil5 = false;
				break;
			case 7:*fil7 = false;
				break;
			case 9:*fil9 = false;
				break;
			}
		}
	}
}

void setFilterCheckboxes(bool* filter3, bool* filter5, bool* filter7, bool* filter9)
{
	if (size3 == false)
	{
		*filter3 = false;
	}
	if (size5 == false)
	{
		*filter5 = false;
	}
	if (size7== false)
	{
		*filter7 = false;
	}
	if (size9 == false)
	{
		*filter9 = false;
	}
}

void modelTrain(int* totalepochs)
{
	int success = 0;

	for (int epoch = 0; epoch < totalimages; epoch++)
	{
		loadpixels(epoch);

		forwardPass();
		MLPforwardPass();
		MLPbackPropogation(epoch);
		backwardPass();

		if (outputvalues[0] > outputvalues[1] && condition[epoch] == 1)
			success += 1;
		else if (outputvalues[1] > outputvalues[0] && condition[epoch] == 0)
			success += 1;

		*totalepochs += 1;

		if (*totalepochs % 1000 == 0)
		{
			accuracy = success / (float)10;
			success = 0;
		}
	}
}

void modelTest()
{
	int success = 0;
	for (int epoch = 0; epoch < totaltestimages; epoch++)
	{
		loadtestpixels(epoch);

		forwardPass();
		MLPforwardPass();

		if (outputvalues[0] > outputvalues[1] && testcondition[epoch] == 1)
			success += 1;
		else if (outputvalues[1] > outputvalues[0] && testcondition[epoch] == 0)
			success += 1;
		testedimages += 1;

		accuracy = (success*100) / (float)(testedimages);
	}
}

void testLineChecker(int* totalepochs)
{
	if (threadchecker == false)
	{
		accuracy = 0;
		testModel = thread(modelTest);
		testModel.detach();
		threadchecker = true;
	}
	
	if (testedimages == totaltestimages)
	{
		static Rectangle Frame = { 400,150,400,400 };
		static int casenumber=0;

		loadpixels(casenumber);
		forwardPass();
		MLPforwardPass();

		//Frame
		DrawRectangleLinesEx(Frame, 10, DARKGRAY);

		//Test Image
		if (testcondition[casenumber] == 1)
		{
			DrawLineEx({ Frame.x + Frame.width / 2 - 5, Frame.y + 10 }, { Frame.x + Frame.width / 2 - 5, Frame.y + Frame.height - 10 }, 20, WHITE);
		}
		else
		{
			DrawLineEx({ Frame.x + 10, Frame.y + Frame.height / 2 - 5, }, { Frame.x + Frame.width - 10, Frame.y + Frame.height / 2 - 5 }, 20, WHITE);
		}

		//Probabilities
		DrawText(TextFormat("CASE #%i",casenumber), 530, 80, 30, WHITE);

		DrawText("Vertical Line Probability", 300, 600, 20, WHITE);
		DrawText("Horizontal Line Probability", 630, 600, 20, WHITE);

		DrawText(TextFormat("%f",outputvalues[0]), 380, 650, 20, WHITE);
		DrawText(TextFormat("%f", outputvalues[1]), 700, 650, 20, WHITE);

		//Interaction
		if (GuiButton({ 900,300,100,100 }, "NEXT")&&casenumber<totaltestimages)
			casenumber += 1;

		if (GuiButton({ 200,300,100,100 }, "PREVIOUS")&&casenumber>0)
			casenumber -= 1;

		if (outputvalues[0] > outputvalues[1] && testcondition[casenumber] == 1 || outputvalues[0]<outputvalues[1] && testcondition[casenumber] == 0)
		{
			DrawText("SUCCESS", 545, 700, 20, GREEN);
		}
		else
			DrawText("FAILURE", 545, 700, 20, RED);

		if (GuiButton({ screenwidth - 200,screenheight - 150,100,50 }, "Change Parameters"))
		{
			threadchecker = false;
			*totalepochs = 0;
			accuracy = 0;
			testedimages = 0;
			screenstate = mlpsettings;
		}

		if (GuiButton({ screenwidth - 200,screenheight - 250,100,50 }, "Back"))
		{
			threadchecker = false;
			testedimages = 0;
			screenstate = simulationoptions;
		}
	}
	else
	{
		DrawText(TextFormat("Test Data Accuracy= %f", accuracy), 450, 350, 20, WHITE);
	}
}

void datasetSelectionScreen()
{
	ClearBackground(BLACK);

	//Frames
	DrawRectangleLinesEx({ 60,200,300,300 }, 10, WHITE);
	DrawRectangleLinesEx({ 460,200,300,300 }, 10, WHITE);
	DrawRectangleLinesEx({ 860,200,300,300 }, 10, WHITE);

	//Images/Text
	DrawRectangleRec({ 70,210,280,280 }, DARKGREEN);
	DrawRectangleRec({ 470,210,280,280 }, GRAY);
	DrawRectangleRec({ 870,210,280,280 }, GRAY);

	DrawText("Line Alignment Checker", 80, 335, 23, WHITE);
	DrawText("Coming Soon!", 540, 335, 23, WHITE);
	DrawText("Coming Soon!", 940, 335, 23, WHITE);

	//Buttons

	if (GuiButton({ 100,550,220,70 }, "PROCEED"))
	{
		screenstate = menu;
	}

}

void menuScreen()
{
	ClearBackground(BLACK);

	static float datasetsizeval;
	static float testdatasetsizeval;
	static float imagedimensionval;

	//GUI
	DrawText("MENU", screenwidth / 2 - 120, screenheight / 4 - 120, 80, WHITE);
	DrawText("Enter Dataset Number:", screenwidth / 2 - 180, screenheight / 2 - 170, 30, WHITE);
	DrawText("Enter Test Dataset Number:", screenwidth / 2 - 220, screenheight / 2 - 20, 30, WHITE);
	DrawText("Enter Image Dimensions:", screenwidth / 2 - 190, screenheight / 2 + 125, 30, WHITE);

	DrawText(TextFormat("%d", (int)datasetsizeval), screenwidth / 2 - 40, screenheight / 2 - 75, 20, WHITE);
	DrawText(TextFormat("%d", (int)testdatasetsizeval), screenwidth / 2 - 40, screenheight / 2 + 80, 20, WHITE);
	DrawText(TextFormat("%d", (int)imagedimensionval), screenwidth / 2 - 20, screenheight / 2 + 225, 20, WHITE);

	DrawText("Valid For Filter Size:", screenwidth / 3, screenheight / 2 + 300, 20, GRAY);

	//Interaction
	GuiSlider({ 200,screenheight / 2 - 120 ,800,30 }, "1", "50000", &datasetsizeval, 1, 50000);
	GuiSlider({ 200,screenheight / 2 + 30 ,800,30 }, "1", "50000", &testdatasetsizeval, 1, 50000);
	GuiSlider({ 200,screenheight / 2 + 175,800,30 }, "10", "100", &imagedimensionval, 10, 100);
	imagedimensionval = ((int)imagedimensionval / 2) * 2;			//STEP SIZE=2

	GuiCheckBox({ screenwidth / 2 + 40,screenheight / 2 + 298,20,20 }, "3", &size3);
	GuiCheckBox({ screenwidth / 2 + 80,screenheight / 2 + 298,20,20 }, "5", &size5);
	GuiCheckBox({ screenwidth / 2 + 120,screenheight / 2 + 298,20,20 }, "7", &size7);
	GuiCheckBox({ screenwidth / 2 + 160,screenheight / 2 + 298,20,20 }, "9", &size9);

	if (GuiButton({ 4 * screenwidth / 5 - 20 ,50,200,100 }, "START"))
	{
		if (size3 + size5 + size7 + size9 > 0)
		{
			imagedimensions = imagedimensionval;
			totalimages = datasetsizeval;
			totaltestimages = testdatasetsizeval;
			screenstate = datasetcreation;
		}
	}

	//Checks filter validity
	checkfiltervalidity((int)imagedimensionval, &size3, &size5, &size7, &size9);
}

void datasetCreationScreen()
{
	ClearBackground(BLACK);
	DrawText("CREATING DATASET", screenwidth / 2 - 270, screenheight / 2 - 50, 50, WHITE);

	DrawText(TextFormat("%d", images.size()), 350, 500, 30, WHITE);
	DrawText(TextFormat("out of %d Train Images", totalimages), 450, 500, 30, WHITE);

	DrawText(TextFormat("%d", test.size()), 350, 550, 30, WHITE);
	DrawText(TextFormat("out of %d Test Images", totaltestimages), 450, 550, 30, WHITE);

	if (threadchecker == false)
	{
		createData = thread(createDataset, 1, totalimages);
		createTest = thread(createDataset, 0, totaltestimages);
		createData.detach();
		createTest.detach();
		threadchecker = true;
	}
	if (totalimages == images.size() && totaltestimages == test.size())
	{
		screenstate = mlpsettings;
		threadchecker = false;
	}
}

void mlpSettingsScreen()
{
	ClearBackground(BLACK);

	static float modellearningrate;
	static float modellayernumber;
	static float layerneuronnumber;
	static float modelfiltersize;
	static bool filter3, filter5, filter7, filter9;

	setFilterCheckboxes(&filter3, &filter5, &filter7, &filter9);

	//GUI
	DrawText("Multi Layer Percepton Configuration", 250, 100, 40, WHITE);
	DrawText("Set Initial Learning Rate:", screenwidth / 2 - 130, screenheight / 3 - 55, 20, WHITE);
	DrawText(TextFormat("%f", modellearningrate), screenwidth / 2 - 30, screenheight / 3 + 55, 20, WHITE);
	DrawText("Number of Hidden Layers:", screenwidth / 2 - 130, screenheight / 3 + 105, 20, WHITE);
	DrawText(TextFormat("%d", (int)modellayernumber), screenwidth / 2 - 10, screenheight / 3 + 205, 20, WHITE);
	DrawText("Neurons Per Layer:", screenwidth / 2 - 100, screenheight / 3 + 255, 20, WHITE);
	DrawText(TextFormat("%d", (int)layerneuronnumber), screenwidth / 2 - 10, screenheight / 3 + 355, 20, WHITE);
	DrawText("Filter Size:", 2 * screenwidth / 10 - 20, screenheight - 85, 20, WHITE);

	//Interaction
	GuiSlider({ 200, screenheight / 3 + -5 ,800,30 }, "0", "1", &modellearningrate, 0, 1);
	GuiSlider({ 200, screenheight / 3 + 155,800,30 }, "2", "10", &modellayernumber, 2, 10);
	GuiSlider({ 200, screenheight / 3 + 305,800,30 }, "1", "500", &layerneuronnumber, 1, 500);
	GuiCheckBox({ 3 * screenwidth / 10 + 20,screenheight - 100,50,50 }, "3", &filter3);
	GuiCheckBox({ 4 * screenwidth / 10 + 20,screenheight - 100,50,50 }, "5", &filter5);
	GuiCheckBox({ 5 * screenwidth / 10 + 20,screenheight - 100,50,50 }, "7", &filter7);
	GuiCheckBox({ 6 * screenwidth / 10 + 20,screenheight - 100,50,50 }, "9", &filter9);

	if (GuiButton({ screenwidth - 150,screenheight - 100,100,50 }, "CONTINUE"))
	{
		CNNLearningRate = (double)modellearningrate;
		learningrate = modellearningrate;
		hiddennodes = (int)layerneuronnumber;
		hiddenlayers = (int)modellayernumber;

		if (filter3 == true)
		{
			filtersize = 3;
		}
		else if (filter5 == true)
		{
			filtersize = 5;
		}
		else if (filter7 == true)
		{
			filtersize = 7;
		}
		else if (filter9 == true)
			filtersize = 9;

		screenstate = simulationoptions;

		initializeArrays();
	}
}

void simulationOptionsScreen()
{
	ClearBackground(BLACK);

	DrawText("CHOOSE MODE", 490, 250, 30, WHITE);

	if (GuiButton({ 300,400,200,100 }, "Train Model"))
	{
		initializeModel();
		screenstate = trainmodel;
	}

	if (GuiButton({ 700,400,200,100 }, "COMING SOON!"))
	{
		//screenstate = simulate;
	}

	if (GuiButton({ 500,600,200,100 }, "Test Model"))
	{
		screenstate = testmodel;
	}
}

void trainModelScreen(int* totalepochs)
{
	ClearBackground(BLACK);
	if (threadchecker == false)
	{
		trainModel = thread(modelTrain, totalepochs);
		trainModel.detach();
		threadchecker = true;
	}
	DrawText("Weights Initialised", 500, 100, 20, WHITE);
	DrawText("Biases Initialised", 500, 150, 20, WHITE);
	DrawText("Filters Initialised", 500, 200, 20, WHITE);

	DrawText("Accuracy: ", 300, 400, 30, WHITE);
	DrawText(TextFormat("%f", accuracy), 570, 400, 30, WHITE);

	DrawText("Loss: ", 300, 450, 30, WHITE);
	DrawText(TextFormat("%f", outputerror[0]), 570, 450, 30, WHITE);

	if (GuiButton({ screenwidth - 200,screenheight - 150,100,50 }, "Change Parameters"))
	{
		threadchecker = false;
		*totalepochs = 0;
		accuracy = 0;
		screenstate = mlpsettings;
	}

	if (GuiButton({ screenwidth - 200,screenheight - 250,100,50 }, "Test Model"))
	{
		threadchecker = false;
		screenstate = testmodel;
	}
}

//Main
int main()
{
	InitWindow(screenwidth, screenheight, "Convolutional Neural Network");
	SetTargetFPS(144);

	srand(time(NULL));

	//SCREENSTATE
	
	screenstate = datasetselection;

	//MOUSEPOS
	Vector2 Mousepos;
	
	//INITIALISATION
	int totalepochs = 0;

	while (!WindowShouldClose())
	{
		Mousepos = GetMousePosition();

		BeginDrawing();

		switch (screenstate)
		{
		case datasetselection://DATASET SELECTION
			datasetSelectionScreen();
			break;

		case menu://MENU
			menuScreen();
			break;

		case datasetcreation://DATASET LOADING
			datasetCreationScreen();
			break;

		case mlpsettings://MLP CONFIGURATION
			mlpSettingsScreen();
			break;

		case simulationoptions://TRAINING OR SIMULATION CHECKPOINT
			simulationOptionsScreen();
			break;

		case trainmodel://TRAINING
			trainModelScreen(&totalepochs);
			break;

		case simulatemodel://SIMULATION
			ClearBackground({ 0,20,100,55 });
			break;

		case testmodel://TEST
			ClearBackground(BLACK);
			testLineChecker(&totalepochs);
			break;
		}
		
		EndDrawing();
	}
}
