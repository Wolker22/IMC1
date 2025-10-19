/*********************************************************************
 * File  : MultilayerPerceptron.cpp
 * Date  : 2020
 *********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
	: layers(nullptr),
	  nOfLayers(0),
	  eta(0.1), // tasa de aprendizaje por defecto
	  mu(0.9)	// momento por defecto
{
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	// Por si venimos de una inicialización previa
	if (this->layers)
		freeMemory();

	this->nOfLayers = nl;
	this->layers = (Layer *)malloc(sizeof(Layer) * nl);
	if (!this->layers)
		return 0;

	for (int l = 0; l < nl; ++l)
	{
		this->layers[l].nOfNeurons = npl[l];
		this->layers[l].neurons = (Neuron *)malloc(sizeof(Neuron) * npl[l]);
		if (!this->layers[l].neurons)
			return 0;

		const int nPrev = (l == 0) ? 0 : npl[l - 1];
		const int nWeights = (l == 0) ? 0 : (nPrev + 1); // +1 por el bias

		for (int j = 0; j < npl[l]; ++j)
		{
			Neuron &neu = this->layers[l].neurons[j];
			neu.out = 0.0;
			neu.delta = 0.0;

			if (nWeights > 0)
			{
				neu.w = (double *)malloc(sizeof(double) * nWeights);
				neu.deltaW = (double *)malloc(sizeof(double) * nWeights);
				neu.lastDeltaW = (double *)malloc(sizeof(double) * nWeights);
				neu.wCopy = (double *)malloc(sizeof(double) * nWeights);
				if (!neu.w || !neu.deltaW || !neu.lastDeltaW || !neu.wCopy)
					return 0;

				// Inicializa acumuladores
				for (int i = 0; i < nWeights; ++i)
				{
					neu.w[i] = 0.0;
					neu.deltaW[i] = 0.0;
					neu.lastDeltaW[i] = 0.0;
					neu.wCopy[i] = 0.0;
				}
			}
			else
			{
				// Capa de entrada: sin pesos
				neu.w = neu.deltaW = neu.lastDeltaW = neu.wCopy = nullptr;
			}
		}
	}
	return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	if (!this->layers)
	{
		this->nOfLayers = 0;
		return;
	}

	for (int i = 0; i < this->nOfLayers; i++)
	{
		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			free(this->layers[i].neurons[j].w);
			free(this->layers[i].neurons[j].deltaW);
			free(this->layers[i].neurons[j].lastDeltaW);
			free(this->layers[i].neurons[j].wCopy);
		}
		free(this->layers[i].neurons);
	}
	free(this->layers);
	this->layers = nullptr;
	this->nOfLayers = 0;
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const int nPrev = this->layers[l - 1].nOfNeurons;
		const int nWeights = nPrev + 1; // bias + entradas

		for (int j = 0; j < this->layers[l].nOfNeurons; ++j)
		{
			Neuron &neu = this->layers[l].neurons[j];
			for (int i = 0; i < nWeights; ++i)
			{
				neu.w[i] = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
				neu.deltaW[i] = 0.0;
				neu.lastDeltaW[i] = 0.0;
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	// Las neuronas de entrada no tienen pesos de entrada ni realizan cálculo alguno.
	// Su función es simplemente recibir los valores de entrada del dataset
	for (int j = 0; j < this->layers[0].nOfNeurons; ++j)
		this->layers[0].neurons[j].out = input[j];
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	Layer &outL = this->layers[this->nOfLayers - 1];
	for (int j = 0; j < outL.nOfNeurons; ++j)
		output[j] = outL.neurons[j].out;
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const int nPrev = this->layers[l - 1].nOfNeurons;
		const int nWeights = nPrev + 1;
		for (int j = 0; j < this->layers[l].nOfNeurons; ++j)
		{
			Neuron &neu = this->layers[l].neurons[j];
			for (int i = 0; i < nWeights; ++i)
				neu.wCopy[i] = neu.w[i];
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const int nPrev = this->layers[l - 1].nOfNeurons;
		const int nWeights = nPrev + 1;
		for (int j = 0; j < this->layers[l].nOfNeurons; ++j)
		{
			Neuron &neu = this->layers[l].neurons[j];
			for (int i = 0; i < nWeights; ++i)
				neu.w[i] = neu.wCopy[i];
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const Layer &prev = this->layers[l - 1];
		Layer &curr = this->layers[l];

		for (int j = 0; j < curr.nOfNeurons; ++j)
		{
			Neuron &neu = curr.neurons[j];

			double net = neu.w[0]; // bias
			for (int i = 0; i < prev.nOfNeurons; ++i)
				net += neu.w[i + 1] * prev.neurons[i].out;

			// Sigmoide inline (si quieres salida lineal en la última capa, usa: if(l==nOfLayers-1) neu.out = net; else ...)
			neu.out = 1.0 / (1.0 + exp(-net));
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double *target)
{
	Layer &outL = this->layers[this->nOfLayers - 1];
	const int nOut = outL.nOfNeurons;

	double mse = 0.0;
	for (int j = 0; j < nOut; ++j)
	{
		const double diff = outL.neurons[j].out - target[j];
		mse += diff * diff;
	}
	return mse / (double)nOut;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double *target)
{
	// Capa de salida
	const int L = this->nOfLayers - 1;
	Layer &outL = this->layers[L];
	for (int j = 0; j < outL.nOfNeurons; ++j)
	{
		Neuron &n = outL.neurons[j];
		// MSE + sigmoide: (y - t) * y*(1 - y)
		n.delta = (n.out - target[j]) * (n.out * (1.0 - n.out));
		// Si última capa lineal + MSE: n.delta = (n.out - target[j]);
	}

	// Capas ocultas
	for (int l = L - 1; l >= 1; --l)
	{
		Layer &curr = this->layers[l];
		Layer &next = this->layers[l + 1];

		for (int j = 0; j < curr.nOfNeurons; ++j)
		{
			double sum = 0.0;
			for (int k = 0; k < next.nOfNeurons; ++k)
				sum += next.neurons[k].w[j + 1] * next.neurons[k].delta; // salta el bias

			const double y = curr.neurons[j].out;
			curr.neurons[j].delta = (y * (1.0 - y)) * sum; // derivada sigmoide inline
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const Layer &prev = this->layers[l - 1];
		Layer &curr = this->layers[l];
		const int nPrev = prev.nOfNeurons; // nº entradas reales a cada neurona
		const int nW = nPrev + 1;		   // +1 por el bias

		for (int j = 0; j < curr.nOfNeurons; ++j)
		{
			Neuron &neu = curr.neurons[j];

			// Gradiente para bias (entrada constante = 1)
			const double dW0 = -this->eta * (neu.delta * 1.0) + this->mu * neu.lastDeltaW[0];
			neu.deltaW[0] += dW0;

			// Gradiente para pesos conectados a la capa previa
			for (int i = 0; i < nPrev; ++i)
			{
				const double x_i = prev.neurons[i].out;
				const double grad = neu.delta * x_i; // dE/dw = delta_j * x_i
				const double dWij = -this->eta * grad + this->mu * neu.lastDeltaW[i + 1];
				neu.deltaW[i + 1] += dWij;
			}
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const int nPrev = this->layers[l - 1].nOfNeurons;
		const int nW = nPrev + 1;

		for (int j = 0; j < this->layers[l].nOfNeurons; ++j)
		{
			Neuron &neu = this->layers[l].neurons[j];

			for (int i = 0; i < nW; ++i)
			{
				const double dW = neu.deltaW[i];
				neu.w[i] += dW;			// aplica actualización acumulada
				neu.lastDeltaW[i] = dW; // guarda para el término de momento
				neu.deltaW[i] = 0.0;	// limpia el acumulador (para siguiente patrón/lote)
			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
	for (int l = 1; l < this->nOfLayers; ++l)
	{
		const int nPrev = this->layers[l - 1].nOfNeurons;
		const int nCurr = this->layers[l].nOfNeurons;

		std::cout << "Layer " << (l - 1) << " -> " << l
				  << " (" << nPrev << " -> " << nCurr << ")\n";

		for (int j = 0; j < nCurr; ++j)
		{
			const Neuron &neu = this->layers[l].neurons[j];
			std::cout << "  Neuron " << j << ":  b=" << neu.w[0];
			for (int i = 0; i < nPrev; ++i)
			{
				std::cout << "  w[" << i << "]=" << neu.w[i + 1];
			}
			std::cout << '\n';
		}
		std::cout << std::endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double *input, double *target)
{
	// 1) Cargar entradas
	feedInputs(input);

	// 2) Propagación hacia delante
	forwardPropagate();

	// 3) Retropropagación del error (debe haber calculado 'delta' en cada neurona)
	backpropagateError(target);

	// 4) Acumular cambios con momento
	accumulateChange();

	// 5) Actualizar inmediatamente (online/SGD)
	weightAdjustment();
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset *trainDataset)
{
	int i;
	for (i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset *testDataset)
{
	double mseSum = 0.0;

	for (int p = 0; p < testDataset->nOfPatterns; ++p)
	{
		feedInputs(testDataset->inputs[p]);
		forwardPropagate();
		mseSum += obtainError(testDataset->outputs[p]);
	}
	return mseSum / (double)testDataset->nOfPatterns;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *obtained = new double[numSalidas];

	cout << "Id,Predicted" << endl;

	for (i = 0; i < pDatosTest->nOfPatterns; i++)
	{

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
	delete[] obtained; // <- AÑADIDO
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	double testError = 0;

	// Learning
	do
	{

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
		if (countTrain == 0 || trainError < minTrainError)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if ((trainError - minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < pDatosTest->nOfPatterns; i++)
	{
		double *prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;
	}

	testError = test(pDatosTest);
	*errorTest = testError;
	*errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
