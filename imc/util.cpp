#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()

#include "util.h"

using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
    return rand() % (High - Low + 1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
    return ((double)rand() / RAND_MAX) * (High - Low) + Low;
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *util::readData(const char *fileName)
{

    ifstream myFile(fileName); // Create an input stream

    if (!myFile.is_open())
    {
        cout << "ERROR: I cannot open the file " << fileName << endl;
        return NULL;
    }

    Dataset *dataset = new Dataset;
    if (dataset == NULL)
        return NULL;

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // Read a line
        istringstream iss(line);
        iss >> dataset->nOfInputs;
        iss >> dataset->nOfOutputs;
        iss >> dataset->nOfPatterns;
    }
    dataset->inputs = new double *[dataset->nOfPatterns];
    dataset->outputs = new double *[dataset->nOfPatterns];

    for (i = 0; i < dataset->nOfPatterns; i++)
    {
        dataset->inputs[i] = new double[dataset->nOfInputs];
        dataset->outputs[i] = new double[dataset->nOfOutputs];
    }

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // Read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset->nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->inputs[i][j] = value;
            }
            for (j = 0; j < dataset->nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}

// ------------------------------
// Print the dataset
void util::printDataset(Dataset *dataset, int len)
{
    if (len == 0)
        len = dataset->nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset->nOfInputs; j++)
        {
            cout << dataset->inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            cout << dataset->outputs[i][j] << ",";
        }
        cout << endl;
    }
}

// ------------------------------
// Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData).
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    return (minAllowed + (((x - minData) * (maxAllowed - minAllowed)) / (maxData - minData)));
}

// ------------------------------
// Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData).
void util::minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                     double *minData, double *maxData)
{
    for (int i = 0; i < dataset->nOfPatterns; i++) // Debería ser nOfPatterns
    {
        for (int j = 0; j < dataset->nOfInputs; j++) // Debería ser nOfInputs
        {
            dataset->inputs[i][j] = minMaxScaler(dataset->inputs[i][j], minAllowed, maxAllowed, *minData, *maxData);
        }
    }
}

// ------------------------------
// Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). Only for regression problems.
void util::minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{
    for (int i = 0; i < dataset->nOfPatterns; i++)
    {
        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            dataset->outputs[i][j] = minMaxScaler(dataset->outputs[i][j], minAllowed, maxAllowed, minData, maxData);
        }
    }
}

// ------------------------------
// Get a vector of minimum values of the dataset inputs
double *util::minDatasetInputs(Dataset *dataset)
{

    double *ret = new double[dataset->nOfInputs];

    for (int i = 0; i < dataset->nOfInputs; i++)
    {
        ret[i] = dataset->inputs[i][0];
        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            if (ret[i] < dataset->inputs[i][j])
            {
                ret[i] = dataset->inputs[i][j];
            }
        }
    }

    return ret;
}

// ------------------------------
// Get a vector of maximum values of the dataset inputs
double *util::maxDatasetInputs(Dataset *dataset)
{
    double *ret = new double[dataset->nOfInputs];

    for (int i = 0; i < dataset->nOfInputs; i++)
    {
        ret[i] = dataset->inputs[i][0];
        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            if (ret[i] > dataset->inputs[i][j])
            {
                ret[i] = dataset->inputs[i][j];
            }
        }
    }

    return ret;
}

// ------------------------------
// Get the minimum value of the dataset outputs
double util::minDatasetOutputs(Dataset *dataset)
{
    double *ret = new double[dataset->nOfInputs];
    ret = minDatasetInputs(dataset);
    double min = ret[0];
    for (int i = 0; i < dataset->nOfInputs; i++)
    {

        if (ret[i]<min)
        {
            min=ret[i];
        }
    }
    return min;
}

// ------------------------------
// Get the maximum value of the dataset outputs
double util::maxDatasetOutputs(Dataset *dataset)
{

    double *ret = new double[dataset->nOfInputs];
    ret = maxDatasetInputs(dataset);
    double max = ret[0];
    for (int i = 0; i < dataset->nOfInputs; i++)
    {

        if (ret[i]>max)
        {
            max=ret[i];
        }
    }
    return max;

}
