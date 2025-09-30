/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()

namespace util
{

    struct Dataset
    {
        int nOfInputs;    /* Number of inputs */
        int nOfOutputs;   /* Number of outputs */
        int nOfPatterns;  /* Number of patterns */
        double **inputs;  /* Matrix with the inputs of the problem */
        double **outputs; /* Matrix with the outputs of the problem */
    };

    // Obtain an integer random number in the range [Low,High]
    int randomInt(int Low, int High);

    // Obtain a real random number in the range [Low,High]
    double randomDouble(double Low, double High);
    
    // Read a dataset from a file name and return it
	Dataset* readData(const char *fileName);
    // Print the dataset
    void printDataset(Dataset *dataset, int len);

    static int *integerRandomVectoWithoutRepeating(int min, int max, int howMany)
    {
        int total = max - min + 1;
        int *numbersToBeSelected = new int[total];
        int *numbersSelected = new int[howMany];
        // Initialize the list of possible selections
        for (int i = 0; i < total; i++)
            numbersToBeSelected[i] = min + i;

        for (int i = 0; i < howMany; i++)
        {
            int selectedNumber = rand() % (total - i);
            // Store the selected number
            numbersSelected[i] = numbersToBeSelected[selectedNumber];
            // We include the last valid number in numbersToBeSelected, in this way
            // all numbers are valid until total-i-1
            numbersToBeSelected[selectedNumber] = numbersToBeSelected[total - i - 1];
        }
        delete[] numbersToBeSelected;
        return numbersSelected;
    }

    // Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the feature in the dataset (minData and maxData). 
    double minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData);

    // Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the feature in the dataset (minData and maxData). 
    void minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                   double *minData, double *maxData);

    // Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
    // and max values of the feature in the dataset (minData and maxData). Only for regression problems. 
    void minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                    double minData, double maxData);

    // Get a vector of maximum values of the dataset inputs
    double *maxDatasetInputs(Dataset *dataset);
    // Get a vector of minimum values of the dataset inputs
    double *minDatasetInputs(Dataset *dataset);
    // Get the minimum value of the dataset outputs
    double minDatasetOutputs(Dataset *dataset);
    // Get the maximum value of the dataset outputs
    double maxDatasetOutputs(Dataset *dataset);
};

#endif /* UTIL_H_ */
