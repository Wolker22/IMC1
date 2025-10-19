//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool tflag = 0, Tflag = 0, wflag = 0, pflag = 0, iflag = 0;
    char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;
    int iterations = 1000; // valor por defecto razonable
    int c;

    opterr = 0;

    // Añadimos -t y -i para poder entrenar
    while ((c = getopt(argc, argv, "t:T:w:i:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'i':
                iflag = true;
                iterations = std::max(1, atoi(optarg));
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'w' || optopt == 'i' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown character `\\x%x'.\n", optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Comprobación mínima de argumentos
        if (!tflag || !Tflag) {
            cerr << "Usage (train): " << argv[0]
                 << " -t <train_file> -T <test_file> [-i iterations] [-w weights_out]\n";
            return EXIT_FAILURE;
        }

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Parameters of the mlp. Por ejemplo:
        // mlp.eta = 0.1; mlp.mu = 0.9;

        // Read training and test data
        Dataset * trainDataset = readData(tvalue);
        Dataset * testDataset  = readData(Tvalue);

        if (trainDataset == NULL || testDataset == NULL) {
            cerr << "Error reading datasets. Check -t and -T paths.\n";
            return EXIT_FAILURE;
        }

        // Initialize topology vector: [nIn, 5, nOut] (1 oculta de 5 neuronas)
        int nIn  = trainDataset->nOfInputs;
        int nOut = trainDataset->nOfOutputs;
        int layers = 1; // nº de capas ocultas
        int * topology = new int[layers + 2];
        topology[0] = nIn;
        topology[1] = 5;     // tamaño por defecto de la capa oculta
        topology[2] = nOut;

        // Initialize the network using the topology vector
        if (!mlp.initialize(layers + 2, topology)) {
            cerr << "Error initializing network topology.\n";
            delete[] topology;
            return EXIT_FAILURE;
        }
        delete[] topology;

        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;

        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);

            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations,
                                         &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        // Obtain training and test averages and standard deviations
        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // medias
        for (int i=0;i<5;++i){ averageTestError += testErrors[i]; averageTrainError += trainErrors[i]; }
        averageTestError /= 5.0;  averageTrainError /= 5.0;

        // desviaciones típicas (poblacionales)
        for (int i=0;i<5;++i){
            stdTestError  += (testErrors[i]  - averageTestError) * (testErrors[i]  - averageTestError);
            stdTrainError += (trainErrors[i] - averageTrainError) * (trainErrors[i] - averageTrainError);
        }
        stdTestError  = sqrt(stdTestError  / 5.0);
        stdTrainError = sqrt(stdTrainError / 5.0);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error  (Mean +- SD): " << averageTestError  << " +- " << stdTestError  << endl;

        delete[] testErrors;
        delete[] trainErrors;

        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading test data
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
