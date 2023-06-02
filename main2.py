import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

import time

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    # train data shape: (3502, 32, 32).
    # test data shape:  (867, 32, 32).

    xtrain, xtest, ytrain, ytest = load_data(args.data)
    if args.method != 'nn' or args.nn_type == 'mlp':
        # data preparation
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)

        if args.normalize:
            means, stds = xtrain.mean(axis=0), xtrain.std(axis=0)
            xtrain, xtest = normalize_fn(xtrain, means, stds), normalize_fn(xtest, means, stds)
        if args.append_bias:
            xtrain, xtest = append_bias_term(xtrain), append_bias_term(xtest)

    ## 2. create a validation set
    if not args.test:
        ### WRITE YOUR CODE HERE
        valid_num = int(xtrain.shape[0] * args.valid_ratio)
        xtest, ytest = xtrain[-valid_num:], ytrain[-valid_num:]
        xtrain, ytrain = xtrain[:-valid_num], ytrain[:-valid_num]

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        print(f'The total variance explained by the first {args.pca_d} principal components is {pca_obj.find_principal_components(xtrain):.3f} %')
        # perform dimension reduction on input data
        xtrain, xtest = pca_obj.reduce_dimension(xtrain), pca_obj.reduce_dimension(xtest)
        print(xtrain.shape, xtest.shape)
    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")
        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":

            cnn = False
            model = MLP(input_size=xtrain.shape[-1], n_classes=n_classes)  ### WRITE YOUR CODE HERE

        elif args.nn_type == "cnn":
            ### WRITE YOUR CODE HERE
            cnn = True
            model = CNN(input_channels=1, n_classes=n_classes)

        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, cnn=cnn)
    elif args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)
    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
    elif args.method == "svm":
        method_obj = SVM(C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma, degree=args.svm_degree, coef0=args.svm_coef0)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    time1 = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    time2 = time.time()

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    time3 = time.time()


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    print(f"Training Time: {time2-time1:3f}")
    print(f"Inference Time: {time3-time2:3f}")

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!
    parser.add_argument('--valid_ratio', type=float, default=0.2, help="the ratio of validation set")
    parser.add_argument('--append_bias', action="store_true", help="append bias to data")
    parser.add_argument('--normalize', action="store_true", help="normalize data")


    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
