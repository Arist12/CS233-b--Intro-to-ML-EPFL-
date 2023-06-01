import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        labels = label_to_onehot(training_labels)
        C = labels.shape[-1]
        D = training_data.shape[-1]
        self.weights = np.random.normal(0, 0.1, (D, C))

        for it in range(self.max_iters):
            gradient = training_data.T @ (self.f_softmax(training_data, self.weights) - labels)
            self.weights -= self.lr * gradient

            predictions = self.predict(training_data)
            if accuracy_fn(predictions, training_labels) == 100:
                break

            if it % 10 == 0:
                print(f'training loss at iteration {it}: {self.loss_logistic_multi(training_data, labels, self.weights)}')
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        preds = self.f_softmax(test_data, self.weights)
        return np.argmax(preds, axis=1)

    def loss_logistic_multi(self, data, labels, w):
        """
        Loss function for multi class logistic regression, i.e., multi-class entropy.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value
        """
        preds = self.f_softmax(data, w)
        return -np.sum(labels * np.log(preds))

    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        y = data @ W
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)