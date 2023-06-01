import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.

    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d

        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT:
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.mean = np.mean(training_data, axis=0)
        training_data_tilde = training_data - self.mean
        C = 1 / training_data.shape[0] * training_data_tilde.T @ training_data_tilde

        eigvals, eigvecs = np.linalg.eigh(C)

        eigvals = eigvals[::-1]
        eg = eigvals[:d]
        exvar = np.sum(eg) / np.sum(eigvals) * 100

        self.W = eigvecs[:, :d]
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        data_reduced = (data - self.mean) @ self.W
        return data_reduced


