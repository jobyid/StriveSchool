import numpy as np
import time

class Knn:
    """A simple function to run the KNN algorithm.

    ...
    Attributes
    ----------
    k : int
        an int which represents the number of neighbours to count.

    Methods
    -------
    predict:
        Makes prediction of class based on training data and data to test
    evaluate:
        Takes true values to score the predictions against the real results
    fit:
        dose nothing """
    def __init__(self, k):
        """
        Parameters
        ----------
        k: int
            The number of neighbours to consider
        """
        self.k = k
        self.start_time = time.time()
        self.execution_time = 0.00

    def fit(self, X, y):
        """This is an empty method just to frustrate you.
         It can take any X and y."""
        # I don't think we need fit, but it was required for my submission
        print("this model is to lazy to fit, just go right to prediction")
        return self

    def __check_for_missing_values__(self, test, train, y_train):
        # the model fails with nan values in the this check prevents this
        tc = np.count_nonzero(np.isnan(test))
        trc = np.count_nonzero(np.isnan(train))
        ytc = np.count_nonzero(np.isnan(y_train))
        if tc > 0 or trc > 0 or ytc > 0:
            raise Exception("Data must not contain nan values")


    def __find_neighbours__(self,x, data, d_class):
        # euc distance for all data points
        # loop all data and find distance from points
        self.neighbours = {}
        for d in data:
            dist = np.linalg.norm(d-x)
            index = np.where(data == d)
            # add to dictionary with distance as key and class as value
            self.neighbours[dist] = d_class[index[0][0]]


    def __vote__(self):
        # grab keys array to sort
        keys = np.array(sorted(self.neighbours.keys()))
        # to save loop time select only the top k number in array
        cla = keys[:self.k]
        options = []
        # loop through shortened list
        for c in cla:
            options.append(self.neighbours[c])
        # count frequency of each class and pick the highest add it to classification array
        unique, frequency = np.unique(options, return_counts = True)
        index = np.where(frequency == max(frequency))
        self.classifcation.append(unique[index[0]][0])

    def predict(self,t,X,y):
        """The predict method is were the magic happens!

        Parameters
        ----------
        t : numpy array,
            Array of data you wish to classify
        X: np.array,
            The training data to train the model on. Must not contain nan
        y: np.array,
            The classifications of the training data, used for training the model

        Returns
        -------
        accuracy: np.array,
            An array of predictions matching your test data. 

        Raises
        ------
        NanValueError
            Exception will be raised if any of you inputs contain NaN values.
        """
        # runs required methods to predict result returns classification array
        self.__execution_time_set__()
        self.__check_for_missing_values__(t,X,y)
        self.classifcation = []
        for x in t:
            self.__find_neighbours__(x,X,y)
            self.__vote__()
        self.execution_time = time.time() - self.start_time
        return self.classifcation

    def __execution_time_set__(self):
        self.start_time = time.time()

    def evaluate(self,true):
        """Check the acucracy % of the model against real values

        Parameters
        ----------
        true: np.array,
            Array of the real values for the test data predicted.

        Returns
        -------
        accuracy: float,
            A float representing the accuracy as a %
        """
        #checks again real values to return and accuracy score
        score = 0
        for i in range(len(self.classifcation)):
            if true[i] == self.classifcation[i]:
                score += 1
        accuracy = score / len(true) * 100
        return accuracy
# square root of n sometimes a good option for k and should be odd
