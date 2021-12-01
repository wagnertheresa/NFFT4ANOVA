"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"Learning in High-Dimensional Feature Spaces Using ANOVA-Based Fast Matrix-Vector Multiplication"
by F. Nestler, M. Stoll, T. Wagner (2021)
"""

import numpy as np
import pandas as pd
import fastadj
import time
import random
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse.linalg import cg, LinearOperator

# import sklearn classifier for comparison
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge


class NFFTKernelRidge:
    """
    NFFT-based Kernel Ridge Regression
    
    Parameters
    ----------
    sigma : float, default=1.0
         Sigma parameter for the Gaussian kernel.
    beta : float, default=1.0
        The regularization parameter within the learning task for the kernel ridge regression, where beta > 0.
    balance : bool, default=True
        Whether the class distribution of the data, the model is fitted on, shall be balanced or not.
    n_samples : int, default=None
        Number of samples to include per class, when balancing the class distribution.
        If None, then the biggest possible balanced subset, i.e. a subset with min(#samples in class -1, #samples in class 1) samples, is used.
        Else, a subset with n_samples randomly chosen samples per class is constructed.
    norm : str, default='z-score'
        The normalization parameter determining how to standardize the features. It is either 'z-score' (z-score normalization) or None (the features are not standardized).
    setup : str, default='default'
        The setup argument loads presets for the parameters of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
    tol : float, default=1e-03
        The tolerance of convergence within the CG-algorithm.
    mis_threshold : float, default=0.0
        Mutual information score threshold determining, which features to include in the kernel. All features with a score below this threshold are dropped, the others are included.
    window_scheme : str, default='mis'
        The window-scheme argument determining, how the windows shall be built.
        If 'mis' is passed, the windows are built within this class based on their mutual information scores. If a pre_list is defined for this window_scheme, the predefined list of windows is taken and the rest of the features are seperated up into the windows by their mis.
        If window_scheme='consec', the windows are built following the feature indices in ascending order.
        If None, the list of windows of features is predefined by the pre_list argument.
    pre_list : list, default=[]
        If window_scheme=None, a predefined list of windows of features must be passed. Those windows are used to realize the kernel-vector multiplication.
        If window_scheme='mis', a predefined list of windows can be passed and the rest of the windows are determined by the features' mis.
    weight_scheme : str, default='equally weighted'
        The weighting-scheme determining, how the weights in the weighted sum of kernels are built.
        If weight_scheme='equally weighted', all weights are equal, so that they sum up to 1.
        If weight_scheme='no weights', all weights are 1.
    
    Attributes
    ----------
    windows : list
        The list of windows determining the feature grouping.
    weights : list
        The list of weights for the weighted sum of kernels.
    alpha : ndarray
        The dual-variable for the KRR-Model.
    trainX : ndarray
        The training data used to fit the model.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> N, d = 25000, 15
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(N, d)
    >>> y = np.sign(rng.randn(N))
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    >>> clf = NFFTKernelRidge
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """
    
    def __init__(self, sigma=1, beta=1, balance=True, n_samples=None, norm='z-score', setup='default', tol=1e-03, mis_threshold=0.0, window_scheme='mis', pre_list=[], weight_scheme='equally weighted'):
        self.sigma = sigma
        self.beta = beta
        self.balance = balance
        self.n_samples = n_samples
        self.norm = norm
        self.setup = setup
        self.tol = tol
        self.mis_threshold = mis_threshold
        self.window_scheme = window_scheme
        self.pre_list = pre_list
        self.weight_scheme = weight_scheme
        
        
    def under_sample(self, X, y):
        """
        Balance the class distribution of the data X by under-sampling the over-represented class.
            
        Parameters
        ----------
        X : ndarray
            The data which is to be under-sampled.
        y : ndarray
            The target vector.
            
        Returns
        -------
        X : ndarray
            The balanced data.
        y : ndarray
            The corresponding target values of the balanced data.
        """
        # save label for all indices
        idx_pos = []
        idx_neg = []
        for i in range(len(y)):
            if y[i] == -1:
                idx_neg.append(i)
            else:
                idx_pos.append(i)
        
        # determine maximal number of samples per class for balanced subset
        n_max = min(len(idx_pos), len(idx_neg))
        if self.n_samples == None:
            num = n_max
        elif self.n_samples > n_max:
            raise Warning("n_samples exceeds the number of samples per class for the biggest possible balanced subset for the input data. Therefore, the biggest possible balanced subset is constructed.")
            num = n_max
        else:
            num = self.n_samples
            
        r1 = random.sample(idx_pos, num)
        r2 = random.sample(idx_neg, num)
        r_samples = r1 + r2
        
        X = X[r_samples,:]
        y = y[r_samples]
            
        return X, y
        
    def z_score_normalization(self, X, dt_train):
        """
        Z-score normalize the training and test data.
        
        Note
        ----
        Only the training data is included in fitting the normalizer.
        The boolean "dt_train" indicates, whether the input data serves as training or test data.
            
        Parameters
        ----------
        X : ndarray
            The data which is to be z-score normalized.
        dt_train : bool
            Whether the input data serve as training or test data.
            If True, the input data serve as training data and the z-score-normalization is fitted on this data.
            If False, the input data serve as test data and the statistics from the training data are used.
            
        Returns
        -------
        X_norm : ndarray
            Z-score normalized data.
        """
        if dt_train == True:
            # define global variable aux_scale to normalize test data using the statistics of the training data
            global aux_scale
            aux_scale = X
            
            scaler = StandardScaler()
            scaler.fit(X)
        else:
            scaler = StandardScaler()
            scaler.fit(aux_scale)
            
            scaler = StandardScaler()
            scaler.fit(X)

        X_norm = scaler.transform(X)
            
        return X_norm
        
    
    def make_mi_scores(self, X, y):
        """
        Compute the mutual information scores.
            
        Parameters
        ----------
        X : ndarray
            The data matrix.
        y : ndarray
            The target vector incorporating the labels.
            
        Returns
        -------
        res_idx : list
            List of feature-indices following their mutual information scores in descending order.   
        """
        threshold = self.mis_threshold
        
        del_list = [elem for pair in self.pre_list for elem in pair]
        
        mi_scores = mutual_info_classif(X, y)
        mi_scores = pd.Series(mi_scores, name="MI Scores")
        mi_scores = mi_scores.tolist()
        
        # sorts scores in descending order
        sorted_scores = np.sort(mi_scores)[::-1]
        sorted_idx = np.argsort(mi_scores)[::-1]
        
        # convert np arrays to lists
        sorted_scores = sorted_scores.tolist()
        sorted_idx = sorted_idx.tolist()
        
        # drop features with mi_score below threshold
        # adjust threshold, if not enough features have a score above threshold
        while len([i for i in sorted_scores if i >= threshold]) < 3:
            threshold = threshold * 0.5
            
        res_idx = [i for i in sorted_idx if sorted_scores[sorted_idx.index(i)] >= threshold and i not in del_list]
        
        res_scores = []
        
        for j in res_idx:
            res_scores.append(sorted_scores[sorted_idx.index(j)])
        
        return res_idx
        
    
    def get_mis_windows(self, mi_idx):
        """
        Construct a list of windows of features based on MIS-ranking.
            
        Parameters
        ----------
        mi_idx : list
            The corresponding feature indices to the mutual information scores in descending order.
        
        Returns
        -------
        windows : list
            List of windows of features.
        """
        windows = []
        
        for l in range(int(np.divide(len(mi_idx),3))):
            I_l = mi_idx[(l*3):(l*3)+3]
            windows.append(I_l)
        
        # if |mi_idx| is not divisible by 3, the last window contains only 1 or 2 indices
        if (int(np.divide(len(mi_idx),3))*3) != len(mi_idx):
            I_l = []
            for i in range(int(np.divide(len(mi_idx),3))*3, len(mi_idx)):
                I_l.append(mi_idx[i])
            windows.append(I_l)
            
        return windows
    
    
    def fast_adjacency(self, p):
        """
        Approximate the matrix-vector product A*p within the cg-algorithm used in train_NFFT_KRR, where A = w_1*K_1 + w_2*K_2 + ... + beta*I.
            
        Note
        ----
        Using the NFFT-approach, the kernel matrices are never computed explicitly. 
        Since this function serves as a LinearOperator for the cg-function from scipy, only one variable, the vector p, can be passed as input parameter.
        The variables which are needed additionally to approximate A*p, are therefore defined as global variables in train_NFFT_KRR, so that they can still be used within this function.
        
        Parameters
        ----------
        p : ndarray
            The vector, whose product A*p with the matrix A shall be approximated.
        
        Returns
        -------
        Ap : ndarray
            The approximated matrix-vector product A*p.
        """
        # number of samples
        N = trainX_fa.shape[0]
        
        ## setup computations with the adjacency matrices
        # set diagonal=1.0, since FastAdjacency package is targeted at graph Laplacian with zeros at the diagonal, but we need 1 at the diagonal
        adjacency_mats = [fastadj.AdjacencyMatrix(trainX_fa[:,self.windows[l]], self.sigma, setup=self.setup, diagonal=1.0) for l in range(len(self.windows))]
        
        # initialize matrix-vector product Ap
        Ap = np.zeros((N,))
    
        for l in range(len(self.windows)):
            Ap += self.weights[l] * adjacency_mats[l].apply(p)
                
        # do not neglect: A = (w_1*K_1 + w_2*K_2 + ...) + beta*I
        Ap += self.beta*p
        
        return Ap
    
    
    def train_NFFT_KRR(self, trainData):    
        """
        Train the model on the training data by solving the underlying system of linear equations using the CG-algorithm with NFFT-approach.
            
        Parameters
        ----------
        trainData : list
            A list holding the data matrix and the target vector of the training data.
        
        Returns
        -------
        alpha : ndarray
            The dual-variable for the KRR-Model.
        """        
        # access training data points
        trainX = trainData[0]
        # f incorporates the y_i values
        f = trainData[1]
    
        # declare global variables for the LinearOperator fast_adjacency
        global trainX_fa
        trainX_fa = trainX
        
        Ap = LinearOperator(shape=(trainX.shape[0],trainX.shape[0]), matvec=(self.fast_adjacency))
    
        # initialize counter to get number of iterations needed in cg-algorithm
        num_iters = 0
    
        # function to count number of iterations needed in cg-algorithm
        def callback(xk):
            nonlocal num_iters
            num_iters += 1
            
        # CG with NFFT
        alpha, info = cg(Ap, f, tol=self.tol, callback=callback)
        
        # print number of iterations needed in cg-algorithm
        #print('num_iters in CG for fitting:', num_iters)
        
        return alpha
           
    
    def fit(self, X, y):
        """
        Fit NFFT-based kernel ridge regression model on training data.

        Parameters
        ----------
        X : ndarray
            The training data matrix.
        y : ndarray
            The corresponding target values.

        Returns
        -------
        self : returns an instance of self
        """
        # balance the class distribution of the data by under-sampling the over-represenetd class
        if self.balance == True:
            X, y = self.under_sample(X, y)
        
        # scale data with z-score-normalization
        if self.norm == 'z-score':
            X = self.z_score_normalization(X, dt_train=True)
        
        # determine windows of features by their mis
        if self.window_scheme == 'mis':
            res_idx = self.make_mi_scores(X, y)
            self.windows = self.get_mis_windows(res_idx)
        # windows are built following the feature indices in ascending order
        elif self.window_scheme == 'consec':
            wind = []
            d = X.shape[1]
        
            for l in range(int(np.divide(d,3))):
                I_l = list(range((l*3),(l*3)+3))
                wind.append(I_l)
        
            # if |d| is not divisible by 3, the last window contains only 1 or 2 indices
            if (int(np.divide(d,3))*3) != d:
                I_l = []
                for l in range(int(np.divide(d,3))*3, d):
                    I_l.append(l)
                wind.append(I_l)
            
            self.windows = wind
        # use predefined list of windows of features
        elif self.window_scheme == None:
            self.windows = self.pre_list
        
        # compute kernel weights
        kweights = []
    
        # equally weighted kernels, so that weights sum up to 1
        if self.weight_scheme == 'equally weighted':
            for j in range(len(self.windows)):
                kweights.append(1/len(self.windows))
        # no weighting, all weights are 1
        elif self.weight_scheme == 'no weights':
            for j in range(len(self.windows)):
                kweights.append(1.0)
        
        self.weights = kweights
        
        self.alpha = self.train_NFFT_KRR([X,y])
        
        self.trainX = X
        
        return self
        
    def predict(self, X):
        """
        Predict class affiliations for the test data after the model has been fitted, using the NFFT-approach.
            
        Note
        ----
        To use the NFFT-based fast summation here, the test samples are appended to the training samples.
        In total, only a fraction of the approximated kernel evaluation is needed here.
        Because of this, the factor consists of alpha and the rest is padded with zeros.
        
        Parameters
        ----------
        X : ndarray
            The data, for which class affiliations shall be predicted.
        
        Returns
        -------
        YPred : ndarray
            The predicted class affiliations.
        """
        
        # scale test data with z-score-normalization
        if self.norm == 'z-score':
            X = self.z_score_normalization(X, dt_train=False)
    
        N_sum = X.shape[0] + self.trainX.shape[0]
        arr = np.append(self.trainX, X, axis=None).reshape(N_sum,self.trainX.shape[1])
    
        adjacency_vals = [fastadj.AdjacencyMatrix(arr[:,self.windows[l]], self.sigma, setup=self.setup) for l in range(len(self.windows))]
        p = np.append(self.alpha, np.zeros(X.shape[0]), axis=None).reshape(N_sum,)
    
        # initialize predicted response vals
        vals = np.zeros((N_sum,))
    
        # predict responses
        for l in range(len(self.windows)):
            vals += self.weights[l] * adjacency_vals[l].apply(p)
        
        # select predicted responses for test data
        YPred = np.sign(vals[-X.shape[0]:])
        
        return YPred
    
    
    
class GridSearch:
    """
    Exhaustive search on candidate parameter values for a classifier.
    
    Parameters
    ----------
    classifier : str, default='NFFTKernelRidge'
        The classifier parameter determines, for which classifier GridSearch shall be performed.
        It is either 'NFFTKernelRidge', 'sklearn KRR' or 'sklearn SVC'.
    param_grid : dict
        Dictionary with parameter names and lists of candidate values for the parameters to try as values.
    scoring : str, default='accuracy'
        The scoring parameter determines, which evaluation metric shall be used.
        It is either 'accuracy', 'precision' or 'recall'.
    balance : bool, default=True
        Whether the class distribution of the data, the model is fitted on, shall be balanced or not.
    n_samples : int, default=None
        Number of samples to include per class, when balancing the class distribution.
        If None, then the biggest possible balanced subset, i.e. a subset with min(#samples in class -1, #samples in class 1) samples, is used.
        Else, a subset with n_samples randomly chosen samples per class is constructed.
    norm : str, default='z-score'
        The normalization parameter determining how to standardize the features. It is either 'z-score' (z-score normalization) or None (the features are not standardized).
    setup : str, default='default'
        The setup argument loads presets for the parameters of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
    tol : float, default=1e-03
        The tolerance of convergence within the CG-algorithm.
    mis_threshold : float, default=0.0
        Mutual information score threshold determining, which features to include in the kernel. All features with a score below this threshold are dropped, the others are included.
    window_scheme : str, default='mis'
        The window-scheme argument determining, how the windows shall be built.
        If 'mis' is passed, the windows are built within this class based on their mutual information scores. If a pre_list is defined for this window_scheme, the predefined list of windows is taken and the rest of the features are seperated up into the windows by their mis.
        If window_scheme='consec', the windows are built following the feature indices in ascending order.
        If None, the list of windows of features is predefined by the pre_list argument.
    pre_list : list, default=[]
        If window_scheme=None, a predefined list of windows of features must be passed. Those windows are used to realize the kernel-vector multiplication.
        If window_scheme='mis', a predefined list of windows can be passed and the rest of the windows are determined by the features' mis.
    weight_scheme : str, default='equally weighted'
        The weighting-scheme determining, how the weights in the weighted sum of kernels are built.
        If weight_scheme='equally weighted', all weights are equal, so that they sum up to 1.
        If weight_scheme='no weights', all weights are 1.
    
    Attributes
    ----------
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> N, d = 25000, 15
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(N, d)
    >>> y = np.sign(rng.randn(N))
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    >>> param_grid = {
        "sigma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "beta": [1, 10, 100, 1000],
    }
    >>> model = GridSearch(classifier="NFFTKernelRidge", param_grid=param_grid)
    >>> model.tune(X_train, y_train, X_test, y_test)
    """
    
    def __init__(self, classifier, param_grid, scoring="accuracy", balance=True, n_samples=None, norm='z-score', setup="default", tol=1e-03, mis_threshold=0.0, window_scheme='mis', pre_list=[], weight_scheme='equally weighted'):
        
        self.classifier = classifier
        self.param_grid = param_grid
        self.scoring = scoring
        self.balance = balance
        self.n_samples = n_samples
        self.norm = norm
        self.setup = setup
        self.tol = tol
        self.mis_threshold = mis_threshold
        self.window_scheme = window_scheme
        self.pre_list = pre_list
        self.weight_scheme = weight_scheme
        
        
    def evaluation_metrics(self, Y, YPred):
        """
        Evaluate the quality of a prediction.
            
        Parameters
        ----------
        Y : ndarray
            The target vector incorporating the true labels.
        YPred : ndarray
            The predicted class affiliations.
            
        Returns
        -------
        accuracy : float
            Share of correct predictions in all predictions.
        precision : float
            Share of true positives in all positive predictions.
        recall : float
            Share of true positives in all positive values.
        """
        # initialize TP, TN, FP, FN (true positive, true negative, false positive, false negative)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(len(Y)):
            if Y[j]==1.0:
                if YPred[j]==1.0:
                    TP += 1
                elif YPred[j]==-1.0:
                    FN += 1
                '''
                else:
                    print('neither predicted class 1 nor -1 for test_sample:', j)
                    print('YPred[j]', YPred[j])
                '''
            elif Y[j]==-1.0:
                if YPred[j]==1.0:
                    FP += 1
                elif YPred[j]==-1.0:
                    TN += 1
                '''
                else:
                    print('neither predicted class 1 nor -1 for test_sample:', j)
                    print('YPred[j]', YPred[j])
                '''
            '''
            else:
                print('Some other class is listed')
            '''
        '''
        if (TP+FN+FP+TN)!=len(Y):
            print('Sum is not correct: error in calculation')
            print('Sum is %d instead of len(y)=%d' % ((TP+FN+FP+TN), len(Y)))
        
        print('share of right predictions for negative class:', np.divide(TN, (TN+FP)))
        '''
        
        if (TP+TN) == 0:
            accuracy = 0
        else:
            accuracy = np.divide((TP+TN), len(Y))
        if TP == 0:
            precision = 0
            recall = 0
        else:
            precision = np.divide(TP, (TP+FP))
            recall = np.divide(TP, (TP+FN))
            
        # return evaluation metrics
        return accuracy, precision, recall
        
        
    def tune(self, X_train, y_train, X_test, y_test):
        """
        Tune over all candidate hyperparameter sets.
        
        Parameters
        ----------
        X_train : ndarray
            The training data.
        y_train : ndarray
            The corresponding labels for the training data.
        X_test : ndarray
            The test data.
        y_test : ndarray
            The corresponding labels for the test data.

        Returns
        -------
        best_params : list
            List of the parameters, which yield the highest value for the chosen scoring-parameter (accuracy, precision or recall).
        best_result : list
            List of the best results, where the chosen scoring-parameter is crucial.
        best_time_fit : float
            Fitting time of the run, that yielded the best result.
        best_time_pred : float
            Prediction time of the run, that yielded the best result.
        mean_total_time_fit : float
            Mean value over the fitting times of all candidate parameters.
        mean_total_time_pred : float
            Mean value over the prediction times of all candidate parameters.
        mean_total_time : float
            Mean value over the total times of all candidate parameters.
        """
        param_names = list(self.param_grid)
        prod = []
        [prod.append(self.param_grid[param_names[i]]) for i in range(len(param_names))]
        params = list(itertools.product(*prod))
        #print("List of Candidate Parameters:", params)
        
        best_params = [0,0]
        best_result = [0,0,0]
        
        total_time_fit = []
        total_time_pred = []
        total_time = []
        
        for j in range(len(params)):
            if self.classifier == "NFFTKernelRidge":
                # measure time needed for fitting
                start_fit = time.time()
                
                clf = NFFTKernelRidge(sigma=params[j][0], beta=params[j][1], balance=self.balance, n_samples=self.n_samples, norm=self.norm, setup=self.setup, tol=self.tol, mis_threshold=self.mis_threshold, window_scheme=self.window_scheme, pre_list=self.pre_list, weight_scheme=self.weight_scheme)
            
                clf.fit(X_train, y_train)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                evaluation = clf.predict(X_test)
                
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
            
            elif self.classifier == "sklearn KRR":
                # measure time needed for fitting
                start_fit = time.time()
                
                clf = KernelRidge(alpha=params[j][0], gamma=params[j][1], kernel='rbf')
                
                clf.fit(X_train, y_train)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                evaluation = np.sign(clf.predict(X_test))
                
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
                
            elif self.classifier == "sklearn SVC":
                # measure time needed for fitting
                start_fit = time.time()
                
                clf = svm.SVC(C=params[j][0], gamma=params[j][1], kernel='rbf')
            
                clf.fit(X_train, y_train)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                evaluation = clf.predict(X_test)
            
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
            
            result = self.evaluation_metrics(y_test,evaluation)
            #print("Candidate Parameter:", params[j])
            #print("Result:", result)
            
            if self.scoring == "accuracy":
                if result[0] > best_result[0]:
                    best_params = params[j]
                    best_result = result
                    best_time_fit = time_fit
                    best_time_pred = time_pred
                
            elif self.scoring == "precision":
                if result[1] > best_result[1]:
                    best_params = params[j]
                    best_result = result
                    best_time_fit = time_fit
                    best_time_pred = time_pred
                
            elif self.scoring == "recall":
                if result[2] > best_result[2]:
                    best_params = params[j]
                    best_result = result
                    best_time_fit = time_fit
                    best_time_pred = time_pred
        
        return best_params, best_result, best_time_fit, best_time_pred, np.mean(total_time_fit), np.mean(total_time_pred), np.mean(total_time)
