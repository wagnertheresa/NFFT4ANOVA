"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"Learning in High-Dimensional Feature Spaces Using ANOVA-Based Fast Matrix-Vector Multiplication"
by F. Nestler, M. Stoll, T. Wagner (2021)
"""

import numpy as np
import pandas as pd
import time
import fastadj

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics.pairwise import rbf_kernel


class kernel_vector_multiplication:
    """
    Kernel-Vector Multiplication: Compare Standard Kernel-Vector Multiplication with Kernel-Vector Multiplication with NFFT-Based Fast Summation in Runtime and Compute Approximation Error.
    
    Parameters
    ----------
    sigma : float, default=1.0
         Sigma parameter for the Gaussian kernel.
    norm : str, default='z-score'
        The normalization parameter determining how to standardize the features. It is either 'z-score' (z-score normalization) or None (the features are not standardized).
    setup : str, default='default'
        The setup argument loads presets for the parameters of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
    mis_threshold : float, default=0.0
        Mutual information score threshold determining, which features to include in the kernel. All features with a score below this threshold are dropped, the others are included.
    window_scheme : str, default='consec'
        The window-scheme argument determining, how the windows shall be built.
        If 'mis' is passed, the windows are built within this class based on their mutual information scores. If a pre_list is defined for this window_scheme, the predefined list of windows is taken and the rest of the features are seperated up into the windows by their mis.
        If window_scheme='consec', the windows are built following the feature indices in ascending order.
        If None, the list of windows of features is predefined by the pre_list argument.
    pre_list : list, default=[]
        If window_scheme=None, a predefined list of windows of features must be passed. Those windows are used to realize the kernel-vector multiplication.
        If window_scheme='mis', a predefined list of windows can be passed and the rest of the windows are determined by the features' mis.
    weight_scheme : str, default='equally weighted'
        The weighting-scheme determining, how the kernels are weighted.
        If 'equally weighted' is passed, all kernels are equally weighted, so that the weights sum up to 1.
        If weight_scheme='no weights', all weights are 1.
    
    Attributes
    ----------
    windows : list
        List of windows of features used to realize the kernel-vector multiplication.
    weights : list
        List of weights for the weighted sum of kernels.
        
    Examples
    --------
    >>> import numpy as np
    >>> N, d = 1000, 9
    >>> X = np.random.randn(N, d)
    >>> multiply = kernel_vector_multiplication(sigma=100)
    >>> p = np.random.rand((X.shape[0]))
    >>> multiply.compare_multiplication(X, p)
    """
    
    def __init__(self, sigma=1.0, norm='z-score', setup='default', mis_threshold = 0.0, window_scheme='consec', pre_list=[], weight_scheme='equally weighted'):
        
        self.sigma = sigma
        self.norm = norm
        self.setup = setup
        self.mis_threshold = mis_threshold
        self.window_scheme = window_scheme
        self.pre_list = pre_list
        self.weight_scheme = weight_scheme
        
    
    def z_score_normalization(self, X):
        """
        Z-score normalize the data.
            
        Parameters
        ----------
        X : ndarray
            The data which is to be z-score normalized.
            
        Returns
        -------
        X_norm : ndarray
            Z-score normalized data.
        """
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
        Approximate the kernel-vector product K*p.
            
        Note
        ----
        Using the NFFT-approach, the kernel matrices are never computed explicitly. 
        Since this function serves as a LinearOperator, only one variable, the vector p, can be passed as input parameter.
        The variables which are needed additionally to approximate K*p, are therefore defined as global variables in multiplication_with_NFFT, so that they can still be used within this function.
        
        Parameters
        ----------
        p : ndarray
            The vector, whose product K*p with the matrix K shall be approximated.
        
        Returns
        -------
        Kp_nfft : ndarray
            The approximated kernel-vector product K*p.
        """
        # number of samples
        N = X_fa.shape[0]
        
        ## setup computations with the adjacency matrices
        # set diagonal=1.0, since FastAdjacency package is targeted at graph Laplacian with zeros at the diagonal, but we need 1 at the diagonal
        adjacency_mats = [fastadj.AdjacencyMatrix(X_fa[:,self.windows[l]], self.sigma, setup=self.setup, diagonal=1.0) for l in range(len(self.windows))]
    
        # initialize matrix-vector product Kp_nfft
        Kp_nfft = np.zeros((N,))
    
        for l in range(len(self.windows)):
            Kp_nfft += self.weights[l] * adjacency_mats[l].apply(p)
        
        return Kp_nfft
    
    
    def multiplication_with_NFFT(self, X, p):    
        """
        Perform the kernel-vector multiplication with NFFT-approach.
            
        Parameters
        ----------
        X : ndarray
            The data matrix.
        p : ndarray
            The vector, the kernel shall be multiplied with.
            
        Returns
        -------
        Kp_nfft : ndarray
            The kernel-vector product, which is approximated using the NFFT-approach.
        """
    
        # declare global variables for the LinearOperator fast_adjacency
        global X_fa
        X_fa = X
        
        Kp_nfft = self.fast_adjacency(p)
        
        return Kp_nfft
    
    
    def standard_multiplication(self, X, p):    
        """
        Perform the standard kernel-vector multiplication.
            
        Parameters
        ----------
        X : ndarray
            The data matrix.
        p : ndarray
            The vector, the kernel shall be multiplied with.
            
        Returns
        -------
        Kp : ndarray
            The kernel-vector product, which is computed using the standard multiplication.
        """
        # compute kernel matrices
        mats = [rbf_kernel(X[:,self.windows[l]], gamma=(1/(self.sigma**2))) for l in range(len(self.windows))]
        
        # initialize matrix-vector product Kp
        Kp = np.zeros((X.shape[0],))
    
        for l in range(len(self.windows)):
            Kp += self.weights[l] * (mats[l] @ p)
        
        return Kp
           
    
    def compare_multiplication(self, X, p, y=None):
        """
        Compare the standard kernel-vector multiplication with kernel-vector multiplication with NFFT-approach in runtime and compute approximation error.

        Parameters
        ----------
        X : ndarray
            The data matrix.
        p : ndarray
            The vector, the kernel shall be multiplied with.
        y : ndarray, default=None
            The target values. (Only needed if the windows shall be determined using the MIS-ranking.)

        Returns
        -------
        time_standard: float
            runtime standard multiplication
        time_nfft: float
            runtime multiplication with NFFT-approach
        abs_error: float
            absolute error arising from the NFFT-based approximation
        rel_error: float
            relative error arising from the NFFT-based approximation
        """
        # scale data with z-score-normalization
        if self.norm == 'z-score':
            X = self.z_score_normalization(X)
    
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
        
        start_standard = time.time()
        
        # perform standard kernel-vector multiplication
        Kp = self.standard_multiplication(X, p)
        
        # runtime standard multiplication
        time_standard = time.time() - start_standard
        
        start_nfft = time.time()
        
        # perform kernel-vector multiplication with NFFT-approach
        Kp_nfft = self.multiplication_with_NFFT(X, p)
        
        # runtime multiplication with NFFT-approach
        time_nfft = time.time() - start_nfft
        
        # absolute approximation error
        abs_error = np.absolute(Kp - Kp_nfft)
        
        # relative approximation error
        rel_error = np.linalg.norm(Kp - Kp_nfft)/np.linalg.norm(Kp)
        
        return time_standard, time_nfft, abs_error, rel_error
        
    