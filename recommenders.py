"""Functions for recommender systems"""

import numpy as np
from numpy import zeros, array, sqrt, log
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix, csr_matrix, eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd


# Functions for set or cosine based methods to calculate similarity ==============

def overlap(a,b):
    """
    Simple set based method to calculate similarity between two items.
    Looks at the number of users that the two items have in common.
    """
    return len(a.intersection(b))

def get_overlaps(item_sets, item_of_interest):
    """Get overlaps of multiple items with any item of interest"""
    for item in item_sets:
        print(item,':', overlap(item_sets[item_of_interest], item_sets[item]))

def norm2(v):
    """L2 norm"""
    return sqrt((v.data ** 2).sum())

def cosine(a, b):
    """Calculate the cosine of the angle between two vectors a and b"""
    return csr_matrix.dot(a, b.T)[0, 0] / (norm2(a) * norm2(b))

def get_sim_with_cos(items, item_of_interest):
    """Get overlaps of multiple items with any item of interest"""
    for item in items:
        print(item,':', cosine(items[item_of_interest], items[item]))


# Functions for matrix factorization methods =====================================

def alternating_least_squares(Cui, factors, regularization=0.01,
                              iterations=15, use_native=True, num_threads=0,
                              dtype=np.float64):
    """
    Factorizes the matrix Cui using an implicit alternating least squares algorithm
    Args:
        Cui (csr_matrix): Confidence Matrix
        factors (int): Number of factors to extract
        regularization (double): Regularization parameter to use
        iterations (int): Number of alternating least squares iterations to
        run
        num_threads (int): Number of threads to run least squares iterations.
        0 means to use all CPU cores.
    Returns:
        tuple: A tuple of (row, col) factors
    """
    #_check_open_blas()
 
    users, items = Cui.shape
 
    X = np.random.rand(users, factors).astype(dtype) * 0.01
    Y = np.random.rand(items, factors).astype(dtype) * 0.01
 
    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()
 
    solver = least_squares
 
    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization, num_threads)
        solver(Ciu, Y, X, regularization, num_threads)
        print("finished iteration %i in %s" % (iteration, time.time() - s))
 
    return X, Y
 
def least_squares(Cui, X, Y, regularization, num_threads):
    """ 
    For each user in Cui, calculate factors Xu for them using least squares on Y.
    """
    users, factors = X.shape
    YtY = Y.T.dot(Y)
 
    for u in range(users):
        # accumulate YtCuY + regularization*I in A
        A = YtY + regularization * np.eye(factors)
 
        # accumulate YtCuPu in b
        b = np.zeros(factors)
 
        for i, confidence in nonzeros(Cui, u):
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor
 
        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(A, b)

def bm25_weight(data, K1=100, B=0.8):
    """ 
    Weighs each row of the matrix data by BM25 weighting
    """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret

def nonzeros(m, row):
    """ 
    Returns the non zeroes of a row in csr_matrix
    """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

class ImplicitMF():
    '''
    Numerical value of implicit feedback indicates confidence that a user prefers an item. 
    No negative feedback- entries must be positive.
    '''
    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
 
    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
 
        for i in range(self.num_iterations):
            t0 = time.time()
            print('Solving for user vectors...')
            self.user_vectors = self.iteration(True, csr_matrix(self.item_vectors))
            print('Solving for item vectors...')
            self.item_vectors = self.iteration(False, csr_matrix(self.user_vectors))
            t1 = time.time()
            print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))
 
    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye1 = eye(num_fixed)
        lambda_eye = self.reg_param * eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))
 
        t = time.time()
        for i in range(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye1).dot(csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print('Solved %i vecs in %d seconds' % (i, time.time() - t))
                t = time.time()
 
        return solve_vecs


# Functions for getting recommendations ==========================================

class TopRelated_useruser(object):
    def __init__(self, user_factors):
        # fully normalize user_factors, so can compare with only the dot product
        norms = np.linalg.norm(user_factors, axis=-1)
        self.factors = user_factors / norms[:, np.newaxis]

    def get_related(self, movieid, N=10):
        scores = self.factors.dot(self.factors[movieid]) # taking dot product
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

class TopRelated_itemitem(object):
    def __init__(self, movie_factors):
        # fully normalize movie_factors, so can compare with only the dot product
        norms = np.linalg.norm(movie_factors, axis=-1)
        self.factors = movie_factors / norms[:, np.newaxis]

    def get_related(self, movieid, N=10):
        scores = self.factors.T.dot(self.factors.T[movieid])
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])
    
def print_top_items(itemname2itemid, recs):
    """Print recommendations and scores"""
    inv_dict = {v: k for k, v in itemname2itemid.items()}
    for item_code, score in recs:
        print(inv_dict[item_code], ":", score)
        
        
