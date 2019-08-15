'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Fast block jackknives.

Everything in this module deals with 2D numpy arrays. 1D data are represented as arrays
with dimension (N, 1) or (1, N), to avoid bugs arising from numpy treating (N, ) as
a fundamentally different shape from (N, 1). The convention in this module is for the
first dimension to represent # of data points (or # of blocks in a block jackknife, since
a block is like a datapoint), and for the second dimension to represent the dimensionality
of the data.

'''

from __future__ import division
import numpy as np
from scipy.optimize import nnls
np.seterr(divide='raise', invalid='raise')
import quadprog
import time
import scipy.stats as stats
import pandas as pd
import scipy.linalg as la
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#minimize ||X*tau - y||^2_2 s.t. C.dot(tau) >= b
def solve_quadprog_quadprog(X, y, C, b):
    #convert pandas dataframes to numpy arrays if needed
    try: X=X.values
    except: pass
    try: y=y.values
    except: pass
    try: C=C.values
    except: pass
    try: b=b.values
    except: pass
    
    G = 2*X.T.dot(X)
    a = 2*X.T.dot(y)
    tau, cost, tau_unconstrained, iter_tuple, lagrangian, iact = quadprog.solve_qp(G, a, C=C.T, b=b, meq=0)
    return tau #, cost, tau_unconstrained, iter_tuple, lagrangian, iact
    


def _check_shape(x, y):
    '''Check that x and y have the correct shapes (for regression jackknives).'''
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError('x and y must be 2D arrays.')
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            'Number of datapoints in x != number of datapoints in y.')
    if y.shape[1] != 1:
        raise ValueError('y must have shape (n_snp, 1)')
    n, p = x.shape
    if p > n:
        raise ValueError('More dimensions than datapoints.')

    return (n, p)


def _check_shape_block(xty_block_values, xtx_block_values):
    '''Check that xty_block_values and xtx_block_values have correct shapes.'''
    if xtx_block_values.shape[0:2] != xty_block_values.shape:
        raise ValueError(
            'Shape of xty_block_values must equal shape of first two dimensions of xty_block_values.')
    if len(xtx_block_values.shape) < 3:
        raise ValueError('xtx_block_values must be a 3D array.')
    if xtx_block_values.shape[1] != xtx_block_values.shape[2]:
        raise ValueError(
            'Last two axes of xtx_block_values must have same dimension.')

    return xtx_block_values.shape[0:2]


class Jackknife(object):

    '''
    Base class for jackknife objects. Input involves x,y, so this base class is tailored
    for statistics computed from independent and dependent variables (e.g., regressions).
    The __delete_vals_to_pseudovalues__ and __jknife__ methods will still be useful for other
    sorts of statistics, but the __init__ method will need to be overriden.

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
    y : np.matrix with shape (n, 1)
        Dependent variable.
    n_blocks : int
        Number of jackknife blocks
    *args, **kwargs :
        Arguments for inheriting jackknives.

    Attributes
    ----------
    n_blocks : int
        Number of jackknife blocks
    p : int
        Dimensionality of the independent varianble
    N : int
        Number of datapoints (equal to x.shape[0])

    Methods
    -------
    jknife(pseudovalues):
        Computes jackknife estimate and variance from the jackknife pseudovalues.
    delete_vals_to_pseudovalues(delete_vals, est):
        Converts delete values and the whole-data estimate to pseudovalues.
    get_separators():
        Returns (approximately) evenly-spaced jackknife block boundaries.
    '''

    def __init__(self, x, y, n_blocks=None, separators=None):
        self.N, self.p = _check_shape(x, y)
        if separators is not None:
            if max(separators) != self.N:
                raise ValueError(
                    'Max(separators) must be equal to number of data points.')
            if min(separators) != 0:
                raise ValueError('Max(separators) must be equal to 0.')
            self.separators = sorted(separators)
            self.n_blocks = len(separators) - 1
        elif n_blocks is not None:
            self.n_blocks = n_blocks
            self.separators = self.get_separators(self.N, self.n_blocks)
        else:
            raise ValueError('Must specify either n_blocks are separators.')

        if self.n_blocks > self.N:
            raise ValueError('More blocks than data points.')

    @classmethod
    def jknife(cls, pseudovalues):
        '''
        Converts pseudovalues to jackknife estimate and variance.

        Parameters
        ----------
        pseudovalues : np.matrix pf floats with shape (n_blocks, p)

        Returns
        -------
        jknife_est : np.matrix with shape (1, p)
            Jackknifed estimate.
        jknife_var : np.matrix with shape (1, p)
            Variance of jackknifed estimate.
        jknife_se : np.matrix with shape (1, p)
            Standard error of jackknifed estimate, equal to sqrt(jknife_var).
        jknife_cov : np.matrix with shape (p, p)
            Covariance matrix of jackknifed estimate.

        '''
        n_blocks = pseudovalues.shape[0]
        jknife_cov = np.atleast_2d(np.cov(pseudovalues.T, ddof=1) / n_blocks)
        jknife_var = np.atleast_2d(np.diag(jknife_cov))
        jknife_se = np.atleast_2d(np.sqrt(jknife_var))
        jknife_est = np.atleast_2d(np.mean(pseudovalues, axis=0))
        return (jknife_est, jknife_var, jknife_se, jknife_cov)

    @classmethod
    def delete_values_to_pseudovalues(cls, delete_values, est):
        '''
        Converts whole-data estimate and delete values to pseudovalues.

        Parameters
        ----------
        delete_values : np.matrix with shape (n_blocks, p)
            Delete values.
        est : np.matrix with shape (1, p):
            Whole-data estimate.

        Returns
        -------
        pseudovalues : np.matrix with shape (n_blocks, p)
            Psuedovalues.

        Raises
        ------
        ValueError :
            If est.shape != (1, delete_values.shape[1])

        '''
        n_blocks, p = delete_values.shape
        if est.shape != (1, p):
            raise ValueError(
                'Different number of parameters in delete_values than in est.')

        return n_blocks * est - (n_blocks - 1) * delete_values

    @classmethod
    def get_separators(cls, N, n_blocks):
        '''Define evenly-spaced block boundaries.'''
        return np.floor(np.linspace(0, N, n_blocks + 1)).astype(int)


class LstsqJackknifeSlow(Jackknife):

    '''
    Slow linear-regression block jackknife. This class computes delete values directly,
    rather than forming delete values from block values. Useful for testing and for
    non-negative least squares (which as far as I am aware does not admit a fast block
    jackknife algorithm).

    Inherits from Jackknife class.

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
    y : np.matrix with shape (n, 1)
        Dependent variable.
    n_blocks : int
        Number of jackknife blocks
    nn: bool
        Non-negative least-squares?

    Attributes
    ----------
    est : np.matrix with shape (1, p)
        FWLS estimate.
    jknife_est : np.matrix with shape (1, p)
        Jackknifed estimate.
    jknife_var : np.matrix with shape (1, p)
        Variance of jackknifed estimate.
    jknife_se : np.matrix with shape (1, p)
        Standard error of jackknifed estimate, equal to sqrt(jknife_var).
    jknife_cov : np.matrix with shape (p, p)
        Covariance matrix of jackknifed estimate.
    delete_vals : np.matrix with shape (n_blocks, p)
        Jackknife delete values.

    Methods
    -------
    delete_values(x, y, func, s):
        Compute delete values of func(x, y) the slow way, with blocks defined by s.

    '''

    def __init__(self, x, y, n_blocks=None, nn=False, separators=None):
        Jackknife.__init__(self, x, y, n_blocks, separators)
        if nn:  # non-negative least squares
            func = lambda x, y: np.atleast_2d(nnls(x, np.array(y).T[0])[0])
        else:
            func = lambda x, y: np.atleast_2d(
                np.linalg.lstsq(x, np.array(y).T[0])[0])

        self.est = func(x, y)
        self.delete_values = self.delete_values(x, y, func, self.separators)
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.delete_values, self.est)
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
            self.jknife(self.pseudovalues)

    @classmethod
    def delete_values(cls, x, y, func, s):
        '''
        Compute delete values by deleting one block at a time.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        func : function (n, p) , (n, 1) --> (1, p)
            Function of x and y to be jackknived.
        s : list of ints
            Block separators.

        Returns
        -------
        delete_values : np.matrix with shape (n_blocks, p)
            Delete block values (with n_blocks blocks defined by parameter s).

        Raises
        ------
        ValueError :
            If x.shape[0] does not equal y.shape[0] or x and y are not 2D.

        '''
        _check_shape(x, y)
        d = [func(np.vstack([x[0:s[i], ...], x[s[i + 1]:, ...]]), np.vstack([y[0:s[i], ...], y[s[i + 1]:, ...]]))
             for i in xrange(len(s) - 1)]

        return np.concatenate(d, axis=0)


class LstsqJackknifeFast(Jackknife):

    '''
    Fast block jackknife for linear regression.

    Inherits from Jackknife class.

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
    y : np.matrix with shape (n, 1)
        Dependent variable.
    n_blocks : int
        Number of jackknife blocks

    Attributes
    ----------
    est : np.matrix with shape (1, p)
        FWLS estimate.
    jknife_est : np.matrix with shape (1, p)
        Jackknifed estimate.
    jknife_var : np.matrix with shape (1, p)
        Variance of jackknifed estimate.
    jknife_se : np.matrix with shape (1, p)
        Standard error of jackknifed estimate, equal to sqrt(jknife_var).
    jknife_cov : np.matrix with shape (p, p)
        Covariance matrix of jackknifed estimate.
    delete_vals : np.matrix with shape (n_blocks, p)
        Jackknife delete values.

    Methods
    -------
    block_values(x, y, n_blocks) :
        Computes block values for the regression y~x.
    block_values_to_est(block_values) :
        Computes whole-data estimate from block values.
    block_values_to_pseudovalues(block_values, est) :
        Computes pseudovalues and delete values in a single pass over the block values.

    '''

    def __init__(self, x, y, n_blocks=None, separators=None, chr=None):
        Jackknife.__init__(self, x, y, n_blocks, separators)
        xty, xtx = self.block_values(x, y, self.separators)
        self.est = self.block_values_to_est(xty, xtx)
        self.delete_values = self.block_values_to_delete_values(xty, xtx)
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.delete_values, self.est)
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
            self.jknife(self.pseudovalues)
            
        #evaluate OOC prediction accuracy
        if chr is not None:
            chromosomes = np.sort(np.unique(chr))
            assert len(chromosomes) > 1
            assert y.shape[1]==1
            y_pred = np.empty(chr.shape[0])
            XTX = x.T.dot(x)
            XTy = y[:,0].dot(x)
            assert np.allclose(np.linalg.solve(XTX, XTy), self.est[0])
            for chr_i, left_out_chr in enumerate(chromosomes):        
                is_chr = (chr == left_out_chr)
                chr_inds = np.where(is_chr)[0]
                assert np.all(chr_inds == np.arange(chr_inds[0], chr_inds[-1]+1))
                chr_slice = slice(chr_inds[0], chr_inds[-1]+1)
                x_chr = x[chr_slice]
                y_chr = y[chr_slice,0]
                XTX_loco = XTX - x_chr.T.dot(x_chr)
                XTy_loco = XTy - y_chr.dot(x_chr)
                tau_loco = np.linalg.solve(XTX_loco, XTy_loco)
                y_pred[chr_slice] = x_chr.dot(tau_loco)
                
            #Estimate OOC score
            self.ooc_score = r2_score(y[:,0], y_pred)
            
            

        

    @classmethod
    def block_values(cls, x, y, s):
        '''
        Compute block values.

        Parameters
        ----------
        x : np.matrix with shape (n, p)
            Independent variable.
        y : np.matrix with shape (n, 1)
            Dependent variable.
        n_blocks : int
            Number of jackknife blocks
        s : list of ints
            Block separators.

        Returns
        -------
        xty_block_values : np.matrix with shape (n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : 3d np array with shape (n_blocks, p, p)
            Block values of X^T X.

        Raises
        ------
        ValueError :
            If x.shape[0] does not equal y.shape[0] or x and y are not 2D.

        '''
        n, p = _check_shape(x, y)
        n_blocks = len(s) - 1
        xtx_block_values = np.zeros((n_blocks, p, p))
        xty_block_values = np.zeros((n_blocks, p))
        for i in xrange(n_blocks):
            xty_block_values[i, ...] = np.dot(
                x[s[i]:s[i + 1], ...].T, y[s[i]:s[i + 1], ...]).reshape((1, p))
            xtx_block_values[i, ...] = np.dot(
                x[s[i]:s[i + 1], ...].T, x[s[i]:s[i + 1], ...])

        return (xty_block_values, xtx_block_values)

    @classmethod
    def block_values_to_est(cls, xty_block_values, xtx_block_values):
        '''
        Converts block values to the whole-data linear regression estimate.

        Parameters
        ----------
        xty_block_values : np.matrix with shape (n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : 3D np.array with shape (n_blocks, p, p)
            Block values of X^T X

        Returns
        -------
        est : np.matrix with shape (1, p)
            Whole data estimate.

        Raises
        ------
        LinAlgError :
            If design matrix is singular.
        ValueError :
            If the last two dimensions of xtx_block_values are not equal or if the first two
        dimensions of xtx_block_values do not equal the shape of xty_block_values.

        '''
        n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
        xty = np.sum(xty_block_values, axis=0)
        xtx = np.sum(xtx_block_values, axis=0)
        return np.linalg.solve(xtx, xty).reshape((1, p))

    @classmethod
    def block_values_to_delete_values(cls, xty_block_values, xtx_block_values):
        '''
        Converts block values to delete values.

        Parameters
        ----------
        xty_block_values : np.matrix with shape (n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : 3D np.array with shape (n_blocks, p, p)
            Block values of X^T X
        est : np.matrix with shape (1, p)
            Whole data estimate

        Returns
        -------
        delete_values : np.matrix with shape (n_blocks, p)
            Delete Values.

        Raises
        ------
        LinAlgError :
            If delete design matrix is singular.
        ValueError :
            If the last two dimensions of xtx_block_values are not equal or if the first two
        dimensions of xtx_block_values do not equal the shape of xty_block_values.

        '''
        n_blocks, p = _check_shape_block(xty_block_values, xtx_block_values)
        delete_values = np.zeros((n_blocks, p))
        xty_tot = np.sum(xty_block_values, axis=0)
        xtx_tot = np.sum(xtx_block_values, axis=0)
        for j in xrange(n_blocks):
            delete_xty = xty_tot - xty_block_values[j]
            delete_xtx = xtx_tot - xtx_block_values[j]
            delete_values[j, ...] = np.linalg.solve(
                delete_xtx, delete_xty).reshape((1, p))

        return delete_values


class RatioJackknife(Jackknife):

    '''
    Block jackknife ratio estimate.

    Jackknife.

    Parameters
    ----------
    est : float or np.array with shape (1, p)
        Whole data ratio estimate
    numer_delete_values : np.matrix with shape (n_blocks, p)
        Delete values for the numerator.
    denom_delete_values: np.matrix with shape (n_blocks, p)
        Delete values for the denominator.

    Methods
    -------
    delete_vals_to_pseudovalues(est, denom, num):
        Converts denominator/ numerator delete values and the whole-data estimate to
        pseudovalues.

    Raises
    ------
    FloatingPointError :
        If any entry of denom_delete_values is zero.

    Note that it is possible for the denominator to cross zero (i.e., be both positive
    and negative) and still have a finite ratio estimate and SE, for example if the
    numerator is fixed to 0 and the denominator is either -1 or 1. If the denominator
    is noisily close to zero, then it is unlikely that the denominator will yield zero
    exactly (and therefore yield an inf or nan), but delete values will be of the form
    (numerator / close to zero) and -(numerator / close to zero), i.e., (big) and -(big),
    and so the jackknife will (correctly) yield huge SE.

    '''

    def __init__(self, est, numer_delete_values, denom_delete_values):
        if numer_delete_values.shape != denom_delete_values.shape:
            raise ValueError(
                'numer_delete_values.shape != denom_delete_values.shape.')
        if len(numer_delete_values.shape) != 2:
            raise ValueError('Delete values must be matrices.')
        if len(est.shape) != 2 or est.shape[0] != 1 or est.shape[1] != numer_delete_values.shape[1]:
            raise ValueError(
                'Shape of est does not match shape of delete values.')

        self.n_blocks = numer_delete_values.shape[0]
        self.est = est
        self.pseudovalues = self.delete_values_to_pseudovalues(self.est,
                                                               denom_delete_values, numer_delete_values)
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
            self.jknife(self.pseudovalues)

    @classmethod
    def delete_values_to_pseudovalues(cls, est, denom, numer):
        '''
        Converts delete values to pseudovalues.

        Parameters
        ----------
        est : np.matrix with shape (1, p)
            Whole-data ratio estimate.
        denom : np.matrix with shape (n_blocks, p)
            Denominator delete values.
        numer : np.matrix with shape (n_blocks, p)
            Numerator delete values.

        Returns
        -------
        pseudovalues :
            Ratio Jackknife Pseudovalues.

        Raises
        ------
        ValueError :
            If numer.shape != denom.shape.

        '''
        n_blocks, p = denom.shape
        pseudovalues = np.zeros((n_blocks, p))
        for j in xrange(0, n_blocks):
            pseudovalues[j, ...] = n_blocks * est - \
                (n_blocks - 1) * numer[j, ...] / denom[j, ...]

        return pseudovalues

        

class Jackknife_Ridge(Jackknife):

    def __init__(self, x, y, n_blocks=None, separators=None, chr=None, verbose=True,
        num_lambdas=100, approx_ridge=False, 
        ridge_lambda=None, use_1se=False, has_intercept=False, standardize=True,
        nonneg_constraints=None
        ):
        
        #sanity checks
        assert chr is not None
        assert len(np.unique(chr)) > 1
        
        #make sure we work with numpy arrays, not dataframes
        try: x=x.values
        except: pass
        try: y=y.values
        except: pass
        try: chr=chr.values
        except: pass
        try: nonneg_constraints=nonneg_constraints.values
        except: pass        
        
        #init stuff
        Jackknife.__init__(self, x, y, n_blocks=n_blocks, separators=separators)
        self.use_1se = use_1se
        self.verbose=verbose        
        self.has_intercept = has_intercept
        self.nonneg_constraints = nonneg_constraints
        
        #define chromosome sets        
        #make y look like a vector
        assert y.shape[1]==1
        y = y[:,0]
      
        #standardize x
        if standardize:
            x_l2 = np.sqrt(np.einsum('ij,ij->j', x, x))
            x /= x_l2
        else:
            x_l2 = None
        
        #Create a set of ridge lambdas to evaluate
        XTX_all = x.T.dot(x)
        XTy_all = y.dot(x)
        mean_diag = np.mean(np.diag(XTX_all))
        self.ridge_lambdas = np.logspace(np.log10(mean_diag*1e-8), np.log10(mean_diag*1e2), num=num_lambdas)
        
        #find best lambda (using OOC estimation) and estimate taus
        if ridge_lambda is not None:
            assert approx_ridge            
            best_lambda = ridge_lambda
            
            #estimate OOC score
            chromosomes = np.sort(np.unique(chr))
            assert len(chromosomes) > 1
            y_pred = np.empty(chr.shape[0])
            XTX = x.T.dot(x)
            XTy = y.dot(x)
            for chr_i, left_out_chr in enumerate(chromosomes):
                is_chr = (chr == left_out_chr)
                chr_inds = np.where(is_chr)[0]
                assert np.all(chr_inds == np.arange(chr_inds[0], chr_inds[-1]+1))
                chr_slice = slice(chr_inds[0], chr_inds[-1]+1)
                x_chr = x[chr_slice]
                y_chr = y[chr_slice]
                XTX_loco = XTX - x_chr.T.dot(x_chr)
                XTy_loco = XTy - y_chr.dot(x_chr)
                tau_loco = self._est_ridge(XTX_loco, XTy_loco, best_lambda)
                y_pred[chr_slice] = x_chr.dot(tau_loco)                    
            #estimate OOC score 
            self.ooc_score = r2_score(y, y_pred)
                
        else:
            best_lambda, ooc_score = self._find_best_lambda(x, y, XTX_all, XTy_all, chr)
            self.ooc_score = ooc_score
        self.est = np.atleast_2d(self._est_ridge(XTX_all, XTy_all, best_lambda))
        if standardize: self.est /= x_l2

        
        #run jackknife        
        self.delete_values = np.empty((len(self.separators)-1, self.est.shape[1]))            
        for block_i in xrange(len(self.separators) - 1):
        
            if verbose:
               print 'Starting Ridge jackknife iteration %d/%d'%(block_i+1, len(self.separators) - 1)
        
            #prepare data structures
            x_block = x[self.separators[block_i]:self.separators[block_i+1], ...]
            y_block = y[self.separators[block_i]:self.separators[block_i+1], ...]
            XTX_noblock = XTX_all - x_block.T.dot(x_block)
            XTy_noblock = XTy_all - y_block.dot(x_block)
            slice_block = slice(self.separators[block_i], self.separators[block_i+1])
            x_noblock = np.delete(x, slice_block, axis=0)
            y_noblock = np.delete(y, slice_block, axis=0)
            chr_noblock = np.delete(chr, slice_block, axis=0)

            #find best lambda for this jackknife block
            if approx_ridge:
                best_lambda_noblock = best_lambda
            else:
                best_lambda_noblock, _ = self._find_best_lambda(x_noblock, y_noblock, XTX_noblock, XTy_noblock, chr_noblock)
            
            #main jackknife estimation
            est_block = self._est_ridge(XTX_noblock, XTy_noblock, best_lambda_noblock)            
            self.delete_values[block_i, ...] = est_block
        if standardize: self.delete_values /= x_l2

        #compute jackknife pseudo-values
        self.pseudovalues = self.delete_values_to_pseudovalues(self.delete_values, self.est)
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) = self.jknife(self.pseudovalues)
        
        #restore original x
        if standardize: x *= x_l2

        
    def _find_best_lambda(self, x, y, XTX, XTy, chr):
        chromosomes = np.sort(np.unique(chr))
        assert len(chromosomes) > 1
        num_lambdas = len(self.ridge_lambdas)
        y_pred_lambdas = np.empty((chr.shape[0], num_lambdas))
        if self.verbose:
            y_pred_lambdas_lstsq = np.empty(chr.shape[0])
        for chr_i, left_out_chr in enumerate(chromosomes):        
            is_chr = (chr == left_out_chr)
            chr_inds = np.where(is_chr)[0]
            assert np.all(chr_inds == np.arange(chr_inds[0], chr_inds[-1]+1))
            chr_slice = slice(chr_inds[0], chr_inds[-1]+1)
            x_chr = x[chr_slice]
            y_chr = y[chr_slice]            
            XTX_loco = XTX - x_chr.T.dot(x_chr)
            XTy_loco = XTy - y_chr.dot(x_chr)
            y_pred_lambdas[chr_slice, :] = self._predict_lambdas(XTX_loco, XTy_loco, x_chr)
            
            if self.verbose:
                tau_lstsq_loco = self._est_ridge(XTX_loco, XTy_loco, 0)
                y_pred_lambdas_lstsq[chr_slice] = x_chr.dot(tau_lstsq_loco)
            
        #find the best ridge lambda
        score_lambdas = np.empty(num_lambdas)
        #score_lambdas_kendall = np.empty(num_lambdas)
        #score_lambdas_spearman = np.empty(num_lambdas)
        for r_i in xrange(num_lambdas):
            score_lambdas[r_i] = r2_score(y, y_pred_lambdas[:,r_i])
            #score_lambdas_kendall[r_i] = stats.kendalltau(y, y_pred_lambdas[:,r_i])[0]
            #score_lambdas_spearman[r_i] = stats.spearmanr(y, y_pred_lambdas[:,r_i])[0]
        #import ipdb; ipdb.set_trace()
        best_lambda_index = np.argmax(score_lambdas)
        
        #choose lambda based on the 1SE rule?
        if self.use_1se:            
            score_folds = np.empty(len(chromosomes))
            for chr_i, left_out_chr in enumerate(chromosomes):
                is_chr = (chr == left_out_chr)
                score_folds[chr_i] = r2_score(y[is_chr], y_pred_lambdas[is_chr, best_lambda_index])
                #score_folds[chr_i] = stats.kendalltau(y[is_chr], y_pred_lambdas[is_chr, best_lambda_index])[0]
            scores_std = np.std(score_folds)
            best_score = score_lambdas[best_lambda_index]
            assert np.isclose(best_score, np.max(score_lambdas))
            best_lambda_index = np.where(score_lambdas > best_score - scores_std)[0][-1]
        
        best_lambda = self.ridge_lambdas[best_lambda_index]        
        ooc_score = score_lambdas[best_lambda_index]
        if self.verbose:
            score_lstsq = r2_score(y, y_pred_lambdas_lstsq)            
            print 'Selected ridge lambda: %0.4e (%d/%d)  score: %0.4e  score lstsq: %0.4e'%(best_lambda, 
                best_lambda_index+1, num_lambdas, ooc_score, score_lstsq)
            
        return best_lambda, ooc_score
        
        
    def _predict_lambdas(self, XTX_train, XTy_train, X_validation):
        tau_est_ridge = np.empty((XTX_train.shape[0], len(self.ridge_lambdas)))
        for r_i, r in enumerate(self.ridge_lambdas):
            tau_est_ridge[:, r_i] = self._est_ridge(XTX_train, XTy_train, r)
        y_pred = X_validation.dot(tau_est_ridge)
        return y_pred
        
    
    def _est_ridge(self, XTX, XTy, ridge_lambda):
        if ridge_lambda==0:
            R=XTX
        else:
            I = np.eye(XTX.shape[0]) * ridge_lambda
            if self.has_intercept: I[-1,-1]=0
            R = XTX+I
        if self.nonneg_constraints is None:
            est = np.linalg.solve(R, XTy)
        else:
            b = np.zeros(self.nonneg_constraints.shape[0])
            G = 2*R
            a = 2*XTy
            est, cost, est_unconstrained, iter_tuple, lagrangian, iact = quadprog.solve_qp(G, a, C=self.nonneg_constraints.T, b=b, meq=0)
        return est
            
            



