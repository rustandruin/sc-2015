from math import sqrt, log

import numpy as np
from numpy.linalg import norm, lstsq

import scipy as sp
from scipy.sparse.linalg import aslinearoperator

from rma_utils import matvec, parse_get_key
import time

def lsqr_spark( A, b, m, n, N, sc, tol=1e-14, iter_lim=None):
    """
    A simple version of LSQR on Spark
    A: rdd with idx
    b: np.array
    """

    lim = 'A'

    eps  = 32*np.finfo(float).eps;      # slightly larger than eps

    if tol < eps:
        tol = eps
    elif tol >= 1:
        tol = 1-eps

    max_n_stag = 3

    u    = b.squeeze().copy()
    beta = norm(u)
    if beta != 0:
        u   /= beta

    v = np.dot(matvec(A,u,'l',sc,lim),N)
    alpha = norm(v)
    if alpha != 0:
       v    /= alpha

    w     = v.copy()
    x     = np.zeros(n)

    phibar = beta
    rhobar = alpha

    nrm_a    = 0.0
    cnd_a    = 0.0
    sq_d     = 0.0
    nrm_r    = beta
    nrm_ar_0 = alpha*beta

    if nrm_ar_0 == 0:                     # alpha == 0 || beta == 0
        return x, 0, 0

    nrm_x  = 0
    sq_x   = 0
    z      = 0
    cs2    = -1
    sn2    = 0

    stag       = 0

    flag = -1
    if iter_lim is None:
        iter_lim = np.max( [20, 2*np.min([m,n])] )

    x_iter = []
    time_iter = []
    t0 = time.time()

    for itn in xrange(int(iter_lim)):

        #u    = A.matvec(v) - alpha*u
        #u    = np.dot(A,v) - alpha*u
        u    = matvec(A,np.dot(N,v),'r',sc,lim) - alpha*u
        beta = norm(u)
        u   /= beta

        # estimate of norm(A)
        nrm_a = sqrt(nrm_a**2 + alpha**2 + beta**2)

        #v     = A.rmatvec(u) - beta*v
        #v     = np.dot(u,A) - beta*v
        v     = np.dot(matvec(A,u,'l',sc,lim),N) - beta*v 
        alpha = norm(v)
        v    /= alpha

        rho    =  sqrt(rhobar**2+beta**2)
        cs     =  rhobar/rho
        sn     =  beta/rho
        theta  =  sn*alpha
        rhobar = -cs*alpha
        phi    =  cs*phibar
        phibar =  sn*phibar

        x      = x + (phi/rho)*w
        w      = v-(theta/rho)*w

        # estimate of norm(r)
        nrm_r   = phibar

        # estimate of norm(A'*r)
        nrm_ar  = phibar*alpha*np.abs(cs)

        # check convergence
        if nrm_ar < tol*nrm_ar_0:
            flag = 0
        #    break

        if nrm_ar < eps*nrm_a*nrm_r:
            flag = 0
        #    break

        # estimate of cond(A)
        sq_w    = np.dot(w,w)
        nrm_w   = sqrt(sq_w)
        sq_d   += sq_w/(rho**2)
        cnd_a   = nrm_a*sqrt(sq_d)

        # check condition number
        if cnd_a > 1/eps:
            flag = 1
        #    break

        # check stagnation
        if abs(phi/rho)*nrm_w < eps*nrm_x:
            stag += 1
        else:
            stag  = 0
        if stag >= max_n_stag:
            flag = 1
        #    break

        # estimate of norm(x)
        delta   =  sn2*rho
        gambar  = -cs2*rho
        rhs     =  phi - delta*z
        zbar    =  rhs/gambar
        nrm_x   =  sqrt(sq_x + zbar**2)
        gamma   =  sqrt(gambar**2 + theta**2)
        cs2     =  gambar/gamma
        sn2     =  theta /gamma
        z       =  rhs   /gamma
        sq_x   +=  z**2

        x_iter.append(x)
        time_iter.append( time.time() - t0 )

    y_iter = x_iter
    x_iter = [np.dot(N,x) for x in x_iter]
    return x_iter, y_iter, time_iter
        
def _test():

    m = 1e2
    n = 1e4
    r = 80
    c = 1e3                            # well-conditioned

    A, b, x_opt = _gen_prob( m, n, c, r )
    
    tol      = 1e-14
    iter_lim = 400 # np.ceil( (log(tol)-log(2.0))/log((c-1.0)/(c+1.0)) )

    x, flag, itn = lsqr(A,b,tol/c,iter_lim)
    relerr       = norm(x-x_opt)/norm(x_opt)

    if flag == 0:
        print "LSQR converged in %d iterations." % (itn,)
    else:
        print "LSQR didn't converge in %d iterations." % (itn,)

    if relerr < 1e-10:
        print "LSQR test passed with relerr %G." % (relerr,)
    else:
        print "LSQR test failed with relerr %G." % (relerr,)    

if __name__ == '__main__':
    _test()
    
