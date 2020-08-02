#!/usr/bin/env/python
import numpy as np
from scipy.stats import binom

def binP(N, p, x1, x2):
    p = float(p)
    q = p/(1-p)
    k = 0.0
    v = 1.0
    s = 0.0
    tot = 0.0

    while(k<=N):
            tot += v
            if(k >= x1 and k <= x2):
                    s += v
            if(tot > 10**30):
                    s = s/10**30
                    tot = tot/10**30
                    v = v/10**30
            k += 1
            v = v*q*(N+1-k)/k
    return s/tot


def calcBin(vx, vN, vCL = 84.13, side=1, tol=10):
    """calcBin(vx, vN, vCL = 84.13, side=1, tol=10)

    Calculate the exact confidence interval for a binomial proportion

    Usage:
    >>> calcBin(13,100)
    (0.07107391357421874, 0.21204372406005856)
    >>> calcBin(4,7)
    (0.18405151367187494, 0.9010086059570312)

    Modifications by N. Secrest:
    - Calculates one-sided for comparison with Gehrels (1986).
    - Tolerance goes from 1e-5 to 1e-tol
    - Rounds to tol decimal place
    """

    vx = float(vx)
    vN = float(vN)
    #Set the confidence bounds
    vTU = (100 - float(vCL))/side
    vTL = vTU

    vP = vx/vN
    if(vx==0):
            dl = 0.0
    else:
            v = vP/2
            vsL = 0
            vsH = vP
            p = vTL/100

            while((vsH-vsL) > 10**-tol):
                    if(binP(vN, v, vx, vN) > p):
                            vsH = v
                            v = (vsL+v)/2
                    else:
                            vsL = v
                            v = (v+vsH)/2
            dl = v

    if(vx==vN):
            ul = 1.0
    else:
            v = (1+vP)/2
            vsL =vP
            vsH = 1
            p = vTU/100
            while((vsH-vsL) > 10**-tol):
                    if(binP(vN, v, 0, vx) < p):
                            vsH = v
                            v = (vsL+v)/2
                    else:
                            vsL = v
                            v = (v+vsH)/2
            ul = v

    n = int(10**tol)
    dl, ul = round(dl * n) / n, round(ul * n) / n

    return (dl, ul)


def bimean(n, N, CL=90, side=2, tol=10):
    """bimean(n, N, CL=90, side=2, tol=10)

    Convenience function to calculate mean (ll,ul). To calculate the
    mean and 99% confidence interval (the interval that contains 99% of
    the confidence of the mean) for n = 6, N = 10:

    u, ll, ul = bimean(6, 10, CL=99.5, side=1)

    CL(%): 99.5 (1 sided)
    Mean (lower, upper): 0.6 (0.191, 0.923)

    in agreement with the example given in Gehrels (1986). Note that
    this is the same as:

    u, ll, ul = bimean(6, 10, CL=99, side=2)

    The default of CL=90 with side=2 means that the confidence interval
    contains 90% of the uncertainty. The corresponding upper and lower
    limits are therefore 95%.

    """

    u = n / N
    ll, ul = calcBin(n, N, vCL=CL, side=side, tol=tol)

    print("CL(%%): %s (%s sided)" % (CL, side))

    print("Mean (lower, upper): %.3g (%.3g, %.3g)" % (u, ll, ul))

    return np.array([u, ll, ul])


def bip(n, N, CL=90, side=2, tol=10):
    """bip(n, N, CL=90, side=2, tol=10)

    Wrapper for bimean that outputs percentages and +/- bounds.

    Returns mean, +, -
    """

    u, ll, ul = bimean(n, N, CL=CL, side=side, tol=tol) * 100

    return u, ul - u, u - ll


def biratio(n0, N0, n1, N1, Ntrial=1e6, CL=95):
    """biratio(n0, N0, n1, N1, Ntrial=1e6, CL=95)

    Calculate ratio of two binomial distributions by using random
    numbers. The confidence limit CL is one-sided, so the upper limit
    is the CL percentile and the lower limit is the 100 - CL percentile.
    """

    Ntrial = int(Ntrial)

    x0 = np.random.binomial(N0, n0 / N0, Ntrial) / N0
    x1 = np.random.binomial(N1, n1 / N1, Ntrial) / N1

    # Remove 0 / 0 rows, since these give no information
    idx = np.where((x0==0) & (x1==0))
    msk = np.ones(Ntrial, dtype=bool)
    msk[idx] = False
    x0, x1 = x0[msk], x1[msk]

    x0x1 = np.inf * np.ones(x0.size)
    idx = np.where(x1>0)
    x0x1[idx] = x0[idx] / x1[idx]

    u  = n0 / n1 * N1 / N0
    ul = np.percentile(x0x1, CL)
    ll = np.percentile(x0x1, 100 - CL)

    return np.array([u, ll, ul])


def bir(n0, N0, n1, N1, Ntrial=1e6, CL=95):
    """bir(n0, N0, n1, N1, Ntrial=1e6, CL=95)

    Wrapper for biratio that ouputs ratio and +/- bounds.

    Returns excess, +, -
    """

    u, ll, ul = biratio(n0, N0, n1, N1, Ntrial=Ntrial, CL=CL)

    return u, ul - u, u - ll


def binull(n0, N0, f, eq='auto', Ntrial=int(1e6)):
    """binull(n0, N0, f, eq, Ntrial=int(1e6))

    Calculate the null hypothesis probability that a number equal to,
    greater than, or less than n0 was obtained from a sample of size N0,
    given a frequency of f. Uses scipy.stats.binom.

    Parameters
    ----------
    n0 : int
        Number from subsample of size N0
    N0 : int
        Size of subsample N0
    f : float
        Expected frequency corresponding to null hypothesis
    eq : str
        "<", "<=", "==", ">=", ">" for less than, less than or equal to,
        equal to, greater than or equal to, greater than. 'auto' will
        determine this from the ratio of n0/N0 to f. E.g., if n0/N0 is
        less than f, eq will be set to '<='.
    """

    if eq=="auto":
        if n0/N0 < f:
            eq = "<="
        if n0/N0 > f:
            eq = ">="
        if n0/N0 == f:
            eq = "=="

    if eq=="<":
        return binom.cdf(n0, N0, f, loc=1)

    if eq=="<=":
        return binom.cdf(n0, N0, f, loc=0)

    if eq=="==":
        return binom(N0, f).pmf(n0)

    if eq==">=":
        return 1 - binom.cdf(n0,N0,f,loc=1)

    if eq==">":
        return 1 - binom.cdf(n0,N0,f,loc=0)


def binull_mc(n0, N0, f, eq='auto', Ntrial=int(1e6)):
    """binull_mc(n0, N0, f, eq, Ntrial=int(1e6))

    Calculate the null hypothesis probability that a number equal to,
    greater than, or less than n0 was obtained from a sample of size N0,
    given a frequency of f. Uses numpy.random.binomial to do Monte Carlo
    simulations. Use this function to check against binull.

    Parameters
    ----------
    n0 : int
        Number from subsample of size N0
    N0 : int
        Size of subsample N0
    f : float
        Expected frequency corresponding to null hypothesis
    eq : str
        "<", "<=", "==", ">=", ">" for less than, less than or equal to,
        equal to, greater than or equal to, greater than. 'auto' will
        determine this from the ratio of n0/N0 to f. E.g., if n0/N0 is
        less than f, eq will be set to '<='.
    Ntrial : int
        Number of trials
    """
    x = np.random.binomial(N0, f, Ntrial)

    if eq=="auto":
        if n0/N0 < f:
            eq = "<="
        if n0/N0 > f:
            eq = ">="
        if n0/N0 == f:
            eq = "=="

    if eq=="<":
        return sum(x<n0)/Ntrial

    if eq=="<=":
            return sum(x<=n0)/Ntrial

    if eq=="==":
            return sum(x==n0)/Ntrial

    if eq==">=":
            return sum(x>=n0)/Ntrial
    if eq==">":
            return sum(x>n0)/Ntrial


if __name__=="__main__":
    # Test binull against binull_mc
    n0 = np.random.randint(0, 10)
    N0 = np.random.randint(n0, n0 + 10)
    f = np.random.uniform(0, 1)
    eq = np.random.choice(['<', '<=', '==', '>=', '>', '>='])
    p, p_mc = binull(n0, N0, f, eq), binull_mc(n0, N0, f, eq)
    err = p - p_mc
    print(n0, N0, f, eq)
    print(p, p_mc)
    print(err * 100)

