import numpy as np
from mpmath import hyp1f1,gamma,exp,power
from itertools import product
import matplotlib.pyplot as plt
import math

def fc(distances, smooth_width=0.5, cutoff=5):
    # Compute values of cutoff function
    cutoffs = np.ones_like(distances)
    mask = distances > cutoff - smooth_width
    cutoffs[mask] = 0.5 + 0.5 * np.cos(math.pi * (distances[mask] - cutoff + smooth_width) / smooth_width)
    cutoffs[distances > cutoff] = 0.

    return cutoffs.reshape(-1)

def draw_radial_basis(nmax=8,lmax=7,rcut=5,sigma=0.4, mesh_size=100):
    c = 0.5/sigma**2
    length = mesh_size
    dists = np.linspace(0, rcut+1e-6, length)
    y = o_ri_gto(rcut, nmax, lmax, dists, c)
    cutoffs = fc(dists)
    for l in range(lmax):
        for n in range(nmax):
            plt.plot(dists,y[:,l,n]*cutoffs, label=f'n={n}')
        plt.title(f'l = {l}')
        plt.legend()
        plt.show()

def dn(n,rcut,nmax):
    sn = rcut*max(np.sqrt(n),1)/nmax
    return 0.5/(sn)**2

def gto_norm(n,rcut,nmax):
    d = dn(n,rcut,nmax)
    return 1/np.sqrt(np.power(d,-n-1.5)*np.power(2,-n-2.5)*float(gamma(n+1.5)))

def ortho_Snn(rcut,nmax):
    Snn = np.zeros((nmax,nmax))
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    for n,m in product(range(nmax),range(nmax)):
        Snn[n,m] = norms[n]*norms[m]*0.5*np.power(ds[n]+ds[m], -0.5*(3+n+m)) * float(gamma(0.5*(3+m+n)))
    eigenvalues, unitary = np.linalg.eig(Snn)
    diagoverlap = np.diag(np.sqrt(eigenvalues) )
    newoverlap = np.dot(np.conj(unitary) ,np.dot(diagoverlap, unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)
    return orthomatrix

def gto(rcut,nmax, r):
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    ortho = ortho_Snn(rcut,nmax)
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    res = np.zeros((r.shape[0],nmax))
    for n in range(nmax):
        res[:,n] = norms[n]*np.power(r,n)*np.exp(-ds[n]*r**2)
    res = np.dot(res,ortho)
    return res

def ri_gto(n,l,rij,c,d, norm):
    res = exp(-c*rij**2)*(gamma(0.5*(l+n+3))/gamma(l+1.5)) * power(c*rij,l) * power(c+d,-0.5*(l+n+3))
    res *= hyp1f1(0.5*(n+l+3),l+1.5,power(c*rij,2)/(c+d))
    return norm*float(res)

def o_ri_gto(rcut,nmax,lmax,rij,c):
    ds = np.array([dn(n,rcut,nmax) for n in range(nmax)])
    norms = np.array([gto_norm(n,rcut,nmax) for n in range(nmax)])
    ortho = ortho_Snn(rcut,nmax)
    res = np.zeros((rij.shape[0],lmax,nmax))
    for ii,dist in enumerate(rij):
        for l in range(lmax):
            for n in range(nmax):
                res[ii,l,n] = ri_gto(n,l,float(dist),c,ds[n],norms[n])
    res = np.dot(res,ortho)
    return res