import quicklens as ql
import numpy as np


r2d = 180./np.pi
d2r = np.pi/180.


def grab(var):
  return var.detach().cpu().numpy()



##################### power spectrum

def estimate_ps_ensemble(samples_true,nx,dx,lbins):
    ell_binned = lbins[:-1] + 0.5*np.diff(lbins)
    nmaps = samples_true.shape[0]
    cl_avg = np.zeros(ell_binned.shape[0])
    for map_id in range(nmaps):
        tmap = samples_true[map_id]
        tmap_cfft = ql.maps.rmap(nx, dx,map=tmap).get_cfft()
        cl = tmap_cfft.get_cl(lbins)
        cl_avg += cl.cl.real
    cl_avg = cl_avg/nmaps
    return cl_avg,ell_binned




###################### local non-gaussianity


def estimate_fnl_local_ensemble(samples_nongauss, samples_gauss, cl_theo, nx, dx):
    #estimates fnl local on a set of maps. you need to provide also gaussian maps which are used to estimate the variance
    #both sets need to be equal size and we need at least 1000 maps to make F converge sufficiently
    cl_theo_inv_nonan = np.copy(cl_theo) 
    #cl_theo_inv_nonan[0]=1.
    #cl_theo_inv_nonan[1]=1.
    cl_theo_inv_nonan[cl_theo_inv_nonan<0.00001] = 0.00001
    
    cl_theo_inv_nonan = 1./cl_theo_inv_nonan 
    
    fnl_unnormed_nongauss = np.zeros(samples_gauss.shape[0])
    fnl_unnormed_gauss = np.zeros(samples_gauss.shape[0])
    for i in range(samples_gauss.shape[0]):
        rmap = samples_nongauss[i]
        fnl_unnormed_nongauss[i] = estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx)

        rmap = samples_gauss[i] 
        fnl_unnormed_gauss[i] = estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx)

    F = np.var(fnl_unnormed_gauss)
    fnl_normed_nongauss = fnl_unnormed_nongauss/F
    fnl_normed_gauss = fnl_unnormed_gauss/F
    return fnl_normed_nongauss,fnl_normed_gauss
    
    
    
def estimate_fnl_local_unnormed_singlemap(rmap, cl_theo_inv_nonan, nx, dx):
    #https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py
    #FT conventions as in np.rfft2 irfft2
    rmap = ql.maps.rmap(nx, dx,map=rmap)
    rfft = rmap.get_rfft()
    
    #A map
    A = rmap.copy()
    
    #B map
    rfft_B = rfft * cl_theo_inv_nonan
    B = rfft_B.get_rmap()
    
    #multiply and integrate
    fnl_unnormed = (1./6.)*6*np.sum(A.map*A.map*B.map) * (dx**2) 
    
    return fnl_unnormed


###################### equilateral non-gaussianity



def estimate_fnl_equilateral_ensemble(samples_nongauss, samples_gauss, cl_theo_ell, nx, dx):
    ell_nonan = np.copy(cl_theo_ell) 
    ell_nonan[0]=1.

    fnlequi_unnormed_nongauss = np.zeros(samples_gauss.shape[0])
    fnlequi_unnormed_gauss = np.zeros(samples_gauss.shape[0])
    for i in range(samples_gauss.shape[0]):
        rmap = samples_nongauss[i]
        fnlequi_unnormed_nongauss[i] = estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx)
    
        rmap = samples_gauss[i] 
        fnlequi_unnormed_gauss[i] = estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx)
  
    Fequi = np.var(fnlequi_unnormed_gauss)
    fnlequi_normed_nongauss = fnlequi_unnormed_nongauss/Fequi
    fnlequi_normed_gauss = fnlequi_unnormed_gauss/Fequi

    return fnlequi_normed_nongauss,fnlequi_normed_gauss

    
def estimate_fnl_equilateral_unnormed_singlemap(rmap,ell_nonan, nx, dx):
    rmap = ql.maps.rmap(nx, dx,map=rmap)
    rfft = rmap.get_rfft()
    
    #A map
    rfft_A = rfft * np.power(ell_nonan,8/3)
    A = rfft_A.get_rmap()
    
    #B map
    rfft_B = rfft * np.power(ell_nonan,-1/3)
    B = rfft_B.get_rmap()
    
    #C map
    rfft_C = rfft * np.power(ell_nonan,5/3)
    C = rfft_C.get_rmap()
    
    #D map
    rfft_D = rfft * np.power(ell_nonan,2/3)
    D = rfft_D.get_rmap()
    
    #multiply and integrate
    As = 1. #placeholder
    fnl_unnormed = (1./As)*np.sum(-3*A.map*B.map*B.map + 6*B.map*C.map*D.map - 2*D.map*D.map*D.map) * (dx**2) 
    
    return fnl_unnormed