from numba import njit

from multiprocessing import shared_memory

import os

import numpy as np

from scipy.ndimage import shift
from scipy.optimize import differential_evolution, minimize
from scipy.signal import correlate
from scipy.signal import correlation_lags

from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

from CRYOtools.io import _read_solar_model, _read_telluric_model

@njit(fastmath=True)
def gaussian_filter1d_numba(input_array, sigma=2):
    ''' Applies a Gaussian 1D convolution to 1d array '''
    # Generate Gaussian kernel
    radius = int(3 * sigma + 0.5) ## USING ONLY 3 STD DEVS FOR SPEED
    x = np.arange(-radius, radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()
    output = np.zeros_like(input_array)
    for i in range(radius,len(input_array)-radius):
        for k in range(len(gaussian_kernel)):
            j = i + k - radius
            output[i] += input_array[j] * gaussian_kernel[k]
    ## edge treatment
    output[:radius] = input_array[:radius]
    output[(-radius):] = input_array[(-radius):]
    return output


@njit(fastmath=True)
def gaussian(x, a, b, c):
    expo = (x-b)/c
    return a*np.exp(-expo**2/2)


    
def fft_shift(input, shift):
    N = input.shape[0]
    x = np.arange(-N / 2, N / 2)
    kx = -1j * 2 * np.pi * x / N
    finput = np.fft.fftshift(np.fft.fftn(input))
    shifted_finput = finput * np.exp(-(kx*shift))
    shifted_input = np.real(np.fft.ifftn(np.fft.ifftshift(shifted_finput)))
    return shifted_input
    


def do_conv(x, y, Rpow_log):
    '''
    Helper function to do convolution in model
    '''
    fwhm_wv = x.mean()/np.exp(Rpow_log)
    sigm_wv = fwhm_wv / (2.*np.sqrt(2.*np.log(2))) 
    dwv = x[0]-x[1]
    kern_pix = sigm_wv / np.abs(dwv) 
    y_conv = gaussian_filter1d_numba(y, sigma=kern_pix)
    return y_conv

def standard(x):
    return (x-x.mean())/x.std()

def parabola(x, y):
  # Return minimum position of parabola given coordinates of 3 data points
  return x[2]-(y[2]-y[1])/(y[2]-2.*y[1]+y[0]) -0.5

def get_lags(data, atlas, spec_coords, Solar=True):
    if Solar:
        index = np.where((spec_coords> 1074.85) & (spec_coords < 1075.05) )[0]
        atlas_cp = deepcopy(atlas)
    else:
        index = np.where((spec_coords> 1074.27) & (spec_coords < 1074.4) )[0]
        # telluric profiles narrow, so broaden by a reasonable amount
        atlas_cp = do_conv(spec_coords, atlas*np.median(data), 10.7)
        
    x1 = data[index]
    x2 = atlas_cp[index]
    corr = correlate(standard(x1), standard(x2) , 'same', method='fft')
    lags = correlation_lags(x1.size, x2.size,'same')

    # get three points around maxima
    max_loc = np.argmax(corr)
    locs = np.array([-1,0,1])+max_loc
    # get sub-pixel estimate of shift
    shift = parabola(lags[locs], corr[locs])
    return -shift

def get_lags_lin(data, atlas, spec_coords, Solar=True):
    if Solar:
        index = np.where((spec_coords> 1074.85) & (spec_coords < 1075.05) )[0]
        atlas_cp = deepcopy(atlas)
    else:
        index = np.where((spec_coords> 1074.27) & (spec_coords < 1074.4) )[0]
        # telluric profiles narrow, so broaden by a reasonable amount
        atlas_cp = do_conv(spec_coords, atlas*np.median(data), 10.7)
    
    # estimate and subtract linear function
    ind_1074 = np.argmin(spec_coords-1074)
    y_est = data[ind_1074]
    c = y_est - 0.25*1074

    x1 = data[index] - (0.25*spec_coords[index]+c)
    x2 = atlas_cp[index]
    corr = corr_sp(standard(x1), standard(x2) , 'same', method='fft')
    lags = correlation_lags(x1.size, x2.size,'same')

    # get three points around maxima
    max_loc = np.argmax(corr)
    locs = np.array([-1,0,1])+max_loc
    # get sub-pixel estimate of shift
    shift = parabola(lags[locs], corr[locs])
    return -shift

def vac2air(wave_vac):
    """ Converts wavelengths from vacuum to air-equivalent """
    wave_air = np.copy(wave_vac)
    ww = (wave_vac >= 2000)
    sigma2 = (1e4 / wave_vac[ww])**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - sigma2) + 0.00015998 / (38.9 - sigma2) 
    wave_air[ww] = wave_vac[ww] / n
    return wave_air

def get_solar_model(wave_len):
    dc = _read_solar_model()

    dcwv = vac2air(1e7/dc[:,0]*10)
    dcsp = dc[:,1]

    ftscor = np.interp(wave_len*10,dcwv[::-1],dcsp[::-1])
    return ftscor

def get_telluric_model(wave_len):
    ## Load HITRAN MODEL TELLURIC SPECTRUM FOR WATER
    hdat = _read_telluric_model()
    hwv = np.flip(hdat[0])
    hitran = np.flip(hdat[1])
    ftsatm = np.interp(wave_len, hwv, hitran)
    return ftsatm

def create_bounds(bounds=None):
    '''
    Bounds for fitting parameters.

    These seems to work okay.
    '''

    if not bounds:
        line_amp = (0,50)
        del_lam = (1074.63 - 6/3e5*1074.63 -1074.,1074.63 + 6/3e5*1074.63-1074.)
        sigma = (0.055, 0.1)
        rpow_log = (np.log(30000), np.log(65000))
        opac = (0.5,10)
        vel_solar = (-2,2)  
        vel_tell = (-3.2,-2.5)
        strayfrac = (0.,0.5)
        icont = (0,0)
        icont_lin = (0,1)

        bounds = [line_amp, del_lam, sigma, rpow_log,
                        opac, vel_solar, vel_tell, strayfrac, icont, icont_lin]

    return bounds

@njit(fastmath=True)
def build_model(params, x, fts_cor, log_fts_atm):

    wv_mean = x.mean()
    amp,lam_0,sigma,Rpow_log,opac,velS,velT,strayfrac,icont, icont_lin = params

    ## ifit -- coronal line
    gfit = gaussian(x-1074., amp, lam_0, sigma)
                  
    ftsSmod = np.copy(fts_cor)
    ## shift coronal data
    #ftsSmod = fft_shift(ftsSmod, velS)
    ftsSmod = np.interp(x,x + velS/3e5*wv_mean,ftsSmod)

    ## scale and shift telluric spectra
    ftsTmod = np.exp(opac*log_fts_atm)
    #ftsTmod = fft_shift(ftsTmod, velT)
    ftsTmod = np.interp(x,x + velT/3e5*wv_mean,ftsTmod)
    

    ftsmod = ftsSmod * ftsTmod
            
    ## add straylight
    ftsmod = (ftsmod + strayfrac) / (1. + strayfrac) 
    ## scale for total
    ftsmod = ftsmod*(icont+icont_lin*x)
            
    ifit = ftsmod + gfit * ftsTmod # note that telluric absorption is also applied to the coronal line 
            
    ## convolution for spectrograph line spread function
    ## Gaussian convolution of the FTS atlas
    fwhm_wv = x.mean()/np.exp(Rpow_log)
    sigm_wv = fwhm_wv / (2.*np.sqrt(2.*np.log(2))) 
    dwv = x[0]-x[1]
    kern_pix = sigm_wv / np.abs(dwv) 
    ifit = gaussian_filter1d_numba(ifit, sigma=kern_pix)
            
    return ifit

@njit(fastmath=True)
def _loss(params,y, x, wgts, fts_cor, log_fts_atm):
    y_hat = build_model(params, x, fts_cor, log_fts_atm)
    return np.mean((y_hat-y)**2 * wgts)

def _fitting_task(res, y, x, wgts):
      #res = differential_evolution(loss_int, res.x, args=(y,x,wgts), tol=1.e-2,maxiter = 50,popsize = 1)#, polish=True)
   return minimize(_loss, res.x, args=(y, x, wgts, fts_cor_shared, log_fts_atm_shared), method='BFGS', tol=1e-4)

def _unpack_and_run(args):
   return _fitting_task(*args)

def worker_init(fts_name, fts_shape, fts_dtype, log_name, log_shape, log_dtype):
    import numpy as np
    from multiprocessing import shared_memory

    # make them globals so your task functions can see them
    existing_fts = shared_memory.SharedMemory(name=fts_name)
    global fts_cor_shared
    fts_cor_shared = np.ndarray(fts_shape, dtype=fts_dtype, buffer=existing_fts.buf)

    existing_log = shared_memory.SharedMemory(name=log_name)
    global log_fts_atm_shared
    log_fts_atm_shared = np.ndarray(log_shape, dtype=log_dtype, buffer=existing_log.buf)


def fit_data_test(data: np.ndarray, spec_coords: np.array, 
                  do_diff: int =0, wgts: np.array = None,
                  use_tqdm: bool = False, cpu_max=4, bounds=None):
    '''
        Perform a fit to a slit/exposure.

        Strategy is to use differential_evolution to find good starting values for
        the model parameters. This is done once per slit (as default) and previous
        profiles values are used as next starting values.

        Input
        img - Cryo slit (from one fits file) with shape [slit_len, wave_len]

        do_diff - modulo number of pixels to run differential evolution

    '''


    n_along_slit = data.shape[0]
    n_wv = data.shape[1]

    if do_diff == 0:
       do_diff = n_along_slit
    else:
        do_diff = do_diff
    
    res_slits = []


    fts_cor = get_solar_model(spec_coords) 
    fts_atm = get_telluric_model(spec_coords)
    log_fts_atm = np.log(fts_atm)

    # create shared memory for each
    shm_fts = shared_memory.SharedMemory(create=True, size=fts_cor.nbytes)
    shm_log = shared_memory.SharedMemory(create=True, size=log_fts_atm.nbytes)

    # wrap them as ndarrays and copy data in
    shared_fts = np.ndarray(fts_cor.shape, dtype=fts_cor.dtype, buffer=shm_fts.buf)
    shared_fts[:] = fts_cor
    shared_log = np.ndarray(log_fts_atm.shape, dtype=log_fts_atm.dtype, buffer=shm_log.buf)
    shared_log[:] = log_fts_atm

    if not bounds:
        bounds = create_bounds()

    if wgts is None:
        wgts = np.ones(n_wv, np.float64)

    #jac_loss = jacobian(loss)

    x= spec_coords
    y = data[0]
    velS_est = get_lags(y, fts_cor, x, Solar=True)
    velT_est = get_lags(y, fts_atm, x, Solar=False)
    bounds[5] = (velS_est-0.5, velS_est+0.5)
    bounds[6] = (velT_est-0.5, velT_est+0.5)
    bounds[8] = (-1000,-500)#(np.nanmedian(y)-10,np.nanmedian(y)+10)
    
   
    res = differential_evolution(_loss, bounds, args=(y,x,wgts, fts_cor, log_fts_atm), tol=1.e-2,maxiter = 50,popsize = 1)

    # Serialize headers to strings (fully picklable) 
    #tasks = [(res, data[i], x, wgts, fts_cor, log_fts_atm)  for i in iterator]

    ncpus = min(os.cpu_count() , cpu_max)

    with ProcessPoolExecutor(max_workers=ncpus,
                            initializer=worker_init,
                            initargs=(
                            shm_fts.name, fts_cor.shape, fts_cor.dtype,
                            shm_log.name, log_fts_atm.shape, log_fts_atm.dtype
                             )
                            ) as executor:
        tasks = [(res, data[i], x, wgts) for i in range(data.shape[0])]
        results = list(tqdm(executor.map(_fitting_task, *zip(*tasks)), total=data.shape[0],desc="Processing"))
        #res_slits = list(tqdm(executor.map(_unpack_and_run, tasks), total=data.shape[0], desc="Processing"))

    return results



def calculate_model(self):

    if hasattr(self, 'res_slits'):
         return [self.build_model(r.x) for r in self.res_slits]
    else:
        return "Modelling has not be undertaken."
   

def plot_model_vs_data(self, vmin=-1, vmax=1):

    self.model = calculate_model(self)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(self.data)
    ax[0].set_title('Data')
    ax[1].imshow(self.model)
    ax[1].set_title('Model')
    pl = ax[2].imshow(self.data-self.model, vmin=vmin, vmax=vmax)
    #plt.colorbar(pl)
    ax[2].set_title('Residual \n clipped {0} to {1}'.format((vmin, vmax)))
    plt.tight_layout()
                
    
