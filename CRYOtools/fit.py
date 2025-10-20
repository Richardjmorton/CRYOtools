import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import glob

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jacobian
from jax.scipy.signal import correlate

import numpy as np

from scipy.ndimage import shift
from scipy.optimize import differential_evolution, minimize
from scipy.signal import correlate as corr_sp 
from scipy.signal import correlation_lags

from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tqdm.auto import tqdm

from CRYOtools.io import _read_solar_model, _read_telluric_model

@jax.jit
def gaussian_filter_1d(input, sigma=2):
    '''
    Remake of scipy function
    '''
    # make the radius of the filter equal to truncate standard deviations
    radius = 30 #jnp.array(4 * sigma + 0.5, jnp.int32)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius+1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    signal_ext = jnp.array([input[::-1], input, input[::-1]])
    ln = len(input)
    smooth = correlate(signal_ext.reshape(3*ln), phi_x[::-1], mode='same')

    return smooth[ln:2*ln]

@jax.jit
def gaussian(x, a, b, c):
    expo = (x-b)/c
    return a*jnp.exp(-expo**2/2)
    

@jax.jit
def fft_shift(input, shift):
    N = input.shape[0]
    x = jnp.arange(-N / 2, N / 2)
    kx = -1j * 2 * np.pi * x / N
    finput = jnp.fft.fftshift(jnp.fft.fftn(input))
    shifted_finput = finput * jnp.exp(-(kx*shift))
    shifted_input = jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(shifted_finput)))
    return shifted_input
    


def do_conv(x, y, Rpow_log):
    '''
    Helper function to do convolution in model
    '''
    fwhm_wv = x.mean()/jnp.exp(Rpow_log)
    sigm_wv = fwhm_wv / (2.*jnp.sqrt(2.*jnp.log(2))) 
    dwv = x[0]-x[1]
    kern_pix = sigm_wv / jnp.abs(dwv) 
    y_conv = gaussian_filter_1d(y, sigma=kern_pix)
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
    corr = corr_sp(standard(x1), standard(x2) , 'same', method='fft')
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
          

def fit_slit(img, spec_coords, do_diff: int =0, wgts: jnp.array = None,
             use_tqdm: bool = False):
    '''
    Perform a fit to a slit/exposure.

    Strategy is to use differential_evolution to find good starting values for
    the model parameters. This is done once per slit (as default) and previous
    profiles values are used as next starting values.

    Input
    img - Cryo slit (from one fits file) with shape [slit_len, wave_len]

    do_diff - modulo number of pixels to run differential evolution

    '''
    res_slits = []
    N = img.shape[0]

    if do_diff == 0:
        do_diff = N

    fts_cor = get_solar_model(spec_coords)
    fts_atm = get_telluric_model(spec_coords)
    log_fts_atm = jnp.log(fts_atm)
    bounds = create_bounds()

    if not wgts:
        wgts = jnp.ones(N, jnp.float64)

    iterator = tqdm(img, desc="Processing", disable=not use_tqdm)
    for i, y in enumerate(iterator):
         
        velS_est = get_lags(y, fts_cor, Solar=True)
        velT_est = get_lags(y, fts_atm, Solar=False)
        bounds[5] = (velS_est-0.2, velS_est+0.2)
        bounds[6] = (velT_est-0.2, velT_est+0.2)
        bounds[8] = (np.nanmedian(y)-10,np.nanmedian(y)+10)
            
        if i % do_diff == 0:
            res = differential_evolution(loss, bounds, args=(spec_coords, y, wgts), tol=1.e-2,maxiter = 800,popsize = 1)#, polish=True)
        res = minimize(loss, res.x, args=(spec_coords, y, wgts), jac=jac_loss, method='BFGS')
        res_slits.append(res)
            
    return res_slits


class fit_data():

    def __init__(self, data: np.ndarray, spec_coords: np.array, 
                           do_diff: int =0):
        self.data = data
        self.spec_coords = spec_coords
        self.n_along_slit = data.shape[0]
        self.n_wv = data.shape[1]
        self.bounds = None

        if do_diff == 0:
           self.do_diff = self.n_along_slit
        else:
            self.do_diff = do_diff

    def create_bounds(self, bounds=None):
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

        self.bounds = bounds
        return

    def build_model(self, params):

        amp,lam_0,sigma,Rpow_log,opac,velS,velT,strayfrac,icont, icont_lin = params

        ## ifit -- coronal line
        gfit = gaussian(self.spec_coords-1074., amp, lam_0, sigma)
                      
        ftsSmod = jnp.copy(self.fts_cor)
        ## shift coronal data
        ftsSmod = fft_shift(ftsSmod, velS)

        ## scale and shift telluric spectra
        ftsTmod = jnp.exp(opac*self.log_fts_atm)
        ftsTmod = fft_shift(ftsTmod, velT)

        ftsmod = ftsSmod * ftsTmod
                
        ## add straylight
        ftsmod = (ftsmod + strayfrac) / (1. + strayfrac) 
        ## scale for total
        ftsmod = ftsmod*(icont+icont_lin*self.spec_coords)
                
        ifit = ftsmod + gfit * ftsTmod # note that telluric absorption is also applied to the coronal line 
                
        ## convolution for spectrograph line spread function
        ## Gaussian convolution of the FTS atlas
        fwhm_wv = self.spec_coords.mean()/jnp.exp(Rpow_log)
        sigm_wv = fwhm_wv / (2.*jnp.sqrt(2.*jnp.log(2))) 
        dwv = self.spec_coords[0]-self.spec_coords[1]
        kern_pix = sigm_wv / jnp.abs(dwv) 
        ifit = gaussian_filter_1d(ifit, sigma=kern_pix)
                
        return ifit

    
    def loss(self,params,y):
        y_hat = self.build_model(params)
        return jnp.mean((y_hat-y)**2 * self.wgts)
 

    def fit_slit(self, wgts: jnp.array = None,
                    use_tqdm: bool = False, min_tol=1e-4):
        '''
        Perform a fit to a slit/exposure.

        Strategy is to use differential_evolution to find good starting values for
        the model parameters. This is done once per slit (as default) and previous
        profiles values are used as next starting values.

        Input
        img - Cryo slit (from one fits file) with shape [slit_len, wave_len]

        do_diff - modulo number of pixels to run differential evolution

        '''
        res_slits = []

        # some parameters are made explicit to work with JAX functions
        fts_cor = get_solar_model(self.spec_coords)
        self.fts_cor = fts_cor
        fts_atm = get_telluric_model(self.spec_coords)
        self.fts_atm = fts_atm
        log_fts_atm = jnp.log(fts_atm)
        self.log_fts_atm = log_fts_atm

        if not self.bounds:
            _ = self.create_bounds()

        if wgts is None:
            self.wgts = jnp.ones(self.n_wv, jnp.float64)
            wgts = self.wgts
        else:
            self.wgts = wgts

        @jax.jit
        def build_model_int(params, x):

            amp,lam_0,sigma,Rpow_log,opac,velS,velT,strayfrac,icont, icont_lin = params

            ## ifit -- coronal line
            gfit = gaussian(x-1074., amp, lam_0, sigma)
                  
            ftsSmod = jnp.copy(fts_cor)
            ## shift coronal data
            ftsSmod = fft_shift(ftsSmod, velS)

            ## scale and shift telluric spectra
            ftsTmod = jnp.exp(opac*log_fts_atm)
            ftsTmod = fft_shift(ftsTmod, velT)

            ftsmod = ftsSmod * ftsTmod
            
            ## add straylight
            ftsmod = (ftsmod + strayfrac) / (1. + strayfrac) 
            ## scale for total
            ftsmod = ftsmod*(icont+icont_lin*x)
            
            ifit = ftsmod + gfit * ftsTmod # note that telluric absorption is also applied to the coronal line 
            
            ## convolution for spectrograph line spread function
            ## Gaussian convolution of the FTS atlas
            fwhm_wv = x.mean()/jnp.exp(Rpow_log)
            sigm_wv = fwhm_wv / (2.*jnp.sqrt(2.*jnp.log(2))) 
            dwv = x[0]-x[1]
            kern_pix = sigm_wv / jnp.abs(dwv) 
            ifit = gaussian_filter_1d(ifit, sigma=kern_pix)
            
            return ifit


        @jax.jit
        def loss_int(params, y,x,wgts):
             y_hat = build_model_int(params, x)
             return jnp.mean((y_hat-y)**2 * wgts)

        jac_loss = jax.jit(jacobian(loss_int))

        x= self.spec_coords
        iterator = tqdm(range(self.n_along_slit), desc="Processing", disable=not use_tqdm)

        for i in iterator:
            y = self.data[i]
            
            if i % self.do_diff == 0:
                velS_est = get_lags_lin(y, fts_cor, x, Solar=True)
                velT_est = get_lags_lin(y, fts_atm, x, Solar=False)
                self.bounds[5] = (velS_est-0.5, velS_est+0.5)
                self.bounds[6] = (velT_est-0.5, velT_est+0.5)
                self.bounds[8] = (-1000,-200)#(np.nanmedian(y)-10,np.nanmedian(y)+10)
                res = differential_evolution(loss_int, self.bounds, args=(y,x,wgts), tol=1.e-2,maxiter = 800,popsize = 1)#, polish=True)
            res = minimize(loss_int, res.x, args=(y,x,wgts), jac=jac_loss, method='BFGS', tol=min_tol)
            res_slits.append(res)

        self.res_slits = res_slits
        return self.res_slits


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
 

def _pull_fit_res(file):
    dat = np.load(file, allow_pickle=True)
        
    return [[m.x, m.fun] for m in dat['res']]          
    
def pull_fit_res(dataset_directory, cpu_max=4):
    '''
    This function pulls all the fit results from the npz files and combines them into a single
    save file.
    '''
    fit_directory = dataset_directory +'spectrum_fits/'

    file_list = glob.glob(fit_directory+'*npz')
    file_list.sort()

    dat = np.load(file_list[0], allow_pickle=True)
    n_params = len(dat['res'][0].x)

    ncpus = min(os.cpu_count() , cpu_max)

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(tqdm(executor.map(_pull_fit_res, file_list), total=len(file_list), desc="Getting fit data"))
    
    fit_results = []
    merit_results = []
    for res in results:
        fit_results.append([m[0] for m in res])
        merit_results.append([m[1] for m in res])
    
    fit_results = np.array(fit_results)
    merit_results = np.array(merit_results)
    return np.transpose(fit_results,[2,0,1]), merit_results

