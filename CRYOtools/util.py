import os

import numpy as np

from astropy.time import Time
from astropy.io.fits import Header

from astropy.wcs import WCS
import astropy.units as u
import astropy.constants as const

import warnings

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

def return_obs_info(hdrs, verbose=True):

    if verbose:
        ## Show information on 1st and 2nd direction scanning (step size and number)
        for k in 'CNP1DSS','CNP1DNSP','CNP2DSS','CNP2DNSP',:
            print(hdrs[0][k+'*'])


    n_scan_steps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_meas_at_step = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))

    n_wv = hdrs[0]['NAXIS1']
    n_along_slit = hdrs[0]['NAXIS2']

    return n_scan_steps, n_meas_at_step, n_wv, n_along_slit


def get_slit_coords(hdrs, output_dir = './outputs/'):
    '''
    Gets slit locations and times from header.

    Returns:

        hpxy_coords array 
                    shape is [2,n_scan_steps, n_meas_at_step, slit_len] 
        time_coords array
                    shape is [n_scan_steps, n_meas_at_step] 
    '''
    n_scan_steps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_meas_at_step = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))

    n_alongSlit = hdrs[0]['NAXIS2']
    hpxy_coords = np.zeros((2,n_scan_steps, n_meas_at_step,n_alongSlit))
    time_coords = np.zeros((n_scan_steps, n_meas_at_step),dtype ='datetime64[ms]')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") ## TO ELIMINATE datafix warnings
        for hd in hdrs:
            wcs = WCS(hd)
            xy = wcs.array_index_to_world(0,np.arange(n_alongSlit),0)[1]
            x,y,obstime = xy.Tx.value,xy.Ty.value,xy[0].obstime
            CNCURSCN,CNCMEAS = hd['CNCURSCN'],hd['CNCMEAS']
            hpxy_coords[0,CNCURSCN-1,CNCMEAS-1,:] = x
            hpxy_coords[1,CNCURSCN-1,CNCMEAS-1,:] = y
            time_coords[CNCURSCN-1,CNCMEAS-1] = obstime.to_datetime()

    ## Save the dataset coordinates to the location of your choice (defaults to working directory)
    ensure_directory(output_dir)
    np.save(output_dir + 'hpxy_coords.npy', hpxy_coords)
    np.save(output_dir + 'time_coords.npy', time_coords)

    print(f"Helioprojective XY Coordinates Shape: {hpxy_coords.shape}")
    print(f"Datetime Coordinates Shape: {time_coords.shape}")
    
    return hpxy_coords, time_coords

def get_spectral_coords(hdrs, output_dir = './outputs/'):
    '''
    GET SPECTRAL DISPERSION AXIS.
    
    Use the WCS tools for this as the dispersion axis in not strictly linear
    the spectral dispersion axis type (CTYPE1) is 'AWAV-GRA' refers to a 
    grating func for air wavelengths.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # TO ELIMINATE datafix warnings
        wcs = WCS(hdrs[0])
        nwv = hdrs[0]['NAXIS1']
        spec_coords = wcs.array_index_to_world(0,0,np.arange(nwv))[0].to(u.nm).value

    # Save the dataset coordinates (defaults to working directory)
    ensure_directory(output_dir)
    np.save(output_dir + 'spec_coords.npy',spec_coords)
    print(f"Spectral Coordinates Shape: {spec_coords.shape}")

    return spec_coords

def get_slit_samp(hpxy_coords, n_scan_steps, verbose=True):
    '''
    Determine the raster slit step size and sampling along the slit.

    Note that these values include any PC_ij matrix transforms of the WCS information:
    Also, the step size is slightly lower on average that that given in the CNP1DSS keyword
    '''

    if n_scan_steps<=1:
        step_width = 0.
        slit_samp = np.sqrt((hpxy_coords[0,0,0,1]-hpxy_coords[0,0,0,0])**2 + (hpxy_coords[1,0,0,1]-hpxy_coords[1,0,0,0])**2 )
    else:
        step_width = np.sqrt((hpxy_coords[0,1,0,0]-hpxy_coords[0,0,0,0])**2 + (hpxy_coords[1,1,0,0]-hpxy_coords[1,0,0,0])**2 )
        slit_samp = np.sqrt((hpxy_coords[0,0,0,1]-hpxy_coords[0,0,0,0])**2 + (hpxy_coords[1,0,0,1]-hpxy_coords[1,0,0,0])**2 )
    
    if verbose:
        print(f'Raster step size in arcsec: {step_width}')
        print(f'Sampling along the slit (arcsec per pixel): {slit_samp}')

    return slit_samp, step_width


def _get_sc_mp(header_str, n_alongSlit,i):
    """
    Worker function: reconstructs WCS from header string and extracts coordinates.

     Returns: 
     current scan step number, current measurement number,
     Tx and Ty helioprojective coordinates, and the observation time.
    """
    header = Header.fromstring(header_str,sep="\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # TO ELIMINATE datafix warnings
        wcs = WCS(header)
        print(wcs)
        xy = wcs.array_index_to_world(0,np.arange(n_alongSlit),0)[1]
        print(type(xy[0]))
        print(type(xy[1]))
        x,y,obstime = xy.Tx.value,xy.Ty.value,xy[0].obstime
        print('x', x)
    return header['CNCURSCN'],header['CNCMEAS'],x,y, obstime.to_datetime()


def _unpack_and_run(args):
    return _get_sc_mp(*args)


def get_slit_coords_mp(hdrs: list, cpu_max: int =4, output_dir: str ='./outputs/'):
    '''
    Extracts slit coordinates from FITS headers using multiprocessing.
    '''

    n_scanSteps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_measAtStep = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))
    n_wv = hdrs[0]['NAXIS1']
    n_alongSlit = hdrs[0]['NAXIS2']
    hpxy_coords = np.zeros((2,n_scanSteps,n_measAtStep,n_alongSlit))
    time_coords = np.zeros((n_scanSteps,n_measAtStep),dtype ='datetime64[ms]')

    # Serialize headers to strings (fully picklable) 
    # Prepare list of (header_str, n_alongSlit) tuples
    tasks = [(hdr.tostring(sep="\n",padding=False, endcard=True), n_alongSlit,i) for i,hdr in enumerate(hdrs)]

    ncpus = min(os.cpu_count() , cpu_max)
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(tqdm(executor.map(_unpack_and_run, tasks), 
                                     total=len(hdrs), desc="Processing"))

        for res in results:
            CNCURSCN,CNCMEAS,x,y,obstime = res
            hpxy_coords[0,CNCURSCN-1,CNCMEAS-1,:] = x
            hpxy_coords[1,CNCURSCN-1,CNCMEAS-1,:] = y
            time_coords[CNCURSCN-1,CNCMEAS-1] = obstime
    
    # Save the dataset coordinates (defaults to working directory)
    ensure_directory(output_dir)
    np.save(output_dir + 'hpxy_coords.npy',hpxy_coords)
    np.save(output_dir + 'time_coords.npy',time_coords)

    print(f"Helioprojective XY Coordinates Shape: {hpxy_coords.shape}")
    print(f"Datetime Coordinates Shape: {time_coords.shape}")
    
    return hpxy_coords, time_coords

def calculate_cadence(tc, plot_cad=False):
    '''
    tc - time_coords
    '''
    tc = tc.squeeze()

    cad = [ (tc[i]-tc[i-1]).item().total_seconds() for i, j in enumerate(tc)]
    med_cad = np.median(cad[1:])

    if plot_cad:
        plt.plot(cad[1:], '.')
        plt.xtitle('Frames')
        plt.ytitle('Seconds')
        plt.savefig('cadence_values.png')

    return med_cad

def shift_to_v(delta_lam, lam_0 = 1074.7, lam_cor = 0.7):
    '''
    delta_lam - array of shift values from fit
    lam_0 - central wavelength
    lam_cor - any correction factor required from fit

    In current fitting method, delta_lam is defined with respect to 1074.0.
    '''
    c_km_s = const.c.to(u.km/u.s)
     # correction to measured shifts (from fitting procedure)
    F = (delta_lam-lam_cor)/lam_0-1
    return (F**2-1)/(F**2+1)*c_km_s # Doppler formula


def calc_ntlw(data, res_pow, wave=1074.7, v_th=21.0):
    '''
    data - line widths
    res_pow - resolving power
    wave - central wavelength
    v_th - thermal velocity (in km/s)
    '''

    if np.ndim(v_th) == 0:
       v_th = np.full_like(data, v_th)

    elif isinstance(v_th, (list, tuple, np.ndarray)):
        v_th = np.asarray(v_th)
        if v_th.shape != data.shape:
            raise ValueError(f"`v_th` must be either a scalar or the same shape as `data` (got {v_th.shape}, expected {data.shape})")

    v_th = v_th *u.km/u.s # km/s
    fac = 2*np.sqrt(np.log(2))
    w_i = wave*u.nm /res_pow # nm
    scale = wave*u.nm/ c.c.to(u.nm/u.s)

    fwhm = np.sqrt(2)*fac*data*u.nm # nm
    ntlw = np.sqrt((fwhm**2-w_i**2)/scale**2/fac**2-v_th.to(u.nm/u.s)**2)

    return ntlw.to(u.km/u.s)


def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def print_exposure(hdrs):
    exposureKeys = ['XPOSURE','TEXPOSUR','CAM_FPS','CNNSCI','CNNNDR','CNMODNST','CNNMEAS']
    for key in exposureKeys:
        print(f"{key.ljust(10)} {hdrs[0].comments[key].ljust(25)} {hdrs[0][key]} ")
    print(f"Time between ramps (i.e. triggers): {1./hdrs[0]['CAM_FPS']*1000.} msec")
    print(f"Times between first and last trigger: {Time(hdrs[-1]['DATE-AVG']).to_datetime() -Time(hdrs[0]['DATE-AVG']).to_datetime()}")