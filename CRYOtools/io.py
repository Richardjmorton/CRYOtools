import dkist

import glob
import os

import numpy as np

from astropy.io import fits
from astropy.io.fits import Header

from CRYOtools.util import return_obs_info

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

from importlib import resources

def load_asdf(dataset_directory):
    '''
    Loads asdf and prints size of data set
    '''
    asdf_filename = glob.glob(dataset_directory+'*.asdf')

    if len(asdf_filename) == 1:
        dset = dkist.load_dataset(asdf_filename[0])
        print(f"Dataset data shape: {dset.data.shape}")
        return dset
    elif len(asdf_filename) == 0:
        print(f"No files found")
    else:
        print(f"Multiple files found.")

    
    return None

def _read_solar_model(file=None) -> str:

    if not file:
        file = "solar_merged_20200720_600_33300_100.out"
    # reads `mypackage/data/sample.txt`
    with resources.path(__package__ + ".models", file) as p:
        return np.loadtxt(p, skiprows =3)

def _read_telluric_model(file=None) -> str:

    if not file:
        file = "hitran_1micron_h20_trans.npy"
    # reads `mypackage/data/sample.txt`
    with resources.path(__package__ + ".models", file) as p:
        return np.load(p)

def find_L1_files(dataset_directory):

    L1_filenames = glob.glob(dataset_directory + '*L1.fits')
    L1_filenames.sort()
    if not L1_filenames:
        print("No L1 FITS files found in the directory.")
        return None
    return L1_filenames


def _get_all_head(file):
    return fits.getheader(file, ext=1)

def get_all_head(dataset_directory, cpu_max=4):
    ## GET ALL HEADERS

    L1_filenames = find_L1_files(dataset_directory)

    if not L1_filenames:
        return None

    ncpus = min(os.cpu_count() , cpu_max)

    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        hdrs = list(tqdm(executor.map(_get_all_head, L1_filenames), total=len(L1_filenames), desc="Getting headers"))
    #hdrs = []
    #for file in L1_filenames:
    #    hdrs.append()

    return hdrs

def restore_slit_coords(dataset_directory):


    coord_list = ['hpxy_coords.npy', 'time_coords.npy', 'spec_coords.npy']

    data = []
    for coord in coord_list:
        filepath = dataset_directory+coord

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")       
        try:
            data.append(np.load(filepath))
        except Exception as e:
            raise RuntimeError(f"Error loading file '{filepath}': {e}")

    return data

def rotate_stokes(data, dataset_directory):
    '''
    Rotate stokes vectors

    DKIST L1 pipeline default polarization frame has +Q oriented along Solar E-W axis (i.e. parallel to solar meridian)
    Here we find and apply the rotation angle that will align +Q with the solar radial direction.
    +U will then be 45 degrees counterclockwise from the solar radial direction as viewed towards the Sun.
    '''

    hpxy_coords, _ = restore_slit_coords(dataset_directory)

    linpol_rotate_angle = np.arctan2(hpxy_coords[1,:,0,:],hpxy_coords[0,:,0,:])
    cos2Rot = np.cos(2.*linpol_rotate_angle)[:,None,:,None]
    sin2Rot = np.sin(2.*linpol_rotate_angle)[:,None,:,None]

    ## apply Mueller Matrix rotation
    dataR = np.copy(data)
    dataR[1] = cos2Rot*data[1] + sin2Rot *data[2]
    dataR[2] = -sin2Rot*data[1] + cos2Rot *data[2]

    return dataR


def _fits_mp(header_str, L1_filename, n_stokes):
    ''' work function for the parallel process'''

    header = Header.fromstring(header_str, sep='\n')

    img = []
    ## get Stokes I image
    img.append(fits.getdata(L1_filename, ext=1).squeeze())
    ## get Stokes Q,U, and V.
    if n_stokes>1:
        img.append(fits.getdata(L1_filename.replace('_I_','_Q_'), ext=1).squeeze())
        img.append(fits.getdata(L1_filename.replace('_I_','_U_'), ext=1).squeeze())
        img.append(fits.getdata(L1_filename.replace('_I_','_V_'), ext=1).squeeze())

    return header['CNCURSCN'], header['CNCMEAS'], img

def _unpack_and_run(args):
    return _fits_mp(*args)

def get_fits_data(dataset_directory: str, n_stokes: int =1, cpu_max: int =4, 
                  file_start: int =None,file_end: int =None,
                  convert_phot: bool =True):
    '''
    Reads in Cryo fits files using multiprocessing

    dataset_directory - path to files
    n_stokes - number of stokes to read in
    cpu_max - max number of cores to use
    file_start - first file index to read in
    file_end l- last file index to read in
    convert_phot - Converts to millionths (or ppm) of the disk center intensity
    '''

    hdrs = get_all_head(dataset_directory)
    files = find_L1_files(dataset_directory)

    if file_end is not None:
        files = files[:file_end]
        hdrs = hdrs[:file_end]

    if file_start is not None:
        files = files[file_start:]
        hdrs = hdrs[file_start:]
    else:
        file_start = 0

    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = return_obs_info(hdrs)
    
    # data buffer setting number of measurements to 1 as we will coadd all repeats
    if n_scan_steps>1:
        data = np.zeros((n_stokes,n_scan_steps,1,n_along_slit,n_wv),dtype = float)
        coadd_index = np.zeros(n_scan_steps)
    else:
        data = np.zeros((n_stokes,1,n_meas_at_step-file_start,n_along_slit,n_wv),dtype = float)
        coadd_index = np.zeros(n_meas_at_step)
    print("Defined data set shape to load: ",data.shape)

    # Serialize headers to strings (fully picklable) 
    tasks = [(hdr.tostring(sep='\n'), file, n_stokes) for hdr, file in zip(hdrs,files)]

    ncpus = min(os.cpu_count() , cpu_max)

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(tqdm(executor.map(_unpack_and_run, tasks), total=len(files), desc="Reading in data"))

        for res in results:
            CNCURSCN,CNCMEAS,imgs = res
            if n_scan_steps>1:
                data[:,CNCURSCN-1,0,:,:] += np.array(imgs)
                coadd_index[CNCURSCN-1] += 1
            else: ## THIS CASE IS MEANT TO HANDLE SIT AND STARE CASES
                data[:,0,CNCMEAS-1-file_start,:,:] = np.array(imgs)

    if n_scan_steps>1:
        data = data / coadd_index[None,:,None,None,None]

    if convert_phot:
        data = data*1e6

    if n_stokes >1:
        data = rotate_stokes(data, dataset_directory)

    return data, files


def write_model_results(folder, file, res):
    '''
    folder - Folder to save to
    file - L1 fits file name
    res - results from model fit
    '''

    if not os.path.exists(folder+'spectrum_fits'):
        os.makedirs(folder+'spectrum_fits')

    file_name = os.path.splitext(os.path.basename(file))[0]
    save_file = folder+'spectrum_fits/'+file_name+'.npz'
    np.savez_compressed(save_file, res=res)

    return save_file