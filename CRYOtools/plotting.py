#from CRYOtools.util import slit_locs, head_to_datetime

from typing import Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import ScalarFormatter

from scipy.ndimage import rotate

import astropy.units as u
from astropy.time import Time

from sunpy.map import Map

import hvpy ## helioviewer api for pulling quick context images from SDO/AIA (JPEG2000 compressed)

from CRYOtools.util import get_slit_samp, return_obs_info, shift_to_v, which_line
from CRYOtools.fit import get_solar_model, get_telluric_model


def plot_slit_locs(
    hpxy_coords: np.ndarray,
    time_coords: np.ndarray,
    slit_pos: int = 0,
) -> None:
    """Plot the solar X/Y slit locations and distance from the Sun centre over time."""

    locs = [hpxy_coords[0, :, :, slit_pos], hpxy_coords[1, :, :, slit_pos]]

    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

    labels = ['Solar X (arcsec)', 'Solar Y (arcsec)']
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)  # turn off 1eX notation
    fmt.set_useOffset(False)  # turn off the “×10…” offset

    for (ax, loc, label) in zip(axes[:2], locs, labels):

        ax.plot(time_coords.ravel(), loc.ravel(), '+')
        ax.set_ylabel(label)

        # Set axis labels for each subplot
        ax.set_xlabel('Time')
        ax.yaxis.set_major_formatter(fmt)

    sol_rad = np.sqrt(locs[0] ** 2 + locs[1] ** 2)
    axes[2].plot(time_coords.ravel(), sol_rad.squeeze().ravel(), '+')
    axes[2].set_ylabel(r'$R_\odot$ (arcsec)')
    axes[2].yaxis.set_major_formatter(fmt)
    # Add a global title
    fig.suptitle("Slit locations")

    # Optional: adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)


def aia_context_plot(
    hdrs: Sequence[dict],
    hpxy_coords: np.ndarray,
    n_scan_steps: int,
    flip_image: bool = False,
) -> None:
    """Plot the scan field of view over helioviewer AIA context images."""

    # corner coordinates of the scan
    spatial_bounds = hpxy_coords[:, [0, 0, -1, -1, 0], 0, [0, -1, -1, 0, 0]]

    for source in hvpy.DataSource.AIA_193, hvpy.DataSource.AIA_171:
        filepath = hvpy.save_file(
            hvpy.getJP2Image(
                date=Time(hdrs[0]['DATE-AVG']).to_datetime(),
                sourceId=source,
            ),
            filename="~/example.jpeg",
            overwrite=True,
        )

        aia_map = Map(filepath)  # assumed to be rotated and centered with Solar North up.

        if flip_image:
            # Rotate just the image data by 180 degrees and mirror to maintain orientation.
            rotated_data = rotate(aia_map.data, angle=180, reshape=False, order=3)

            # Create a new map with rotated data and original header
            aia_map = Map(rotated_data[:, ::-1], aia_map.meta)

        ny, nx = aia_map.data.shape

        extaia = (
            aia_map.pixel_to_world(0 * u.pix, 0 * u.pix).Tx.value,
            aia_map.pixel_to_world(nx * u.pix, 0 * u.pix).Tx.value,
            aia_map.pixel_to_world(0 * u.pix, 0 * u.pix).Ty.value,
            aia_map.pixel_to_world(0 * u.pix, ny * u.pix).Ty.value,
        )

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax = ax.flatten()
        for axi in ax:
            # show AIA data
            imn = axi.imshow(
                aia_map.data,
                extent=extaia,
                cmap='sdoaia' + str(aia_map.meta['wavelnth']),
            )
            imn.set_clim(np.nanpercentile(aia_map.data, [5, 99.5]))

            # CryoNIRSP dataset spatial bounds
            axi.plot(spatial_bounds[0], spatial_bounds[1], lw=2, color='white', ls='solid')
            axi.plot(
                spatial_bounds[0],
                spatial_bounds[1],
                lw=1,
                color='black',
                ls='dashdot',
                label="Scanned FOV",
            )
            axi.plot(
                hpxy_coords[0, n_scan_steps // 2, 0, :],
                hpxy_coords[1, n_scan_steps // 2, 0, :],
                lw=1,
                color='black',
                ls='solid',
                label="Center Slit",
            )

            # overplot solar limb
            rsun = hdrs[0]['SOLARRAD']
            tx = np.linspace(0, 2.0 * np.pi, 500)
            axi.plot(rsun * np.cos(tx), rsun * np.sin(tx), lw=0.8, ls='dashed', color='white')
            axi.set_facecolor('black')
            axi.legend()

        # CREATE A ZOOMED IN PANEL
        xw = spatial_bounds[0].max() - spatial_bounds[0].min()
        yw = spatial_bounds[1].max() - spatial_bounds[1].min()
        ww = np.max((xw, yw))
        ax[1].set_xlim(spatial_bounds[0].min() - 50, spatial_bounds[0].min() + ww + 50)
        ax[1].set_ylim(spatial_bounds[1].min() - 50, spatial_bounds[1].min() + ww + 50)
        fig.suptitle(
            f"CryoNIRSP SP Field of View Over SDO/AIA {aia_map.meta['wavelnth']} $\\AA$ "
            f"{aia_map.meta['date-obs']}"
        )
        fig.tight_layout()



def plot_line_example(
    data: np.ndarray,
    spec_coords: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
    scan_step: int = 0,
    along_pos: int = 1200,
) -> None:
    """Plot a spectral image and extracted spectral profile for a given scan step."""

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, _ = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)

    wv_cen, wv_line, wv_cont = which_line(spec_coords, verbose=False)

    extent = (spec_coords[0], spec_coords[-1], 0, n_along_slit * slit_samp)

    if along_pos < 0:
        along_pos = n_along_slit + along_pos
    if (along_pos < 0) or (along_pos > n_along_slit):
        raise ValueError("along_pos must be between 0 and {0}, got {1}".format(n_along_slit - 1, along_pos))

    if (scan_step < 0) or (scan_step > n_scan_steps):
        raise ValueError("along_pos must be between 0 and {0}, got {1}".format(n_scan_steps - 1, scan_step))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[0.3, 0.7])
    ax = ax.flatten()

    # Display the full spectrogram for the selected scan step.
    im0 = ax[0].imshow(data[0, scan_step, 0, :, :], extent=extent, aspect='auto', origin='lower')
    im0.set_clim(np.nanpercentile(data[0, scan_step, 0, :, :], [1, 99]))

    ax[0].set_xlabel("Wavelength [nm]")
    ax[0].set_ylabel("Arcseconds along slit")
    ax[0].set_title("Spectral Image")

    ax[0].axhline(along_pos * slit_samp, ls='dashed', color='blue')
    cax = ax[0].inset_axes([1.04, 0.05, 0.05, 0.95], transform=ax[0].transAxes)
    cbar = fig.colorbar(ax[0].get_images()[0], ax=ax[0], cax=cax)

    ax[1].plot(spec_coords, data[0, scan_step, 0, along_pos, :], color='blue')
    ax[1].set_xlabel("Wavelength [nm]")
    ax[1].set_title("Extracted Spectral Profile")
    ax[1].set_ylabel(r"Spectral Radiance [$\mu$B$_{\odot}$]")

    ax[1].locator_params(axis='x', nbins=6)
    ax[1].grid(ls='solid', lw=0.2, color='grey')
    ax[1].axvline(
        wv_cen,
        label='Coronal Nominal Center Wavelength',
        color='magenta',
        lw=1,
        ls='dashed',
    )
    ax[1].axvline(
        spec_coords[wv_cont],
        label='Continuum Reference Position',
        color='green',
        lw=1,
        ls='dashed',
    )

    ax[1].legend()

    plt.tight_layout()


def quick_plot_emission(
    data: np.ndarray,
    spec_coords: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
) -> None:
    """Generate a quick-look emissivity map by integrating line intensity."""

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, step_width = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)

    wv_cen, wv_line, wv_cont = which_line(spec_coords, verbose=False)

    extent = (spec_coords[0], spec_coords[-1], 0, n_along_slit * slit_samp)

    # Integrate over a narrow spectral window around the line and subtract continuum.
    line_emiss = data[0, :, 0, :, wv_line - 30 : wv_line + 30].sum(axis=2) - data[0, :, 0, :, wv_cont] * 60

    extent = [0, n_along_slit * slit_samp, 0, n_scan_steps * step_width]

    im = plt.imshow(line_emiss, cmap='gray', extent=extent, origin='lower')
    im.set_clim(np.nanpercentile(line_emiss, [1, 99]))
    plt.colorbar(label=r"$\mu B_\odot$")
    plt.xlabel("Arcseconds along slit")
    plt.ylabel("Arcseconds in scan direction")
    plt.title('Coarse total line emissivity {} nm'.format(wv_cen))


def plot_solar_atlas(spec_coords: np.ndarray) -> None:
    """Plot the solar atlas model over the supplied spectral coordinates."""

    solar_atlas = get_solar_model(spec_coords)

    plt.plot(spec_coords, solar_atlas)
    plt.xlabel('Wavelength (nm)')


def plot_telluric_atlas(spec_coords: np.ndarray) -> None:
    """Plot the telluric transmission model over the supplied spectral coordinates."""

    telluric_atlas = get_telluric_model(spec_coords)

    plt.plot(spec_coords, telluric_atlas)
    plt.xlabel('Wavelength (nm)')


def plot_all_params_scan(
    fit_results: np.ndarray,
    metrics: np.ndarray,
    spec_coords: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
    aspect: float = 1,
) -> None:
    """Visualise all fitted parameter maps for a raster scan observation."""

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, step_width = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)
    # Determine the wavelength indices so downstream scaling is consistent.
    wv_cen, _, _ = which_line(spec_coords, verbose=False)

    n_maps = fit_results.shape[0]

    plotTitles= [r"Line Peak Amplitude [$\mu$B$_{\odot}$]",
    "Doppler Shift [km/s]\n(relative to median)",'Coronal Line Width [nm]',
    'Estimated Resolving\nPower','Telluric Line \nOpacity Factor',
    'Velocity shift applied to\nscat. phot. spectrum',
    'Velocity shift applied to\ntelluric transmission','Inferred Spectral\nStraylight Fraction',
    'Background Const.','Background Grad.','Merit Function']


    default_cmap = plt.get_cmap('viridis')
    default_cmap.set_bad('indigo')
    doppvel_cmap = plt.get_cmap('RdBu')
    doppvel_cmap.set_bad('white')

    if step_width != 0:
        img_extent = (0,n_along_slit*slit_samp,0,n_scan_steps*step_width)
    else:
        img_extent = (0,n_along_slit*slit_samp,0,n_scan_steps)

    fig,ax = plt.subplots(3,4,figsize = (12,8),sharex=True,sharey=True)
    ax = ax.flatten()

    for n, para_map in enumerate(fit_results):

        if n ==0:
            amp = para_map
            # Limit extreme amplitudes that can dominate the colour scaling.
            amp[np.abs(amp)> 150] =0
            imn = ax[n].imshow(amp,extent = img_extent,cmap = default_cmap,
                               norm = PowerNorm(0.85), aspect=aspect)
            #imn.set_clim(np.nanpercentile(para_map,[2,98]))
        if n ==1:
            dopp_vels = shift_to_v(para_map)
            median_vel = np.nanmedian(dopp_vels.value)
            imn = ax[n].imshow(dopp_vels.value-median_vel,extent = img_extent,cmap = doppvel_cmap, aspect=aspect)
            imn.set_clim(-3,3)
        if n >1:
            imn = ax[n].imshow(para_map,extent = img_extent,cmap = default_cmap, aspect=aspect)
            imn.set_clim(np.nanpercentile(para_map,[2,98]))
        #if n==8: imn.set_clim(0,300)
        if n==9: imn.set_clim(0,2)
        ax[n].set_title(plotTitles[n],fontsize = 10)

    cbars = []

    imn = ax[n_maps].imshow(metrics,extent = img_extent, aspect=aspect)
    ax[n_maps].set_title(plotTitles[n_maps],fontsize = 10)
    imn.set_clim(np.nanpercentile(metrics,[2,98]))


    for axi in ax[:n_maps+1]:
        cax = axi.inset_axes([1.04, 0.05, 0.05, 0.95], transform=axi.transAxes)
        cbar1 = fig.colorbar(axi.get_images()[0], ax=axi,cax=cax)
        cbars.append(cbar1)

    for axi in ax[n_maps+1:]: axi.set_visible(False)

    fig.suptitle(f"Spectroscopic Fit Parameters near {wv_cen:.2f} nm",fontsize = 18 )
    fig.supylabel("Arcseconds in Stepping Direction",fontsize = 18)
    fig.supxlabel("Arcseconds along slit",fontsize = 18)
    fig.tight_layout()


def plot_all_params_ss(
    fit_results: np.ndarray,
    metrics: np.ndarray,
    spec_coords: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
    cadence: float,
    aspect: float = 0.1,
) -> None:
    """Visualise fitted parameter maps for a single-slit stare observation."""

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, step_width = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)
    wv_cen, _, _ = which_line(spec_coords, verbose=False)

    n_maps = fit_results.shape[0]

    plotTitles= [r"Line Peak Amplitude [$\mu$B$_{\odot}$]",
    "Doppler Shift [km/s]\n(relative to median)",'Coronal Line Width [nm]',
    'Estimated Resolving\nPower','Telluric Line \nOpacity Factor',
    'Velocity shift applied to\nscat. phot. spectrum',
    'Velocity shift applied to\ntelluric transmission','Inferred Spectral\nStraylight Fraction',
    'Background Const.','Background Grad.','Merit Function']

    default_cmap = plt.get_cmap('viridis')
    default_cmap.set_bad('indigo')
    doppvel_cmap = plt.get_cmap('RdBu')
    doppvel_cmap.set_bad('white')

    imgExtent = (0,n_along_slit*slit_samp,0,n_meas_at_step*cadence)

    fig,ax = plt.subplots(3,4,figsize = (12,8),sharex=True,sharey=True)
    ax = ax.flatten()

    for n, para_map in enumerate(fit_results):

        if n ==0:
            # Suppress outliers to keep the amplitude map readable.
            imn = ax[n].imshow(para_map,extent = imgExtent,cmap = default_cmap,
                               norm = PowerNorm(0.85), aspect=aspect)
            #imn.set_clim(np.nanpercentile(para_map,[2,98]))
        if n ==1:
            dopp_vels = shift_to_v(para_map)
            median_vel = np.nanmedian(dopp_vels.value)
            imn = ax[n].imshow(dopp_vels.value-median_vel,extent = imgExtent,cmap = doppvel_cmap, aspect=aspect)
            imn.set_clim(-3,3)
        if n >1:
            imn = ax[n].imshow(para_map,extent = imgExtent,cmap = default_cmap, aspect=aspect)
            imn.set_clim(np.nanpercentile(para_map,[2,98]))
        #if n==8: imn.set_clim(0,300)
        if n==9: imn.set_clim(0,2)
        ax[n].set_title(plotTitles[n],fontsize = 10)

    cbars = []

    imn = ax[n_maps].imshow(metrics,extent = imgExtent, aspect=aspect)
    ax[n_maps].set_title(plotTitles[n_maps],fontsize = 10)
    imn.set_clim(np.nanpercentile(metrics,[2,98]))


    for axi in ax[:n_maps+1]:
        cax = axi.inset_axes([1.04, 0.05, 0.05, 0.95], transform=axi.transAxes)
        cbar1 = fig.colorbar(axi.get_images()[0], ax=axi,cax=cax)
        cbars.append(cbar1)

    for axi in ax[n_maps+1:]: axi.set_visible(False)

    fig.suptitle(f"Spectroscopic Fit Parameters near {wv_cen:.2f} nm",fontsize = 18 )
    fig.supylabel("Time (s)",fontsize = 18)
    fig.supxlabel("Arcseconds along slit",fontsize = 18)
    fig.tight_layout()
    

def plot_coronal_params(
    results: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
    spec_coords: np.ndarray,
    aspect: float = 1,
    fold: int = 1,
    figsize: Tuple[float, float] = (15, 5),
    amp_clip: float = 50,
) -> None:
    """Plot a subset of coronal parameters (amplitude, velocity, width)."""

    amp = results[0]
    # Suppress extreme amplitudes that may arise from fitting artefacts.
    amp[np.abs(amp) > amp_clip] = 0
    vel = results[1]
    vel = shift_to_v(vel)
    width = results[2]

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, step_width = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)
    wv_cen, _, _ = which_line(spec_coords, verbose=False)

    fig, ax = plt.subplots(3, 1, figsize=figsize)
    plotTitles= [r"Peak Amp. [$\mu$B$_{\odot}$]",
    "Doppler Shift [km/s]",'Line Width [nm]']

    default_cmap = plt.get_cmap('viridis')
    default_cmap.set_bad('indigo')
    doppvel_cmap = plt.get_cmap('RdBu')
    doppvel_cmap.set_bad('white')

    if step_width != 0:
        img_extent = (0,n_along_slit*slit_samp,0,n_scan_steps*step_width/fold)
    else:
        img_extent = (0,n_along_slit*slit_samp,0,n_scan_steps/fold)

    imn = ax[0].imshow(amp, extent = img_extent,cmap = default_cmap,
                     aspect=aspect, origin='lower')
    imn.set_clim(np.nanpercentile(amp,[3,98]))


    median_vel = np.nanmedian(vel.value)
    imn = ax[1].imshow(vel.value-median_vel,extent = img_extent,cmap = doppvel_cmap,
                         aspect=aspect, origin='lower')
    imn.set_clim(-3,3)

    imn = ax[2].imshow(width, extent = img_extent,cmap = 'inferno', aspect=aspect,
                       origin='lower')
    imn.set_clim(np.nanpercentile(width,[3,98]))

    for axi, title in zip(ax, plotTitles):
        cax = axi.inset_axes([1.04, 0.05, 0.05, 0.95], transform=axi.transAxes)
        cbar1 = fig.colorbar(axi.get_images()[0], ax=axi,cax=cax)
        cbar1.set_label(title, fontsize=8)
        #axi.set_title(title,fontsize = 10)

    fig.suptitle(f"Coronal Fit Parameters near {wv_cen:.2f} nm",fontsize = 14 )
    fig.supylabel("Raster Direction [arcsec]",fontsize = 12)
    fig.supxlabel("Along slit [arcsec]",fontsize = 12)

    plt.tight_layout()



def plot_spectrogram_wf(
    data: np.ndarray,
    model: np.ndarray,
    spec_coords: np.ndarray,
    hpxy_coords: np.ndarray,
    hdrs: Sequence[dict],
    trim_end: bool = False,
) -> None:
    """Plot observed spectra, the model, and residuals for comparison."""

    res = return_obs_info(hdrs, verbose=False)
    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = res

    slit_samp, _ = get_slit_samp(hpxy_coords, n_scan_steps, verbose=False)

    wv_cen, wv_line, wv_cont = which_line(spec_coords, verbose=False)

    extent = (spec_coords[0], spec_coords[-1], 0, n_along_slit*slit_samp)


    fig,ax = plt.subplots(1,3,figsize = (9,4))
    ax = ax.flatten()

    im0 = ax[0].imshow(data,extent=extent, aspect = 'auto')
    im0.set_clim(np.nanpercentile(data,[1,99]))
    ax[0].set_title("Spectrogram")


    cax = ax[0].inset_axes([1.04, 0.05, 0.05, 0.95], transform=ax[0].transAxes)
    cbar = fig.colorbar(ax[0].get_images()[0],ax=ax[0],cax=cax)

    im1 = ax[1].imshow(model,extent=extent, aspect = 'auto')
    im1.set_clim(np.nanpercentile(data,[1,99]))
    ax[1].set_title("Model")

    resid = data-model
    im2 = ax[2].imshow(resid,extent=extent, aspect = 'auto', vmax=1, vmin=-1)
    ax[2].set_title("Residual")
    cax = ax[2].inset_axes([1.04, 0.05, 0.05, 0.95], transform=ax[2].transAxes)
    cbar = fig.colorbar(ax[2].get_images()[0],ax=ax[2],cax=cax)

    for axes in ax:
        axes.set_xlabel("Wavelength [nm]")
        axes.set_ylabel("Arcseconds along slit")
        if trim_end:
            # Hide the region most affected by the dual-beam boundary artefact.
            val = 1076 if np.floor(wv_cen) == 1074 else 1080
            axes.set_xlim(spec_coords[0], val)

    plt.tight_layout()
