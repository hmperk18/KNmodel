import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance, SkyCoord
import astropy.coordinates as coord
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import schwimmbad
import pickle
import scipy.stats as sts
from scipy.interpolate import interp1d
import sys
import argparse
import os


from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_addition import AfterglowAddition
from dns_mass_distribution import Galaudage21, Farrow19
from monte_carlo_sims import get_ejecta_mass
from afterglow_params import get_logn0, get_opening_angle, get_loge0, get_p, get_distances, get_corr_loge0_logn0

# 10 days out
idx_10 = np.where(np.isclose(phases, 10.1))[0][0]
phases_10 = phases[:idx_10]

# common band from uv to ir in sncosmo
sncosmo_bands = ['uvot::uvw2', 'uvot::uvw1', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty'] # , 'f125w', 'f160w', 'f200w'
labels = ["uv2", "uv1", r"$u$-band", r"$g$-band", r"$r$-band", r"$i$-band", r"$z$-band", r"$y$-band"]#, "J", "H", "K"
labels_idx = np.arange(len(labels))


def get_params(n, save=False, filename=''):

    if save:

        # loop over kn params - stolen from Ved unless otherwise noted
            # mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av steal from ved
        mass1, mass2 = Galaudage21(n)
        masses = np.array([mass1, mass2]).T
        mej_dyns, mej_winds = get_ejecta_mass(mass1, mass2)
        
        # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
        box_size = 600*u.Mpc
        coords, dists = get_distances(n, box_size, shape='sphere')
        
        thetaCores = np.deg2rad(get_opening_angle(n, distr='RE23'))
        cos_thetas = np.random.uniform(np.cos(np.deg2rad(30)), 1, size=n) # for EK_aft
        #cos_thetas = np.cos(thetaCores/2) # for EK_aft0tc
        #cos_thetas = np.cos(np.deg2rad(cos_thetas))
        #cos_thetas = np.full((n,), 1) # on-axis for all 
        phis = np.random.uniform(15, 75, size=n)
        avs = np.random.exponential(0.334, size=n)*0.334
        rvs = np.full((n,), 3.1)

        param_names = ["mej_dyn", "mej_wind", "phi", "cos_theta", "dist", "coord", "av", "rv"]
        params_array = np.array([mej_dyns, mej_winds, phis, cos_thetas, dists, coords, avs, rvs]).T
        kn_params = [{param_names[i]: value for i, value in enumerate(row)} for row in params_array]
        print(f'done KN params {filename}', flush=True)
        # loop over aft params
        # self.grb_params = { # use the same params I used for nsf fig
        #         'jetType':     grb.jet.Gaussian,   # Gaussian jet!! - not flat
        #         'specType':    0,                  # Basic Synchrotron Spectrum
        #         'thetaObs':    theta_v,            # Viewing angle in radians
        #         'E0':          E0,                 # Isotropic-equivalent energy in erg
        #         'thetaCore':   0.07,               # Half-opening angle in radians
        #         'thetaWing':   0.47,               # Wing angle in radians
        #         'n0':          n0,                 # circumburst density in cm^{-2}
        #         'p':           2.17,               # electron energy distribution index
        #         'epsilon_e':   10**-1.4,           # epsilon_e
        #         'epsilon_B':   10**-4,             # epsilon_B
        #         'xi_N':        1.0,                # Fraction of electrons accelerated
        #         'd_L':         10*u.pc.to(u.cm),   # Luminosity distance in cm
        #     }
        
        # from Zhu et al 2022 I unless otherwise specified
        #logE0s = np.random.normal(49.3, 0.4**2, n) # ergs
        #logn0s = np.random.normal(-2, 0.4**2, n) # cm^-2
        #ps = np.random.normal(2.25, 0.1**2, n)   # spectral index
        # logees = np.random.normal(-1, 0.3**2, n)
        # logebs = np.random.normal(-2, 0.4**2, n)

        logE0s = get_loge0(n, distr='Zhu22')
        ps = get_p(n, distr='Fong15')
        logn0s = get_logn0(n, distr='Zhu22') #sts.norm.rvs(-2, 0.4**2, size=n)
        # fix ee and eb and use n0 distribution
            # e_e = 0.1, e_B = 0.01
        
        # logn0s, logE0s = get_corr_loge0_logn0(n) #, distr='clipped 1'
        logees = np.full((n,), -1.0) #sts.norm.rvs(-1, 0.3**2, size=n)
        logebs = np.full((n,), -2.0) #sts.norm.rvs(-2, 0.4**2, size=n)

        # make afterglow param dicts
        aft_array = np.array([10**logE0s, thetaCores, 10**logn0s, ps, 10**logees, 10**logebs]).T
        aft_names = ["E0", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B"]
        aft_params = [{aft_names[i]: value for i, value in enumerate(row)} for row in aft_array]
        print(f'done aft params {filename}', flush=True)
        params = list(np.array([kn_params, aft_params]).T)
        #print(params, flush=True)

        # save them
        with open(f'data/sims/{n}_params_{filename}.pkl', 'wb') as f:
            print(f'done params {filename}', flush=True)
            pickle.dump(params, f)
        #np.savetxt(f"{n}_events.csv", params, delimiter=",")
        # with open(f'data/sims/{n}_masses_{filename}.pkl', 'wb') as f:
        #     print(f'done masses {filename}', flush=True)
        #     pickle.dump(masses, f)
    else:
        # load in the values
        with open(f'data/sims/{n}_params_{filename}.pkl', 'rb') as f:
            params = pickle.load(f)

    # list of dicts
    return params

# simulate one event
def gen_event(params):

    kn_params, aft_params = params

    KN = SEDDerviedLC(**kn_params)
    afterglow = AfterglowAddition(KN, **aft_params, addKN=False)

    mag_band_aft = afterglow.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_aft = np.array([list(item) for item in mag_band_aft.values()])
    print(mag_band_aft[:, 0], flush=True) # rm
    afterglow.sed += afterglow.KNsed # add the KN on top

    # get the diff and save those
    # abs mag of KN+afterglow
    mag_band_aftKN = afterglow.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_aftKN = np.array([list(item) for item in mag_band_aftKN.values()])
    
    # abs mag of KN
    mag_band_KN = KN.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_KN = np.array([list(item) for item in mag_band_KN.values()])

    return np.array([mag_band_aft, mag_band_aftKN, mag_band_KN]), np.array(params) #, kn_params, aft_params


# issue with NaNs in UV: check and interpolate over them
def smooth_out_Nans(lc):    
    if np.isnan(lc).any():
        not_nan_indices = np.logical_not(np.isnan(lc))
        interpolator = interp1d(phases[not_nan_indices], lc[not_nan_indices], kind='linear', fill_value="extrapolate")
        interpolated_data = interpolator(phases)
        return interpolated_data
    else:
        return lc

# TO DO - undo ext mods 
def gen_events(n, save=False, filename='', pfilename=None):

    if save: 
        # edit to use same params as past version
        if pfilename is None:
            params = get_params(n, save, filename) # fix. filename.replace('z', 'n')
        else:
            # if specified use the parameters from a different file
            params = get_params(n, False, pfilename)

        with schwimmbad.JoblibPool(6) as pool:
             values = list(pool.map(gen_event, params)) # fix

        # filename += 'Ext
        with open(f'data/sims/{n}_events_{filename}.pkl', 'wb') as f:
             pickle.dump(values, f)

    else:
        # load in the values
        with open(f'data/sims/{n}_events_{filename}.pkl', 'rb') as f:
            values = pickle.load(f)

    return values

def plot(n, save, filename=''):

    # load them in
    events = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN

    # values is a 3d array each entry is an 2d grid of enhancements
    values = np.array([a[0] for a in events]) # the first entry in all events is the mags

    distr = np.percentile(values[:,0], [16, 50, 84], axis=0)
    distr_net = np.percentile(values[:,1], [16, 50, 84], axis=0)

    fig, axs = plt.subplots(int(len(sncosmo_bands)/2), 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one
    for idx in labels_idx:
        ax = axs[idx]
        # plot distr 1 - median
        # fill btwn distr 0 and 2
        ax.fill_between(phases, distr[0][idx, :], distr[2][idx, :], alpha=0.3, color='C0')
        ax.plot(phases, distr[1][idx, :], color='C0')

        ax.fill_between(phases, distr_net[0][idx, :], distr_net[2][idx, :], alpha=0.3, color='C1')
        ax.plot(phases, distr_net[1][idx, :], color='C1')

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$\Delta M$')
        ax.invert_yaxis()
        ax.set_title(labels[idx])

    # ax = axs[-1]
    
    # event = values[6]
    # for idx in labels_idx[:3]:
    #     ax.plot(phases, event[1][idx, :], linestyle='-')
    #     ax.plot(phases, event[2][idx, :], linestyle='--')   

    #     ax.set_xlabel('time (days)')    
    #     ax.set_ylabel(r'$\Delta M$')
    #     ax.invert_yaxis()
    #     ax.set_xscale('log')
    #     ax.set_title(labels[idx])

    fig.tight_layout()
    fig.savefig(f'img/{n}_events_{filename}.png')
    plt.show()

def plot_avglc(n, save, filename='', log=False):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN
    
    values = np.array([v for v, p in values])

    # overlay GW170817 like event
        # at: Dietrich fit Fig 
    # ang = 0.03 # core = 0.07
    # at2017gfo = SEDDerviedLC(mej_dyn=10**-2.27, mej_wind=10**-1.28, phi=49.5, cos_theta=np.cos(ang), 
    #                          coord=SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"), dist = 40*u.Mpc, av=0.0)
    #     # afterglow: Table 3 (default in afterglow addition)
    # gw170817 = AfterglowAddition(at2017gfo, addKN=False)
    # lcs = gw170817.getAbsMagsInPassbands(sncosmo_bands)

    # values is a 3d array each entry is an 2d grid of enhancements
    #distr = np.percentile(values[:,0], [16, 50, 84], axis=0) # get Z
    distr_aft = np.percentile(values[:,0], [16, 50, 84], axis=0)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)

    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 4*n_plots))
    plt.subplots_adjust(wspace=0.15, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one

    idx5 = np.where(np.isclose(phases, 5.1))[0][0]
    axs[0].set_ylabel(r'$M$')
    for i, idx in enumerate(labels_idx):
        ax = axs[i]
        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')
        #ax.plot(phases, lcs[sncosmo_bands[idx]], color='k', linestyle='--', label='GW170817 @ 2deg')
            # just KN
        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        ax.fill_between(phases, smooth_out_Nans(distr_aft[0][idx, :]), smooth_out_Nans(distr_aft[2][idx, :]), alpha=0.1, color='g')
        ax.plot(phases, smooth_out_Nans(distr_aft[1][idx, :]), color='g', label='aft only', linewidth=0.5)

        print(labels[idx], phases[idx5], flush=True)
        print(distr[1][idx, idx5] - distr_KN[1][idx, idx5], flush=True)

        # if idx == 2:
        #     print(distr_KN[1][idx, :], flush=True)

        ax.set_xlabel('time (days)')    
        ax.invert_yaxis()

        if log:
            ax.set_xscale('log')
        #ax.set_ylabel(r'$M$')
        ax.set_title(labels[idx])
        ax.legend()

    fig.tight_layout()
    if log:
        filename += 'log'

    fig.savefig(f'img/caps/{n}_events_{filename}_lc_all.png')
    #fig.savefig(f'img/{n}_events_{filename}_lc_lsst_noaft.png')
    #fig.savefig(f'img/caps/lsst.png')
    plt.show()

def plot_color(n, save, filename):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN

    # values is a 3d array each entry is an 2d grid of enhancements
    #distr = np.percentile(values[:,0], [16, 50, 84], axis=0) # get Z

    idx_uv2 = 0
    idx_r = 4
    # for all events get the total/KN Mag in uv and r and take the diff
    color_diff = np.squeeze(values[:, 1:, idx_uv2, :] - values[:, 1:, idx_r, :])

    # take all combined events and get the percentiles
        # repeat for KN only
    distr = np.percentile(color_diff[:, 0], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(color_diff[:, 1], [16, 50, 84], axis=0) # get magKN

    fig, ax = plt.subplots(1,1)

    ax.fill_between(phases, distr[0,:], distr[2,:], alpha=0.3, color='b')
    ax.plot(phases, distr[1,:], color='b')
        # just KN
    ax.fill_between(phases, distr_KN[0,:], distr_KN[2,:], alpha=0.3, color='orange')
    ax.plot(phases, distr_KN[1,:], color='orange')

    ax.set_ylabel(r'$M_{uv2} - M_{r}$ [mag]')
    ax.set_xscale('log')
    ax.set_xlabel(r'phase [day]')
    fig.savefig(f'img/{n}_events_{filename}_uv-r.png')

    plt.show()

def plot_distance(n, save, filename, limiting_mags):

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)

    def max_distance(M, limiting_mag):
        mu = limiting_mag - smooth_out_Nans(M)
        return 10**(1 + (mu/5)) / 1e6
    
    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        lim_mag = limiting_mags[idx]
        ax = axs[i]

        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        ax.fill_between(phases, max_distance(distr[0][idx, :], lim_mag), max_distance(distr[2][idx, :], lim_mag), alpha=0.3, color='b')
        ax.plot(phases, max_distance(distr[1][idx, :], lim_mag), color='b')
            # just KN
        ax.fill_between(phases, max_distance(distr_KN[2][idx, :], lim_mag), max_distance(distr_KN[0][idx, :], lim_mag), alpha=0.3, color='orange')
        ax.plot(phases, max_distance(distr_KN[1][idx, :], lim_mag), color='orange')

        ax.set_title(labels[idx] + ' Limiting Magnitude: '+str(lim_mag))

        ax.set_ylabel(r'distance [Mpc]')
        ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
    
    fig.tight_layout()
    fig.savefig(f'img/caps/{n}_events_{filename}_distlsst.png')
    plt.show()

def merge(n, n_files, fname):

    # join the parameter arrays
    # params_arr = []
    # param_files = [f'data/sims/{n}_params_{fname}{i}.pkl' for i in range(1,n_files+1)]
    # for f in param_files:
    #     with open(f, 'rb') as f:
    #             params = pickle.load(f)
    #             params_arr.append(params)
            
    # params = np.vstack(params_arr)
    # # print(params.shape, flush=True)
    # with open(f'data/sims/{n*n_files}_params_{fname}.pkl', 'wb') as f:
    #     pickle.dump(params, f)

    # join the value arrays
    values_arr = []
    val_files = [f'data/sims/{n}_events_{fname}{i}.pkl' for i in range(1,n_files+1)]
    for f in val_files:
        with open(f, 'rb') as f:
            values = pickle.load(f)
            values_arr += values

    values = values_arr # bc params are in there 
    # print(values.shape, flush=True)
    with open(f'data/sims/{n*n_files}_events_{fname}.pkl', 'wb') as f:
        pickle.dump(values, f)

    # join the mass arrays
    # mass_arr = []
    # mass_files = [f'data/sims/{n}_masses_{fname}{i}.pkl' for i in range(1,n_files+1)]
    # for f in mass_files:
    #     with open(f, 'rb') as f:
    #         params = pickle.load(f)
    #         mass_arr.append(params)
            
    # masses = np.vstack(mass_arr)
    # # # print(masses.shape, flush=True)
    # with open(f'data/sims/{n*n_files}_masses_{fname}.pkl', 'wb') as f:
    #         pickle.dump(masses, f)

    # clean up
    # for f in mass_files + param_files + val_files: # TODO: put back 
    #     os.remove(f)

def compare_GW170817():
    ang = 0.03 # core = 0.07
    at2017gfo = SEDDerviedLC(mej_dyn=10**-2.27, mej_wind=10**-1.28, phi=49.5, cos_theta=np.cos(ang), 
                             coord=SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"), dist = 40*u.Mpc, av=0.0)
        # afterglow: Table 3 (default in afterglow addition)
    gw170817 = AfterglowAddition(at2017gfo, addKN=False)
    lcs = gw170817.getAbsMagsInPassbands(sncosmo_bands)

    theta_c = np.deg2rad(6) # 6 is ~ peak of RE
    mean_p = {"E0": 10**(49.3) * (1/(1-np.cos(theta_c))), 
              "thetaCore": theta_c, 
              "n0": 1e-2, 
              "p": 2.3, 
              "epsilon_e": 0.1, 
              "epsilon_B": 0.01}
    mean = AfterglowAddition(at2017gfo, **mean_p, addKN=False)
    mean_lcs = mean.getAbsMagsInPassbands(sncosmo_bands)
    mean.sed += mean.KNsed # then get combined curve
    comb_lcs = mean.getAbsMagsInPassbands(sncosmo_bands)

    fig, axs = plt.subplots(int(len(sncosmo_bands)/2)+1, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T

    for idx, band in enumerate(sncosmo_bands):

        ax=axs[idx]

        ax.plot(phases, lcs[band], color='k', linestyle='--', label='GW170817')
        ax.plot(phases, mean_lcs[band], color='g', linestyle='-', label='mean aft')
        ax.plot(phases, comb_lcs[band], color='b', linestyle='-', label='mean aft')
        ax.set_title(band)

        ax.set_ylabel(r'M')
        ax.invert_yaxis()
        ax.set_xlabel(r'phase [day]')

    fig.tight_layout()
    fig.savefig(f'img/mean_lc.png')
    plt.show()

def afterglows(n, save, filename='', log=False):

    # load them in
    params = get_params(n, save, filename)
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)

    # get just afterglows
    afterglows = values[:,0]

    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one
    for i, idx in enumerate(labels_idx):
        ax = axs[i]

        for j in range(n):
            ax.plot(phases, smooth_out_Nans(afterglows[j][idx, :]), color='gray', 
                    alpha=0.1, linewidth=0.5)

            if afterglows[j][idx, 0] > 80:
                print(np.arccos(params[j][0]['cos_theta']), params[j][1], params[j][0], flush = True)


        # if idx == 2:
        #     print(distr_KN[1][idx, :], flush=True)

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$M$')
        ax.invert_yaxis()

        if log:
            ax.set_xscale('log')

        ax.set_title(labels[idx])

    fig.tight_layout()
    if log:
        filename += 'log'
    fig.savefig(f'img/{n}_events_{filename}_afts_lsst.png')
    plt.show()

def splitNfix(n, save, filename):

    params = get_params(n, save, filename)
    _, dists = get_distances(n, 600, shape='sphere')

    for i in range(10):

        p_segment = params[500*i: (500*i)+500]
        dist_segment = np.array(dists[500*i: (500*i)+500])*u.Mpc 
        print(len(p_segment))

        # insert the new distances
        for j, (kn, _) in enumerate(p_segment):
            kn['dist'] = dist_segment[j]


        with open(f'data/sims/{500}_params_{filename}{i+1}.pkl', 'wb') as f:
                print(f'done params {filename}{i+1}', flush=True)
                pickle.dump(params, f)        


def plot_extremes_corr():

    fig, axs = plt.subplots(1, 2, figsize=(12,7))
    axs = axs.ravel()

    band = 'lsstg'
    params_grb = { # from Troja 2020
        'E0': 10**52.9,
        'thetaCore': 0.05,
        'n0':10**-2.7,
        'p':2.25,
        'epsilon_e':10**-1, 
        'epsilon_B':10**-2.,
        'coord': SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"),
        'dist': 40*u.Mpc
    }

    distr_types = ['gmm', 'gmm1', 'clipped 2', 'clipped 1']
    for i, d_type in enumerate(distr_types):

        logn0s, logE0s = get_corr_loge0_logn0(5000, distr=d_type)

        ax = axs[0]
        maxv = np.argmax(logE0s)
        params_grb['E0'] = 10**(logE0s[maxv])
        params_grb['n0'] = 10**(logn0s[maxv])

        afterglow = AfterglowAddition(None, **params_grb)
        mag = afterglow.getAbsMagsInPassbands([band,], apply_extinction=False)
        ax.plot(phases, mag[band], label=f'{d_type}', color=f'C{i}')

        ax = axs[1]
        maxv = np.argmax(logn0s)
        params_grb['E0'] = 10**(logE0s[maxv])
        params_grb['n0'] = 10**(logn0s[maxv])

        afterglow = AfterglowAddition(None, **params_grb)
        mag = afterglow.getAbsMagsInPassbands([band,], apply_extinction=False)
        ax.plot(phases, mag[band], label=f'{d_type}', color=f'C{i}')

        
    axs[0].set_xlabel('days')
    axs[0].set_ylabel('M')
    axs[0].set_title('with Max e0 value')
    axs[0].invert_yaxis()
    axs[0].legend()
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('M')
    axs[1].set_title('with Max n0 value')
    axs[1].invert_yaxis()
    fig.savefig('img/extremes_.png')
    plt.show()


def add_gamma0_parameters(n, filename, pfilename, gamma0=None):

    # load in the parameters to update
    params = get_params(n, False, filename)

    for param in params:
        aft = param[1]
        # add gamma0
        aft['gamma0'] = gamma0

    # save with the new name
    with open(f'data/sims/{n}_params_{pfilename}.pkl', 'wb') as f:
             pickle.dump(params, f)


if __name__ == '__main__':

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_events', default=500, type=int, required=False, help='number of events')
    parser.add_argument('--iter', type=int, required=False, help='Filename of simulation results')
    parser.add_argument('--plot', help='If true, plot else iter', action='store_true')
    parser.add_argument('--fname', type=str, required=False)
    parser.add_argument('--pfname', type=str, required=False, default=None)

    args = parser.parse_args(args=argv)

    #dir = args.dir

    #np.random.seed(1674 % i) # each i will be different

    # https://www.lsst.org/scientists/keynumbers
        # u, g, r, i, z, y
    #limiting_mags = [23.8, 24.5, 24.03, 23.41, 22.74, 22.96]


    UV_bands = ['UVEX::FUV', 'UVEX::NUV']
    UV_labels = ['UVEX FUV', 'UVEX NUV']
    #labels_idx = np.arange(len(labels))

    sncosmo_bands = UV_bands + sncosmo_bands
    labels = UV_labels + labels
    # labels_idx = np.arange(len(labels))

    # # STAR-X: http://star-x.xraydeep.org/observatory/
    # # UVEX: https://www.uvex.caltech.edu/page/about
    # # UVOT: https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html
    # # LSST: Bianco+ 2022
        #https://www.lsst.org/scientists/keynumbers : 23.8, 24.5, 24.03, 23.41, 22.74, 22.96
    UV_limiting_mags = [24.5, 24.5]
    sncosmo_lim_mags = [22.3, 22.3, 23.9, 25.0, 24.7, 24.0, 23.3, 22.1] # , 26, 26, 26
    sncosmo_lim_mags = UV_limiting_mags + sncosmo_lim_mags

    # NIR 
    # JWST (10k s exposures): https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-sensitivity#NIRCamSensitivity-Imaging
        # as of Aug 1, 2024
    # Roman: https://roman.gsfc.nasa.gov/science/WFI_technical.html, 
        #as of June 3, 2024
    sncosmo_bands += ['f070w', 'f277w', 'f444w', 'f062', 'f146', 'f213']
    labels += ['JWST 70w', 'JWST 200w', 'JWST 444w', 'Roman 62', 'Roman 146wide', 'Roman 213']
    sncosmo_lim_mags += [28.5, 28.7, 28.3, 24.77, 25.37, 23.14]

    labels_idx = np.arange(len(labels))

    n = args.n_events
    n_files = 10
    # fname = 'mediann0' #'trunc' #Ext
    fname = args.fname
    pfname = args.pfname
    if not args.plot:
        if args.iter:
            i = args.iter
            
            fname += str(i)
            seed_new = hash(f"{fname}") % 2**32
            np.random.seed(seed_new)

            
            gen_events(n, save=True, filename=fname)
            print(f'done {fname}, seed {seed_new}', flush=True)
        else:
            i = 1
            # np.random.seed(2667 % i)

            if pfname is not None:
                add_gamma0_parameters(n, fname, pfname, gamma0=100)
            gen_events(n, save=True, filename=pfname, pfilename=pfname)


        # TODO: re-run param gen for Ek_aft
            # if params are the same, 
            # then do again but just save the m1,m2
        #get_params(n, save=True, filename=fname)
        # done - now check that these are the correct ones, then save the masses

    if args.plot:
        print(f'merging {fname} now', flush=True)
        merge(n, n_files=n_files, fname=fname)
        # print('now plotting', flush=True)

        # select bands for plotting
        # labels_idx = np.array([0, 1, 4, 5, 6, 7, 8, 9]) # UV + LSST
        # labels_idx = np.array([0,1])
        # labels_idx = np.arange(len(labels))


        # kn_p = {'mej_dyn': 0.004321779762824195, 
        #     'mej_wind': 0.04182826654986341, 
        #     'phi': 16.08937008294327, 
        #     'cos_theta': 0.9575188803585961, 
        #     'dist': 587.88021042*u.Mpc, 
        #     'coord': SkyCoord(ra = 315.77266398*u.deg, dec =  5.85190817*u.deg),
        #     'av': 0.5505275977803256, 
        #     'rv': 3.1}
        # aft_p = {'E0': 5.467852735022001e+52, 
        #     'thetaCore': 0.06303483850999154, 
        #     'n0': 4.211736092225102e-05, 
        #     'p': 2.4813287049836688, 
        #     'epsilon_e': 0.09999999999999999, 
        #     'epsilon_B': 0.01}

        # gen_event((kn_p, aft_p))


        # font = {'family' : 'normal',
        #          'size'   : 15}
        # matplotlib.rc('font', **font)

        # values = gen_events(n*n_files, save=False, filename=fname)
        # print(len(values), len(values[0]), flush=True)

        # plot_extremes_corr()

        # #compare_GW170817()
        #plot(n, save=False, filename=fname)
        # #afterglows(n*n_files, save=False, filename=fname)
        # plot_avglc(n*n_files, save=False, filename=fname)
        #plot_color(n, save=False, filename=fname)
        #plot_distance(n*n_files, save=False, filename=fname, limiting_mags=sncosmo_lim_mags)
    # params = get_params(500, False, filename=fname)
    # values = gen_events(500, False, filename=fname)

    # # for idx in labels_idx:
    # #     band = idx
    # #     KN_v = values[:,2, band, :] # check all the  u light curves

    # #     print(values.shape, KN_v.shape, flush=True)
    # #     # for i in [12, 70, 87, 90, 99]:
    # #     #     print(params[i], flush=True)
    # #     #     print(values[i, 2, 2,:], flush=True)
    # #     print(np.argwhere(np.isnan(KN_v)),'\n',flush=True)

    # band = 1
    # KN_v = values[:,2, band, :]
    # #for event, phase in np.argwhere(np.isnan(KN_v))[0]:
    # print(np.argwhere(np.isnan(KN_v)))
    # event, phase = np.argwhere(np.isnan(KN_v))[7]
    
    
    # print(params[event])
    
    # kn_p, aft_p = params[event]

    # KN = SEDDerviedLC(**kn_p)
    # print(KN.getAbsMagsInPassbands([sncosmo_bands[band],]), flush=True)
    # #print(np.argwhere(np.isnan(KN.sed[phase])),'\n',flush=True)
    # idx_130nm = np.where(lmbd == 1300)[0][0]
    # idx_300nm = np.where(lmbd == 3100)[0][0]
    # source = sncosmo.TimeSeriesSource(phase=phases[phase-2:phase+3], wave=lmbd, flux = KN.sed[phase-2:phase+3], name='test', zero_before=True)
    # model = sncosmo.Model(source)

    # print(KN.sed[phase-2:phase+3, idx_130nm:idx_300nm], flush=True)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)

    # model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
    # model.set(hostebv = KN.host_ebv)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)
    # # add MW extinction to observing frame
    # model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')
    # model.set(mwebv=KN.mw_ebv)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)


    # fig, axs = plt.subplots(2,1)
    # axs = axs.ravel()
    # #ax.plot(lmbd*u.AA.to(u.nm), KN.sed[phase])
    # axs[0].plot(phases, KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]])
    # axs[0].plot(phases, KN.getAbsMagsInPassbands([sncosmo_bands[band],], apply_extinction=False)[sncosmo_bands[band]])
    # axs[0].axvline(phases[phase],ymin=min(KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]]), ymax=max(KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]]),
    #                linewidth=5, alpha=0.5, label=phases[phase])
    # axs[0].set_xlabel('day')
    # axs[0].invert_yaxis()
    # axs[0].legend()

    # for p in range(phase-2, phase+3):
    #     axs[1].plot(lmbd[:idx_300nm], KN.sed[p, :idx_300nm], label=phases[p])
    # axs[1].set_xlabel('AA')
    # axs[1].legend()

    # # there keeps being a weird big spike in the flux
    # print(lmbd[np.argmax(KN.sed[phase, :idx_300nm])])
    # fig.savefig(f'img/sed_check.png')
        #print(KN.getAbsMagsInPassbands([sncosmo_bands[band],]))

    # distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)
    # print(np.argwhere(np.isnan(distr_KN)), flush=True)








    

    









