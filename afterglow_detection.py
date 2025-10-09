import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.gridspec import GridSpec
import astropy.units as u
import astropy.constants as const
import matplotlib.ticker as ticker
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
from scipy.interpolate import interp1d
import scipy.stats as sts
import schwimmbad
import sncosmo
from tqdm import tqdm
import pickle
import corner
import os
import sys
import argparse
from functools import partial

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import phases
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_distribution import gen_events, get_params, sncosmo_bands, labels, labels_idx
from waveforms import get_snr
from afterglow_params import get_distances

import rubin_sim.maf as maf
from rubin_sim.data import get_baseline

def get_internightgap(n):
    uniform_samples = np.random.uniform(0, 1, n)
    t = Table.read('./data/internightgap.csv', format='csv')
    interp_func = interp1d(t['cdf'], t['gap'], kind='linear', fill_value='extrapolate')

    return np.array(interp_func(uniform_samples))


# check if a point is brighter than the detection limit
# 0 = no, 1 = yes
def check_above_limit(mag, limit):
    return (mag <= limit).astype(int)

# helper to avoid out of bound error when checking next bin
    # if out of bounds, use same thing
def check_phase_index(idx):
    if idx + 1 < len(phases):
        return idx+1
    else:
        return idx

def get_mags(event, idx):
    return np.array(event[:, idx])

# for mags and time idx get the mag in that idx phase bin and the next one
def get_mags_adj_bins(event, idx):
    idx1 = check_phase_index(idx)
    return get_mags(event, idx), get_mags(event, idx1)

def get_mags_peak(event):
    res = np.min(event, axis=1)
    assert res.shape == (6,)
    return res
# list of ugrizy mags at a given phase and also the next bin
# return if detection in two bands and also the list so can see which bands
def check_detection(event, event_ne, t_idx, limits):

    mags, mags1 = get_mags_adj_bins(event, t_idx)
    mags_ne, mags_ne1 = get_mags_adj_bins(event_ne, t_idx)
    #print(limits, mags, mags1, flush=True)
    real_detection = 0

    # check for detections across bands
    dets = check_above_limit(mags, limits)

    # if 2 or more bands see it, thats a detection
        # I only check the next time bin if I only get 1 filter
    #print(dets, np.sum(dets), flush=True)
    if np.sum(dets) >= 2:
        real_detection = 1

    # if only one detection, check all other bands in the next time bin    
    elif np.sum(dets) == 1:
        dets1 = check_above_limit(mags1, limits)

        mask = (dets == 0) & (dets1 == 1)
        if np.sum(dets1) >= 1: # yay seen in different band
            real_detection = 1
            mags[mask] = mags1[mask]  # replace the non detections with detections in the next time bin
            dets[mask] = dets1[mask]
            mags_ne[mask] = mags_ne1[mask] # if we like the next bin, use that for t0 in unextincted too

        else:
            real_detection = 0 # nope just saw it in 1 # TODO: flag different types of fails
            mags = [-890, ]*6

    else:
        real_detection = 0
        mags = [-891, ]*6

    # real_detection = 1 if seen in two bands, mags is the mags of the two time bins, dets is the flags for det in those bands
    #print(real_detection, mags, dets, flush=True)

    return real_detection, mags, mags_ne, dets

# check if t0+n or t0+(n+0.2) days has a detection 
def check_two_detections(event, event_ne, idx_t0, idx_next, limits):

    mags_peak = get_mags_peak(event_ne) # no extinction, peak mag

    real_detection, mags, mags_ne, dets_t0 = check_detection(event, event_ne, idx_t0, limits)
    # if mags[0] == -890 or mags[0] == -891:
    #     print("error", flush=True)
    #     print(mags, flush=True)
    #     print(mags_ne, flush=True)
    #     print(event)
    
    real_detection_next, mags_next, mags_next_ne, dets_next = check_detection(event, event_ne, idx_next, limits)

    # unextincted dms
    dm_peak = mags_peak - mags_ne
    dm_peakn = mags_peak - mags_next_ne

    # if 0 in dm_peak or 0 in dm_peakn:
    #     print("zero in dm_peak or dm_peakn", flush=True)
    #     print(idx_t0, idx_next, flush=True)
    #     print(phases[idx_t0], phases[idx_next], flush=True)
    #     print(event_ne[0, :], flush=True)
    #     print(mags_peak, flush=True)
    #     print(dm_peak, flush=True)
    #     print(dm_peakn, flush=True)
    #     print(mags_ne, flush=True)
    #     print(mags_next_ne, flush=True)
    
    # if there is a 1st detection, the consider the next
    if real_detection:

        # check the same bands
        mask = dets_t0 == 1
        
        #print("tnext", real_detection_next, mags_next, dets_next, flush=True)
        # if the sum is 1 or more with the mask then there was a detection in the same band at this later phase
        if real_detection_next and np.sum(dets_next[mask]) >= 1:

            ndets = 2
            deltam = mags - mags_next
            band_det = np.zeros(6)
            band_det[np.where((np.array(dets_t0)==1) & (np.array(dets_next)==1))[0]] = 1
        else:
            ndets =1
            deltam = [-898,]*6
            band_det = dets_t0
            # return 1, [-898,]*6, dets_t0  # wasnt detected in a same band at the next obs
    
    else: # not detected at t0
        ndets = 0
        deltam = [-899,]*6
        band_det = [-899,]*6
        # return 0, [-899,]*6, [-899,]*6 # not detected
    
    return ndets, deltam, band_det, dm_peak, dm_peakn

# potentially make this have a t0?
    # will want to map/parallelize the construction of the rows over the events (bc they dont interact)
def lsst_detection_sim(n, filename, t0, limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1]), event_type='KN'):
    # load in the events
    values = gen_events(n, False, filename)
    values_ne = gen_events(n, False, filename+'ne')

    # choose a t0 - first obs
    # idx_t0 = 0
    idx_t0 = np.where(np.isclose(phases, t0))[0][0]

    tprevious = t0 - get_internightgap(n)
    tnext = t0 + get_internightgap(n)

    # tnext = t0 + 3 # 3 days for now
    # tprevious = t0 - 3
    # idx_next = -1
    # idx_prev = -1

    idx_next = np.full(n, -1)
    idx_prev = np.full(n, -1)

    # check if in the bounds and get the time that is closest to the randomly sampled time
    for i in range(n):
        if tprevious[i] < min(phases):
            tprevious[i] = -1
        else:
            idx_prev[i] = np.argmin(np.abs(phases - tprevious[i]))

        if tnext[i] > max(phases):# to do - in big run go up to max(phase)-3 and no more
            raise ValueError("next detection attempt beyond simulated phases") 
        else:
            idx_next[i] = np.argmin(np.abs(phases - tnext[i]))

    #print(idx_prev, flush=True)
    # the last params are needed but fixed per batch
    # proc_1_event = partial(process_1_event, limits=limits, idx_prev=idx_prev, idx_t0=idx_t0, idx_next=idx_next, event_type=event_type)
    proc_1_event = partial(process_1_event_internightgap, limits=limits, idx_t0=idx_t0, event_type=event_type)
    data_in = []
    for i in range(n):
        data_in.append((values[i], values_ne[i], idx_prev[i], idx_next[i]))

    with schwimmbad.JoblibPool(1) as pool:
        res = list(pool.map(proc_1_event, data_in))

    columns = ['t0', 'tprev', 'tnext', 'ndet'] + [f'dm-{b}' for b in 'ugrizy']+[f'det-{b}' for b in 'ugrizy']
    columns +=  [f'dm-peak-{b}' for b in 'ugrizy']+[f'dm-peak-next-{b}' for b in 'ugrizy']

    df = pd.DataFrame(np.array(res), columns=columns)
    df.to_csv(f'./data/det_sims/{event_type}/dets_{np.round(t0, 1)}_ing.csv')



# check detection for each event, with varying internight gaps
def process_1_event_internightgap(data_in, limits, idx_t0, event_type):
    data, data_ne, idx_prev, idx_next = data_in
    return process_1_event((data, data_ne), limits, idx_prev, idx_t0, idx_next, event_type)

# check detections for each event
def process_1_event(data, limits, idx_prev, idx_t0, idx_next, event_type):

    data, data_ne = data

    if event_type == "KN":
        event = data[0][2, 4:10, :] # KN data
        event_ne = data_ne[0][2, 4:10, :]
    else:
        event = data[0][1, 4:10, :] # KN+aft data
        event_ne = data_ne[0][1, 4:10, :]


    # extract just the lsst bands
        # all events, all combinations of KN, lsst only bands, all phases
        # bands=[4, 5, 6, 7, 8, 9]
    # event = dat÷a[0][1, 4:10, :]
    params = data[1][0] # get kn_params 
    event += params['dist'].distmod.value # adjust to observed mag

    # save the times, ndets, deltam (m_tnext - m_t0), flags for which bands detected
        # if ndet = 1, band_flag = bands det'd at t0
        # if ndet = 2, band_flag = bands det'd at t0 AND tnext
    res = [phases[idx_t0], phases[idx_prev], phases[idx_next]] + [-999, ]*7 + [-999,]*6

    # now check for detection
    # ensure for a detection at t0 and if so check for detection at tnext
        # just use the same point if not bc if it wasnt detected then it wasnt in the next but if it is then found the point
    n_det, deltam, bands_dets, dm_peak, dm_peakn = check_two_detections(event, event_ne, idx_t0, idx_next, limits)
    
    # check for non-detection: tprevious before first t-merge or no detect in any band
    if idx_prev != -1:
        # if there is a detection, then skip event
        if check_detection(event, event_ne, idx_prev, limits)[0]:
            r = np.array(res + list(dm_peak) + list(dm_peakn))
            return r # still want the dms
        else: # if there isnt then check regular
            r = np.array(res[:3] + [n_det, ] + list(deltam) + list(bands_dets) + list(dm_peak) + list(dm_peakn))
            return r
    else:
        # can assume we didnt detect it because it wasnt born yet
            # so update res to use the values from checked_dets
        res[1] = -1 #reset t_prev
        r = np.array(res[:3] + [n_det, ] + list(deltam) + list(bands_dets) + list(dm_peak) + list(dm_peakn))
        return r




def vary_t0(n, filename):

    t0 = phases.copy()
    t0 = t0[t0 < 17]
    for t in t0: # every timestep # ~every half day, afterglows all have inf at 0.1d????
        try:
            lsst_detection_sim(n, filename, t, limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1]), event_type='KN')
            lsst_detection_sim(n, filename, t, limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1]), event_type='aft')
        except ValueError as e: # eventually will hit a t0 where tnext is outside of the phases
            print(f"Skipping t0 {t} due to error: {e}", flush=True)
            break
    
    # merge into one df and pkl
    columns = ['t0', 'tprev' 'tnext', 'ndet'] + [f'dm-{b}' for b in 'ugrizy']+[f'det-{b}' for b in 'ugrizy']
    columns += [f'dm-peak-{b}' for b in 'ugrizy']+[f'dm-peak-next-{b}' for b in 'ugrizy']
    df_KN = pd.DataFrame(columns=columns)
    df_aft = pd.DataFrame(columns=columns)
    
    # load all the KN only values and pickle the dataframe
    path = './data/det_sims/KN/'
    for f in os.listdir('./data/det_sims/KN/'):
        df_to_add = pd.read_csv(path+f, index_col=0)
        df_KN = pd.concat([df_KN, df_to_add])
    with open('./data/det_sims/KN_dets.pkl', 'wb') as f:
        pickle.dump(df_KN, f)

    # same with the KN+aft
    path = './data/det_sims/aft/'
    for f in os.listdir(path):
        df_to_add = pd.read_csv(path+f, index_col=0)
        df_aft = pd.concat([df_aft, df_to_add])
    with open('./data/det_sims/aft_dets.pkl', 'wb') as f:
        pickle.dump(df_aft, f)
    
    return "I am done :)"


# worked!!!!!
    # the extincted and no extincted files are ordered the same way
def check_ext_vs_noext(n, filename1, filename2):

    values1 = gen_events(n, False, filename1)
    values2 = gen_events(n, False, filename2)

    for i in range(n):

        params1 = values1[i][1][0]
        params2 = values2[i][1][0]

        assert params1 == params2

    return "all good"


def plots(limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1])):

    # load in the pickles
    with open('./data/det_sims/KN_dets.pkl', 'rb') as f:
        df_KN = pickle.load(f)
    with open('./data/det_sims/aft_dets.pkl', 'rb') as f:
        df_aft = pickle.load(f)

    # get all the t0s (unsure if one has more non-nan entries than the other)
    # print(np.unique(df_KN['t0']), flush=True)
    # print(np.unique(df_aft['t0']), flush=True)
    t0s = np.unique(df_KN['t0'])

    res_dict_KN = {key: [] for key in ['eff',] + [f'dm-{b}' for b in 'ugrizy'] + [f'dm-peak-{b}' for b in 'ugrizy']}
    res_dict_aft = {key: [] for key in ['eff',] + [f'dm-{b}' for b in 'ugrizy'] + [f'dm-peak-{b}' for b in 'ugrizy']}

    for res_d, df in zip([res_dict_KN, res_dict_aft], [df_KN, df_aft]):

        for t in t0s:
            dft0 = df[df['t0']==t]
            res_d['eff'].append(len(dft0[dft0['ndet']==2.0]) / len(dft0.index))

            # get average delta m per band
            for b in 'ugrizy':
                col = f'dm-{b}'
                try:
                    dms = dft0[col]
                    m = np.nanmean(dms[dms > -800])
                except:
                    m = np.inf
                res_d[col].append(m)
            
                # get the average volume
                # col_peak = f'M-peak-{b}' #
                col = f'dm-peak-{b}' # Mpeak - M_t0
                col_next = f'dm-peak-next-{b}' # Mpeak - M_tnext

                dms_0 = dft0[col]
                dms_next = dft0[col_next]

                # TODO: check that this is selecting the correct one
                dms = np.select([dms_0 <= dms_next, dms_0 > dms_next], [dms_0, dms_next])
                print(dms.shape, flush=True)
                print(dms, flush=True)
                print(np.nanmean(dms), flush=True)
                res_d[col].append(np.nanmean(dms))

    # grid spec
        # efficiency plot on top
        # dm plots in 2 cols, 3 rows
    fig = plt.figure(figsize=(10, 16))
    gs = GridSpec(4, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t0s, res_dict_aft['eff'], label='aft')
    ax1.plot(t0s, res_dict_KN['eff'], label='KN')
    ax1.set_xlabel('time of first detection rel to t_merge')
    ax1.set_ylabel('fraction of events detected by LSST')
    ax1.legend()

    for ax, b in zip([gs[r, c] for r in range(1,4) for c in range(0,2)], [b for b in 'ugrizy']):
        ax = fig.add_subplot(ax)
        col = f'dm-{b}'
        ax.plot(t0s, -1*np.array(res_dict_aft[col])) # combined is blue
        ax.plot(t0s, -1*np.array(res_dict_KN[col]))
        ax.set_xlabel(r'$t_0$ (days since merge)')

        # down to zero means the amount of fading is decreasing
        ax.set_ylabel(r'$dm = M(t_{\rm next}) - M(t_0)$')
        ax.set_title(col)

    print('hello?', flush=True)
    plt.subplots_adjust(hspace=0.3)
    fig.savefig('./img/det_plots/det_eff_dm_ing.png')

    fig, axs = plt.subplots(3, 2, figsize=(10, 16))
    axs = axs.flatten()
    for ax, b in zip(axs, [b for b in 'ugrizy']):
        col = f'dm-peak-{b}'
        ax.plot(t0s, res_dict_aft[col], label='aft')
        ax.plot(t0s, res_dict_KN[col], label='KN')
        ax.set_xlabel(r'$t_0$ (days since merge)')
        ax.set_ylabel(r'$dm = M_{\rm peak} - M(t_{\rm next})$')
        ax.set_title(r'avg $\delta M = M_{\rm peak} - M_{\rm obs}$')
        ax.legend()
    fig.savefig('./img/det_plots/det_eff_dm_ing_dM.png')


# def get_median_internightgap():

#     opsdb = 'baseline_v3.6_10yrs.db'
#     run_name = os.path.split(opsdb)[-1].replace('.db', '')

#     metric = maf.CountMetric('observationStartMJD', metric_name='NVisits')
#     slicer = maf.HealpixSlicer(nside=64)
#     constraint = None
#     plot_dict = {'color_min': 0, 'color_max': 1200, 'extend': 'max'}
#     plot_funcs = [maf.HealpixSkyMap(),]
#     bundle = maf.MetricBundle(metric, slicer, constraint, run_name=run_name, 
#                             plot_dict=plot_dict, plot_funcs=plot_funcs)
#     g = maf.MetricBundleGroup({'nvisits': bundle}, opsdb, verbose=True)
#     g.run_all()

#     bundle.plot()



# end goal table:
    # rows = event id (0-4999)
    # columns = [t0, tnext, n_detections (0, 1, 2),  deltaM per band if detected at t0, flag for bands where det at t0 and tnext]
        # should make deltaM np.NaN if band of no interest

# then could average for a given t0 each deltaM per band
if __name__ == '__main__':

    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_events', default=500, type=int, required=False, help='number of events')
    parser.add_argument('--fname', type=str, required=False)
    args = parser.parse_args(args=argv)

    fname = args.fname
    n = args.n_events

    print(vary_t0(n, fname), flush=True)
    # t = phases[0]
    # lsst_detection_sim(n, fname, t, limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1]), event_type='KN')
    # lsst_detection_sim(n, fname, t, limits=np.array([23.9, 25.0, 24.7, 24.0, 23.3, 22.1]), event_type='aft')
    # plots()

    # values = gen_events(n, False, fname)

    # for ev in values[:5]:

    #     params = ev[1]
    #     ev = ev[0]

    #     aft_only = ev[0]
    #     aftkn = ev[1]
    #     kn = ev[2]

    #     # check first bin
    #     if np.any(np.isinf(aftkn[:, 0])):
    #         print(params, flush=True)
    #         print('combined', flush=True)
    #         print(aftkn[:, 0], flush=True)
    #         print('only aft', flush=True)
    #         print(aft_only[:, 0], flush=True) 
    #         print('kn')
    #         print(kn[:, 0], flush=True) 
        

    # lsst_detection_sim(n, fname, t0=0.1, event_type='aft')

    # print(check_ext_vs_noext(n, fname, fname[:-1]+'ne'), flush=True)