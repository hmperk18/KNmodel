import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
import matplotlib.ticker as ticker
from astropy.coordinates import SkyCoord, Distance
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import pickle
import corner
import os
import sys
import argparse

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import phases
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_distribution import gen_events, get_params, sncosmo_bands, labels, labels_idx
from waveforms import get_snr
from afterglow_params import get_distances

import healpy as hp
import scipy.stats as sts
from astropy.coordinates import SkyCoord


# check if a point is brighter than the detection limit
# 0 = no, 1 = yes
def check_above_limit(mag, limit):
    return int(mag <= limit)

# helper to avoid out of bound error when checking next bin
    # if out of bounds, use same thing
def check_phase_index(idx):
    if idx + 1 < len(phases):
        return idx+1
    else:
        return idx

def get_mags(event, idx):
    return event[:, idx]

# for mags and time idx get the mag in that idx phase bin and the next one
def get_mags_adj_bins(event, idx):
    idx1 = check_phase_index(idx)
    return get_mags(event, idx), get_mags(event, idx1)



# list of ugrizy mags at a given phase and also the next bin
# return if detection in two bands and also the list so can see which bands
def check_detection(event, t_idx, limits):

    mags, mags1 = get_mags_adj_bins(event, t_idx)

    real_detection = 0

    # check for detections across bands
    dets = list(map(check_above_limit(mags, limits)))

    # if 2 or more bands see it, thats a detection
        # I only check the next time bin if I only get 1 filter
    if np.sum(dets) <= 2:
        real_detection = 1

    # if only one detection, check all other bands in the next time bin    
    elif np.sum(dets) == 1:
        mask = mags == 0
        dets_1 = list(map(check_above_limit(mags1[mask], limits)))

        if np.sum(dets_1) <= 1: # yay seen in different band
            real_detection = 1
        else:
            real_detection = 0 # nope just saw it in 1

    else:
        real_detection = 0

    # return the t0 values - if true, then dets will indicate which bands
    return real_detection, dets

# check if t0+n or t0+(n+0.2) days has a detection 
    
def check_two_detections(event, idx_t0, idx_next, limits):

    real_detection, dets_t0 = check_detection(event, idx_t0, limits)

    # if there is a 1st detection, the consider the next
    if real_detection:

        # check the same bands
        mask = dets_t0 = 1
        det_next_flag, dets_tnext = check_detection(event, idx_next, limits)

        # if the sum is 1 or more with the mask then there was a detection in the same band at this later phase
        if np.sum(dets_tnext[mask]) >= 1 or np.sum(dets_tnext1[mask]) >= 1: 
            return 2
        else:
            return 1 # wasnt detected in a same band at the next obs
    else:
        return 0 # not detected


def lsst_detection_sim(n, filename, limits=[???]):

    # load in the events
    values = gen_events(n, save, filename)
    events = np.array([v for v, p in values])

    # extract just the lsst bands
        # all events, all combinations of KN, lsst only bands, all phases
        # bands=[4, 5, 6, 7, 8, 9]
    events = events[:, :, 4:10, :]

    # choose a t0 - first obs
    idx_t0 = 0
    t0 = phases[idx_t0] # t-merge for now
    tnext = t0 + 3 # 3 days for now
    tprevious = t0 - 3

    idx_next = -1
    idx_prev = -1

    # check if in the bounds
    if tprevious < min(phases):
        tprevious = -1
    else:
        idx_prev = phases[np.where(np.isclose(phases, tprevious))[0][0]]

    if tnext > max(phases):
        raise ValueError("next detection attempt beyond simulated phases") 
    else:
        idx_next = phases[np.where(np.isclose(phases, tprevious))[0][0]]

    for i, event in enumerate(events):

        # check for non-detection: tprevious before first t-merge or no detect in any band
        if tprevious != -1:

            # if there is a detection, then skip event
            if check_detection(event, idx_prev, limits)[0]:
                continue

        else:
            pass # can assume we didnt detect it because it wasnt born yet

        # now check for detection

        # ensure for a detection at t0 and if so check for detection at tnext
            # just use the same point if not bc if it wasnt detected then it wasnt in the next but if it is then found the point
        n_det = check_two_detection(event, idx_t0, idx_next, limits)

        if n_det > 1:
            # measure delta M
        



        




    
    






