import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tabulate import tabulate as tab
import warnings


def read_data_txt(IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], folder='../rawdata/txt_files'):
    """
    read waveforms and labels from text files 
    use Parser.py to create the text files from the binaries

    return
        data: array of waveforms
        labels: 1 for neutrons, 0 for gammas
    """

    data,labels = [], []
    headers, lines = ['case', 'total', 'gammas', 'neutrons', 'ratio (g/n)', 'Amax total', 'Amax gammas', 'Amax neutrons'], []
    sumTot, sumG, sumN = 0, 0, 0

    for ID in IDs:
        caseID = f'case{ID}'
        d=np.loadtxt(f'{folder}/Pulse_waveform_{caseID}.txt')
        l=np.loadtxt(f'{folder}/PSD_label_{caseID}.txt')
        n = d[l == 1]
        g = d[l == 0]
        data.append(d)
        labels.append(l)

        ## information display
        Ntot = d.shape[0]
        Ngammas = g.shape[0]
        Nneutrons = n.shape[0]
        sumTot+=Ntot
        sumG+=Ngammas
        sumN+=Nneutrons
        lines.append([f'case{ID}',
                    Ntot,
                    Ngammas,
                    Nneutrons,
                    f'{Ngammas/Nneutrons:.1f}',
                    f'{np.max(d)}',
                    f'{np.max(g)}',
                    f'{np.max(n)}',
                    ])
    lines.append(['---',  '---', '---', '---', '---', '---', '---', '---'])
    lines.append(['Total', sumTot, sumG, sumN, f'{sumG/sumN:.1f}' '---', '---', '---'])
    table = tab(lines, headers=headers, tablefmt="GitHub")
    print(table)

    return data, labels 

def merge_cases_together(data, labels):
    """
    merge (concatenate) all the cases together into one dataset
    return
        - data and labels of merge cases
        - neutrons and gammas samples
    """
    data_merged   = np.concatenate(data)
    labels_merged = np.concatenate(labels)
    neutrons_merged = data_merged[labels_merged == 1]
    gammas_merged   = data_merged[labels_merged == 0]

    return data_merged, labels_merged, neutrons_merged, gammas_merged

def pulse_selection_tight(data, labels, voltage_range = (0.05, 0.5), late_start = 30, late_end = 200, afterpulse_frac = 0.08, peak_position_max = 80):
    """
    Tight selection for the templates

    Cuts:
     - peak_position_max: if the main peak is situated after this position then the event is rejected
     - late_start, late_end: define a search window after the peak where to look for afterpulses
     - afterpulse_frac: fraction of main peak to be cosidered as an afterpulse 

    """

    lines = []
    headers = ['', 'count', 'percentage of total [%]']
    Ntot = data.shape[0]
    lines.append(['Ntot',  Ntot])

    # from the paper: they average between 0.05 and 0.5 V 
    min_threshold = voltage_range[0] 
    max_threshold = voltage_range[1]

    ## get rid of very late pulses
    peak_idx = data.argmax(axis=1)
    mask_peak_position= (peak_idx <= peak_position_max) 
    data   = data[mask_peak_position]
    labels = labels[mask_peak_position]

    peak_idx = np.argmax(data, axis=1)
    peak_val = np.max(data, axis=1)
    mask_clean = np.ones(len(data), dtype=bool)

    count_minThreshold = 0
    count_maxThreshold = 0
    count_switch = 0
    count_afterPulses = 0
    count_beforePulses = Ntot - data.shape[0]

    for i in range(len(data)):
        pidx = peak_idx[i]
        pval = peak_val[i]

        if pval < min_threshold:
                count_minThreshold+=1
                mask_clean[i] = False
                continue
        if pval > max_threshold:
                max_threshold+=1
                mask_clean[i] = False
                continue
        
        # after-pulse / pile-up rejection
        start = pidx + late_start
        end   = min(pidx + late_end, data.shape[1])
        if start >= end:
            count_switch+=1
            mask_clean[i] = False
            continue
        
        late_max = np.max(data[i, start:end]) # get the maximum in the search window
        if late_max > afterpulse_frac * pval:
            count_afterPulses+=1
            mask_clean[i] = False

    lines.append(['count_minThreshold', count_minThreshold,  f'{count_minThreshold*100/Ntot:.1f}'])
    lines.append(['count_maxThreshold', count_maxThreshold,  f'{count_maxThreshold*100/Ntot:.1f}'])
    lines.append(['count_switch',       count_switch,        f'{count_switch*100/Ntot:.1f}'])
    lines.append(['count_beforePulses', count_beforePulses,  f'{count_beforePulses*100/Ntot:.1f}'])
    lines.append(['count_afterPulses',  count_afterPulses,   f'{count_afterPulses*100/Ntot:.1f}'])

    data_clean = data[mask_clean]
    labels_clean = labels[mask_clean]
    Nsel = data_clean.shape[0]
    lines.append(['---',  '---', '---'])
    lines.append(['Nsel',  Nsel, f'{Nsel*100/Ntot:.1f}'])

    table = tab(lines, headers=headers, tablefmt="GitHub")
    print(table )
    return data_clean, labels_clean

def make_templates(data, bin_edges = np.linspace(0.05, 0.5, 11)):
    """
    make templates by averaging waveforms in bins of voltage
    paper: 10 bins (= 11 edges), in range 0.05V to 0.5V
    """

    print(' sample shape', data.shape) #(Npulses, 296)
    print(' peak amplitude (min, max)', np.min(data), np.max(data))
    print(' average peak amplitude', np.average(data))

    Nbins = bin_edges.shape[0]-1

    amplitudes = data.max(axis=1) ## get the maximum of each pulse, shape (Npulses,)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # bin index for each pulse: 0..9, or -1 if outside range
    bin_idx = np.digitize(amplitudes, bin_edges) - 1
    templates = np.zeros((Nbins, data.shape[1]))
    counts = np.zeros(Nbins, dtype=int)

    for i in range(Nbins):
        mask = bin_idx == i
        counts[i] = mask.sum()

        if counts[i] > 0:
            templates[i] = data[mask].mean(axis=0)

    templates_norm = templates / templates.max(axis=1, keepdims=True)
    print(" counts per bin:", counts)
    return bin_centers, templates, templates_norm

def align_templates(templates, target_idx=50):
    """
    align the position of the peaks between the different templates
    target_idx: desired position for the peak
    """

    # get peak index (for each pulse)
    current_peaks = np.argmax(templates, axis=1)
    # get shift 
    shifts = target_idx - current_peaks
    # apply shift
    aligned_templates = np.zeros_like(templates)
    for i, shift in enumerate(shifts):
        aligned_templates[i] = np.roll(templates[i], shift)
        
    return aligned_templates

def interpolate_template(A, shapes, bin_centers):
    """
    Linear interpolation of NORMALIZED pulse templates (shapes).

    Parameters
    ----------
    A : float
        Target amplitude
    shapes : array (Nbins, Nsamples)
        Templates normalized to peak = 1
    bin_centers : array (Nbins,)
        Amplitude associated with each template

    Returns
    -------
    pulse : array (Nsamples,)
        Interpolated pulse with peak amplitude = A

    CAUTION: the interpolation doesn't work for values of A that are not comprise between bin_centers[0] and bin_centers[-1]
     solution we use the closest (minimal or maximal) template if we want to go out of range

    IMPORTANT: for this ot work fine, the templates must be normalized (we then multiply by the amplitude to get not normalized pulses)
    """

    # --- Clamp outside range (no extrapolation) ---
    if A <= bin_centers[0]:
        warnings.warn(
            f"A={A:.3f} below template range "
            f"[{bin_centers[0]:.3f}, {bin_centers[-1]:.3f}] — clamping to lowest template"
        )
        return A * shapes[0].copy()

    if A >= bin_centers[-1]:
        warnings.warn(
            f"A={A:.3f} above template range "
            f"[{bin_centers[0]:.3f}, {bin_centers[-1]:.3f}] — clamping to highest template"
        )
        return A * shapes[-1].copy()

    # --- Find surrounding bins ---
    idx_hi = np.searchsorted(bin_centers, A)
    idx_lo = idx_hi - 1

    A_lo, A_hi = bin_centers[idx_lo], bin_centers[idx_hi]

    # --- Linear interpolation in amplitude space ---
    w = (A - A_lo) / (A_hi - A_lo)

    shape = (
        (1.0 - w) * shapes[idx_lo] +
        w * shapes[idx_hi]
    )

    # --- Scale to target amplitude ---
    pulse = A * shape

    return pulse.copy()

def generate_synthetic_pulse(A, templates, bin_centers, sigma, Normalize=True):
    """
    generate a synthetic pulse with noise form the interpolated templates
    A: target amplitude
    simga: width of the gaussian noise
    Normalize: return pulses normalized to the maximum (useful for the ML implementation, but turn it off for PSD)

    """
    # interpolate
    pulse = interpolate_template(A, templates, bin_centers)

    # add Gaussian noise
    noise = np.random.normal(0, sigma, size=pulse.shape)
    pulse = pulse + noise

    # normalize to peak = 1
    if Normalize:
        pulse = pulse / np.max(pulse)

    return pulse

def generate_sample(templates, bin_centers, Npulses, sigma, A_min=None, A_max=None, Normalize=True): 
    """
    create a sample of Npulses pulses with random (flat) amplitudes between A_min and A_max
    return
        - X: sythetic pulses
        - amplitudes: randomly generated array of amplitudes (useful for sanity check)
    """
    if A_min is None:
        A_min = bin_centers.min()
    if A_max is None:
        A_max = bin_centers.max()

    # sample amplitude
    amplitudes = np.random.uniform(A_min, A_max, Npulses)

    X = np.zeros((Npulses, templates.shape[1]))

    for i in range(Npulses):
        X[i] = generate_synthetic_pulse(amplitudes[i], templates, bin_centers, sigma, Normalize=Normalize)
    
    print("Clamped fraction:",
      np.mean(amplitudes >= bin_centers[-1])) # if this is too large there will be a distortion in the PSD shape -> reduce A_max

    return X, amplitudes 

def get_psd_integrals(data, total_start=2, total_end=185, tail_start=9):
    """
    Compute total charge and tail-to-total ratio (PSD).

    Parameters
    ----------
    data : (Npulses, Nsamples)
        Pulse waveforms (NOT normalized)
    total_start : int
        Start of total integration window (relative to peak)
    total_end : int
        End of total integration window (relative to peak)
    tail_start : int
        Start of tail integration window (relative to peak)

    Returns
    -------
    totals : array
        Total integrated charge
    ttr : array
        Tail-to-total ratio
    """

    totals, ttr = [], []

    for pulse in data:
        peak_position = np.argmax(pulse)

        lo = peak_position - total_start
        hi = peak_position + total_end

        if lo < 0 or hi > len(pulse):
            continue

        total = np.sum(pulse[lo:hi])
        tail  = np.sum(pulse[peak_position + tail_start : hi])

        if total > 0:
            totals.append(total)
            ttr.append(tail / total)

    return np.asarray(totals), np.asarray(ttr)



