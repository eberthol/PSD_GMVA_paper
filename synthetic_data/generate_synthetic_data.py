import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tabulate import tabulate 
import generate_synthetic_data as gsd
from collections import OrderedDict
import pandas as pd

import warnings
warnings.filterwarnings("ignore") # for now

#------ read data ------
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
        l=np.loadtxt(f'{folder}/Pulse_label_{caseID}.txt')
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
    table = tabulate(lines, headers=headers, tablefmt="GitHub")
    print(table)

    return np.array(data), np.array(labels) 

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

#------ pulse selection ------

def get_total_integral(data):
    total_start=2
    total_end=185
    totals = []
    for pulse in data:
        peak_position = np.argmax(pulse)
        lo = peak_position - total_start
        hi = peak_position + total_end

        if lo < 0 or hi > len(pulse):
            totals.append(np.nan)
            continue
        
        total = np.sum(pulse[lo:hi])
        if total > 0:
            totals.append(total)
        else:
            totals.append(np.nan)
    totals = np.array(totals)
    return totals

def reject_afterpulses(data, late_start, late_end, afterpulse_frac):
    """
        reject after pulses contained in a window [late_start, late_end] defined w.r.t peak
        if they are larger than (afterpulse_frac * peak amplitude)
        returns a mask to apply to the data
    """
    N, L = data.shape

    # get peak positions and amplitudes
    peak_idx = np.argmax(data, axis=1)
    peak_val = np.max(data, axis=1)

    # Late window indices
    late_offsets = np.arange(late_start, late_end)
    late_indices = peak_idx[:, None] + late_offsets[None, :]
    late_indices = np.clip(late_indices, 0, L - 1)

    # Late window values
    late_values = data[np.arange(N)[:, None], late_indices]
    late_max = np.max(late_values, axis=1)

    # Afterpulse / pile-up rejection
    mask_afterpulse = late_max <= afterpulse_frac * peak_val

    return  mask_afterpulse

def cutflow_table(cutflow):
    table_data = [[k] + list(v.values()) for k, v in cutflow.items()]
    headers = ["Step"] + list(cutflow["Input"].keys())
    col_formats = ("", ".0f", ".4f", ".4f", ".0f", ".4f", ".0f", ".4f")
    print(tabulate(table_data, headers=headers, tablefmt="", floatfmt=col_formats))
    df = pd.DataFrame(cutflow)
    return df

def tight_selection(data, labels, Vsamples_range, Vpeak_range, peak_position_max, late_window, afterpulse_frac, return_cutflow=True):
    """
    Docstring for tight_selection
    
    :param Vsamples_range: energy cut - range of the integral of the pulse in volts per sample
    :param Vpeak_range: amplitude cut - voltage range of the peak
    :param peak_position_max: keep only pulses that occur before this position
    :param late_window: min and max posion for the "late window" w.r.t peak position
    :param afterpulse_frac: Description
    :param return_cutflow: boolean
    """
    label_map = {1: "neutron", 0: "gamma"}
    N0 = len(data)
    mask0 = np.ones(N0, dtype=bool)

    cutflow = OrderedDict()
    cutflow["Input"] = {"N": N0, "eff_abs": 1.0, "eff_rel": 1.0}
    for lbl, name in label_map.items():
        cutflow["Input"][f"N_{name}"] = np.sum(labels == lbl)
        cutflow["Input"][f"eff_{name}"] = 1.0

    mask_full = mask0.copy()

    # --------------------------------------------------
    # Step 1: peak-position cut (optional)
    # --------------------------------------------------
    if peak_position_max is not None:
        peak_idx = np.argmax(data, axis=1)
        mask_pos = peak_idx <= peak_position_max
        mask_full &= mask_pos

        cutflow["Peak position cut"] = {
            "N": int(mask_full.sum()),
            "eff_abs": float(mask_full.sum() / N0),
            "eff_rel": float(mask_pos[mask0].mean()),  # relative to previous (Input)
        }
        for lbl, name in label_map.items():
            denom = np.sum((labels == lbl) & mask0)
            num = np.sum((labels == lbl) & mask_full)
            cutflow["Peak position cut"][f"N_{name}"] = int(num)
            cutflow["Peak position cut"][f"eff_{name}"] = float(num / denom) if denom else 0.0

    # Work on the selected subset so far
    data_sel = data[mask_full]
    labels_sel = labels[mask_full]

    # --------------------------------------------------
    # Step 2: energy / integral selection (optional)
    # --------------------------------------------------
    if Vsamples_range is not None:
        pulse_integrals = get_total_integral(data_sel)
        mask_E_local = (pulse_integrals >= Vsamples_range[0]) & (pulse_integrals <= Vsamples_range[1])

        # update global mask
        idx_global = np.where(mask_full)[0]
        mask_E_full = np.zeros_like(mask_full)
        mask_E_full[idx_global[mask_E_local]] = True
        mask_full = mask_E_full

        cutflow["Energy selection"] = {
            "N": int(mask_full.sum()),
            "eff_abs": float(mask_full.sum() / N0),
            "eff_rel": float(mask_E_local.mean()),
        }
        for lbl, name in label_map.items():
            denom = np.sum((labels == lbl) & (mask_E_full | (~mask_E_full)))  # == total per class
            denom = np.sum(labels == lbl)
            num = np.sum((labels == lbl) & mask_full)
            cutflow["Energy selection"][f"N_{name}"] = int(num)
            cutflow["Energy selection"][f"eff_{name}"] = float(num / denom) if denom else 0.0

        data_sel = data[mask_full]
        labels_sel = labels[mask_full]
    
    # --------------------------------------------------
    # Step 3: amplitude cut
    # --------------------------------------------------
    Vpeak = np.max(data_sel, axis=1)
    mask_amp_local = (Vpeak >= Vpeak_range[0]) & (Vpeak <= Vpeak_range[1])

    idx_global = np.where(mask_full)[0]
    mask_amp_full = np.zeros_like(mask_full)
    mask_amp_full[idx_global[mask_amp_local]] = True
    mask_full = mask_amp_full

    cutflow["Amplitude cut"] = {
        "N": int(mask_full.sum()),
        "eff_abs": float(mask_full.sum() / N0),
        "eff_rel": float(mask_amp_local.mean()) if len(mask_amp_local) else 0.0,
    }
    for lbl, name in label_map.items():
        denom = np.sum(labels == lbl)
        num = np.sum((labels == lbl) & mask_full)
        cutflow["Amplitude cut"][f"N_{name}"] = int(num)
        cutflow["Amplitude cut"][f"eff_{name}"] = float(num / denom) if denom else 0.0

    data_sel = data[mask_full]
    labels_sel = labels[mask_full]

    # --------------------------------------------------
    # Step 4: afterpulse rejection
    # --------------------------------------------------
    mask_after_local = reject_afterpulses(
        data_sel,
        late_start=late_window[0],
        late_end=late_window[1],
        afterpulse_frac=afterpulse_frac
    )

    idx_global = np.where(mask_full)[0]
    mask_after_full = np.zeros_like(mask_full)
    mask_after_full[idx_global[mask_after_local]] = True
    mask_full = mask_after_full

    cutflow["Afterpulse rejection"] = {
        "N": int(mask_full.sum()),
        "eff_abs": float(mask_full.sum() / N0),
        "eff_rel": float(mask_after_local.mean()) if len(mask_after_local) else 0.0,
    }
    for lbl, name in label_map.items():
        denom = np.sum(labels == lbl)
        num = np.sum((labels == lbl) & mask_full)
        cutflow["Afterpulse rejection"][f"N_{name}"] = int(num)
        cutflow["Afterpulse rejection"][f"eff_{name}"] = float(num / denom) if denom else 0.0
    
    data_final = data[mask_full]
    labels_final = labels[mask_full]

    if return_cutflow:
        return data_final, labels_final, cutflow
    return data_final, labels_final

#------ create templates ------
def make_templates(data, bin_edges = np.linspace(0.05, 0.5, 11), target_idx=60, align = True):
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

    if align:
        templates = align_templates(templates, target_idx=target_idx)

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

#------ generate neutron and gamma pulses from templates------
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

#------ perform PSD ------
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

#------ pileup ------
def sample_dt_exponential(rate_hz, dt_sampling_s, window_len_samples):
    """
    Sample inter-arrival time in ADC samples

    rate_hz      : average event rate (Hz)
    dt_sampling_s : sampling period (s)
    window_len_samples  : truncate to waveform length

    ATTENTION: 
    if window_len_samples is set at the legnth of the trigger window, then the distribution will peak at this value
    it is not a bug, it is expected (we are clipping the distribution)
    QUESTION: what to do with these events?
        - they are really pile up (the pile up is over 2 trigger windows)
        - maybe we should throw them away?
        - maybe we should redraw a random number until it is inside the trigger window?
    """
    rate_per_sample = rate_hz * dt_sampling_s    

    if rate_per_sample <= 0:
        print('Sanity Check: Someting is fishy...')
        return window_len_samples

    # exponential in *samples*
    delta_samples = np.random.exponential(
        scale=1.0 / rate_per_sample
    )
    return int(min(delta_samples, window_len_samples))

def sample_dt_paper(
    rate_hz: float,
    dt_sampling_s: float, # in seconds per sample 
    window_len_samples: int,
    margin_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Sample inter-pulse time shift (in ADC samples) using the paper's Eq. (2):
        P(r,t) = exp(-r t) * (1 - exp(-r t)) = exp(-r t) - exp(-2 r t)

    We interpret this as a *sampling distribution over t within the trigger window* [0, T),
    normalize it on that interval, sample t by inverse CDF, then convert to integer samples.

    Args:
        rate_hz: detector count rate r [Hz]
        dt_sampling_s: ADC sampling period [seconds per sample]
        window_len_samples: total waveform length in samples
        margin_samples: require dt <= window_len_samples - 1 - margin_samples
                        (useful to ensure the 2nd pulse is not too close to the end)
        rng: optional np.random.Generator for reproducibility

    Returns:
        dt_samples: integer shift in [0, max_shift]
    """
    if rng is None:
        rng = np.random.default_rng()

    if rate_hz <= 0:
        # No meaningful rate; choose 0 shift or max shift depending on your preference
        return 0

    N = int(window_len_samples)
    margin = int(margin_samples)
    if N <= 1:
        return 0

    max_shift = N - 1 - margin
    if max_shift < 0:
        # margin bigger than window; fall back
        return 0

    r = float(rate_hz)
    dt = float(dt_sampling_s)

    # Allowed time interval length
    T = (max_shift + 1) * dt  # shifts 0..max_shift correspond to times in [0, T)

    # Inverse-CDF sampling for the normalized CDF:
    #   F(t) = ((1 - exp(-r t)) / (1 - exp(-r T)))^2
    u = rng.random()
    a = 1.0 - np.exp(-r * T)             # (1 - e^{-rT})
    y = np.sqrt(u)                       # sqrt(u)
    inside = 1.0 - a * y                 # 1 - (1-e^{-rT})*sqrt(u)

    # Numerical safety: keep inside in (0,1]
    inside = np.clip(inside, 1e-15, 1.0)

    t = -np.log(inside) / r              # sampled continuous time in [0,T)
    dt_samples = int(np.floor(t / dt))   # convert to integer shift

    # Clamp (should rarely be needed due to clipping + finite precision)
    if dt_samples > max_shift:
        dt_samples = max_shift
    if dt_samples < 0:
        dt_samples = 0

    return dt_samples

def shift_pulse(pulse, dt):
    """
    Shift pulse by dt samples to the right 
    """
    shifted_pulse = np.zeros_like(pulse)
    if dt < len(pulse):
        shifted_pulse[dt:] = pulse[:-dt] if dt > 0 else pulse
    return shifted_pulse

def generate_pileup_event(
    A1, A2, # amlitudes of the two peaks
    template1_normalized, template2_normalized, 
    bin_centers,
    noise_sigma, # Guassian noise
    time_shift, # shift in time (number of samples)
):
    """
    Generate a pile-up pulse from 2 pulses.
    """

    p1 = interpolate_template(A1, template1_normalized, bin_centers)
    p2 = interpolate_template(A2, template2_normalized, bin_centers)

    # --- time offset ---
    p2_shifted = shift_pulse(p2, time_shift)

    # --- superposition ---
    pileup = p1 + p2_shifted

    # --- add noise ---
    pileup += np.random.normal(0.0, noise_sigma, pileup.shape)

    return pileup

def generate_random_pileup_event(
        neutron_templates_normalized, gamma_templates_normalized, bin_centers, # templates
        peak_amplitude_range_V = [0.05, 0.5],
        rate_hz = 1e6, dt_sampling_s = 2e-9, margin_samples = 0, 
        noise_sigma = 0., # Guassian noise
        Normalize = True
        ):
    
    window_len_samples = neutron_templates_normalized.shape[1]

    # randomly chose between neutrons and gamma templates
    type1, type2  = np.random.choice(['n', 'g'], size=2) 
    shapes1 = neutron_templates_normalized if type1 == 'n' else gamma_templates_normalized
    shapes2 = neutron_templates_normalized if type2 == 'n' else gamma_templates_normalized

    # --- random amplitude (uniform) ---
    A1 = np.random.uniform(peak_amplitude_range_V[0], peak_amplitude_range_V[1])
    A2 = np.random.uniform(peak_amplitude_range_V[0], peak_amplitude_range_V[1])
    p1 = interpolate_template(A1, shapes1, bin_centers)
    p2 = interpolate_template(A2, shapes2, bin_centers)

    # --- random time offset (exponential) ---
    ## do not use time offset if it is superior to window_len_samples
    ## CAREFUL: it may be a mistake to do that
    ## 60 is the HARDCODED position of the first peak - need to make that cleaner
    # time_shift = 1e10
    # while time_shift >= window_len_samples - 60:
        # time_shift = sample_dt_exponential(rate_hz, dt_sampling_s, window_len_samples)

    time_shift = sample_dt_paper(rate_hz, dt_sampling_s, window_len_samples, margin_samples, rng=None)
        
    p2_shifted = shift_pulse(p2, time_shift)

    # --- add pulses ---
    pileup = p1 + p2_shifted

    # --- add noise (Gaussian) ---
    pileup += np.random.normal(0.0, noise_sigma, pileup.shape)

    # normalize to peak = 1
    if Normalize:
        pileup = pileup / np.max(pileup)

    return pileup, time_shift

def generate_pileup_sample(
    Npulses, # how many events to generate
    neutron_templates_normalized, gamma_templates_normalized, bin_centers, # templates
    peak_amplitude_range_V = [0.05, 0.5],
    rate_hz = 1e6, dt_sampling_s = 2e-9, margin_samples = 0, 
    noise_sigma = 0., # Guassian noise 
    Normalize=True
    ):
    """
    create a sample of Npulses pulses with pile up
    """
    X = np.zeros((Npulses, neutron_templates_normalized.shape[1]))
    time_shifts = np.zeros(Npulses)
    for i in range(Npulses):
        X[i], time_shifts[i] = generate_random_pileup_event(
            neutron_templates_normalized, gamma_templates_normalized, bin_centers,
            peak_amplitude_range_V, 
            rate_hz, dt_sampling_s, margin_samples,  
            noise_sigma, 
            Normalize
        )
    return X, time_shifts

