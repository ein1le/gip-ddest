import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
import pywt
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial
from matplotlib import font_manager as fm


font_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\computer-modern\cmunrm.ttf"
fm.fontManager.addfont(font_path)
cm_font = fm.FontProperties(fname=font_path)
font_name = cm_font.get_name()
plt.rcParams['font.family'] = font_name

scales = np.arange(1, 128)

df_DRY = pd.read_csv(r"LFT friction avg\DRY_ratio_array.csv")
df_SG = pd.read_csv(r"LFT friction avg\SG_ratio_array.csv")
df_PTFE = pd.read_csv(r"LFT friction avg\PTFE_ratio_array.csv")
df_2L = pd.read_csv(r"LFT friction avg\TwoL_ratio_array.csv")
df_2LR = pd.read_csv(r"LFT friction avg\TwoLR_ratio_array.csv")

test_type = {
    "DRY": df_DRY,
    "SG": df_SG,
    "PTFE": df_PTFE,
    "2L": df_2L,
    "2LR": df_2LR
}



def make_stationary(signal, method='polynomial', poly_order=3, window_length=51, plot=True):
    """
    Removes a nonlinear trend from the signal to make it stationary.
    """
    signal = np.asarray(signal)
    n = len(signal)
    t = np.arange(n)

    if method == 'polynomial':
        coefs = Polynomial.fit(t, signal, poly_order).convert().coef
        trend = Polynomial(coefs)(t)
    elif method == 'savgol':
        if window_length >= n:
            window_length = n - 1 if n % 2 == 0 else n
        if window_length % 2 == 0:
            window_length += 1
        trend = savgol_filter(signal, window_length=window_length, polyorder=poly_order)
    else:
        raise ValueError("Method must be 'polynomial' or 'savgol'.")

    signal_stationary = signal - trend

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(t, signal, label='Original Signal', lw=2)
        plt.plot(t, trend, label=f'Trend ({method})', lw=2)
        plt.plot(t, signal_stationary, label='Stationary Signal', lw=1.5, linestyle='--')
        plt.legend()
        plt.title('Trend Removal for Stationarity')
        plt.xlabel('Sample Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return signal_stationary, trend


avg_spectra = {} 

def fft_plot_df(df, sampling_rate=10.0, log_scale=False, freq_limit=None, detrend_method='polynomial', name=None):
    """
    Perform FFT on each column of a DataFrame (each column = one test),
    then create:
      1) A single figure with overlaid FFT plots for all tests.
      2) A second figure showing the aggregated (mean) FFT and ±1 std dev region.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Each column corresponds to a different test/time-series.
    sampling_rate : float
        Sampling rate in Hz (default 10 Hz).
    log_scale : bool
        If True, plots amplitude on a log scale.
    freq_limit : float or None
        If set, only show frequency components up to this value (in Hz).
    detrend_method : str
        'polynomial' or 'savgol' for stationarizing the signal.
    """

    # Store each test's freq, amplitude for aggregation
    all_freqs = []
    all_amps = []

    # First figure: overlayed FFTs

    plt.figure(figsize=(10, 6))


    for col in df.columns:
        signal = np.asarray(df[col].dropna())  # ensure no NaNs

        # Make stationary
        signal_stationary, _ = make_stationary(signal, method=detrend_method, plot=False)

        n = len(signal_stationary)
        dt = 1.0 / sampling_rate

        # FFT
        fft_vals = np.fft.fft(signal_stationary)
        fft_freqs = np.fft.fftfreq(n, d=dt)

        # Keep only positive freqs
        pos_mask = fft_freqs > 0
        fft_vals = fft_vals[pos_mask]
        fft_freqs = fft_freqs[pos_mask]
    
        # Convert to amplitude
        amplitude = 2.0 / n * np.abs(fft_vals)

        # If freq_limit is given, filter frequencies
        if freq_limit is not None:
            freq_mask = fft_freqs <= freq_limit
            fft_freqs = fft_freqs[freq_mask]
            amplitude = amplitude[freq_mask]

        # Plot on the overlay figure
        plt.plot(fft_freqs, amplitude, label=col, lw=1.5)


        # Save for aggregation
        all_freqs.append(fft_freqs)
        all_amps.append(amplitude)

    # Formatting for the overlay figure

    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel("Amplitude" + (" [log scale]" if log_scale else ""), fontsize=12)
    plt.title(f"FFT Spectra - {name}", fontsize=14)
    if log_scale:
        plt.yscale('log')
    #plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if freq_limit is not None:
        plt.xlim([0, freq_limit])
    plt.show()



    # Let's pick the first freq array as reference; you can also find the min freq array length etc.
    ref_freqs = all_freqs[0]
    # Convert others if needed (here, ignoring if they differ).
    # Build a matrix of amplitudes
    amp_matrix = []
    for freqs, amps in zip(all_freqs, all_amps):
        # if freq arrays differ, you'd do something like:
        # amps_common = np.interp(ref_freqs, freqs, amps)
        # amp_matrix.append(amps_common)
        # but assuming they're identical:
        amp_matrix.append(amps)

    amp_matrix = np.vstack(amp_matrix)
    mean_amp = np.mean(amp_matrix, axis=0)
    std_amp = np.std(amp_matrix, axis=0)

    # Plot aggregated

    plt.figure(figsize=(10, 6))
    plt.plot(ref_freqs, mean_amp, label='Mean FFT Amplitude', lw=2, color='C0')
    plt.fill_between(ref_freqs, mean_amp - std_amp, mean_amp + std_amp,
                     alpha=0.3, color='C0', label='±1 STD')

    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel("Amplitude" + (" [log scale]" if log_scale else ""), fontsize=12)
    plt.title(f"Aggregated FFT - {name}", fontsize=14)
    if log_scale:
        plt.yscale('log')
    #plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if freq_limit is not None:
        plt.xlim([0, freq_limit])
    plt.show()

    avg_spectra[name] = {
        "frequency": ref_freqs,
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp
    }


for name, test in test_type.items():
    fft_plot_df(
        test,
        sampling_rate=10.0,
        log_scale=False,
        freq_limit=2,
        detrend_method='savgol',
        name = name
    )


# Choose the subplot you want to use — for example, the first one (top-left)
filtered_keys = [key for key in avg_spectra if key != "SG"]

# Set up 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(22, 10))
axs = axs.flatten()  # Flatten to make indexing easier

# Plot each key in a separate subplot
for i, key in enumerate(filtered_keys[:4]):  # Only take up to 4 keys
    data = avg_spectra[key]
    freq = data["frequency"] 
    mean_amp = data["mean_amplitude"]* 10e3
    std_amp = data["std_amplitude"] * 10e3


    ax = axs[i]
    ax.plot(freq, mean_amp, label=f'{key}',color = "black")
    ax.fill_between(freq, mean_amp - std_amp, mean_amp + std_amp, alpha=0.4,color = "black")
    
    ax.set_xlabel("Frequency (Hz)",fontsize=30)
    ax.set_ylim( -0.1, 15)
    ax.tick_params(axis='both', labelsize=24)
    ax.set_ylabel("Mean Amplitude (mN)",fontsize=28)
    ax.set_title(f"{key}",fontsize=32)
    ax.grid(True, alpha=0.3)
    #ax.legend()

# Remove unused subplots if fewer than 4 keys

for j in range(len(filtered_keys), 4):
    fig.delaxes(axs[j])

plt.subplots_adjust(wspace=0.3)
plt.tight_layout()
plt.savefig(r"fft_plots.png")
plt.show()


def wavelet_features(signal, wavelet='morl', sampling_rate=10.0, max_scale=128, detrend_method="savgol"):
    """
    Compute average integrated wavelet power and average coefficient amplitude
    for a given 1D signal using Continuous Wavelet Transform (CWT), with
    a frequency axis matching standard Hz for 'sampling_rate'.
    """

    # 1) Detrend the signal
    signal = np.asarray(signal)
    signal_stationary, _ = make_stationary(signal, method=detrend_method, plot=False)
    
    # 2) Define scales from 1 to max_scale
    scales = np.arange(1, max_scale + 1)
    
    # 3) Perform the CWT
    coeffs, _ = pywt.cwt(
        signal_stationary, 
        scales, 
        wavelet, 
        sampling_period=1.0 / sampling_rate
    )
    # coeffs.shape -> (num_scales, num_time_points)

    # 4) Convert scales to actual frequencies in Hz
    #    The default freq array from pywt.cwt can be wavelet-dependent,
    #    so we do an explicit scale->freq conversion for the 'morl' wavelet
    freqs_hz = pywt.scale2frequency(wavelet, scales) / (1.0 / sampling_rate)
    # freqs_hz[0] is typically the highest frequency, freqs_hz[-1] the lowest

    # 5) Compute wavelet power
    wavelet_power = np.abs(coeffs)**2
    total_power = np.sum(wavelet_power)
    num_coeffs = coeffs.size  # total elements in coeffs (scales * time length)
    
    # 6) Average integrated wavelet power
    avg_int_power = total_power / num_coeffs
    
    # 7) Average coefficient amplitude
    avg_coef_amplitude = np.mean(np.abs(coeffs))
    
    # Return freq axis in Hz (descending order from high freq -> low freq)
    return avg_int_power, avg_coef_amplitude, coeffs, freqs_hz

def wavelet_features_df(df, wavelet='morl', sampling_rate=10.0, max_scale=128, detrend_method="savgol"):
    """
    For each column in 'df', compute average integrated wavelet power and average coefficient amplitude
    using Continuous Wavelet Transform (CWT).

    Parameters
    ----------
    df : pandas.DataFrame
        Each column corresponds to a different time-series test.
    wavelet : str
        Mother wavelet (default 'morl').
    sampling_rate : float
        Sampling rate in Hz (default 10.0).
    max_scale : int
        Maximum scale for CWT (controls frequency resolution).
    detrend_method : str
        'polynomial' or 'savgol' method for make_stationary.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with columns:
          ['Test', 'avg_int_power', 'avg_amp', 'coeffs', 'freqs_hz']
        containing wavelet features for each test.
    aggregated_stats : dict
        Dictionary with mean and std for 'avg_int_power' and 'avg_amp' across all columns.
    """

    # We'll store results for each column here
    results = []

    # Loop over columns (each test)
    for col in df.columns:
        # 1) Extract the signal as a 1D numpy array (dropping NaN if any)
        signal = df[col].dropna().to_numpy()

        # 2) Compute wavelet features using your existing wavelet_features function
        avg_int_power, avg_amp, coeffs, freqs_hz = wavelet_features(
            signal, 
            wavelet=wavelet, 
            sampling_rate=sampling_rate, 
            max_scale=max_scale, 
            detrend_method=detrend_method
        )

        # 3) Append to the list
        results.append({
            'Test': col,
            'avg_int_power': avg_int_power,
            'avg_amp': avg_amp,
            'coeffs': coeffs,
            'freqs_hz': freqs_hz
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # 4) Compute aggregate mean and std for the scalar features across tests
    mean_int_power = results_df['avg_int_power'].mean()
    std_int_power  = results_df['avg_int_power'].std()
    mean_amp       = results_df['avg_amp'].mean()
    std_amp        = results_df['avg_amp'].std()

    aggregated_stats = {
        'mean_avg_int_power': float(mean_int_power),
        'std_avg_int_power': float(std_int_power),
        'mean_avg_amp': float(mean_amp),
        'std_avg_amp': float(std_amp)
    }

    return results_df, aggregated_stats

def plot_wavelet_scalogram(coeffs, freqs, sampling_rate=10.0, cmap='viridis', title="Wavelet Scalogram"):
    """
    Visualizes a wavelet scalogram from CWT coefficients.

    Parameters
    ----------
    coeffs : 2D ndarray
        Wavelet coefficients from pywt.cwt (shape: [scales, time]).
    freqs : 1D ndarray
        Frequencies corresponding to scales.
    sampling_rate : float
        Sampling rate in Hz for time axis.
    cmap : str
        Colormap for heatmap.
    title : str
        Plot title.
    """
    time_points = coeffs.shape[1]
    time = np.arange(time_points) / sampling_rate

    plt.figure(figsize=(12, 6))
    plt.imshow(
        np.abs(coeffs), 
        extent=[time[0], time[-1], freqs[-1], freqs[0]],  # flip y-axis
        aspect='auto', 
        cmap=cmap,
        origin='upper'
    )
    plt.colorbar(label='|Coefficient Magnitude|')
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Frequency [Hz]", fontsize=12)
    plt.title(title, fontsize=14)
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

for name,test in test_type.items():
    results_df, agg_stats = wavelet_features_df(
    test,
    wavelet='morl',
    sampling_rate=10.0,
    max_scale=64,
    detrend_method='savgol'
    )      
    print(f"Wavelet Features for {name}:")
    print(results_df)
    print(agg_stats)

for name,test in test_type.items():
    plot_wavelet_scalogram(
        results_df['coeffs'].values[0],
        results_df['freqs_hz'].values[0],
        sampling_rate=10.0,
        cmap='viridis',
        title=f"Wavelet Scalogram for {name}"
    )

def get_dominant_amplitude_over_time(coeffs):
    """
    Given the wavelet coefficient matrix 'coeffs' (shape: [num_scales, num_time_points]),
    returns a 1D array 'dominant_amp' of shape [num_time_points],
    where dominant_amp[t] = max(|coeffs[:, t]|) across scales.
    """
    # Take absolute value across scales and find the max for each time index
    # coeffs.shape = (n_scales, n_time)
    amplitude = np.abs(coeffs)
    dominant_amp = amplitude.max(axis=0)  # shape = (n_time,)
    return dominant_amp
def average_dominant_amplitude_over_time(df, 
                                         wavelet='morl', 
                                         sampling_rate=10.0, 
                                         max_scale=128, 
                                         detrend_method="savgol"):
    """
    For each column (test) in 'df':
      1) Compute wavelet coefficients,
      2) Find the maximum wavelet amplitude (across scales) at each time index,
      3) Average these "dominant amplitude" curves across all columns.

    Returns
    -------
    avg_dom_amp : 1D numpy array (length = time_points)
        The time series of average dominant amplitude across all columns.
    time : 1D numpy array
        The time axis (seconds) for plotting, matching the length of avg_dom_amp.
    dom_amp_matrix : 2D (num_tests x num_time_points)
        The individual dominant amplitude curves for each column (test).
    """

    from copy import deepcopy

    # We'll store the dominant amplitude time-series for each column
    dom_amp_list = []
    min_len = None  # to handle possibly different test lengths

    # Re-use your wavelet_features_df logic, but we need the 'coeffs' from each column
    # We'll do it more directly below, or you could reuse wavelet_features_df.

    for col in df.columns:
        signal = df[col].dropna().to_numpy()

        # wavelet_features returns (avg_int_power, avg_amp, coeffs, freqs_hz)
        _, _, coeffs, freqs_hz = wavelet_features(
            signal, wavelet=wavelet, sampling_rate=sampling_rate, 
            max_scale=max_scale, detrend_method=detrend_method
        )

        # coeffs.shape = (num_scales, num_time_points)
        dom_amp = get_dominant_amplitude_over_time(coeffs)  # shape: (num_time_points,)

        # Keep track of the smallest length among columns (if they differ)
        if min_len is None or len(dom_amp) < min_len:
            min_len = len(dom_amp)

        dom_amp_list.append(dom_amp)

    # Now dom_amp_list is a list of 1D arrays, possibly different lengths
    # We'll unify them by cutting to 'min_len'
    dom_amp_matrix = []
    for dom_amp in dom_amp_list:
        dom_amp_matrix.append(dom_amp[:min_len])

    # Convert to array => shape (num_tests, min_len)
    dom_amp_matrix = np.vstack(dom_amp_matrix)

    # Compute average across tests (axis=0 => average across rows)
    avg_dom_amp = dom_amp_matrix.mean(axis=0)

    # Build time axis in seconds => same as the length
    time = np.arange(min_len) / sampling_rate

    return avg_dom_amp, time, dom_amp_matrix

for name, df_tests in test_type.items():
    avg_dom_amp, time, _ = average_dominant_amplitude_over_time(
        df_tests,
        wavelet='morl',
        sampling_rate=10.0,
        max_scale=64,
        detrend_method='savgol'
    )
    plt.plot(time, avg_dom_amp, label=name)

plt.xlabel("Time [s]")
plt.ylabel("Dominant Wavelet Amplitude (A.U.)")
plt.title("Average Dominant Stick-Slip Amplitude Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
