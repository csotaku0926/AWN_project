import numpy as np
import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "../../data/single_scenario/data_generation_package/data"
N_IMG = 5000

# parameters for OFDM channel matrix
CR_FREQ = 6e10 # 60 GHz
TOTAL_BW = 5e8 # total bandwidth is 0.5 GHz
AN_SPACING = 0.5 # half antenna spacing
N_ANN = 128
N_CR = 32
C = 3e8
WVLEN = CR_FREQ / C
N_SRC = 1
D = 0.5

def process_mat(task_name:str) -> pd.DataFrame:
    """
    `task_name`: e.g., colo_direct_wireless_dataset

    read .mat file
    output pd.daraframe
    """
    wireless_dataset_path = os.path.join(DATA_PATH, f"{task_name}.mat")
    f = h5py.File(wireless_dataset_path, 'r')
    raw_key = list(f.keys())
    ds_obj = f[raw_key[1]][0][0]
    
    dataset = f[ds_obj]["user"]
    df = pd.DataFrame(columns=['loc', 'channel'])

    for i in range(N_IMG):
        user_i = f[dataset[i][0]]

        # location: (3, )
        loc_i = user_i["loc"][()]
        loc_i = np.array(loc_i).flatten()

        # channel: (# OFDM symbol, # antenna) of `numpy.void` type data (e.g.: by default (32, 128) size of (-6.51345623e-07, -8.99951125e-07))
        chnl_i = user_i["channel"][()]
        chnl_i = np.array(chnl_i)
        # convert [('real', '<f8'), ('imag', '<f8')] dtype to complex
        cmplx_chnl_i = chnl_i['real'] + chnl_i['imag'] * 1j

        df.loc[i] = [loc_i, cmplx_chnl_i]

    # print(df.head())
    # print(df.shape)

    return df


def steering_vector(angle, n_antennas, wavelength, d=0.5) -> np.array:
    """
    parameters
    - `angle`: AoA in degrees
    - `n_antennas`: num of antennas
    - `wavelength`:single wavelength
    - `d`: Antenna spacing in terms of `wavelength`

    returns
    steering vector
    """
    angle_rad = np.radians(angle)
    k = 2 * np.pi / wavelength
    array_pos = np.arange(n_antennas) * d * wavelength
    return np.exp(-1j * k * array_pos * np.sin(angle_rad))


def music_algorithm(H:np.array, n_source=1, wavelength=5e-3, d=0.5, do_plot=False) -> tuple[np.array, np.array]:
    """
    MUSIC algorithm for AoA estimation

    parameters
    - `H`: Channel matrix (M, N), where M is antenna num and N is subcarrier nums
    - `n_source`: num of signal source
    - `wavelength`: signal wavelength
    - `d`: Antenna spacing in terms of `wavelength`

    returns
    AoA spectrum (np.array) and angles (np.array)
    """
    n_antenna, n_subcarrier = H.shape

    # Spatial correlation matrix
    R = np.zeros((n_antenna, n_antenna), dtype=complex)
    for f in range(n_subcarrier):
        h_f = H[:, f].reshape(-1, 1) # channel vector for sc f
        R += np.dot(h_f, h_f.conj().T)
    R /= n_subcarrier

    # eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eigh(R)
    # last M - `n_source` eigenvectors
    noise_subspace = eigen_vectors[:, :-n_source]

    # angle range for AoA estimation
    angles = np.linspace(-90, 90, 361)
    spectrums = []

    for angle in angles:
        a_theta = steering_vector(angle, n_antenna, wavelength, d)
        pseudo_spectrum = 1 / np.abs(np.dot(
            a_theta.conj().T,
            np.dot(noise_subspace, noise_subspace.conj().T).dot(a_theta)
        ))
        spectrums.append(pseudo_spectrum)

    spectrums = np.array(spectrums)

    # plot (optional)
    if (do_plot):
        plt.figure(figsize=(10, 6))
        plt.plot(angles, 10 * np.log10(spectrums / np.max(spectrums))) # relative strength (dB)
        plt.title("MUSIC")
        plt.xlabel("AoA")
        plt.ylabel("Pseudo-Spectrum")
        plt.grid()
        plt.show()

    return spectrums, angles


def get_channel_gain(H:np.array, per='ant'):
    """
    calculate channel gain

    parameter
    - `H`: Channel matrix (M, N), where M is antenna num and N is subcarrier nums
    - `per`: 
    "ant" -- per ant, "sc" -- per sub-carrier, else overall gain is returned

    returns
    desired channel gain
    """
    gain = None
    # per-antenna gain
    if (per == "ant"):
        gain = np.sqrt(np.sum(np.abs(H) ** 2, axis=1))
    # per-subcarrier gain
    elif (per == "sc"):
        gain = np.sqrt(np.sum(np.abs(H) ** 2, axis=0))
    # overall channel gain (Frobenius form)
    else:
        gain = np.linalg.norm(H, 'fro')

    return gain


def get_path_delay(H:np.array, threshold=1.0, do_plot=False):
    """
    Estimate path delay leverage the time-domain proporties of channel response

    parameters
    - `H`: OFDM matrix
    - `threshold`: threshold for IFFT peak detection

    return
    list of path delays for each antenna (only for those magnitutde that exceed `threshold`)
    """
    n_ant, n_sc = H.shape

    # sub-carrier spacing = BW / # sub-carrier (in Hz)
    sc_spacing = TOTAL_BW / N_CR
    delays = []

    for m in range(n_ant): # n_ant
        # IFFT to transform freq response to time domain
        h_t = np.fft.ifft(H[m, :])
        h_t_mag = np.abs(h_t)

        # normalize and detect peaks
        max_h_t_mag = np.max(h_t_mag)
        peak_indices = np.where(h_t_mag >= threshold * max_h_t_mag)[0]
        
        # convert indices to delay
        delay_time = peak_indices / (sc_spacing * n_sc)
        delays.append(delay_time)

        # plot impulse response (optional)
        if (do_plot):
            plt.figure()
            plt.stem(np.arange(len(h_t_mag)), h_t_mag)
            plt.title(f"Impulse Response for Antenna {m + 1}")
            plt.xlabel("Delay index")
            plt.ylabel("Magnitude")
            plt.grid()
            plt.show()

    return delays


def get_doppler(H:np.array, symbol_duration=1e-3, do_plot=False):
    """
    Estimate Doppler Shift from OFDM channel matrix
    
    parameters
    - `H`: a sequence containing multiple Hs along timeslots, shape: (n_time, n_ant, n_sc)
    - `symbol_duration`: OFDM symbol duration (s) (assumed)

    return
    doppler shift array, shape: (`n_ant`, `n_sc`)
    """
    n_sym, n_ant, n_sc = H.shape
    doppler_shifts = np.zeros((n_ant, n_sc))

    for m in range(n_ant):
        for f in range(n_sc):
            # extract channel response for each antenna and sub-carrier
            h_k = H[:, m, f]

            # compute DFT to get Doppler spectrum
            doppler_spectrum = np.fft.fftshift(np.fft.fft(h_k))
            freqs = np.fft.fftshift(np.fft.fftfreq(n_sym, d=symbol_duration))

            # find doppler corresponding to peak of spectrum
            peak_index = np.argmax(np.abs(doppler_spectrum) ** 2)
            doppler_shift = freqs[peak_index]
            doppler_shifts[m, f] = doppler_shift

            # plot (optional)
            if (do_plot and m == 0 and f == 0):
                plt.figure()
                plt.plot(freqs, np.abs(doppler_spectrum))
                plt.title(f"Antenna {m}, SC {f}")
                plt.xlabel("doppler freq")
                plt.ylabel("Magnitude")
                plt.grid()
                plt.show()
    
    return doppler_shifts


def get_temporal_correlation(H:np.array, symbol_duration=1e-3, max_lag=None, do_plot=False):
    """
    Compute temporal channel correlation of OFDM channel matrices

    - `H`: 3D OFDM channel array
    - `symbol_duration`: duration of one OFDM symbol
    - `max_lag`: maximum lag to compute correlation
    """
    n_sym, n_ant, n_sc = H.shape
    if (max_lag is None):
        max_lag = n_sym - 1
    
    # focus on single antenna and sc
    h_k = H[:, 0, 0]
    # normalize
    h_k = h_k / np.sqrt(np.mean(np.abs(h_k) ** 2))

    # compute temporal correlation for each lag
    lags = np.arange(0, max_lag + 1)
    correlation = []
    for lag in lags:
        corr = np.mean(h_k[:n_sym - lag] * np.conj(h_k[lag : n_sym]))
        correlation.append(corr)

    correlation = np.array(correlation)
    time_lags = lags * symbol_duration

    # plot (optional)
    if (do_plot):
        plt.figure()
        plt.plot(time_lags, np.abs(correlation), marker='o')
        plt.title("Temporal Channel Correlation")
        plt.xlabel("Time Lag (s)")
        plt.ylabel("Correlation Magnitude")
        plt.grid()
        plt.show()

    return time_lags, correlation


"""
TODO:
1. User tracking problem definition

                                image feature extraction V
channel parameters prediction --> sensing aided refinition (?) --> predicted channel states

- channel states: 
    - AoA -- AoA estimation, MUSIC algorithm
    - path gain -- compute norm of signal's multipath components
    - path delay -- IFFT peak detection
    - Doppler shift -- relative velocity / signal strength
     
- image data:

- output predicted user location, velocity
"""
def main():
    df = process_mat("colo_direct_wireless_dataset")
    loc = df.iloc[0]["loc"]
    print(round(loc[0], 4))
    
    # remember to transpose
    chnl = df.iloc[20]["channel"].T
    print("channel shape:", chnl.shape)

    # estimate AoA
    spectrums, angles = music_algorithm(chnl, n_source=N_SRC, wavelength=WVLEN)
    spec_idx = np.argmax(spectrums)
    print("estimated angle:", angles[spec_idx])

    # path gain
    gain = get_channel_gain(chnl)
    print("channel gain shape:", gain.shape)

    # path delay
    for i in [0, 20, 40]:
        ch = df.iloc[i]["channel"].T
        delays = get_path_delay(ch, threshold=1.0)
        print(delays[0])

    # Doppler shift
    Hs = [df.iloc[i]["channel"].T for i in range(10)]
    Hs = np.stack(Hs, axis=0) # (#time, channel.shape)
    dp_spectrum = get_doppler(Hs)


if __name__ == '__main__':
    main()