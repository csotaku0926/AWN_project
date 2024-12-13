from channel_param import *
from img_processing import *
import torch.optim as optim

BS_POS = (45, 0)

def estimate_pos(AoA_deg, path_delay, tx_pos=(0, 0)):
    distance = (C * path_delay) / 2

    AoA_rad = np.radians(AoA_deg)

    x = tx_pos[0] + distance * np.cos(AoA_rad)
    x = float(x[0])
    y = tx_pos[1] + distance * np.sin(AoA_rad)
    y = float(y[0])
    return (x, y)


def estimate_vec(doppler_shifts, aoa_degs):
    """
    Estimate user velocity in 2D using Doppler shifts and AoAs

    returns estimated velocity vector (vx, vy) in m/s
    """
    aoa_rads = np.radians(aoa_degs)

    # v * cos(theta) = fd * c / fc
    # this can be solved using LS
    A = np.vstack((np.cos(aoa_rads), np.sin(aoa_rads))).T
    b = np.array(doppler_shifts) * C / CR_FREQ

    # LS
    est_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return est_vec


def train_model(model:UserTrackingModel, dataloader:DataLoader, 
                n_epoch=10, lr=0.001):
    """
    Start train process
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # print(model.path_gain_fc.weight.dtype)

    # train loop
    for epoch in range(1):
        model.train()
        running_loss = 0

        for batch in dataloader:
            img_data, dm_data, _wireless_data, _gain, coord = batch

            # zero gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(_wireless_data, _gain, img_data)

            print(outputs.shape)

"""
1. AoA and path delay to get initial estimate of user's position
2. Doppler shift to estimate user's velocity
3. Depth map to verify user's location within 3D env
4. Use visual data to update user position based on env features
"""
def foo():
    df = process_mat("colo_direct_wireless_dataset")
    idx = 0
    loc = df.iloc[idx]["loc"]
    chnl = df.iloc[idx]["channel"]

    idx2 = 1
    loc2 = df.iloc[idx2]["loc"]
    chnl2 = df.iloc[idx2]["channel"]

    # AoA estimation
    spectrums, angles = music_algorithm(chnl, n_source=N_SRC, wavelength=WVLEN)
    spec_idx = np.argmax(spectrums)
    est_AoA = angles[spec_idx]

    spectrums2, angles2 = music_algorithm(chnl2, n_source=N_SRC, wavelength=WVLEN)
    spec_idx2 = np.argmax(spectrums2)
    est_AoA2 = angles2[spec_idx2]

    # path delay
    path_delay = get_path_delay(chnl)[0]
    path_delay2 = get_path_delay(chnl2)[0]

    print(loc)
    print(estimate_pos(est_AoA, path_delay, BS_POS))
    print(estimate_pos(est_AoA2, path_delay2, BS_POS))

    # requires sequences of H
    locs = df.iloc[:idx]["loc"]

    Hs = [df.iloc[i]["channel"].T for i in range(3)]
    Hs = np.stack(Hs, axis=0) # (#time, channel.shape)

    dp_spectrum = get_doppler(Hs) # (idx, `N_CR`, `N_ANN`)
    dp = np.median(dp_spectrum[:, 0])

    Hs2 = [df.iloc[i]["channel"].T for i in range(1, 4)]
    Hs2 = np.stack(Hs2, axis=0) # (#time, channel.shape)

    dp_spectrum2 = get_doppler(Hs2) # (idx, `N_CR`, `N_ANN`)
    dp2 = np.median(dp_spectrum2[:, 0])

    print(estimate_vec([dp, dp2], [est_AoA, est_AoA2]))


def main():
    ds = UserTrackingDataset()
    dataloader = DataLoader(ds, batch_size=32, shuffle=False)
    model = UserTrackingModel()
    
    train_model(model, dataloader)


if __name__ == '__main__':
    main()