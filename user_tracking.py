from channel_param import *
from img_processing import *
import torch.optim as optim



def get_ground_truth_channel(UE_pos, BS_pos=BS_POS):
    """
    Calculate channel states (AoA, path delay) based on ground turth UE position

    returns:
    AoA, path delay
    """
    # extract coord
    x_ue, y_ue = UE_pos
    x_bs, y_bs = BS_pos
    dx = x_ue - x_bs
    dy = y_ue - y_bs

    # path delay
    dist = np.sqrt(dx ** 2 + dy ** 2)
    path_delay = dist / C

    # Azimuth angle
    azimuth = np.arctan2(dy, dx)

    return azimuth, path_delay


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
            wireless_data = (_wireless_data, _gain)
            # zero gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(wireless_data, img_data, dm_data) # (B, 4)

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
    # print(estimate_pos(est_AoA, path_delay, BS_POS))
    # print(estimate_pos(est_AoA2, path_delay2, BS_POS))

    # # requires sequences of H
    # locs = df.iloc[:idx]["loc"]

    # Hs = [df.iloc[i]["channel"].T for i in range(3)]
    # Hs = np.stack(Hs, axis=0) # (#time, channel.shape)

    # dp_spectrum = get_doppler(Hs) # (idx, `N_CR`, `N_ANN`)
    # dp = np.median(dp_spectrum[:, 0])

    # Hs2 = [df.iloc[i]["channel"].T for i in range(1, 4)]
    # Hs2 = np.stack(Hs2, axis=0) # (#time, channel.shape)

    # dp_spectrum2 = get_doppler(Hs2) # (idx, `N_CR`, `N_ANN`)
    # dp2 = np.median(dp_spectrum2[:, 0])

    # print(estimate_vec([dp, dp2], [est_AoA, est_AoA2]))


def main():
    ds = UserTrackingDataset()
    dataloader = DataLoader(ds, batch_size=32, shuffle=False)
    model = UserTrackingModel()
    
    train_model(model, dataloader)


if __name__ == '__main__':
    main()
