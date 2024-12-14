from channel_param import *
from img_processing import *
import torch.optim as optim


def train_model(model:UserTrackingModel, dataloader:DataLoader, 
                n_epoch=10, lr=0.001, save_dir="model/"):
    """
    Start train process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # print(model.path_gain_fc.weight.dtype)
    best_loss = 1e4
    # train loop
    for epoch in range(n_epoch):
        model.train()
        running_loss = 0
        print(f"Start episode {epoch + 1}")

        for batch in dataloader:
            img_data, dm_data, _wireless_data, _gain, coord = batch # [(B, 3, 64, 64), (B, 3, 64, 64), (B, 4), (B, 128), (B, 2)]

            # forward pass
            _wireless_data = _wireless_data.to(device)
            _gain = _gain.to(device)
            img_data = img_data.to(device)
            dm_data = dm_data.to(device)
            # print(_wireless_data.is_cuda, _gain.is_cuda, img_data.is_cuda, dm_data.is_cuda, next(model.parameters()).is_cuda)
            outputs = model(_wireless_data, _gain, img_data, dm_data) # (B, 4)

            # compute loss
            gt_channel = get_ground_truth_channel(coord) # (B, 2)
            az, delay = gt_channel
            az = torch.unsqueeze(az, 1)
            delay = torch.unsqueeze(delay, 1)
            gt_channel = torch.cat((az, delay), dim=1)
            real_output = torch.cat((coord, gt_channel), dim=1)

            real_output = real_output.to(device=device)
            loss = criterion(outputs, real_output)

            # zero gradient
            optimizer.zero_grad()
            # backward prop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss/len(dataloader)
        print(f"Epoch [{epoch + 1}/{n_epoch}] Loss: {avg_loss:.4f}")
        # save model if better loss
        # can be load with `model.load_state_dict(torch.load)`
        if (avg_loss < best_loss):
            best_loss = avg_loss
            save_path = os.path.join(save_dir, f"{avg_loss}.pt")
            torch.save(model.state_dict(), save_path)

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
    
    train_model(model, dataloader, n_epoch=100)


if __name__ == '__main__':
    main()