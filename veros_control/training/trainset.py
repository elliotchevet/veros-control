import h5py

def load_snapshot(filename, keys):
    data = {}
    with h5py.File(filename, "r") as f:
        core = f["core"]
        for key in keys:
            data[key] = core[key][:]
    return data


def build_training_windows(files, rollout_length):
    windows = []
    for i in range(len(files) - rollout_length):
        windows.append(
            {
                "initial": files[i],
                "targets": files[i + 1 : i + rollout_length + 1],
            }
        )
    return windows


class RestartDataset:
    def __init__(self, files, rollout_length, keys=("temp", "salt")):
        self.keys = tuple(keys)
        self.windows = build_training_windows(files, rollout_length)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample = self.windows[idx]
        initial = load_snapshot(sample["initial"], self.keys)
        target = load_snapshot(sample["targets"][-1], self.keys)
        return initial, target
