# File: sem_record.py

# Refactored for integration
import h5py
import numpy as np
import os

class DatasetRecorder:
    def __init__(self, folder="sem_dataset", img_height=240, img_width=320):
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "carla_dataset.hdf5")
        self.h5_file = h5py.File(filename, "w")
        self.img_height = img_height
        self.img_width = img_width
        self.frame_count = 0

        self.rgb_ds = self.h5_file.create_dataset("observations/rgb",
            (0, img_height, img_width, 3), maxshape=(None, img_height, img_width, 3), dtype=np.uint8)
        self.speed_ds = self.h5_file.create_dataset("observations/speed",
            (0, 1), maxshape=(None, 1), dtype=np.float32)
        self.steer_ds = self.h5_file.create_dataset("actions/steer",
            (0, 1), maxshape=(None, 1), dtype=np.float32)
        self.throttle_ds = self.h5_file.create_dataset("actions/throttle",
            (0, 1), maxshape=(None, 1), dtype=np.float32)
        self.brake_ds = self.h5_file.create_dataset("actions/brake",
            (0, 1), maxshape=(None, 1), dtype=np.float32)

    def record(self, img, speed, steer, throttle, brake):
        # Resize datasets dynamically
        for ds in [self.rgb_ds, self.speed_ds, self.steer_ds, self.throttle_ds, self.brake_ds]:
            ds.resize((self.frame_count + 1, *ds.shape[1:]))
        self.rgb_ds[self.frame_count] = img
        self.speed_ds[self.frame_count] = [speed]
        self.steer_ds[self.frame_count] = [steer]
        self.throttle_ds[self.frame_count] = [throttle]
        self.brake_ds[self.frame_count] = [brake]
        self.frame_count += 1
        print(f"Recorded frame {self.frame_count}")

    def close(self):
        self.h5_file.close()
