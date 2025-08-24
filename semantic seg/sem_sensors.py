import numpy as np
import carla

class SensorHandler:
    def __init__(self):
        self.semantic_image = None
        self.imu_data = None  # Store latest IMU data

    def on_semantic_image(self, image: carla.Image):
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
            self.semantic_image = arr.copy()
        except Exception as e:
            print(f"⚠️ Sensor error: {e}")
            self.semantic_image = None

    def on_imu(self, imu: carla.IMUMeasurement):
        try:
            self.imu_data = {
                "accelerometer": np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]),
                "gyroscope": np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]),
                "compass": imu.compass
            }
        except Exception as e:
            print(f"⚠️ IMU sensor error: {e}")
            self.imu_data = None
