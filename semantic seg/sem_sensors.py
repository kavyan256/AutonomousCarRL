import numpy as np
import carla

class SensorHandler:
    def __init__(self):
        self.semantic_image = None

    def on_semantic_image(self, image: carla.Image):
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
            arr = arr[:, :, ::-1]
            self.semantic_image = arr.copy()
        except Exception as e:
            print(f"⚠️ Sensor error: {e}")
            self.semantic_image = None
