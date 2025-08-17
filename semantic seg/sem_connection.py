import carla
import time

class ConnectionManager:
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self.spectator = None
        self.original_settings = None

    def connect(self):
        try:
            print(f"Connecting to {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.load_world("Town04")

            # Backup and set synchronous settings
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            settings.no_rendering_mode = False
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)

            self.spectator = self.world.get_spectator()
            print("✅ Connected and configured")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
