import carla
import time

class ConnectionManager:
    def __init__(self, host="localhost", port=2000, town="Town04"):
        self.host = host
        self.port = port
        self.town = town  # specify the map to load
        self.max_retries = 5
        self.retry_delay = 2.0
        self.client = None
        self.world = None
        self.spectator = None
        self.original_settings = None

    def connect(self) -> bool:
        for attempt in range(1, self.max_retries + 1):
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(10.0)

                # Load specified town
                self.world = self.client.load_world(self.town)

                # Save original settings
                self.original_settings = self.world.get_settings()

                # Apply synchronous settings
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)

                # Get spectator
                self.spectator = self.world.get_spectator()
                return True
            except Exception as e:
                print(f"[ConnectionManager] Attempt {attempt} failed: {e}")
                time.sleep(self.retry_delay)
        return False

    def disconnect(self):
        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except:
                pass
        self.client = None
        self.world = None
        self.spectator = None
