import random
import carla

class SpawnManager:
    def __init__(self, world, client, max_npc_speed=20.0, tm_port=8000):
        """
        world: carla.World object
        client: carla.Client object
        max_npc_speed: Maximum NPC speed in km/h
        tm_port: Traffic Manager port
        """
        self.world = world
        self.client = client
        self.vehicle = None
        self.vehicles = []
        self.semantic_camera = None
        self.max_npc_speed = max_npc_speed

        # Get traffic manager via client
        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_global_distance_to_leading_vehicle(2.5)  # safe distance

    def spawn_vehicle_and_camera(self, npc_vehicles=50):
        try:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

            spawn_points = self.world.get_map().get_spawn_points()

            # Spawn ego vehicle
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if not self.vehicle:
                return False
            self.vehicle.set_autopilot(False)

            spawned = 0
            for sp in spawn_points:
                if spawned >= npc_vehicles:
                    break
                npc_bp = random.choice(bp_lib.filter("vehicle.*"))
                npc = self.world.try_spawn_actor(npc_bp, sp)
                if npc:
                    # Enable autopilot via traffic manager
                    npc.set_autopilot(True, self.tm.get_port())
                    # Limit speed by setting percentage speed difference
                    speed_percentage = max(0, int((self.max_npc_speed / 100) * 100))
                    self.tm.vehicle_percentage_speed_difference(npc, 100 - speed_percentage)
                    self.vehicles.append(npc)
                    spawned += 1
            return True
        except Exception as e:
            print(f"Spawn error: {e}")
            return False

    def setup_semantic_camera(self, vehicle, callback):
        bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", "800")
        bp.set_attribute("image_size_y", "600")
        bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))
        self.semantic_camera = self.world.spawn_actor(bp, cam_tf, attach_to=vehicle)
        self.semantic_camera.listen(callback)
        return self.semantic_camera
