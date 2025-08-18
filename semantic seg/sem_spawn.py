import random
import carla

class SpawnManager:
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.vehicles = []
        self.semantic_camera = None

    def spawn_vehicle_and_camera(self, npc_vehicles=50):
        try:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if not self.vehicle: return False
            self.vehicle.set_autopilot(False)

            spawned = 0
            for sp in spawn_points:
                if spawned >= npc_vehicles: break
                npc_bp = random.choice(bp_lib.filter("vehicle.*"))
                npc = self.world.try_spawn_actor(npc_bp, sp)
                if npc:
                    npc.set_autopilot(True)
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
