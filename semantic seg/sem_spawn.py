import random
import carla

class SpawnManager:
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.semantic_camera = None

    def spawn_vehicle_and_camera(self, npc_vehicles=50):
        try:
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

            if not self.vehicle:
                print("❌ Ego spawn failed")
                return False
            self.vehicle.set_autopilot(False)
            print("✅ Ego vehicle spawned")

            # NPCs
            spawned = 0
            for sp in spawn_points:
                if spawned >= npc_vehicles: break
                npc_bp = random.choice(bp_lib.filter("vehicle.*"))
                npc = self.world.try_spawn_actor(npc_bp, sp)
                if npc:
                    npc.set_autopilot(True)
                    # Reduce NPC speed
                    try:
                        physics_control = npc.get_physics_control()
                        physics_control.max_velocity = 4.0  # meters/second (adjust as needed)
                        npc.apply_physics_control(physics_control)
                    except Exception as e:
                        print(f"Could not set NPC speed: {e}")
                    spawned += 1
            print(f"✅ Spawned {spawned} NPCs")

            return True
        except Exception as e:
            print(f"❌ Spawn error: {e}")
            return False

    def setup_semantic_camera(self, vehicle, callback):
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(carla.Location(x=0, y=0, z=50),
                                 carla.Rotation(pitch=-90))
        self.semantic_camera = self.world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        self.semantic_camera.listen(callback)
        print("📷 Semantic camera ready")
        return self.semantic_camera
