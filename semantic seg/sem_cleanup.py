import time
import pygame

class CleanupManager:
    def __init__(self, world, vehicle=None, semantic_camera=None, original_settings=None):
        self.world = world
        self.vehicle = vehicle
        self.semantic_camera = semantic_camera
        self.original_settings = original_settings

    def clean_environment(self):
        print("🔄 Cleaning environment")
        for filt in ["sensor.*", "vehicle.*", "walker.*"]:
            for actor in self.world.get_actors().filter(filt):
                try: actor.destroy()
                except: pass
        st = self.world.get_settings()
        st.synchronous_mode = False
        self.world.apply_settings(st)
        time.sleep(0.1)
        st.synchronous_mode = True
        st.fixed_delta_seconds = 0.05
        self.world.apply_settings(st)
        print("✅ Cleaned")

    def cleanup(self):
        print("🧹 Cleanup")
        if self.semantic_camera:
            try: self.semantic_camera.stop(); self.semantic_camera.destroy()
            except: pass
        if self.vehicle:
            try: self.vehicle.destroy()
            except: pass
        if self.original_settings:
            try: self.world.apply_settings(self.original_settings)
            except: pass
        pygame.quit()
        print("✅ Done")
