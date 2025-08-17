import numpy as np
import math
import carla

class SemanticAwareness:
    def __init__(self, world, ego_vehicle):
        """
        world: carla.World handle (to query actors)
        ego_vehicle: carla.Actor of the ego vehicle
        """
        self.world = world
        self.ego_vehicle = ego_vehicle

    def update_ego(self, ego_vehicle):
        """Update ego vehicle handle (if respawned)."""
        self.ego_vehicle = ego_vehicle

    def analyze_vehicle(self, bbox_center, vehicle_actor):
        """
        Uses CARLA ground-truth data for detected vehicle.
        bbox_center: (cx, cy) image center of bounding box (for drawing)
        vehicle_actor: carla.Vehicle actor corresponding to detection
        """
        # Ego state
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_vel = self.ego_vehicle.get_velocity()
        ego_speed = np.linalg.norm([ego_vel.x, ego_vel.y, ego_vel.z])

        # Vehicle state
        veh_tf = vehicle_actor.get_transform()
        veh_loc = veh_tf.location
        veh_yaw = math.radians(veh_tf.rotation.yaw)
        veh_vel = vehicle_actor.get_velocity()
        veh_speed = np.linalg.norm([veh_vel.x, veh_vel.y, veh_vel.z])

        # --- Relative distance ---
        dx = veh_loc.x - ego_loc.x
        dy = veh_loc.y - ego_loc.y
        rel_distance = math.sqrt(dx**2 + dy**2)

        # --- Relative bearing ---
        bearing_global = math.atan2(dy, dx)
        rel_bearing = self._wrap_angle(bearing_global - ego_yaw)

        # --- Relative speed ---
        rel_speed = veh_speed - ego_speed

        return {
            "distance": rel_distance,
            "bearing": rel_bearing,
            "heading": veh_yaw,
            "relative_speed": rel_speed,
            "bbox_center": bbox_center
        }

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
