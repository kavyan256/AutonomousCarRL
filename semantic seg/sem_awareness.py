import numpy as np
import math
import carla

def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_to_matrix(rotation: carla.Rotation):
    """Return 3x3 rotation matrix from CARLA Rotation (degrees)."""
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    Ry = np.array([
        [math.cos(roll), 0, math.sin(roll)],
        [0, 1, 0],
        [-math.sin(roll), 0, math.cos(roll)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

class SemanticAwareness:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle

    def update_ego(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle

    def analyze_vehicle(self, vehicle_actor):
        """Return detailed info about a vehicle relative to ego."""
        # Vehicle
        veh_tf = vehicle_actor.get_transform()
        veh_loc = veh_tf.location
        veh_yaw = math.radians(veh_tf.rotation.yaw)
        veh_vel = vehicle_actor.get_velocity()
        # Ensure velocity is valid
        veh_vel_vec = np.array([veh_vel.x if veh_vel else 0.0,
                                veh_vel.y if veh_vel else 0.0,
                                veh_vel.z if veh_vel else 0.0], dtype=float)
        veh_speed = float(np.linalg.norm(veh_vel_vec))

        # Ego
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_vel = self.ego_vehicle.get_velocity()
        ego_vel_vec = np.array([ego_vel.x if ego_vel else 0.0,
                                ego_vel.y if ego_vel else 0.0,
                                ego_vel.z if ego_vel else 0.0], dtype=float)
        ego_speed = float(np.linalg.norm(ego_vel_vec))

        dx, dy = veh_loc.x - ego_loc.x, veh_loc.y - ego_loc.y
        rel_distance = math.hypot(dx, dy)

        bearing_global = math.atan2(dy, dx)
        rel_bearing = _wrap_angle(bearing_global - ego_yaw)
        heading_rel = _wrap_angle(veh_yaw - ego_yaw)

        rel_vel_vec = veh_vel_vec - ego_vel_vec
        los_xy = np.array([dx, dy, 0.0]) / max(rel_distance, 1e-6)
        rel_speed_along_los = float(np.dot(rel_vel_vec, los_xy))
        closing_speed = -rel_speed_along_los
        ttc = rel_distance / closing_speed if closing_speed > 0.1 else float('inf')

        actor_type = getattr(vehicle_actor, 'type_id', str(type(vehicle_actor)))
        lane_id = None
        try:
            waypoint = self.world.get_map().get_waypoint(veh_loc)
            lane_id = getattr(waypoint, 'lane_id', None)
        except:
            pass

        return {
            "distance": float(rel_distance),
            "bearing": float(rel_bearing),
            "heading": float(veh_yaw),
            "heading_rel": float(heading_rel),
            "relative_speed_along_los": float(rel_speed_along_los),
            "closing_speed": float(closing_speed),
            "ego_speed": float(ego_speed),
            "veh_speed": float(veh_speed),
            "speed_kmh": veh_speed * 3.6,
            "ttc": float(ttc),
            "bbox_center": None,
            "actor_id": vehicle_actor.id,
            "actor_type": actor_type,
            "lane_id": lane_id
        }
