import cv2
import numpy as np
from sem_awareness import SemanticAwareness, rotation_to_matrix
from scipy.spatial import cKDTree
import carla
import math

class SemanticDetector:
    def __init__(self, world, ego_vehicle, camera_sensor, image_size=(800,600), fov=90.0, match_threshold_px=80):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.camera = camera_sensor
        self.img_w, self.img_h = image_size
        self.fov = fov
        self.match_threshold_px = match_threshold_px
        self.awareness = SemanticAwareness(world, ego_vehicle)

        fov_rad = math.radians(fov)
        self.fx = self.img_w / (2 * math.tan(fov_rad / 2))
        self.fy = self.fx
        self.cx = self.img_w / 2
        self.cy = self.img_h / 2

        self.classes = {
            "Car": np.array([0,0,142], dtype=np.uint8),
            "Truck": np.array([0,0,70], dtype=np.uint8),
            "Bus": np.array([0,60,100], dtype=np.uint8),
            "Motorcycle": np.array([0,0,230], dtype=np.uint8)
        }

    def world_to_camera(self, world_point, cam_tf):
        wp = np.array([world_point.x, world_point.y, world_point.z], dtype=float)
        cam_loc = cam_tf.location
        cam_pos = np.array([cam_loc.x, cam_loc.y, cam_loc.z], dtype=float)
        R = rotation_to_matrix(cam_tf.rotation).T
        t = -R @ cam_pos
        return R @ wp + t

    def camera_to_image(self, cam_coords):
        x, y, z = cam_coords
        if z <= 0.001: return None
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy
        return int(u), int(v), float(z)

    def _project_actor(self, actor):
        try:
            bb = actor.bounding_box
            bb_loc = actor.get_transform().transform(bb.location)
            img_pt = self.camera_to_image(self.world_to_camera(bb_loc, self.camera.get_transform()))
            if img_pt is None: return None
            u, v, z = img_pt
            if u<0 or u>=self.img_w or v<0 or v>=self.img_h: return None
            return u,v,z
        except:
            return None

    def detect_and_draw(self, frame):
        if frame.shape[2]==4: frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame_out = frame.copy()
        counts = {}

        actors = self.world.get_actors().filter("vehicle.*")
        actor_proj_points = []
        actor_list = []

        for actor in actors:
            proj = self._project_actor(actor)
            if proj:
                u,v,_ = proj
                actor_proj_points.append((u,v))
                actor_list.append(actor)

        kdtree = cKDTree(np.array(actor_proj_points)) if actor_proj_points else None
        matches = {}

        for cls_name, color in self.classes.items():
            mask = cv2.inRange(frame, color, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            counts[cls_name]=0

            for cnt in contours:
                x,y,w,h=cv2.boundingRect(cnt)
                if w<8 or h<8: continue
                cx,cy = x+w//2, y+h//2
                counts[cls_name]+=1

                box_color = (0,255,0) if cls_name=="Car" else \
                            (255,0,0) if cls_name=="Truck" else \
                            (0,255,255) if cls_name=="Bus" else (255,0,255)

                matched_actor=None
                match_dist=None
                if kdtree:
                    dists, idxs = kdtree.query([(cx,cy)], k=3, distance_upper_bound=self.match_threshold_px)
                    dists, idxs = dists[0], idxs[0]
                    for dist, idx in zip(dists, idxs):
                        if idx>=len(actor_list) or np.isinf(dist): continue
                        matched_actor=actor_list[idx]
                        match_dist=dist
                        break

                info={}
                if matched_actor:
                    info=self.awareness.analyze_vehicle(matched_actor)
                    info["bbox_center"]=(cx,cy)
                    info["detection_class"]=cls_name
                    info["match_dist_px"]=match_dist
                    cv2.rectangle(frame_out,(x,y),(x+w,y+h),box_color,2)
                    txt1=f"{cls_name} id:{matched_actor.id} d:{info['distance']:.1f}m v:{info['veh_speed']*3.6:.0f}km/h"
                    cv2.putText(frame_out,txt1,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,box_color,1)
                else:
                    cv2.rectangle(frame_out,(x,y),(x+w,y+h),box_color,1)
                    cv2.putText(frame_out,cls_name,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)

        return cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR), counts
