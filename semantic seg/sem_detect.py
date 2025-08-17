import cv2
import numpy as np
import time
import math
from sem_awareness import SemanticAwareness
from scipy.spatial import cKDTree
import carla

class SemanticDetector:
    def __init__(self, world, ego_vehicle):
        self.classes = {
            "Car": np.array([0, 0, 142], dtype=np.uint8),
            "Truck": np.array([0, 0, 70], dtype=np.uint8),
            "Bus": np.array([0, 60, 100], dtype=np.uint8),
            "Motorcycle": np.array([0, 0, 230], dtype=np.uint8),
        }
        self.awareness = SemanticAwareness(world, ego_vehicle)
        self.world = world
        self.ego_vehicle = ego_vehicle

    def detect_and_draw(self, frame):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        frame_out = frame.copy()
        counts = {}

        # Get all vehicle actors
        actors = self.world.get_actors().filter("vehicle.*")

        for cls_name, color in self.classes.items():
            mask = cv2.inRange(frame, color, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            counts[cls_name] = 0

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 10 or h < 10:
                    continue

                cx, cy = x + w // 2, y + h // 2
                counts[cls_name] += 1

                # Pick bbox color
                if cls_name == "Car": box_color = (0, 255, 0)
                elif cls_name == "Truck": box_color = (255, 0, 0)
                elif cls_name == "Bus": box_color = (0, 255, 255)
                elif cls_name == "Motorcycle": box_color = (255, 0, 255)
                else: box_color = (255, 255, 255)

                # --- Match detection to nearest actor ---
                nearest_actor = None
                min_dist = 9999
                ego_loc = self.ego_vehicle.get_transform().location
                for actor in actors:
                    loc = actor.get_transform().location
                    dist = ego_loc.distance(loc)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_actor = actor

                if nearest_actor:
                    info = self.awareness.analyze_vehicle((cx, cy), nearest_actor)

                    # Draw bbox
                    cv2.rectangle(frame_out, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(frame_out, cls_name, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Overlay awareness info
                    hd = np.degrees(info["heading"])
                    rs = info["relative_speed"]
                    cv2.putText(frame_out,
                                f"d:{info['distance']:.1f} br:{np.degrees(info['bearing']):.1f} hd:{hd:.1f} rs:{rs:.1f}",
                                (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                    # Draw heading arrow
                    arrow_len = max(w, h) // 2
                    arrow_angle = -info["heading"] + np.pi / 2
                    end_x = int(cx + arrow_len * np.cos(arrow_angle))
                    end_y = int(cy + arrow_len * np.sin(arrow_angle))
                    cv2.arrowedLine(frame_out, (cx, cy), (end_x, end_y), box_color, 2, tipLength=0.3)

        frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        summary = " | ".join([f"{k}:{v}" for k, v in counts.items()])
        cv2.putText(frame_bgr, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame_bgr, counts
