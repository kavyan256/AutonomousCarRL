# sem_display.py
import pygame
import cv2
import numpy as np
from sem_detect import SemanticDetector
from sem_track import Sort
import carla

class DisplayManager:
    def __init__(self, world, vehicle, sensors, img_size=(800,600)):
        pygame.init()
        self.display = pygame.display.set_mode(img_size)
        pygame.display.set_caption("CARLA Semantic Feed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.world = world
        self.vehicle = vehicle
        self.semantic_sensor = sensors
        self.img_w, self.img_h = img_size

        # Initialize SemanticDetector
        self.detector = SemanticDetector(world, vehicle, None, image_size=img_size)

        # Initialize SORT tracker
        self.tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

        # OpenCV window for bounding box visualization
        cv2.namedWindow("Bounding Boxes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bounding Boxes", self.img_w, self.img_h)

    def update_spectator(self, vehicle, spectator):
        vt = vehicle.get_transform()
        fwd = vt.get_forward_vector()
        sp_tf = carla.Transform(
            carla.Location(x=vt.location.x - 8*fwd.x,
                           y=vt.location.y - 8*fwd.y,
                           z=vt.location.z + 4),
            carla.Rotation(pitch=-15, yaw=vt.rotation.yaw))
        spectator.set_transform(sp_tf)

    def draw_pygame_feed(self, semantic_image):
        """
        Display the raw semantic feed in a Pygame window
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.display.fill((0, 0, 0))
        if semantic_image is not None:
            surf = pygame.surfarray.make_surface(semantic_image.swapaxes(0, 1))
            self.display.blit(surf, (0, 0))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def draw_with_detection(self, recorder=None):
        """
        Process semantic image, draw bounding boxes in OpenCV window.
        Tracks vehicles across frames with SORT.
        """
        semantic_image = self.semantic_sensor.semantic_image
        bbox_image = None
        detections = []

        if semantic_image is not None:
            # Prepare detections for SORT
            for cls_name, color in self.detector.classes.items():
                mask = cv2.inRange(semantic_image, color, color)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
                contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if w < 8 or h < 8: continue
                    detections.append([x, y, x+w, y+h, cls_name])

            # Update tracker
            tracked_objects = self.tracker.update(detections)

            # Draw tracked boxes
            bbox_image = semantic_image.copy()
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id, cls_name = obj
                color = (0,255,0) if cls_name=="Car" else \
                        (255,0,0) if cls_name=="Truck" else \
                        (0,255,255) if cls_name=="Bus" else (255,0,255)
                cv2.rectangle(bbox_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(bbox_image, f"{cls_name} id:{obj_id}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Show bounding box feed in OpenCV window
            cv2.imshow("Bounding Boxes", cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            # Record if enabled
            if recorder:
                vel = self.vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y, vel.z])
                ctrl = self.vehicle.get_control()
                rec_img = cv2.resize(semantic_image, (800, 600)) \
                    if semantic_image.shape[0:2] != (600, 800) else semantic_image
                recorder.record(rec_img, speed, ctrl.steer, ctrl.throttle, ctrl.brake)

        # Show semantic feed in Pygame
        running = self.draw_pygame_feed(semantic_image)
        return running, tracked_objects if semantic_image is not None else []

    def close(self):
        """
        Cleanly close OpenCV windows
        """
        cv2.destroyAllWindows()
        pygame.quit()
