# sem_display.py
import pygame
import cv2
import numpy as np
from sem_detect import SemanticDetector
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
        Raw feed is shown in Pygame window.
        """
        semantic_image = self.semantic_sensor.semantic_image
        bbox_image, counts = None, {}

        if semantic_image is not None:
            # Run detection
            bbox_image, counts = self.detector.detect_and_draw(semantic_image.copy())

            # Show bbox feed in OpenCV window
            cv2.imshow("Bounding Boxes", bbox_image)
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
        return running, counts

    def close(self):
        """
        Cleanly close OpenCV windows
        """
        cv2.destroyAllWindows()
        pygame.quit()
