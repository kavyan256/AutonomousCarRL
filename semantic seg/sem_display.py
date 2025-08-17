import pygame
import carla

class DisplayManager:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        # Add a second window for bounding box feed
        self.bbox_display = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        pygame.display.set_caption("CARLA Semantic View")
        self.bbox_caption = None

    def update_spectator(self, vehicle, spectator):
        vt = vehicle.get_transform()
        fwd = vt.get_forward_vector()
        sp_tf = carla.Transform(
            carla.Location(x=vt.location.x - 8*fwd.x,
                           y=vt.location.y - 8*fwd.y,
                           z=vt.location.z + 4),
            carla.Rotation(pitch=-15, yaw=vt.rotation.yaw))
        spectator.set_transform(sp_tf)

    def draw(self, semantic_image):
        self.display.fill((0,0,0))
        if semantic_image is not None:
            surf = pygame.surfarray.make_surface(semantic_image.swapaxes(0,1))
            self.display.blit(surf,(0,0))
        pygame.display.flip()
        self.clock.tick(60)

    def draw_bbox_feed(self, bbox_image):
        # Show bounding box feed in a separate window
        if bbox_image is not None:
            surf = pygame.surfarray.make_surface(bbox_image.swapaxes(0,1))
            self.bbox_display.blit(surf, (0,0))
        if not self.bbox_caption:
            self.bbox_caption = pygame.font.Font(None, 36).render("Bounding Box Feed", True, (255,255,255))
        self.bbox_display.blit(self.bbox_caption, (10, 10))
        pygame.display.update()
        self.clock.tick(60)
