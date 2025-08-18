import pygame
import carla

class DisplayManager:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((1600, 600))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("CARLA Semantic + BBox View")
        self.font = pygame.font.Font(None, 36)

    def update_spectator(self, vehicle, spectator):
        vt = vehicle.get_transform()
        fwd = vt.get_forward_vector()
        sp_tf = carla.Transform(
            carla.Location(x=vt.location.x - 8*fwd.x,
                           y=vt.location.y - 8*fwd.y,
                           z=vt.location.z + 4),
            carla.Rotation(pitch=-15, yaw=vt.rotation.yaw))
        spectator.set_transform(sp_tf)

    def draw(self, semantic_image, bbox_image=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        self.display.fill((0, 0, 0))
        if semantic_image is not None:
            surf = pygame.surfarray.make_surface(semantic_image.swapaxes(0, 1))
            self.display.blit(surf, (0, 0))
        if bbox_image is not None:
            surf2 = pygame.surfarray.make_surface(bbox_image.swapaxes(0, 1))
            self.display.blit(surf2, (800, 0))
        pygame.display.flip()
        self.clock.tick(60)
        return True
