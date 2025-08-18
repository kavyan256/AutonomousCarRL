import pygame
import carla

class ControlManager:
    def __init__(self):
        self.reverse = False

    def process_keyboard(self, vehicle):
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()
        control.throttle = 0.6 if keys[pygame.K_w] else 0.0
        if keys[pygame.K_s]:
            if self.reverse: control.throttle = 0.2
            else: control.brake = 0.6
        control.steer = -0.5 if keys[pygame.K_a] else 0.5 if keys[pygame.K_d] else 0.0
        control.hand_brake = keys[pygame.K_SPACE]
        control.reverse = self.reverse
        vehicle.apply_control(control)

    def handle_events(self, client):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return False
                if event.key == pygame.K_r:
                    self.reverse = not self.reverse
                    print(f"Reverse: {'ON' if self.reverse else 'OFF'}")
        return True
