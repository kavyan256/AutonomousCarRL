import argparse
import sys
import signal
import time
import cv2
import numpy as np
import pygame

from sem_connection import ConnectionManager
from sem_spawn import SpawnManager
from sem_sensors import SensorHandler
from sem_control import ControlManager
from sem_display import DisplayManager
from sem_cleanup import CleanupManager
from sem_detect import SemanticDetector
from sem_record import DatasetRecorder  # optional for recording

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--vehicles", type=int, default=50)
    parser.add_argument("--record", action="store_true", help="Enable dataset recording")
    args = parser.parse_args()

    print("[INFO] Connecting to CARLA...")
    conn = ConnectionManager(args.host, args.port)
    if not conn.connect():
        print("❌ Could not connect")
        return
    print("[INFO] Connected to CARLA")

    spawner = SpawnManager(conn.world)
    sensors = SensorHandler()
    controls = ControlManager()
    display = DisplayManager()

    def cleanup_all():
        cleaner = CleanupManager(
            conn.world,
            vehicle=spawner.vehicle,
            semantic_camera=spawner.semantic_camera,
            vehicles=spawner.vehicles,
            sensors=[sensors.semantic_image],
            original_settings=conn.original_settings
        )
        cleaner.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda s, f: cleanup_all())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_all())

    for ep in range(args.episodes):
        print(f"[INFO] Starting episode {ep+1}/{args.episodes}")

        if not spawner.spawn_vehicle_and_camera(npc_vehicles=args.vehicles):
            print("❌ Vehicle or NPC spawn failed")
            break
        print("✅ Vehicle and NPCs spawned")

    spawner.semantic_camera = spawner.setup_semantic_camera(spawner.vehicle, sensors.on_semantic_image)
    semantic_cam = spawner.semantic_camera  # keep reference
    print("📷 Semantic camera ready")

    detector = SemanticDetector(conn.world, spawner.vehicle, semantic_cam)

    recorder = DatasetRecorder(folder="sem_dataset", img_height=600, img_width=800) \
                   if args.record else None

    running = True
    while running:
        conn.world.tick()

        # Controls
        running = controls.handle_events(spawner)
        controls.process_keyboard(spawner.vehicle)

        # Spectator update
        display.update_spectator(spawner.vehicle, conn.spectator)

        # Semantic detection
        semantic_img = sensors.semantic_image
        bbox_img = None
        if semantic_img is not None:
            bbox_img, counts = detector.detect_and_draw(semantic_img.copy())

            # Record if enabled
            if recorder:
                vel = spawner.vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y, vel.z])
                ctrl = spawner.vehicle.get_control()
                rec_img = cv2.resize(semantic_img, (800, 600)) \
                            if semantic_img.shape[0:2] != (600, 800) else semantic_img
                recorder.record(rec_img, speed, ctrl.steer, ctrl.throttle, ctrl.brake)

        # Draw
        running = display.draw(semantic_img, bbox_img)

    if recorder:
        recorder.close()

    # Cleanup after episode
    cleaner = CleanupManager(
        conn.world,
        vehicle=spawner.vehicle,
        semantic_camera=semantic_cam,
        vehicles=spawner.vehicles,
        sensors=[sensors.semantic_image],
        original_settings=conn.original_settings
    )
    cleaner.cleanup()
    time.sleep(1)

pygame.quit()


if __name__ == "__main__":
    main()
