# sem_main.py
import argparse
import sys
import signal
import time
import pygame

from sem_connection import ConnectionManager
from sem_spawn import SpawnManager
from sem_sensors import SensorHandler
from sem_control import ControlManager
from sem_display import DisplayManager
from sem_cleanup import CleanupManager
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

    spawner = SpawnManager(conn.world, conn.client, max_npc_speed=30.0)
    sensors = SensorHandler()
    controls = ControlManager()
    display = DisplayManager(conn.world, spawner.vehicle, sensors)

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
    print("📷 Semantic camera ready")

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

        # Draw semantic + bounding boxes (detection handled internally)
        running, bbox_counts = display.draw_with_detection(recorder)

    if recorder:
        recorder.close()

    # Cleanup after episode
    cleaner = CleanupManager(
        conn.world,
        vehicle=spawner.vehicle,
        semantic_camera=spawner.semantic_camera,
        vehicles=spawner.vehicles,
        sensors=[sensors.semantic_image],
        original_settings=conn.original_settings
    )
    cleaner.cleanup()
    time.sleep(1)
    pygame.quit()


if __name__ == "__main__":
    main()
