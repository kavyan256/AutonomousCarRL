import argparse, signal, sys, time
import cv2
import numpy as np
from sem_connection import ConnectionManager
from sem_spawn import SpawnManager
from sem_sensors import SensorHandler
from sem_control import ControlManager
from sem_display import DisplayManager
from sem_cleanup import CleanupManager
from sem_detect import SemanticDetector
from sem_record import DatasetRecorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--vehicles", type=int, default=50)
    parser.add_argument("--record", action="store_true", help="Enable dataset recording")
    args = parser.parse_args()

    conn = ConnectionManager(args.host, args.port)
    if not conn.connect():
        print("❌ Could not connect")
        return

    spawner = SpawnManager(conn.world)
    sensors = SensorHandler()
    controls = ControlManager()
    display = DisplayManager()

    def cleanup_all():
        cleaner = CleanupManager(conn.world, spawner.vehicle,
                                 spawner.semantic_camera,
                                 conn.original_settings)
        cleaner.cleanup(); sys.exit(0)

    signal.signal(signal.SIGINT, lambda s,f: cleanup_all())
    signal.signal(signal.SIGTERM, lambda s,f: cleanup_all())

    for ep in range(args.episodes):
        if not spawner.spawn_vehicle_and_camera(args.vehicles):
            break
        spawner.setup_semantic_camera(spawner.vehicle, sensors.on_semantic_image)

        # ✅ Keep a reference so GC doesn’t kill the camera
        semantic_cam = spawner.semantic_camera

        # ✅ Create detector AFTER vehicle exists
        detector = SemanticDetector(conn.world, spawner.vehicle)

        def cleanup_all():
            cleaner = CleanupManager(conn.world, spawner.vehicle,
                                     spawner.semantic_camera,
                                     conn.original_settings)
            cleaner.cleanup(); sys.exit(0)

        signal.signal(signal.SIGINT, lambda s,f: cleanup_all())
        signal.signal(signal.SIGTERM, lambda s,f: cleanup_all())

        for ep in range(args.episodes):
            if not spawner.spawn_vehicle_and_camera(args.vehicles): break
            spawner.setup_semantic_camera(spawner.vehicle, sensors.on_semantic_image)

            # Recorder setup (only if recording enabled)
            recorder = None
            if args.record:
                recorder = DatasetRecorder(folder="sem_dataset", img_height=600, img_width=800)

            running = True
            while running:
                conn.world.tick()
                running = controls.handle_events(spawner)
                controls.process_keyboard(spawner.vehicle)
                display.update_spectator(spawner.vehicle, conn.spectator)
                semantic_img = sensors.semantic_image
                bbox_img = None
                car_count = None
                if semantic_img is not None:
                    bbox_img, car_count = detector.detect_and_draw(semantic_img.copy())
                    # Record data only if enabled
                    if recorder is not None:
                        velocity = spawner.vehicle.get_velocity()
                        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                        control = spawner.vehicle.get_control()
                        rec_img = cv2.resize(semantic_img, (800, 600)) if semantic_img.shape[0:2] != (600, 800) else semantic_img
                        recorder.record(rec_img, speed, control.steer, control.throttle, control.brake)
                display.draw(semantic_img)
                if bbox_img is not None and isinstance(bbox_img, np.ndarray):
                    cv2.imshow("Bounding Box Feed", bbox_img)
                else:
                    print("Warning: bbox_img is not a valid image array")
                    cv2.waitKey(1)

            if recorder is not None:
                recorder.close()
            cleaner = CleanupManager(conn.world, spawner.vehicle,
                                     spawner.semantic_camera,
                                     conn.original_settings)
            cleaner.cleanup()
            time.sleep(1)
