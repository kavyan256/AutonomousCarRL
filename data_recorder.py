#!/usr/bin/env python3
# acreate a vision model, that as soon as it detects a obstacle, it starts a live feed
# the live feed is inserted into a llm which then describes the sceen, model predicts the next move() ex slow down or speed up) if correct llm says good work if incorrect, llm asks user reason for his decision and then updates that to its memory and trains the model

"""
CARLA Behavior Cloning Data Recorder - Phase 2
Phase 1: Connection, vehicle spawn, and 3rd person spectator view
Phase 2: Keyboard control for the vehicle
"""

import carla
import pygame
import time
import argparse
import random
import cv2
import numpy as np
import threading
import queue
from computer_vision import ComputerVisionProcessor

class CARLADataRecorder:
    def __init__(self, host='localhost', port=2000, timeout=5.0):
        """Initialize the CARLA data recorder."""
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.spectator = None
        self.npc_vehicles = []  # List to track spawned NPCs
        
        print("🚀 CARLA Data Recorder - Phase 2")
        print("Phase 1: Connection + Vehicle Spawn + 3rd Person View")
        print("Phase 2: Keyboard Control (WASD)")
        print("=" * 50)
        
        # Initialize pygame for keyboard input
        pygame.init()
        self.display = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("CARLA Vehicle Control - WASD to drive")
        self.clock = pygame.time.Clock()
        
        # Control inputs
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = False
        
        # === CAMERA SYSTEM CONFIGURATION ===
        # Camera enable/disable flags - Set to True/False to control which cameras are active
        self.front_camera_enabled = False     # Front camera (driver's POV)
        self.left_camera_enabled = False     # Left side camera
        self.right_camera_enabled = False     # Right side camera
        self.top_camera_enabled = True      # Top camera (bird's eye view)
        self.rear_camera_enabled = False     # Rear camera (back view)
        
        # Camera objects
        self.front_camera = None
        self.left_camera = None
        self.right_camera = None
        self.top_camera = None
        self.rear_camera = None
        
        # Camera data storage
        self.current_front_image = None
        self.current_left_image = None
        self.current_right_image = None
        self.current_top_image = None
        self.current_rear_image = None
        
        # Camera data queues for real-time processing
        self.front_image_queue = queue.Queue(maxsize=5)
        self.left_image_queue = queue.Queue(maxsize=5)
        self.right_image_queue = queue.Queue(maxsize=5)
        self.top_image_queue = queue.Queue(maxsize=5)
        self.rear_image_queue = queue.Queue(maxsize=5)
        
        # Vision system status
        self.vision_active = False
        
        # Initialize Computer Vision Processor
        self.cv_processor = ComputerVisionProcessor()
        
        print(f"📹 Camera Configuration:")
        print(f"   Front Camera: {'ENABLED' if self.front_camera_enabled else 'DISABLED'}")
        print(f"   Left Camera:  {'ENABLED' if self.left_camera_enabled else 'DISABLED'}")
        print(f"   Right Camera: {'ENABLED' if self.right_camera_enabled else 'DISABLED'}")
        print(f"   Top Camera:   {'ENABLED' if self.top_camera_enabled else 'DISABLED'}")
        print(f"   Rear Camera:  {'ENABLED' if self.rear_camera_enabled else 'DISABLED'}")
        print("=" * 50)
    
    def connect_to_carla(self):
        """Connect to CARLA server with proper error handling."""
        try:
            print(f"🔗 Attempting to connect to CARLA at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)

            try:
                print("🌍 Loading Town04...")
                self.world = self.client.load_world('Town04')
                print("✅ Loaded Town04")
            except Exception as e:
                print(f"⚠️ Failed to load Town04 ({e}), using current world instead")
                self.world = self.client.get_world()

            print(f"📍 Current map: {self.world.get_map().name}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def spawn_vehicle(self):
        """Spawn a vehicle in the world."""
        try:
            print("\n🚗 Spawning vehicle...")
            
            # Get blueprint library
            blueprint_library = self.world.get_blueprint_library()
            print(f"   📚 Available blueprints: {len(blueprint_library)}")
            
            # Choose Tesla Model 3
            vehicle_blueprints = blueprint_library.filter('vehicle.tesla.model3')
            if not vehicle_blueprints:
                print("❌ Tesla Model 3 not found, trying any vehicle...")
                vehicle_blueprints = blueprint_library.filter('vehicle.*')
            
            vehicle_bp = vehicle_blueprints[0]
            print(f"   🏎️ Selected vehicle: {vehicle_bp.id}")
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"   📍 Available spawn points: {len(spawn_points)}")
            
            if not spawn_points:
                print("❌ No spawn points available on this map!")
                return False
            
            # Try to spawn at first available point
            spawn_point = spawn_points[0]
            print(f"   🎯 Spawning at: {spawn_point.location}")
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            print(f"✅ Vehicle spawned successfully!")
            print(f"   🆔 Vehicle ID: {self.vehicle.id}")
            print(f"   📍 Location: {self.vehicle.get_location()}")
            
            # Enable physics for the vehicle
            self.vehicle.set_simulate_physics(True)
            
            # Wait a moment for the vehicle to settle
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to spawn vehicle: {e}")
            return False
    
    def setup_spectator_view(self):
        """Set up 3rd person spectator view behind the vehicle."""
        try:
            print("\n👁️ Setting up 3rd person spectator view...")
            
            # Get the spectator (camera)
            self.spectator = self.world.get_spectator()
            
            # Get vehicle location and rotation
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            print(f"   🚗 Vehicle location: {vehicle_location}")
            print(f"   🧭 Vehicle rotation: {vehicle_rotation}")
            
            # Calculate spectator position (behind and above the vehicle)
            # Move backwards along the vehicle's forward vector
            import math
            
            # Convert rotation to radians
            yaw_rad = math.radians(vehicle_rotation.yaw)
            
            # Calculate offset position (behind the vehicle)
            offset_distance = 8.0  # Distance behind vehicle
            offset_height = 4.0    # Height above vehicle
            
            # Calculate new position
            spectator_x = vehicle_location.x - offset_distance * math.cos(yaw_rad)
            spectator_y = vehicle_location.y - offset_distance * math.sin(yaw_rad)
            spectator_z = vehicle_location.z + offset_height
            
            # Create spectator transform
            spectator_location = carla.Location(x=spectator_x, y=spectator_y, z=spectator_z)
            
            # Point camera towards the vehicle (adjust pitch to look down slightly)
            spectator_rotation = carla.Rotation(
                pitch=-15.0,  # Look down slightly
                yaw=vehicle_rotation.yaw,  # Same yaw as vehicle
                roll=0.0
            )
            
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            
            # Apply the transform to spectator
            self.spectator.set_transform(spectator_transform)
            
            print(f"✅ Spectator view set!")
            print(f"   📍 Spectator location: {spectator_location}")
            print(f"   🎥 Looking at vehicle from behind")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup spectator view: {e}")
            return False
    
    def update_spectator_view(self):
        """Update spectator to follow the vehicle."""
        if not self.vehicle or not self.spectator:
            return
        
        try:
            # Get current vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # Calculate spectator position
            import math
            yaw_rad = math.radians(vehicle_rotation.yaw)
            
            offset_distance = 8.0
            offset_height = 4.0
            
            spectator_x = vehicle_location.x - offset_distance * math.cos(yaw_rad)
            spectator_y = vehicle_location.y - offset_distance * math.sin(yaw_rad)
            spectator_z = vehicle_location.z + offset_height
            
            spectator_location = carla.Location(x=spectator_x, y=spectator_y, z=spectator_z)
            spectator_rotation = carla.Rotation(
                pitch=-15.0,
                yaw=vehicle_rotation.yaw,
                roll=0.0
            )
            
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            self.spectator.set_transform(spectator_transform)
            
        except Exception as e:
            print(f"⚠️ Failed to update spectator view: {e}")
    
    def spawn_npc_vehicles(self, num_vehicles=20):
        """Spawn NPC vehicles around the map."""
        try:
            print(f"\n🚦 Spawning {num_vehicles} NPC vehicles...")
            
            # Get blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Get all vehicle blueprints (excluding bicycles and motorcycles for simplicity)
            vehicle_blueprints = blueprint_library.filter('vehicle.*')
            vehicle_blueprints = [bp for bp in vehicle_blueprints if 
                                'bicycle' not in bp.id.lower() and 
                                'motorcycle' not in bp.id.lower() and
                                'bike' not in bp.id.lower()]
            
            print(f"   🚗 Available vehicle types: {len(vehicle_blueprints)}")
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            print(f"   📍 Available spawn points: {len(spawn_points)}")
            
            # Limit number of vehicles to available spawn points (leave one for player)
            max_vehicles = min(num_vehicles, len(spawn_points) - 1)
            
            # Skip the first spawn point (used for player vehicle)
            available_spawn_points = spawn_points[1:max_vehicles + 1]
            
            # Spawn vehicles
            spawned_count = 0
            for spawn_point in available_spawn_points:
                try:
                    # Choose random vehicle blueprint
                    vehicle_bp = random.choice(vehicle_blueprints)
                    
                    # Set random color if vehicle supports it
                    if vehicle_bp.has_attribute('color'):
                        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                        vehicle_bp.set_attribute('color', color)
                    
                    # Set autopilot attribute
                    if vehicle_bp.has_attribute('driver_id'):
                        driver_id = random.choice(vehicle_bp.get_attribute('driver_id').recommended_values)
                        vehicle_bp.set_attribute('driver_id', driver_id)
                    
                    # Spawn the vehicle
                    npc_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    
                    if npc_vehicle:
                        # Enable autopilot for NPC
                        npc_vehicle.set_autopilot(True)
                        
                        # Add to our tracking list
                        self.npc_vehicles.append(npc_vehicle)
                        spawned_count += 1
                        
                except Exception as spawn_error:
                    # Skip this spawn point if there's an error (probably occupied)
                    print(f"   ⚠️ Failed to spawn at point {spawn_point.location}: {spawn_error}")
                    continue
            
            print(f"✅ Successfully spawned {spawned_count} NPC vehicles!")
            print(f"   🤖 All NPCs have autopilot enabled")
            
            if spawned_count < max_vehicles:
                print(f"   ⚠️ Only spawned {spawned_count}/{max_vehicles} vehicles (some spawn points were occupied)")
            
            return spawned_count > 0
            
        except Exception as e:
            print(f"❌ Failed to spawn NPC vehicles: {e}")
            return False
    
    def clean_environment(self):
        """Clean up the environment by removing trees, foliage, and vegetation."""
        try:
            print(f"\n🌲 Cleaning environment...")
            
            # Get all static actors like trees, foliage, etc.
            vegetation_tags = ['foliage', 'tree', 'grass', 'bush', 'vegetation', 'plant', 'flower']
            
            removed_count = 0
            total_actors = len(self.world.get_actors())
            
            print(f"   🔍 Scanning {total_actors} actors in the world...")
            
            # Debug: Print ALL actor types to understand what's in the world
            actor_types = {}
            vegetation_found = []
            
            for actor in self.world.get_actors():
                actor_type = actor.type_id.lower()
                
                # Count each type of actor
                if actor.type_id in actor_types:
                    actor_types[actor.type_id] += 1
                else:
                    actor_types[actor.type_id] = 1
                
                # Check if actor type contains any vegetation keywords
                if any(tag in actor_type for tag in vegetation_tags):
                    vegetation_found.append(actor.type_id)
                    try:
                        print(f"   🌿 Found vegetation: {actor.type_id} - removing...")
                        actor.destroy()
                        removed_count += 1
                    except Exception as removal_error:
                        print(f"   ⚠️ Could not remove {actor.type_id}: {removal_error}")
                        continue
            
            # Print summary of all actor types found
            print(f"   📊 Actor types in world:")
            for actor_type, count in sorted(actor_types.items()):
                print(f"      {actor_type}: {count}")
            
            print(f"✅ Environment cleanup complete!")
            if removed_count > 0:
                print(f"   🗑️ Removed {removed_count} vegetation objects: {vegetation_found}")
            else:
                print(f"   ℹ️ No vegetation objects found in this world")
                print(f"   🌍 This map appears to be vegetation-free already!")
            print(f"   🌍 {total_actors - removed_count} actors remaining")
            print(f"   📷 Camera views should be optimal!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to clean environment: {e}")
            return False
    
    def setup_cameras(self):
        """Setup multiple cameras with individual toggle control."""
        try:
            print("\n📹 Setting up camera system...")
            
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Set camera attributes for better top view
            camera_bp.set_attribute('image_size_x', '800')  # Increased resolution
            camera_bp.set_attribute('image_size_y', '600')  # Increased resolution
            camera_bp.set_attribute('fov', '120')  # Wider field of view for top camera
            
            enabled_cameras = []
            
            # Front camera (driver's POV)
            if self.front_camera_enabled:
                front_transform = carla.Transform(
                    carla.Location(x=2.5, z=0.7),  # Dashboard position
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                self.front_camera = self.world.spawn_actor(
                    camera_bp, front_transform, attach_to=self.vehicle
                )
                self.front_camera.listen(self._on_front_image)
                enabled_cameras.append("Front")
                print("   📷 Front camera: ENABLED")
            else:
                print("   📷 Front camera: DISABLED")
            
            # Left camera
            if self.left_camera_enabled:
                left_transform = carla.Transform(
                    carla.Location(x=1.5, y=-1.5, z=1.2),  # Left side position
                    carla.Rotation(pitch=0, yaw=-90, roll=0)  # Looking left
                )
                self.left_camera = self.world.spawn_actor(
                    camera_bp, left_transform, attach_to=self.vehicle
                )
                self.left_camera.listen(self._on_left_image)
                enabled_cameras.append("Left")
                print("   📷 Left camera: ENABLED")
            else:
                print("   📷 Left camera: DISABLED")
            
            # Right camera
            if self.right_camera_enabled:
                right_transform = carla.Transform(
                    carla.Location(x=1.5, y=1.5, z=1.2),  # Right side position
                    carla.Rotation(pitch=0, yaw=90, roll=0)  # Looking right
                )
                self.right_camera = self.world.spawn_actor(
                    camera_bp, right_transform, attach_to=self.vehicle
                )
                self.right_camera.listen(self._on_right_image)
                enabled_cameras.append("Right")
                print("   📷 Right camera: ENABLED")
            else:
                print("   📷 Right camera: DISABLED")
            
            # Top camera (bird's eye view)
            if self.top_camera_enabled:
                top_transform = carla.Transform(
                    carla.Location(x=0, z=25),  # Even higher altitude for maximum coverage
                    carla.Rotation(pitch=-90, yaw=0, roll=0)  # Looking down
                )
                self.top_camera = self.world.spawn_actor(
                    camera_bp, top_transform, attach_to=self.vehicle
                )
                self.top_camera.listen(self._on_top_image)
                enabled_cameras.append("Top")
                print("   🛰️ Top camera: ENABLED (Maximum Coverage View)")
            else:
                print("   🛰️ Top camera: DISABLED")
            
            # Rear camera
            if self.rear_camera_enabled:
                rear_transform = carla.Transform(
                    carla.Location(x=-2.5, z=1.2),  # Behind the vehicle
                    carla.Rotation(pitch=0, yaw=180, roll=0)  # Looking backward
                )
                self.rear_camera = self.world.spawn_actor(
                    camera_bp, rear_transform, attach_to=self.vehicle
                )
                self.rear_camera.listen(self._on_rear_image)
                enabled_cameras.append("Rear")
                print("   📷 Rear camera: ENABLED")
            else:
                print("   📷 Rear camera: DISABLED")
            
            if enabled_cameras:
                self.vision_active = True
                print(f"✅ Camera system setup complete!")
                print(f"   🎥 Active cameras: {', '.join(enabled_cameras)}")
            else:
                print("⚠️ No cameras enabled!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup cameras: {e}")
            return False
    
    def _on_front_image(self, image):
        """Process front camera image."""
        if not self.front_camera_enabled:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Process with computer vision
        processed_image = self.cv_processor.process_front_view(cv_image)
        self.current_front_image = processed_image
        
        if not self.front_image_queue.full():
            self.front_image_queue.put((image.timestamp, processed_image))
    
    def _on_left_image(self, image):
        """Process left camera image."""
        if not self.left_camera_enabled:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Process with computer vision
        processed_image = self.cv_processor.process_side_view(cv_image, "left")
        self.current_left_image = processed_image
        
        if not self.left_image_queue.full():
            self.left_image_queue.put((image.timestamp, processed_image))
    
    def _on_right_image(self, image):
        """Process right camera image."""
        if not self.right_camera_enabled:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Process with computer vision
        processed_image = self.cv_processor.process_side_view(cv_image, "right")
        self.current_right_image = processed_image
        
        if not self.right_image_queue.full():
            self.right_image_queue.put((image.timestamp, processed_image))
    
    def _on_top_image(self, image):
        """Process top camera image."""
        if not self.top_camera_enabled:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Process with computer vision - Top view gets advanced processing
        processed_image = self.cv_processor.process_top_view(cv_image)
        self.current_top_image = processed_image
        
        if not self.top_image_queue.full():
            self.top_image_queue.put((image.timestamp, processed_image))
    
    def _on_rear_image(self, image):
        """Process rear camera image."""
        if not self.rear_camera_enabled:
            return
            
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        cv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Process with computer vision
        processed_image = self.cv_processor.process_rear_view(cv_image)
        self.current_rear_image = processed_image
        
        if not self.rear_image_queue.full():
            self.rear_image_queue.put((image.timestamp, processed_image))
    
    def display_vision_system(self):
        """Display all enabled camera feeds."""
        # Display front camera if enabled and image exists
        if self.front_camera_enabled and self.current_front_image is not None:
            front_display = self.current_front_image.copy()
            cv2.putText(front_display, "Front Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Front Camera", front_display)
        
        # Display left camera if enabled and image exists
        if self.left_camera_enabled and self.current_left_image is not None:
            left_display = self.current_left_image.copy()
            cv2.putText(left_display, "Left Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Left Camera", left_display)
        
        # Display right camera if enabled and image exists
        if self.right_camera_enabled and self.current_right_image is not None:
            right_display = self.current_right_image.copy()
            cv2.putText(right_display, "Right Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow("Right Camera", right_display)
        
        # Display top camera if enabled and image exists
        if self.top_camera_enabled and self.current_top_image is not None:
            top_display = self.current_top_image.copy()
            cv2.putText(top_display, "Top Camera (Bird's Eye)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Top Camera", top_display)
        
        # Display rear camera if enabled and image exists
        if self.rear_camera_enabled and self.current_rear_image is not None:
            rear_display = self.current_rear_image.copy()
            cv2.putText(rear_display, "Rear Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Rear Camera", rear_display)
        
        cv2.waitKey(1)  # Process OpenCV events
    
    def check_vehicle_status(self):
        """Check vehicle status for debugging."""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            location = self.vehicle.get_location()
            
            # Only print if there's some movement or control input
            if speed > 0.1 or self.throttle > 0 or self.brake > 0:
                print(f"🚗 Vehicle Status: Speed:{speed:.2f} m/s, Pos:({location.x:.1f},{location.y:.1f})")
    
    def apply_control(self):
        """Apply control inputs to the vehicle."""
        if self.vehicle:
            control = carla.VehicleControl()
            control.throttle = self.throttle
            control.steer = self.steer
            control.brake = self.brake
            control.reverse = self.reverse
            
            # Debug output (only when controls are active)
            if self.throttle > 0 or self.steer != 0 or self.brake > 0:
                print(f"🎮 Controls: T:{self.throttle:.2f} S:{self.steer:.2f} B:{self.brake:.2f} R:{self.reverse}")
            
            self.vehicle.apply_control(control)
    
    def process_keyboard_input(self):
        """Process keyboard input for vehicle control."""
        keys = pygame.key.get_pressed()
        
        # Reset controls
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        
        # Throttle and brake
        if keys[pygame.K_w]:
            self.throttle = 0.6  # Forward throttle
        if keys[pygame.K_s]:
            self.brake = 0.8     # Brake
        
        # Steering
        if keys[pygame.K_a]:
            self.steer = -0.4    # Turn left
        if keys[pygame.K_d]:
            self.steer = 0.4     # Turn right
        
        # NOTE: Reverse toggle is handled in the event loop, not here

    def display_control_info(self):
        """Display control information on pygame window."""
        # Clear display
        self.display.fill((0, 0, 0))
        
        # Create font
        font = pygame.font.Font(None, 24)
        
        # Control status
        y_offset = 10
        
        # Title
        title = font.render("CARLA Vehicle Control", True, (255, 255, 255))
        self.display.blit(title, (10, y_offset))
        y_offset += 30
        
        # Current controls
        throttle_text = font.render(f"Throttle: {self.throttle:.2f}", True, (0, 255, 0) if self.throttle > 0 else (100, 100, 100))
        self.display.blit(throttle_text, (10, y_offset))
        y_offset += 25
        
        steer_text = font.render(f"Steering: {self.steer:.2f}", True, (0, 255, 255) if self.steer != 0 else (100, 100, 100))
        self.display.blit(steer_text, (10, y_offset))
        y_offset += 25
        
        brake_text = font.render(f"Brake: {self.brake:.2f}", True, (255, 0, 0) if self.brake > 0 else (100, 100, 100))
        self.display.blit(brake_text, (10, y_offset))
        y_offset += 25
        
        reverse_text = font.render(f"Reverse: {'ON' if self.reverse else 'OFF'}", True, (255, 255, 0) if self.reverse else (100, 100, 100))
        self.display.blit(reverse_text, (10, y_offset))
        y_offset += 35
        
        # Instructions
        instructions = [
            "Controls:",
            "W - Throttle",
            "S - Brake", 
            "A - Steer Left",
            "D - Steer Right",
            "R - Toggle Reverse",
            "V - Toggle Cameras",
            "",
            "Detection Controls:",
            "1 - Road Detection",
            "2 - Lane Detection",
            "3 - Vehicle Detection",
            "4 - YOLO Detection",
            "5 - YOLO Segmentation",
            "",
            "Fine-tuning:",
            "6 - Increase Confidence",
            "7 - Decrease Confidence", 
            "8 - Show Settings",
            "9 - Reduce Sensitivity",
            "0 - Increase Sensitivity",
            "",
            "ESC - Exit"
        ]
        
        for instruction in instructions:
            text = font.render(instruction, True, (200, 200, 200))
            self.display.blit(text, (10, y_offset))
            y_offset += 20
        
        pygame.display.flip()
    
    def run_phase2(self, num_npcs=15):
        """Run Phase 2: Phase 1 + Keyboard control."""
        print("\n🚀 Starting Phase 2...")
        
        # Step 1: Run Phase 1 setup
        if not self.connect_to_carla():
            print("\n❌ Phase 2 failed at connection step")
            return False
        
        if not self.spawn_vehicle():
            print("\n❌ Phase 2 failed at vehicle spawn step")
            return False
        
        if not self.setup_spectator_view():
            print("\n❌ Phase 2 failed at spectator setup step")
            return False
        
        # Step 3: Setup camera system
        if not self.setup_cameras():
            print("\n❌ Phase 2 failed at camera setup step")
            return False
        
        # Step 4: Spawn NPC vehicles
        if num_npcs > 0:
            print(f"\n🚦 Setting up {num_npcs} NPC vehicles for realistic environment...")
            npc_success = self.spawn_npc_vehicles(num_vehicles=num_npcs)
            if not npc_success:
                print("⚠️ Warning: Failed to spawn NPCs, continuing without them")
        else:
            print("\n🚦 Skipping NPC vehicle spawning (disabled)")
        
        # Step 5: Clean environment automatically
        print(f"\n🌲 Automatically cleaning environment for optimal detection...")
        cleanup_success = self.clean_environment()
        if cleanup_success:
            print(f"✅ Environment cleaned! All vegetation removed for clear camera views.")
        else:
            print(f"⚠️ Environment cleanup had some issues, but continuing...")
        
        print("\n✅ Phase 1 setup complete!")
        print("\n🎮 Starting Phase 2: Keyboard Control + Multi-Camera System")
        print("\n📋 Controls:")
        print("   W - Throttle forward")
        print("   S - Brake")
        print("   A - Steer left") 
        print("   D - Steer right")
        print("   R - Toggle reverse")
        print("   V - Toggle camera display")
        print("\n🔧 Detection Controls:")
        print("   1 - Toggle road detection")
        print("   2 - Toggle lane detection") 
        print("   3 - Toggle vehicle detection")
        print("   4 - Toggle YOLO detection")
        print("   5 - Toggle YOLO segmentation")
        print("\n⚙️ Fine-tuning Controls:")
        print("   6 - Increase YOLO confidence (+0.1)")
        print("   7 - Decrease YOLO confidence (-0.1)")
        print("   8 - Show current detection settings")
        print("   9 - Apply preset: Reduce sensitivity (larger zones)")
        print("   0 - Apply preset: Increase sensitivity (smaller zones)")
        print("   ESC - Exit")
        print("\n� Camera System:")
        
        active_cameras = []
        if self.front_camera_enabled:
            active_cameras.append("Front")
        if self.left_camera_enabled:
            active_cameras.append("Left")
        if self.right_camera_enabled:
            active_cameras.append("Right")
        if self.top_camera_enabled:
            active_cameras.append("Top")
        if self.rear_camera_enabled:
            active_cameras.append("Rear")
        
        if active_cameras:
            print(f"   📹 Active: {', '.join(active_cameras)} camera(s)")
        else:
            print("   ⚠️ No cameras active")
        
        print("\n🤖 Computer Vision Features:")
        print("   🛣️ Road Detection: Advanced road surface segmentation")
        print("   🛤️ Lane Detection: Real-time lane marking detection with crack filtering")
        print("   🚗 Vehicle Detection: Distance-based classification (CLOSE/MEDIUM/FAR)")
        print("   🎯 YOLOv8 Integration: Advanced AI-powered object detection")
        print("   📊 Top View Analysis: Bird's eye perspective with proximity warnings")
        print("\n⚠️ Distance-Based Safety System:")
        print("   🔴 RED = CLOSE vehicles (potential collision risk)")
        print("   🟠 ORANGE = MEDIUM distance vehicles")
        print("   🟢 GREEN = FAR vehicles (safe distance)")
        print("\n🎯 Drive around and monitor vehicle proximity in real-time!")
        
        try:
            running = True
            while running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            # Toggle reverse on keypress (not hold)
                            self.reverse = not self.reverse
                            print(f"🔄 Reverse: {'ON' if self.reverse else 'OFF'}")
                        elif event.key == pygame.K_v:
                            # Toggle vision display
                            self.vision_active = not self.vision_active
                            if not self.vision_active:
                                cv2.destroyAllWindows()
                            print(f"👁️ Camera Display: {'ON' if self.vision_active else 'OFF'}")
                        elif event.key == pygame.K_1:
                            # Toggle road detection
                            road_state = self.cv_processor.toggle_detection("road")
                            print(f"🛣️ Road Detection: {'ON' if road_state else 'OFF'}")
                        elif event.key == pygame.K_2:
                            # Toggle lane detection
                            lane_state = self.cv_processor.toggle_detection("lane")
                            print(f"🛤️ Lane Detection: {'ON' if lane_state else 'OFF'}")
                        elif event.key == pygame.K_3:
                            # Toggle vehicle detection
                            vehicle_state = self.cv_processor.toggle_detection("vehicle")
                            print(f"🚗 Vehicle Detection: {'ON' if vehicle_state else 'OFF'}")
                        elif event.key == pygame.K_4:
                            # Toggle YOLO detection
                            yolo_state = self.cv_processor.toggle_detection("yolo")
                            print(f"🚀 YOLO Detection: {'ON' if yolo_state else 'OFF'}")
                        elif event.key == pygame.K_5:
                            # Toggle YOLO segmentation
                            yolo_seg_state = self.cv_processor.toggle_detection("yolo_seg")
                            print(f"🎯 YOLO Segmentation: {'ON' if yolo_seg_state else 'OFF'}")
                        elif event.key == pygame.K_6:
                            # Increase YOLO confidence
                            current_conf = self.cv_processor.yolo_confidence
                            new_conf = min(1.0, current_conf + 0.1)
                            self.cv_processor.adjust_yolo_confidence(new_conf)
                        elif event.key == pygame.K_7:
                            # Decrease YOLO confidence
                            current_conf = self.cv_processor.yolo_confidence
                            new_conf = max(0.1, current_conf - 0.1)
                            self.cv_processor.adjust_yolo_confidence(new_conf)
                        elif event.key == pygame.K_8:
                            # Print current detection settings
                            self.cv_processor.print_current_settings()
                        elif event.key == pygame.K_9:
                            # Quick preset: Reduce sensitivity (larger detection zones)
                            self.cv_processor.adjust_detection_thresholds(
                                min_vehicle_area=500,      # Increase minimum size
                                close_distance=120,        # Larger close zone
                                medium_distance=250        # Larger medium zone
                            )
                            print("🔧 Applied preset: Reduced sensitivity (larger zones)")
                        elif event.key == pygame.K_0:
                            # Quick preset: Increase sensitivity (smaller detection zones)
                            self.cv_processor.adjust_detection_thresholds(
                                min_vehicle_area=200,      # More sensitive to small objects
                                close_distance=80,         # Smaller close zone
                                medium_distance=150        # Smaller medium zone
                            )
                            print("🔧 Applied preset: Increased sensitivity (smaller zones)")
                
                # Process continuous keyboard input
                self.process_keyboard_input()
                
                # Apply controls to vehicle
                self.apply_control()
                
                # Check vehicle status (for debugging)
                self.check_vehicle_status()
                
                # Update spectator view to follow vehicle
                self.update_spectator_view()
                
                # Display camera system if active
                if self.vision_active:
                    self.display_vision_system()
                
                # Display control information
                self.display_control_info()
                
                # Control update rate
                self.clock.tick(60)  # 60 FPS for smooth control
                
        except KeyboardInterrupt:
            print("\n🛑 Phase 2 stopped by user")
        
        return True
    
    def run_phase1(self, spawn_npcs=True):
        """Run Phase 1: Connection, spawn, and spectator setup (no controls)."""
        print("\n🚀 Starting Phase 1...")
        
        # Step 1: Connect to CARLA
        if not self.connect_to_carla():
            print("\n❌ Phase 1 failed at connection step")
            return False
        
        # Step 2: Spawn vehicle
        if not self.spawn_vehicle():
            print("\n❌ Phase 1 failed at vehicle spawn step")
            return False
        
        # Step 3: Setup spectator view
        if not self.setup_spectator_view():
            print("\n❌ Phase 1 failed at spectator setup step")
            return False
        
        # Step 4: Optionally spawn NPCs
        if spawn_npcs:
            print("\n🚦 Setting up NPC vehicles...")
            npc_success = self.spawn_npc_vehicles(num_vehicles=10)
            if not npc_success:
                print("⚠️ Warning: Failed to spawn NPCs, continuing without them")
        
        print("\n✅ Phase 1 completed successfully!")
        print("\n📋 What you should see:")
        print("   - Vehicle spawned in CARLA world")
        print("   - Camera positioned behind vehicle in 3rd person view")
        if spawn_npcs:
            print("   - NPC vehicles driving around with autopilot")
        print("   - Vehicle should be visible in the spectator view")
        
        # Keep the demo running for observation
        print("\n🎮 Demo mode: Vehicle will stay stationary")
        print("   Press Ctrl+C to exit")
        
        try:
            while True:
                # Keep updating spectator view
                self.update_spectator_view()
                time.sleep(0.1)  # 10 FPS update
                
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Close vision windows
        if self.vision_active:
            cv2.destroyAllWindows()
            print("   👁️ Camera windows closed")
        
        # Destroy cameras
        cameras_to_destroy = [
            (self.front_camera, self.front_camera_enabled, "Front"),
            (self.left_camera, self.left_camera_enabled, "Left"),
            (self.right_camera, self.right_camera_enabled, "Right"),
            (self.top_camera, self.top_camera_enabled, "Top"),
            (self.rear_camera, self.rear_camera_enabled, "Rear")
        ]
        
        for camera, enabled, name in cameras_to_destroy:
            if camera and enabled:
                try:
                    camera.destroy()
                    print(f"   📷 {name} camera destroyed")
                except:
                    pass
        
        # Destroy NPC vehicles
        if self.npc_vehicles:
            print(f"   🚦 Destroying {len(self.npc_vehicles)} NPC vehicles...")
            for npc in self.npc_vehicles:
                try:
                    if npc.is_alive:
                        npc.destroy()
                except:
                    pass  # Vehicle might already be destroyed
            self.npc_vehicles.clear()
        
        # Destroy player vehicle
        if self.vehicle:
            self.vehicle.destroy()
            print("   🚗 Player vehicle destroyed")
        
        pygame.quit()
        print("   🎮 Pygame cleaned up")
        
        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='CARLA Data Recorder - Phase 2')
    parser.add_argument('--host', default='localhost', help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Connection timeout (default: 5.0)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=2, help='Run specific phase (1 or 2, default: 2)')
    parser.add_argument('--no-npcs', action='store_true', help='Disable NPC vehicle spawning')
    parser.add_argument('--num-npcs', type=int, default=15, help='Number of NPC vehicles to spawn (default: 15)')
    
    args = parser.parse_args()
    
    recorder = CARLADataRecorder(args.host, args.port, args.timeout)
    
    try:
        spawn_npcs = not args.no_npcs
        
        if args.phase == 1:
            print("🎯 Running Phase 1 only (connection, spawn, spectator)")
            success = recorder.run_phase1(spawn_npcs=spawn_npcs)
        else:  # Phase 2
            print("🎯 Running Phase 2 (Phase 1 + keyboard control)")
            if spawn_npcs:
                print(f"🚦 Will spawn {args.num_npcs} NPC vehicles")
            success = recorder.run_phase2(num_npcs=args.num_npcs if spawn_npcs else 0)
        
        if not success:
            print(f"\n❌ Phase {args.phase} failed!")
            return
            
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        recorder.cleanup()

if __name__ == '__main__':
    main()
