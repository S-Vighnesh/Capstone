# env2.py
from env1 import *
from env3 import *
import random
import numpy as np

def generate_server_speed(is_moving):
    if is_moving:
        # Randomly choose if the server is a moving pedestrian or a vehicle
        if random.choice([True, False]):
            # Moving pedestrian with speed in Gaussian distribution (mean=5 km/h, std=1 km/h)
            return max(0, random.gauss(5, 1))  # Ensure non-negative speed
        else:
            # Moving vehicle with speed in Gaussian distribution (mean=20 km/h, std=5 km/h)
            return max(0, random.gauss(20, 5))  # Ensure non-negative speed
    else:
        # Stationary server with speed 0 km/h
        return 0

# Server class definition
class Server:
    def __init__(self, server_id, is_moving, position, speed, direction, computing_cycles_available, remaining_time_in_range):
        self.server_id = server_id  # Unique server ID
        self.is_moving = is_moving  # Whether the server is moving or stationary
        self.position = position  # Current position within RSU's coverage
        self.speed = speed  # Speed of the server (only relevant if moving)
        self.direction = direction  # Direction of movement (+1 for right, -1 for left)
        self.computing_cycles_available = computing_cycles_available  # Computing cycles offered by the server
        self.remaining_time_in_range = remaining_time_in_range  # Time remaining until the server leaves RSU's coverage

    def update_position(self, time_slot_duration):
        if self.is_moving:
            # Update server's position based on speed and direction
            self.position += self.direction * (self.speed * (time_slot_duration / 60))  # Convert speed to distance per minute

        # Helper function to generate server speed

# Environment class
class RSUEnvironment:
    def __init__(self, num_servers, area_size, min_required_time_slots, time_slot_duration):
        self.num_servers = num_servers  # Initial number of servers
        self.area_size = area_size  # Size of the RSU's coverage area
        self.min_required_time_slots = min_required_time_slots  # Minimum time slots for server viability
        self.time_slot_duration = time_slot_duration  # Duration of each time slot (in minutes)
        self.servers = []  # List to store all servers
        self.server_classes = []  # List to store server classes
        self.server_id_counter = 0  # Counter to ensure unique IDs for dynamically added servers
        self.generate_servers()

    def calculate_remaining_time_slots(self, position, speed, direction):
        if speed == 0:  # Stationary server
            return float('inf')  # Stays indefinitely within coverage
        distance_to_edge = self.area_size / 2 - abs(position)  # Distance to nearest edge of coverage
        time_to_exit = distance_to_edge / (speed / 60)  # Time to exit in minutes
        return int(time_to_exit / self.time_slot_duration)  # Convert to time slots

    def generate_servers(self):
        # Generate the initial set of servers
        for _ in range(self.num_servers):
            self.add_server()

    def add_server(self):
        # Dynamically add a new server to the environment
        is_moving = random.choice([True, False])  # Randomly decide if server is moving or stationary
        position = random.uniform(-self.area_size / 2, self.area_size / 2)  # Random position within the coverage area
        speed = generate_server_speed(is_moving)  # Generate speed based on whether it's moving
        direction = random.choice([-1, 1]) if is_moving else 0  # Assign direction for moving servers

        # Calculate remaining time slots dynamically
        remaining_time_in_range = self.calculate_remaining_time_slots(position, speed, direction)

        # Skip servers that don't meet the minimum time slots
        if remaining_time_in_range < self.min_required_time_slots:
            return

        # Compute server resources (e.g., computing cycles)
        computing_cycles_available = random.randint(100, 1000)

        # Create the server object and append to the list
        server = Server(self.server_id_counter, is_moving, position, speed, direction, computing_cycles_available, remaining_time_in_range)
        self.server_id_counter += 1  # Increment the unique server ID counter
        self.servers.append(server)

    def generate_new_servers(self):
        # Generate new servers at each time slot based on a Poisson distribution
        num_new_servers = np.random.poisson(1)  # Average 1 new server per time slot
        for _ in range(num_new_servers):
            self.add_server()

    def assign_servers_to_request_classes(self, request_classes):
        # Function to assign servers to request classes
        for request_class in request_classes:
            total_cycles_needed = sum(client['computing_cycles_needed'] for client in request_class['clients'])  # Total cycles required
            total_cycles_assigned = 0  # Track total cycles assigned to this request class

            server_class = []  # Initialize the server class for this request class

            # Sequentially allocate servers to this request class until requirements are met
            for server in self.servers[:]:  # Iterate over a copy of the server list
                if total_cycles_assigned >= total_cycles_needed:
                    break  # Stop when the required computing cycles are met

                server_class.append(server)
                total_cycles_assigned += server.computing_cycles_available
                self.servers.remove(server)  # Remove server from available list after allocation

            request_class['server_class'] = server_class  # Assign the server class to this request class

    def update_servers(self):
        # Update all servers (move their positions, update remaining time, etc.)
        for server in self.servers[:]:  # Iterate over a copy of the server list
            server.update_position(self.time_slot_duration)

            # Remove servers that exit RSU coverage
            if abs(server.position) > self.area_size / 2:
                self.servers.remove(server)

            # Reduce the remaining time slots for the server
            server.remaining_time_in_range -= 1
            if server.remaining_time_in_range <= 0:
                self.servers.remove(server)

        # Dynamically generate new servers during the time slot
        self.generate_new_servers()
