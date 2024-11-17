import random
import numpy as np
import math


class RSUEnvironment:
    RSU_COVERAGE_DIAMETER = 1.0  # km
    RSU_COVERAGE_RADIUS = RSU_COVERAGE_DIAMETER / 2
    TIME_SLOT_DURATION = 1 / 60  # 1 minute in hours
    available_BRBs = 50  # Number of bandwidth resource blocks available
    bandwidth_block_size = 0.18  # Bandwidth size per block in MHz

    def __init__(self, mean_arrival_rate, compute_cycles_factor, delay_tolerance_range, num_servers, area_size, min_required_time_slots):
        # Client-related initialization
        self.mean_arrival_rate = mean_arrival_rate  # Lambda for Poisson distribution
        self.compute_cycles_factor = compute_cycles_factor  # Factor for computing cycles based on task size
        self.delay_tolerance_range = delay_tolerance_range
        self.clients = []
        self.request_classes = [RequestClass(i + 1) for i in range(4)]

        # Server-related initialization
        self.num_servers = num_servers  # Initial number of servers
        self.area_size = area_size  # Size of the RSU's coverage area
        self.min_required_time_slots = min_required_time_slots  # Minimum time slots for server viability
        self.servers = []  # List to store all servers
        self.server_classes = []  # List to store server classes
        self.server_id_counter = 0  # Counter to ensure unique IDs for dynamically added servers

        # Generate initial servers
        self.generate_servers()

    def run_time_slot(self):
        self.generate_and_assign_clients()
        self.update_clients_and_allocate_bandwidth()
        self.generate_and_update_servers()
        self.assign_servers_to_request_classes()

    # ---- Client-Related Methods ----
    def generate_and_assign_clients(self):
        num_new_clients = np.random.poisson(self.mean_arrival_rate)

        for _ in range(num_new_clients):
            initial_x = random.uniform(-self.RSU_COVERAGE_RADIUS, self.RSU_COVERAGE_RADIUS)
            initial_y = random.uniform(-self.RSU_COVERAGE_RADIUS, self.RSU_COVERAGE_RADIUS)
            speed = max(0, np.random.normal(20, 3))  # Ensure speed is non-negative
            direction = random.choice([-1, 1])
            time_slots_available = self.calculate_time_in_range(initial_x, speed, direction, self.RSU_COVERAGE_RADIUS)
            task_size = random.randint(1, 10)
            computing_cycles = task_size * self.compute_cycles_factor
            delay_tolerance = random.randint(*self.delay_tolerance_range)
            upper_delay_tolerance = delay_tolerance + 0.15

            if delay_tolerance <= time_slots_available:
                client = Client(
                    client_id=len(self.clients),
                    position=(initial_x, initial_y),
                    speed=speed,
                    direction=direction,
                    computing_cycles=computing_cycles,
                    task_size=task_size,
                    delay_tolerance=delay_tolerance,
                    upper_delay_tolerance=upper_delay_tolerance,
                    time_slots_available=time_slots_available,
                    bandwidth=0
                )
                self.clients.append(client)
                self.assign_to_request_class(client)

    def update_clients_and_allocate_bandwidth(self):
        total_bandwidth = self.available_BRBs * self.bandwidth_block_size
        num_clients_in_range = len(self.clients)

        if num_clients_in_range > 0:
            bandwidth_per_client = total_bandwidth / num_clients_in_range

        for client in self.clients[:]:
            client.update_position()
            if abs(client.position[0]) > self.RSU_COVERAGE_RADIUS or client.computing_cycles <= 0:
                self.remove_from_request_class(client)
                self.clients.remove(client)
            else:
                client.bandwidth = bandwidth_per_client if num_clients_in_range > 0 else 0

    def remove_from_request_class(self, client):
        for request_class in self.request_classes:
            for client_info in request_class.clients:
                if client_info["client_id"] == client.client_id:
                    request_class.clients.remove(client_info)
                    request_class.total_computing_cycles -= client.computing_cycles
                    break

    def assign_to_request_class(self, client):
        if client.computing_cycles <= 100:
            class_index = 0
        elif client.computing_cycles <= 200:
            class_index = 1
        elif client.computing_cycles <= 300:
            class_index = 2
        else:
            class_index = 3

        self.request_classes[class_index].add_client({
            "client_id": client.client_id,
            "computing_cycles": client.computing_cycles
        })

    # ---- Server-Related Methods ----
    def generate_servers(self):
        for _ in range(self.num_servers):
            self.add_server()

    def add_server(self):
        is_moving = random.choice([True, False])
        position_x = random.uniform(-self.area_size / 2, self.area_size / 2)
        position_y = random.uniform(-self.area_size / 2, self.area_size / 2)
        speed = self.generate_server_speed(is_moving)
        direction = random.choice([-1, 1]) if is_moving else 0

        remaining_time_in_range = self.calculate_time_in_range(position_x, speed, direction, self.area_size / 2)

        if remaining_time_in_range < self.min_required_time_slots:
            return

        computing_cycles_available = random.randint(100, 1000)
        server = Server(
            server_id=self.server_id_counter,
            is_moving=is_moving,
            position=(position_x, position_y),
            speed=speed,
            direction=direction,
            computing_cycles_available=computing_cycles_available,
            remaining_time_in_range=remaining_time_in_range
        )
        self.server_id_counter += 1
        self.servers.append(server)

    def generate_and_update_servers(self):
        num_new_servers = np.random.poisson(1)  # Average 1 new server per time slot
        for _ in range(num_new_servers):
            self.add_server()

        for server in self.servers[:]:
            server.update_position(self.TIME_SLOT_DURATION)

            if abs(server.position[0]) > self.area_size / 2:
                self.servers.remove(server)

            server.remaining_time_in_range -= 1
            if server.remaining_time_in_range <= 0:
                self.remove_server_from_request_classes(server)
                self.servers.remove(server)

    def remove_server_from_request_classes(self, server):
        for server_class in self.server_classes:
            for assigned_server in server_class.servers:
                if assigned_server.server_id == server.server_id:
                    server_class.servers.remove(assigned_server)
                    break

    def generate_server_speed(self, is_moving):
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
        
    def assign_servers_to_request_classes(self):
        # Clear previous server-class associations
        self.server_classes = [RequestClass(i + 1) for i in range(4)]

        # Sort servers based on arrival order
        sorted_servers = sorted(self.servers, key=lambda s: s.server_id)

        for request_class in self.request_classes:
            while request_class.total_computing_cycles > 0 and sorted_servers:
                server = sorted_servers.pop(0)  # Take the next server in the list and remove it

                # Allocate server resources to the request class
                if server.computing_cycles_available >= request_class.total_computing_cycles:
                    server.computing_cycles_available -= request_class.total_computing_cycles
                    request_class.total_computing_cycles = 0
                    self.server_classes[request_class.class_id - 1].add_server(server)
                else:
                    request_class.total_computing_cycles -= server.computing_cycles_available
                    server.computing_cycles_available = 0
                    self.server_classes[request_class.class_id - 1].add_server(server)

    # ---- Helper Methods ----
    def calculate_time_in_range(self, position_x, speed, direction, coverage_radius):
        if speed == 0:  # Stationary entity
            return float('inf')  # Stays indefinitely within range

        distance_to_exit = coverage_radius - abs(position_x)  # Distance to boundary
        speed_km_per_min = speed / 60  # Convert speed to km/min
        time_slots = distance_to_exit / (speed_km_per_min * abs(direction))
        
        return int(time_slots)

class RequestClass:
    def __init__(self, request_class_id):
        self.request_class_id = request_class_id
        self.clients = []
        self.total_computing_cycles = 0

    def add_client(self, client_info):
        self.clients.append(client_info)
        self.total_computing_cycles += client_info["computing_cycles"]

class Client:
    def __init__(self, client_id, position, speed, direction, computing_cycles, task_size, delay_tolerance,
                 upper_delay_tolerance, time_slots_available, bandwidth):
        self.client_id = client_id
        self.position = position
        self.speed = speed
        self.direction = direction
        self.computing_cycles = computing_cycles
        self.task_size = task_size
        self.delay_tolerance = delay_tolerance
        self.upper_delay_tolerance = upper_delay_tolerance
        self.time_slots_available = time_slots_available
        self.bandwidth = bandwidth

    def update_position(self):
        x, y = self.position
        x += self.speed * RSUEnvironment.TIME_SLOT_DURATION * self.direction
        self.position = (x, y)

class Server:
    def __init__(self, server_id, is_moving, position, speed, direction, computing_cycles_available, remaining_time_in_range):
        self.server_id = server_id
        self.is_moving = is_moving
        self.position = position
        self.speed = speed
        self.direction = direction
        self.computing_cycles_available = computing_cycles_available
        self.remaining_time_in_range = remaining_time_in_range

    def update_position(self, time_slot_duration):
        if self.is_moving:
            x, y = self.position
            x += self.speed * time_slot_duration * self.direction
            self.position = (x, y)