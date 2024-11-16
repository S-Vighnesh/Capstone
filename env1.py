# env1.py

from env2 import *
from env3 import *

import random
import numpy as np

class RSUEnvironment:
    RSU_COVERAGE_DIAMETER = 1.0  # km
    RSU_COVERAGE_RADIUS = RSU_COVERAGE_DIAMETER / 2
    TIME_SLOT_DURATION = 1 / 60  # 1 minute in hours
    available_BRBs = 50  # Number of bandwidth resource blocks available
    bandwidth_block_size = 0.18  # Bandwidth size per block in MHz

    def __init__(self, mean_arrival_rate, compute_cycles_factor, delay_tolerance_range):
        self.mean_arrival_rate = mean_arrival_rate  # Lambda for Poisson distribution
        self.compute_cycles_factor = compute_cycles_factor  # Factor for computing cycles based on task size
        self.delay_tolerance_range = delay_tolerance_range
        self.clients = []
        self.request_classes = [RequestClass(i + 1) for i in range(4)]

    def run_time_slot(self):
        self.generate_clients()
        self.update_clients()
        self.allocate_bandwidth()

    def generate_clients(self):
        num_new_clients = np.random.poisson(self.mean_arrival_rate)

        for _ in range(num_new_clients):
            initial_position = random.uniform(-self.RSU_COVERAGE_RADIUS, self.RSU_COVERAGE_RADIUS)
            speed = max(0, np.random.normal(20, 3))  # Ensure speed is non-negative
            direction = random.choice([-1, 1])
            time_slots_available = self.calculate_time_slots(initial_position, speed, direction)
            task_size = random.randint(1, 10)
            computing_cycles = task_size * self.compute_cycles_factor
            delay_tolerance = random.randint(*self.delay_tolerance_range)
            upper_delay_tolerance = delay_tolerance + 0.15

            if delay_tolerance <= time_slots_available:
                client = Client(
                    client_id=len(self.clients),
                    position=initial_position,
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

    def update_clients(self):
        for client in self.clients[:]:
            client.update_position()
            if self.calculate_distance(client) > self.RSU_COVERAGE_RADIUS or client.computing_cycles <= 0:
                self.remove_from_request_class(client)
                self.clients.remove(client)

    def remove_from_request_class(self, client):
        for request_class in self.request_classes:
            for client_info in request_class.clients:
                if client_info["client_id"] == client.client_id:
                    request_class.clients.remove(client_info)
                    request_class.total_computing_cycles -= client.computing_cycles
                    break

    def calculate_distance(self, client):
        return abs(client.position)

    def calculate_time_slots(self, position, speed, direction):
        distance_to_exit = self.RSU_COVERAGE_RADIUS - abs(position)
        speed_km_per_min = speed / 60
        time_slots = distance_to_exit / (speed_km_per_min * abs(direction))
        return int(time_slots)

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

    def allocate_bandwidth(self):
        total_bandwidth = self.available_BRBs * self.bandwidth_block_size
        num_clients_in_range = len(self.clients)

        if num_clients_in_range > 0:
            bandwidth_per_client = total_bandwidth / num_clients_in_range
            for client in self.clients:
                client.bandwidth = bandwidth_per_client


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
        self.position += self.speed * RSUEnvironment.TIME_SLOT_DURATION * self.direction
        self.time_slots_available -= 1


class RequestClass:
    def __init__(self, class_id):
        self.class_id = class_id
        self.clients = []
        self.total_computing_cycles = 0

    def add_client(self, client_info):
        self.clients.append(client_info)
        self.total_computing_cycles += client_info['computing_cycles']
