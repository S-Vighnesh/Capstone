import math
import numpy as np
from scipy.stats import rice  # Importing rice for Rician fading

# Constants for the environment
C = 3 * 10**8  # Speed of light in m/s
FREQ = 5.9 * 10**9  # Frequency in Hz (5.9 GHz)
D0 = 100  # Reference distance in meters
ALPHA = 3.75  # Path loss exponent
TRANSMIT_POWER_DBM = 23  # Transmit power in dBm
NOISE_SPECTRAL_DENSITY_DBM = -174  # Noise spectr**al density in dBm/Hz
BANDWIDTH = 180 * 10**3  # Bandwidth in Hz (180 kHz)
K_FACTOR = 6  # Rician K-factor

RSU_coverage = 1000  # RSU coverage in meters (1 km)
RSU_COVERAGE_RADIUS = RSU_coverage / 2
TIME_SLOT_DURATION = 1 / 60  # 1 minute in hours
task_sizes = [5, 7, 10]  # Task sizes in MB
delay_tolerance_l = [0.35, 0.5]  # Lower delay tolerance in seconds
delay_tolerance_u = [0.5, 0.65]  # Upper delay tolerance in seconds
compute_cycles_needed = [0.2, 0.3, 0.4]  # Computing cycles in Gigacycles
bandwidth_block_size = 180  # Bandwidth block size in kHz
available_BRBs = 50  # Available bandwidth resource blocks per RSU

class RequestClass:
    def __init__(self, class_id):
        self.class_id = class_id
        self.clients = []  # List to store clients in this request class
        self.total_computing_cycles_required = 0  # Total computing cycles needed by all clients in this class
    
    def add_client(self, client):
        """
        Adds a client to the request class and updates the total computing cycles required.
        """
        self.clients.append(client)
        self.total_computing_cycles_required += client["computing_cycles"]

class Environment:
    def _init_(self, num_clients, num_servers, area_size, mean_arrival_rate, compute_cycles_needed, delay_tolerance, compute_cycles_factor):
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.area_size = area_size
        self.mean_arrival_rate = mean_arrival_rate
        self.compute_cycles_needed = compute_cycles_needed
        self.compute_cycles_factor = compute_cycles_factor
        self.delay_tolerance = delay_tolerance
        self.clients = []


        # Define the four request classes
        self.request_classes = [
            RequestClass(class_id=1),  # Request Class 1 (computing cycles <= 100)
            RequestClass(class_id=2),  # Request Class 2 (101 <= computing cycles <= 200)
            RequestClass(class_id=3),  # Request Class 3 (201 <= computing cycles <= 300)
            RequestClass(class_id=4)   # Request Class 4 (computing cycles > 300)
        ]
        
        self.clients = []  # List to store client objects
        self.servers = []  # List to store server objects
        self.request_classes = [RequestClass(i+1) for i in range(4)]
        self.server_classes = []  # List to store server classes
        
        # Initialize clients and servers
        self.generate_clients()
        self.generate_servers()
    
    def reset(self):
        # Reset environment and client/server states for a new episode
        self.clients = []
        self.servers = []
        self.request_classes = []
        self.server_classes = []
        self.generate_clients()
        self.generate_servers()
    
    def run_time_slot(self):
        # Simulate a single time slot in which clients may arrive
        self.generate_clients()

    def generate_clients(self):
        # Determine the number of new clients using Poisson distribution
        num_new_clients = np.random.poisson(self.mean_arrival_rate)
        
        for _ in range(num_new_clients):
            # Generate client position randomly within RSU coverage
            initial_position = random.uniform(-self.RSU_COVERAGE_RADIUS, self.RSU_COVERAGE_RADIUS)
            
            # Randomly set speed between 15 and 25 km/h
            speed = random.uniform(15, 25)
            
            # Randomly set direction (+1 or -1)
            direction = random.choice([-1, 1])
            
            # Calculate time slots available based on speed and initial position
            time_slots_available = self.calculate_time_slots(initial_position, speed, direction)
            
            # Randomly assign task size and calculate computing cycles
            task_size = random.randint(1, 10)
            computing_cycles = task_size * self.compute_cycles_factor
            
            # Assign delay tolerance and check if client is acceptable
            delay_tolerance = random.randint(*self.delay_tolerance_range)
            
            # Check if the client meets the delay tolerance criteria
            if delay_tolerance <= time_slots_available:
                # Create client
                client = Client(
                    client_id=len(self.clients),  # Unique client ID
                    position=initial_position,
                    speed=speed,
                    direction=direction,
                    computing_cycles=computing_cycles,
                    task_size=task_size,
                    delay_tolerance=delay_tolerance,
                    time_slots_available=time_slots_available
                )
                
                # Add client to list and assign to request class
                self.clients.append(client)
                self.assign_to_request_class(client)

    def calculate_time_slots(self, position, speed, direction):
        """
        Calculate the time slots before the client leaves RSU coverage.
        position: initial x-coordinate position of the client within RSU coverage.
        speed: client's speed in km/h.
        direction: +1 or -1, representing movement along the x-axis.
        """
        distance_to_exit = self.RSU_COVERAGE_RADIUS - abs(position)
        # Convert speed to km per minute, then calculate time slots based on the distance to exit
        speed_km_per_min = speed / 60
        time_slots = distance_to_exit / (speed_km_per_min * abs(direction))
        
        return int(time_slots)  # Convert to integer time slots
    
    def generate_servers(self):
        # Function to randomly generate server objects (moving and stationary)
        pass
    
    def update_clients(self):
        """Update client states, remove those that leave RSU coverage or complete tasks."""
        for client in self.clients[:]:  # Use a copy of the list to avoid modification issues during iteration
            # Update client position if they are moving
            client.update_position(self.TIME_SLOT_DURATION)
            
            # Check if the client is still within RSU coverage
            if self.calculate_distance(client.position) > self.RSU_COVERAGE_RADIUS or client.task_completed():
                # Remove client from request class and update computing cycle requirements
                self.remove_client_from_request_class(client)
                self.clients.remove(client)  # Remove client from main client list
    
    def update_servers(self):
        for server in self.servers:
            # Update server position (if moving) or status
            server.update_position()
            
            # Remove server if out of RSU's coverage
            if self.calculate_distance(server) > RSU_coverage:
                self.servers.remove(server)
            else:
                # Update server's remaining available time/resources
                server.remaining_time_in_range -= 1
                if server.remaining_time_in_range <= 0:
                    self.servers.remove(server)
    
    def assign_to_request_class(self, client):
        """
        Assign client to the appropriate request class based on computing cycles needed.
        """
        if client["computing_cycles"] <= 100:
            class_index = 0  # Request Class 1
        elif client["computing_cycles"] <= 200:
            class_index = 1  # Request Class 2
        elif client["computing_cycles"] <= 300:
            class_index = 2  # Request Class 3
        else:
            class_index = 3  # Request Class 4
        
        # Add client to the selected request class and update total computing cycles required
        self.request_classes[class_index].add_client({
            "client_id": client.client_id,
            "computing_cycles": client.computing_cycles
        })
        self.request_classes[class_index].remove_client(client.client_id)   
    
    def assign_servers_to_request_class(self):
        # Assign servers to request classes 
        pass
    
    def calculate_fspl(distance):
        """Calculate Free Space Path Loss (FSPL) in linear scale."""
        fspl = ((4 * math.pi * FREQ * D0 / C) ** 2) * ((distance / D0) ** ALPHA)
        return fspl

    def calculate_fspl_db(distance):
        """Calculate Free Space Path Loss (FSPL) in dB."""
        term1 = 20 * math.log10((4 * math.pi * FREQ * D0) / C)
        term2 = 10 * ALPHA * math.log10(distance / D0)
        fspl_db = term1 + term2
        return fspl_db

    def apply_rice_fading(fspl_db):
        """Apply Rice fading to the FSPL (in dB) with a specified Rician K-factor."""
        # Generate a Rician fading sample with the K-factor
        rice_fading = rice.rvs(K_FACTOR)
        path_loss_db = fspl_db + rice_fading
        return path_loss_db

    def calculate_received_power(path_loss_db):
        """Calculate the received power in dBm."""
        received_power_dbm = TRANSMIT_POWER_DBM - path_loss_db
        return received_power_dbm

    def calculate_snr(received_power_dbm):
        """Calculate the Signal-to-Noise Ratio (SNR) in linear scale."""
        noise_power_dbm = NOISE_SPECTRAL_DENSITY_DBM + 10 * math.log10(BANDWIDTH)
        noise_power_mw = 10 ** (noise_power_dbm / 10)  # Convert dBm to mW
        received_power_mw = 10 ** (received_power_dbm / 10)  # Convert dBm to mW
        snr = received_power_mw / noise_power_mw
        return snr

    def calculate_link_rate(snr):
        """Calculate the link rate (or data rate) in bps using the Shannon capacity formula."""
        link_rate = BANDWIDTH * math.log2(1 + snr)
        return link_rate

    def calculate_transmission_delay(client, server):
        """Calculate transmission delay between a client and server."""
        distance = calculate_distance(client.position, server.position)
        fspl_db = calculate_fspl_db(distance)
        path_loss_db = apply_rice_fading(fspl_db)
        received_power_dbm = calculate_received_power(path_loss_db)
        snr = calculate_snr(received_power_dbm)
        link_rate = calculate_link_rate(snr)
        transmission_delay = client.task_size * 8 * 10**6 / link_rate  # Convert MB to bits
        return transmission_delay

    def calculate_computational_delay(server, client):
        """Calculate computational delay based on server's compute capacity and client's computational needs."""
        computational_delay = client.compute_cycles_needed / server.compute_capacity
        return computational_delay

    def calculate_total_delay(client, server):
        """Calculate the total delay as the sum of transmission and computational delays."""
        transmission_delay = calculate_transmission_delay(client, server)
        computational_delay = calculate_computational_delay(server, client)
        total_delay = transmission_delay + computational_delay
        return total_delay

    def calculate_distance(self, entity):
        # Calculate the distance between a client/server and the RSU or between client-server pair
        pass
    
    def random_entry_for_clients(self):
        # Simulate random entry of clients into the RSU's range
        pass
    
    def random_entry_for_servers(self):
        # Simulate random entry of servers into the RSU's coverage
        pass
    
    def get_state(self):
        # Return the current state of the environment (client-server pairings, available resources, etc.)
        pass
    
    def reward(self):
        # Calculate and return the reward based on current actions (e.g., successful task offloading)
        pass
    
    def bandwidth(self):
        # Assign bandwidth to clients based on the available resources and clients in range
        total_bandwidth = available_BRBs * bandwidth_block_size  # Total bandwidth
        num_clients_in_range = len(self.clients)
        
        if num_clients_in_range > 0:
            # Divide the total bandwidth equally among the clients
            bandwidth_per_client = total_bandwidth / num_clients_in_range
            for client in self.clients:
                client.bandwidth = bandwidth_per_client