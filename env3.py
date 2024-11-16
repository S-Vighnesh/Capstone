# env3.py

import math
import numpy as np
from scipy.stats import rice  # Importing rice for Rician fading
from env1 import *  # Assuming the Client class is defined in env1.py
from env2 import *  # Assuming the Server class is defined in env2.py

# Constants for the environment
C = 3 * 10**8  # Speed of light in m/s
FREQ = 5.9 * 10**9  # Frequency in Hz (5.9 GHz)
D0 = 100  # Reference distance in meters
ALPHA = 3.75  # Path loss exponent
TRANSMIT_POWER_DBM = 23  # Transmit power in dBm
NOISE_SPECTRAL_DENSITY_DBM = -174  # Noise spectral density in dBm/Hz
BANDWIDTH = 180 * 10**3  # Bandwidth in Hz (180 kHz)
K_FACTOR = 6  # Rician K-factor

RSU_coverage = 1000  # RSU coverage in meters (1 km)
task_sizes = [5, 7, 10]  # Task sizes in MB
delay_tolerance_l = [0.35, 0.5]  # Lower delay tolerance in seconds
delay_tolerance_u = [0.5, 0.65]  # Upper delay tolerance in seconds
compute_cycles_needed = [0.2, 0.3, 0.4]  # Computing cycles in Gigacycles
bandwidth_block_size = 180  # Bandwidth block size in kHz
available_BRBs = 50  # Available bandwidth resource blocks per RSU

class Environment:
    def __init__(self, num_clients, num_servers, area_size, mean_arrival_rate, compute_cycles_needed, delay_tolerance):
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.area_size = area_size
        self.mean_arrival_rate = mean_arrival_rate
        self.compute_cycles_needed = compute_cycles_needed
        self.delay_tolerance = delay_tolerance
        
        self.clients = []  # List to store client objects
        self.servers = []  # List to store server objects
        self.request_classes = []  # List to store client request classes
        self.server_classes = []  # List to store server classes
        
        # Initialize clients and servers
        self.generate_clients()
        self.generate_servers()
    
    def reset(self):
        """Reset environment and client/server states for a new episode."""
        self.clients = []
        self.servers = []
        self.request_classes = []
        self.server_classes = []
        self.generate_clients()
        self.generate_servers()

    def calculate_fspl(self, distance):
        """Calculate Free Space Path Loss (FSPL) in linear scale."""
        fspl = ((4 * math.pi * FREQ * D0 / C) ** 2) * ((distance / D0) ** ALPHA)
        return fspl

    def calculate_fspl_db(self, distance):
        """Calculate Free Space Path Loss (FSPL) in dB."""
        term1 = 20 * math.log10((4 * math.pi * FREQ * D0) / C)
        term2 = 10 * ALPHA * math.log10(distance / D0)
        fspl_db = term1 + term2
        return fspl_db

    def apply_rice_fading(self, fspl_db):
        """Apply Rice fading to the FSPL (in dB) with a specified Rician K-factor."""
        rice_fading = rice.rvs(K_FACTOR)  # Generate a Rician fading sample
        path_loss_db = fspl_db + rice_fading
        return path_loss_db

    def calculate_received_power(self, path_loss_db):
        """Calculate the received power in dBm."""
        received_power_dbm = TRANSMIT_POWER_DBM - path_loss_db
        return received_power_dbm

    def calculate_snr(self, received_power_dbm):
        """Calculate the Signal-to-Noise Ratio (SNR) in linear scale."""
        noise_power_dbm = NOISE_SPECTRAL_DENSITY_DBM + 10 * math.log10(BANDWIDTH)
        noise_power_mw = 10 ** (noise_power_dbm / 10)  # Convert dBm to mW
        received_power_mw = 10 ** (received_power_dbm / 10)  # Convert dBm to mW
        snr = received_power_mw / noise_power_mw
        return snr

    def calculate_link_rate(self, snr):
        """Calculate the link rate (or data rate) in bps using the Shannon capacity formula."""
        link_rate = BANDWIDTH * math.log2(1 + snr)
        return link_rate

    def calculate_transmission_delay(self, client, server):
        """Calculate transmission delay between a client and server."""
        distance = self.calculate_distance(client.position, server.position)
        fspl_db = self.calculate_fspl_db(distance)
        path_loss_db = self.apply_rice_fading(fspl_db)
        received_power_dbm = self.calculate_received_power(path_loss_db)
        snr = self.calculate_snr(received_power_dbm)
        link_rate = self.calculate_link_rate(snr)
        transmission_delay = client.task_size * 8 * 10**6 / link_rate  # Convert MB to bits
        return transmission_delay

    def calculate_computational_delay(self, server, client):
        """Calculate computational delay based on server's compute capacity and client's computational needs."""
        computational_delay = client.compute_cycles_needed / server.compute_capacity
        return computational_delay

    def calculate_total_delay(self, client, server):
        """Calculate the total delay as the sum of transmission and computational delays."""
        transmission_delay = self.calculate_transmission_delay(client, server)
        computational_delay = self.calculate_computational_delay(server, client)
        total_delay = transmission_delay + computational_delay
        return total_delay

    def calculate_distance(self, position1, position2):
        """Calculate the Euclidean distance between two positions."""
        return np.linalg.norm(np.array(position1) - np.array(position2))


    def calculate_distance(self, client, server):
        dx = client['x'] - server['x']
        dy = client['y'] - server['y']
        return math.sqrt(dx**2 + dy**2)

    def get_state(self):
        # Return the current state of the environment (client-server pairings, available resources, etc.)
        pass
    
    def reward(self):
        # Calculate and return the reward based on current actions (e.g., successful task offloading)
        pass