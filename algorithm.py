
import random
import numpy as np
from bees_algorithm import BeesAlgorithm
from collections import Counter
import matplotlib.pyplot as plt
import time

class BooleanNetwork:
    def __init__(self, W, Theta):
        self.W = np.array(W)  # Convert to NumPy array for speed
        self.Theta = np.array(Theta)  # Convert to NumPy array
        self.n = len(W)  # Number of nodes
    
    def update(self, state):
        """Update state using parallel synchronous rule."""
        weighted_sum = np.dot(self.W, state)
        # Return 1 for activation (positive sum) and -1 for repression (negative sum)
        return np.sign(weighted_sum - self.Theta).astype(int)  # Vectorized update
    
    def simulate(self, initial_state, max_steps=500):
        """Simulate network until all attractors are found."""
        history = set()
        state = np.array(initial_state)
        attractors = set()  # Track attractors found

        for _ in range(max_steps):
            state_tuple = tuple(state)
            if state_tuple in history:
                attractors.add(state_tuple)  # Add the attractor if we revisit a state
                return attractors  # Return the set of attractors
            history.add(state_tuple)
            state = self.update(state)
        return np.array(list(attractors))  # Return the set of attractors if no cycle is found

# Predefined target attractors (fixed points of the original model)
target_attractors = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sepal
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],  # Petal
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Carpel
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Stamen
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Inflorescence
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]   # Mutant
]

# def fitness_function(params):
#     """Simplified fitness function: Focuses on attractor mismatch only."""
#     W_candidate = np.array(params[:144]).reshape(12, 12).round().astype(int)
#     Theta_candidate = np.array(params[144:]).round().astype(int)

#     # Ensure matrix dimensions are correct
#     if W_candidate.shape != (12, 12) or Theta_candidate.shape != (12,):
#         return float('-inf')  # Invalid solution

#     network = BooleanNetwork(W_candidate, Theta_candidate)

#     # Validate attractors (only the error term for mismatch)
#     error = 0
#     found_attractors = set()
#     for target_state in target_attractors:  # target_attractors is already in binary (0 or 1)
#         initial_state = np.random.randint(0, 2, size=12)
#         final_state = network.simulate(initial_state)
#         error += sum(10 for i in range(12) if final_state[i] != target_state[i])
#         found_attractors.add(tuple(final_state))

#     # Penalize if there are more than the expected number of attractors
#     extra_attractors = len(found_attractors) - len(target_attractors)
#     if extra_attractors > 0:
#         error += extra_attractors * 10  # Strong penalty for extra attractors

#     return -error  # Return negative error for minimization

def fitness_function(params):
    W_candidate = np.array(params[:144]).reshape(12, 12).round().astype(int)
    Theta_candidate = np.array(params[144:]).round().astype(int)

    # Ensure matrix dimensions are correct
    if W_candidate.shape != (12, 12) or Theta_candidate.shape != (12,):
        return float('-inf')  # Invalid solution

    network = BooleanNetwork(W_candidate, Theta_candidate)

    # Validate attractors (only the error term for mismatch)
    error = 0
    found_attractors = set()

    # Iterate over the target attractors and simulate to match
    for target_state in target_attractors:
        initial_state = np.random.randint(0, 2, size=12)
        attractors = network.simulate(initial_state)  # List of attractors found by the simulation
        
        # Convert the attractors to tuples for easy comparison
        attractors = {tuple(a) for a in attractors}

        # Calculate error for mismatches between the found attractors and target attractors
        for attractor in attractors:
            if attractor in target_attractors:
                # Correct attractor, no error
                pass
            else:
                # Mismatch found, penalize
                error += 10  # You can adjust the penalty

        # Add the found attractors to the set
        found_attractors.update(attractors)

    # Penalize if there are extra attractors (unexpected ones)
    extra_attractors = len(found_attractors) - len(target_attractors)
    if extra_attractors > 0:
        error += extra_attractors * 10  # Strong penalty for extra attractors

    # Penalize for missing attractors (if there are any missing)
    missing_attractors = len(target_attractors) - len(found_attractors)
    if missing_attractors > 0:
        error += missing_attractors * 10  # Strong penalty for missing attractors

    # Penalize sparse networks (too few edges or too many edges)
    edge_count = np.count_nonzero(W_candidate)
    
    # Reward or penalty for the number of edges being in the range of 15-25
    if 15 <= edge_count <= 25:
        edge_penalty = 0  # No penalty for edge count in this optimal range
    else:
        # Penalize for being outside the range of 15-25
        edge_penalty = abs(edge_count - 20)  # The penalty increases as it deviates from 20
        if edge_count < 15:
            edge_penalty *= 0.5  # Less severe penalty for fewer edges
        elif edge_count > 25:
            edge_penalty *= 1.5  # More severe penalty for too many edges

    # Add edge penalty to the error
    error += edge_penalty

    return -error  # Return negative error for minimization

# def fitness_function(params):
#     """Evaluate how well a candidate network replicates the target attractors while penalizing extra edges."""
#     W_candidate = np.array(params[:144]).reshape(12, 12).round().astype(int)
#     Theta_candidate = np.array(params[144:]).round().astype(int)

#     # Ensure matrix dimensions are correct
#     if W_candidate.shape != (12, 12) or Theta_candidate.shape != (12,):
#         return float('-inf')  # Invalid solution

#     network = BooleanNetwork(W_candidate, Theta_candidate)

#     # Validate attractors
#     error = 0
#     found_attractors = set()
#     for target_state in target_attractors:
#         initial_state = np.random.randint(0, 2, size=12)
#         final_state = network.simulate(initial_state)
#         error += sum(1 for i in range(12) if final_state[i] != target_state[i])
#         found_attractors.add(tuple(final_state))

#     # Penalize excessive edges
#     edge_count = np.count_nonzero(W_candidate)
#     edge_penalty = edge_count * 0.2

#     # Penalize extra attractors
#     extra_attractors = len(found_attractors) - len(target_attractors)
#     extra_attractor_penalty = 0.2 * extra_attractors

#     # Add additional penalties if the network doesn't match expected fixed points
#     if len(found_attractors) != len(target_attractors):
#         attractor_penalty = abs(len(found_attractors) - len(target_attractors)) * 10  # Strong penalty for missing or extra attractors
#     else:
#         attractor_penalty = 0

#     # Combine penalties (error, edge, extra attractor, attractor mismatch)
#     return -(error + edge_penalty + extra_attractor_penalty + attractor_penalty)

def calculate_basin_sizes(network, num_simulations=1000):
    """
    Simulate the network multiple times and track the basin sizes for each attractor.
    
    Args:
    network -- The BooleanNetwork instance
    num_simulations -- The number of random initial states to simulate (default 1000)
    
    Returns:
    basin_sizes -- A dictionary where the keys are attractors and the values are the basin sizes.
    """
    basin_sizes = Counter()  # To count how many times each attractor is reached
    for _ in range(num_simulations):
        initial_state = np.random.randint(0, 2, size=12)
        attractor = network.simulate(initial_state)
        basin_sizes[tuple(attractor)] += 1  # Count the attractor's occurrence
    
    return basin_sizes


def is_valid_network(W, Theta, expected_fixed_points, num_simulations=1000):
    """
    Check if a network has the correct number of fixed points and the correct basin sizes.
    
    Args:
    W -- Weight matrix of the network (12x12 matrix)
    Theta -- Threshold vector of the network (12,)
    expected_fixed_points -- List of expected fixed points (as tuples) for validation
    num_simulations -- Number of random initial states to simulate (default 1000)
    
    Returns:
    bool -- True if the network is valid, False otherwise
    """
    network = BooleanNetwork(W, Theta)
    
    # Calculate the basin sizes for each attractor
    basin_sizes = calculate_basin_sizes(network, num_simulations)
    
    # Check for expected fixed points
    found_attractors = set(basin_sizes.keys())
    correct_attractors = [a for a in found_attractors if a in expected_fixed_points]
    extra_attractors = [a for a in found_attractors if a not in expected_fixed_points]
    
    # If there are extra attractors, the network is invalid
    if extra_attractors:
        return False
    
    # Ensure the correct number of attractors
    if len(correct_attractors) != len(expected_fixed_points):
        print("Missing expected attractors.")
        return False
    
    # Now you can check if the basin sizes are evenly distributed or penalize networks with uneven basins
    basin_size_values = list(basin_sizes.values())
    avg_basin_size = sum(basin_size_values) / len(basin_size_values)  # Calculate the average basin size
    basin_imbalance_penalty = sum(abs(size - avg_basin_size) for size in basin_size_values)
    
    # If the basin sizes are too imbalanced, penalize this network (you can set a threshold for imbalance)
    if basin_imbalance_penalty > 100:  # Example threshold, can be adjusted
        print("Basin size imbalance detected:", basin_imbalance_penalty)
        return False
    
    return True

expected_fixed_points = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sepal
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],  # Petal
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Carpel
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Stamen
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Inflorescence
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]   # Mutant
]


def calculate_edge_counts(W):
    """Calculates total, positive, and negative edges in the weight matrix."""
    # Total edges: count non-zero values
    total_edges = np.count_nonzero(W)
    
    # Positive edges: count non-zero positive values
    positive_edges = np.count_nonzero(W > 0)
    
    # Negative edges: count non-zero negative values
    negative_edges = np.count_nonzero(W < 0)
    
    return total_edges, positive_edges, negative_edges

# Function to run multiple optimization runs sequentially
def run_optimization(n):
    best_score_overall = float('-inf')
    best_W_overall = None
    best_Theta_overall = None
    valid_networks = []
    all_total_edges = []
    all_positive_edges = []
    all_negative_edges = []

    for i in range(n):
        print(f"Optimization run {i+1} of {n}...")

        bee_optimizer = BeesAlgorithm(
            score_function=fitness_function,
            range_min=[-5] * 156,
            range_max=[5] * 156,
            ns=50,
            nb=20,
            ne=5,
            nrb=8,
            nre=15,
            stlim=10,
            initial_ngh=None,
            shrink_factor=0.2,
            useSimplifiedParameters=False
        )

        iterations, best_score = bee_optimizer.performFullOptimisation(max_iteration=200)
        best_params = bee_optimizer.best_solution.values
        best_W = np.array(best_params[:144]).reshape(12, 12)
        best_Theta = np.array(best_params[144:])
        
        # Calculate the edge counts for the best W found in this optimization run
        total_edges, positive_edges, negative_edges = calculate_edge_counts(best_W)
        
        # Store the edge counts for plotting
        all_total_edges.append(total_edges)
        all_positive_edges.append(positive_edges)
        all_negative_edges.append(negative_edges)

        edge_count = np.count_nonzero(best_W)

        if is_valid_network(best_W, best_Theta, expected_fixed_points):
            print("Added valid network")
            valid_networks.append((best_W, best_Theta))

        print(f"Current Best Score: {best_score_overall}")
        print(f"Current Best W: {best_W}")
        print(f"Current Best Theta: {best_Theta}")
        print(f"Edge Count in Current Best: {edge_count}")

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_W_overall = best_W
            best_Theta_overall = best_Theta

    return best_score_overall, best_W_overall, best_Theta_overall, valid_networks, all_total_edges, all_positive_edges, all_negative_edges

start = time.time()
# Run optimization n times sequentially
n = 10  # Set the number of optimization runs
best_score, best_W, best_Theta, valid_networks, all_total_edges, all_positive_edges, all_negative_edges = run_optimization(n)
end = time.time()

diff = end - start

print(f"Time (min): {diff / 60}")

print("Valid: ", valid_networks)
try:
    with open("runFiles/results/valid_networks.txt", "a") as f:
        for network in valid_networks:
            # Convert each network (W and Theta) to a string and write it to the file
            f.write(f"W:\n{network[0]}\nTheta:\n{network[1]}\n\n")
except Exception as e:
    print("Error: ", e)

try:
    with open("runFiles/results/best_networks.txt", "a") as f:
        f.write(f"W:\n{best_W}\nTheta: \n{best_Theta}\nNumber of Edges: {np.count_nonzero(best_W)}\n\n")
except Exception as e:
    print(f"Error: {e}")

# Visualize histograms for edge distributions across all iterations
plt.figure(figsize=(18, 5))

# Histogram for total number of edges
plt.subplot(1, 3, 1)
plt.hist(all_total_edges, bins=20, color='blue', alpha=0.7)
plt.xlabel("Total Number of Edges")
plt.ylabel("Frequency")
plt.title("Total Number of Edges")

# Histogram for positive edges
plt.subplot(1, 3, 2)
plt.hist(all_positive_edges, bins=10, color='green', alpha=0.7)
plt.xlabel("Positive Edges")
plt.ylabel("Frequency")
plt.title("Positive Edges")

# Histogram for negative edges
plt.subplot(1, 3, 3)
plt.hist(all_negative_edges, bins=10, color='red', alpha=0.7)
plt.xlabel("Negative Edges")
plt.ylabel("Frequency")
plt.title("Negative Edges")

# Adjust layout for better spacing
plt.tight_layout()

# Show the histograms
plt.show()

# Evaluate results for the best network found
best_network_model = BooleanNetwork(best_W, best_Theta)

# Visualize the basin sizes
basin_sizes = calculate_basin_sizes(best_network_model)
attractor_labels = [str(k) for k in basin_sizes.keys()]
plt.bar(attractor_labels, basin_sizes.values())
plt.xlabel('Attractors')
plt.ylabel('Basin Size')
plt.title('Attractor Basin Sizes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()