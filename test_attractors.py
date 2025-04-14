
import random
import numpy as np
from bees_algorithm import BeesAlgorithm
from collections import Counter
import matplotlib.pyplot as plt
import time
import benchmark_functions as bf

class AttractorBooleanNetwork:
    def __init__(self, W, Theta):
        self.W = np.array(W)  # Convert to NumPy array for speed
        self.Theta = np.array(Theta)  # Convert to NumPy array
        self.n = len(W)  # Number of nodes
    
    def update(self, state):
        """Update state using parallel synchronous rule."""
        weighted_sum = np.dot(self.W, state)
        binary = (weighted_sum > self.Theta).astype(int)
        return np.clip(binary, 0, 1)  # Just in case
    
    def simulate(self, initial_state, max_steps=500):
        """Simulate network and detect fixed points or cycles."""
        state = np.array(initial_state)
        history = []

        for step in range(max_steps):
            history.append(tuple(state))
            next_state = self.update(state)

            if np.array_equal(next_state, state):
                return tuple(state), step + 1  # Fixed point

            if tuple(next_state) in history:
                cycle_start = history.index(tuple(next_state))
                cycle = history[cycle_start:]
                return tuple(cycle), step + 1  # Cycle detected

            state = next_state

        return None, max_steps  # No attractor found within max steps

# Predefined target attractors (fixed points of the original model)
target_attractors = [
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

def fitness_function(params):
    W_candidate = np.array(params[:144]).reshape(12, 12).round().astype(int)
    Theta_candidate = np.array(params[144:]).round().astype(int)

    # Ensure matrix dimensions are correct
    if W_candidate.shape != (12, 12) or Theta_candidate.shape != (12,):
        return float('-inf')  # Invalid solution

    network = AttractorBooleanNetwork(W_candidate, Theta_candidate)

    error = 0

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

expected_fixed_points = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sepal
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],  # Petal
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Carpel
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Stamen
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Inflorescence
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]   # Mutant
]

# Function to run multiple optimization runs sequentially
def run_optimization(n):
    best_score_overall = float('-inf')
    best_W_overall = None
    best_Theta_overall = None
    valid_networks = []
    all_total_edges = []
    all_positive_edges = []
    all_negative_edges = []

    
    func = bf.Schwefel(n_dimensions=2)
    lb, ub = func.suggested_bounds()

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

        iterations, best_score = bee_optimizer.performFullOptimisation(max_iteration=1000, verbose=1)
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
n = 1  # Set the number of optimization runs
best_score, best_W, best_Theta, valid_networks, all_total_edges, all_positive_edges, all_negative_edges = run_optimization(n)
end = time.time()

diff = end - start

print(f"Time (min): {diff / 60}")


def find_attractors(network, max_steps=500, sample_size=None):
    attractors = {}
    num_nodes = network.n

    # All possible initial binary states (or random sample)
    if sample_size:
        initial_states = [np.random.randint(0, 2, num_nodes).tolist() for _ in range(sample_size)]
    else:
        initial_states = [list(map(int, format(i, f'0{num_nodes}b'))) for i in range(2 ** num_nodes)]

    print(initial_states)

    for state in initial_states:
        attractor, steps = network.simulate(state, max_steps=max_steps)
        if attractor is not None:
            attractors.setdefault(attractor, []).append(steps)

    return attractors

# Create final Boolean network with the best W and Theta
final_network = AttractorBooleanNetwork(best_W, best_Theta)

# Use the full space (4096) or a sample (e.g., 1000)
attractors_dict = find_attractors(final_network, max_steps=500, sample_size=1000)
print(attractors_dict)

# Extract attractor counts and steps
attractor_labels = []
attractor_counts = []
steps_to_attractor = []

for attractor, steps_list in attractors_dict.items():
    label = ''.join(map(str, attractor))  # Convert tuple to binary string
    attractor_labels.append(label)
    attractor_counts.append(len(steps_list))
    steps_to_attractor.extend(steps_list)  # Collect all steps for histogram
    
# Bar chart of attractor frequencies
plt.figure(figsize=(10, 5))
plt.bar(attractor_labels, attractor_counts)
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Attractors")
plt.xlabel("Attractors (Binary States)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Histogram of steps to reach attractors
plt.figure(figsize=(8, 5))
plt.hist(steps_to_attractor, bins=20, edgecolor='black')
plt.title("Steps to Reach Attractors")
plt.xlabel("Steps")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

