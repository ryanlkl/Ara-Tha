from bees_algorithm import BeesAlgorithm
import random

# Define the Boolean network update function without NumPy
def boolean_update(state, W, Theta):
    # Perform the dot product manually and compare it to Theta
    return [1 if sum(w * s for w, s in zip(row, state)) - theta > 0 else 0 for row, theta in zip(W, Theta)]

# Initialize the original Mendoza & Alvarez-Buylla network as lists
W_original = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2, -1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 5, 0, 0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, -2, 1, -2, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 0, -2],
    [0, 0, 4, 0, 0, 0, 1, 1, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

Theta_original = [0, 0, 3, -1, 1, 0, 0, 1, -1, 0, 0, 0]

# Define the fitness function without NumPy
def fitness_function(candidate):
    W_candidate = [candidate[i:i + len(W_original[0])] for i in range(0, len(candidate), len(W_original[0]))]  # Reshape into a 2D list
    errors = 0
    for _ in range(100):  # Sample 100 random initial states
        state = [random.randint(0, 1) for _ in range(len(W_original))]
        original_attractor = boolean_update(state, W_original, Theta_original)
        candidate_attractor = boolean_update(state, W_candidate, Theta_original)
        errors += sum(o != c for o, c in zip(original_attractor, candidate_attractor))  # Sum differences
    return errors

# Set up Bees Algorithm optimization with plain lists
n_var = len(W_original) * len(W_original[0])
lower_bound = [-5] * n_var
upper_bound = [5] * n_var

# Bees Algorithm setup (keeping parameters as they are)
bees = BeesAlgorithm(
    score_function=fitness_function,
    range_min=lower_bound,
    range_max=upper_bound,
    ns=20,
    ne=5,
    nb=10,
    nre=5,
    nrb=3,
)

bees.visualize_iteration_steps()

# # Run the optimization
# bees.performFullOptimisation(max_iteration=1000)
# best = bees.best_solution
# best_solution = best.score
# best_coords = best.values

# # Output results
# print(best_solution)
# print(best_coords)
