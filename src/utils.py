import numpy as np
import random
import copy
import warnings
from dataclasses import dataclass
from MyNode import Node

operators=[np.add, np.subtract, np.multiply, np.sin, np.cos, np.exp]
one_arg_op=[np.sin, np.cos, np.exp]

def import_prova():
    print("Pass")

def best_worst_subset(fitness, n):
    """
    Perform tournament selection
    
    Args:
        fitness (array-like): Fitness values of the population.
        n (int): Subset size of winners and losers.
        
    Returns:
        best_n (list): Indices of best n individuals.
        worst_n (list): Indices of worst n individuals
    """
    sorted_indices = np.argsort(fitness)
    best_n = sorted_indices[:n]
    worst_n = sorted_indices[-n:][::-1]
    return best_n, worst_n

def tournament_selection(arr, n, k, sorted=False):
    """
    Select k winners from the array by repeatedly comparing random n elements.
    
    Parameters:
    - arr (np.ndarray): The input array.
    - n (int): Number of elements to randomly select for each comparison.
    - k (int): Number of winners to select.
    
    Returns:
    - np.ndarray: Array of the k largest winners.
    """
    
    # Initialize list of indices
    indices = list(range(len(arr)))
    
    while len(indices) > k:
        # Randomly select `n` indices
        selected_indices = np.random.choice(indices, n, replace=False)
        
        # Find the index of the maximum among the selected
        max_idx = selected_indices[np.argmax(arr[selected_indices])]
        
        # Remove all selected indices except the winner
        for idx in selected_indices:
            if idx != max_idx:
                indices.remove(idx)

    return indices

# Collect all nodes in the tree
def collect_nodes(n, nodes):
    if n is None:
        return
    nodes.append(n)
    collect_nodes(n.left, nodes)
    collect_nodes(n.right, nodes)

def mutation(individual, feature_count):
    """
    Randomly modifies a node's value or feature_index in the tree.
    
    Args:
        node (Node): The root of the tree.
        feature_count (int): The number of input features (to determine valid feature indices).
        max_constant (float): The maximum absolute value for random constants.

    Returns:
        bool: True if a node was modified, False otherwise.
    """
    node = copy.deepcopy(individual)
    nodes = []
    collect_nodes(node, nodes)

    # Randomly pick a node
    if not nodes:
        return False

    target_node = random.choice(nodes)

    # Modify the target node
    if target_node.feature_index is not None: # Node is Xn
        if random.random() < 0.5:
        # Modify the feature index
            target_node.feature_index = random.randint(0, feature_count - 1) #TODO: find a better way to exclude existing feature index
        else:
            # Assign constant value to feature node
            target_node.feature_index = None
            target_node.value = np.random.normal(0,1,1)
    else:
    # Modify the operator or constant
        if target_node.value in operators:
            if target_node.value in one_arg_op: # If the operator is one-argument, pick another one-argument operator
                target_node.value = np.random.choice([op for op in one_arg_op if op != target_node.value])
            else: # If the operator is two-argument, pick another two-argument operator
                target_node.value = np.random.choice([op for op in set(operators)-set(one_arg_op) if op != target_node.value])
        else: # If the node is a constant, assign a new constant value
            if random.random() < 0.5:
                # Replace the constant value with constant value
                target_node.value = np.random.normal(0,1,1)
            else:
                # Replace the constant value with a feature
                target_node.value = None
                target_node.feature_index = random.randint(0, feature_count - 1)
    return node


def crossover(parent1, parent2):
    """
    Perform crossover between two parents.
    
    Args:
        parent1 (Node): First parent.
        parent2 (Node): Second parent.
        
    Returns:
        child1 (Node): First child.
        child2 (Node): Second child.
    """
    # Copy parent to create new children
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Collect all nodes in the trees
    nodes1 = []
    collect_nodes(child1, nodes1)
    nodes2 = []
    collect_nodes(child2, nodes2)
    
    # Randomly pick a node from each parent
    if not nodes1 or not nodes2:
        return None
    target_node1 = random.choice(nodes1)
    target_node2 = random.choice(nodes2)

    # Copy the target node from parent1
    copy_target_node1 = copy.deepcopy(target_node1)

    # Replace the target node with the target node from parent2
    target_node1.value = target_node2.value
    target_node1.feature_index = target_node2.feature_index
    target_node1.left = target_node2.left
    target_node1.right = target_node2.right

    # Replace the target node with the target node from parent1
    target_node2.value = copy_target_node1.value
    target_node2.feature_index = copy_target_node1.feature_index
    target_node2.left = copy_target_node1.left
    target_node2.right = copy_target_node1.right
    
    return child1, child2

# def simplify(population, x, y):
#     """
#     Simply population by removing overflowed individuals.

#     Args:
#         population (list): List of individuals.
#         x (array-like): Input data.

#     Returns:
#         simplified_population (list): List of simplified individuals.
#     """
#     simplified_population = []
#     for individual in population:
#         with warnings.catch_warnings(record=True)as w:
#             warnings.simplefilter("always")
#             cost(individual, x, y)
#             if len(w) == 0:
#                 simplified_population.append(individual)
#             else:
#                 pass
#     return simplified_population

def random_tree(depth, num_features):
    if depth == 0:
        if random.random() < 0.5:
            return Node(feature_index=random.randint(0, num_features - 1))
        else:
            return Node(value=np.random.normal(0,1,1))

    operator = random.choice(operators)
    node = Node(value=operator)

    if operator in one_arg_op:
        node.left = random_tree(depth - 1, num_features)
        node.right = None
    else:
        node.left = random_tree(depth - 1, num_features)
        node.right = random_tree(depth - 1, num_features)

    return node

def create_population(num_peop,depth,num_features):
    population = []
    num_ones = num_peop//2
    for i in range(num_ones):
        baby=random_tree(1,num_features)
        population.append(baby)
    for i in range(num_peop-num_ones):
        baby=random_tree(depth,num_features)
        population.append(baby)
    return population

def cost(genome,x,y):
    predictions = np.array([genome.evaluate(x[:, i]) for i in range(x.shape[1])])
    mse = np.mean((predictions - y) ** 2)
    return mse

# def cost_population(population, x, y):
#     costs = np.array([cost(population[j],x,y) for j in range(len(population))])
#     return costs

def cost_population(population, x, y):
    cost_list = []
    removed_el = []
    for ind in range(len(population)):
        with warnings.catch_warnings(record=True)as w:
            warnings.simplefilter("always")
            ind_cost = cost(population[ind], x, y)
            if len(w) == 0:
                cost_list.append(ind_cost)
            else:
                removed_el.append(ind)
    for ind in sorted(removed_el, reverse=True):
        del population[ind]
            
    return np.asarray(cost_list)

def migration(population_1,population_2,num_peop,x,y):
    costs_1 = cost_population(population_1,x,y)
    costs_2 = cost_population(population_2,x,y)
    best_1, worst_1 = tournament_selection(costs_1,num_peop)
    best_2, worst_2 = tournament_selection(costs_2,num_peop)
    elements_1 = [population_1[i] for i in best_1]
    elements_2 = [population_2[i] for i in best_2]
    for i, idx in enumerate(best_1):
        population_1[idx] = elements_2[i]
    for i, idx in enumerate(best_2):
        population_2[idx] = elements_1[i]
    return population_1,population_2
