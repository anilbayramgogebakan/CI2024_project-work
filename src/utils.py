import numpy as np
import random

def import_prova():
    print("Pass")

def tournament_selection(fitness, n):
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

# Collect all nodes in the tree
def collect_nodes(n, nodes):
    if n is None:
        return
    nodes.append(n)
    collect_nodes(n.left, nodes)
    collect_nodes(n.right, nodes)

def modify_random_node(node, feature_count, max_constant=10):
    """
    Randomly modifies a node's value or feature_index in the tree.
    
    Args:
        node (Node): The root of the tree.
        feature_count (int): The number of input features (to determine valid feature indices).
        max_constant (float): The maximum absolute value for random constants.

    Returns:
        bool: True if a node was modified, False otherwise.
    """

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
            target_node.value = random.uniform(-max_constant, max_constant) #TODO: check the initiliazation of the constant
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
                target_node.value = random.uniform(-max_constant, max_constant) #TODO: check the initiliazation of the constant
            else:
                # Replace the constant value with a feature
                target_node.value = None
                target_node.feature_index = random.randint(0, feature_count - 1)
    return True

def crossover():
    pass