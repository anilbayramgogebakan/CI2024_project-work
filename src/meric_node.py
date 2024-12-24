import numpy as np
import random
import copy
import warnings
from src.node import Node
from src.Individual import Individual

operators=[np.add, np.subtract, np.multiply, np.sin, np.cos, np.exp, np.abs]
one_arg_op=[np.sin, np.cos, np.exp, np.abs]

def import_prova():
    print("Pass")


def tournament_selection(population, n, k):
    """
    Perform tournament selection on a population of individuals.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    - n (int): Number of individuals to randomly select for each comparison.
    - k (int): Number of winners to select.

    Returns:
    - list of Individual: The selected individuals.
    """
    # Ensure all individuals have their fitness calculated
    if any(ind.fitness is None for ind in population):
        raise ValueError("All individuals must have a fitness value assigned before tournament selection.")

    # Create an index list for the population
    indices = list(range(len(population)))

    while len(indices) >= k:
        # Randomly select `n` indices for the tournament
        selected_indices = np.random.choice(indices, n, replace=False)
        
        # Find the index of the individual with the best (lowest) fitness in the selected group
        best_idx = selected_indices[np.argmin([population[i].fitness for i in selected_indices])]

        # Remove all other indices except the winner of the tournament
        for idx in selected_indices:
            if idx != best_idx:
                indices.remove(idx)

    # Return the selected individuals
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

    child = Individual(genome=copy.deepcopy(individual.genome))
    node = child.genome
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
    return child


def crossover(parent1, parent2):
    """
    Perform crossover between two parent individuals.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.

    Returns:
        tuple: Two offspring individuals (child1, child2).
    """
    import copy
    # Create new children from parents
    child1 = Individual(genome=copy.deepcopy(parent1.genome))
    child2 = Individual(genome=copy.deepcopy(parent2.genome))

    genome1 = child1.genome
    genome2 = child2.genome

    # Collect all nodes in the genomes
    nodes1 = []
    collect_nodes(genome1, nodes1)
    nodes2 = []
    collect_nodes(genome2, nodes2)

    # Randomly pick a node from each parent
    if not nodes1 or not nodes2:
        return child1, child2  # No crossover occurs if any parent has no nodes

    target_node1 = random.choice(nodes1)
    target_node2 = random.choice(nodes2)

    # Swap the nodes between the two genomes
    target_node1.value, target_node2.value = target_node2.value, target_node1.value
    target_node1.feature_index, target_node2.feature_index = target_node2.feature_index, target_node1.feature_index
    target_node1.left, target_node2.left = target_node2.left, target_node1.left
    target_node1.right, target_node2.right = target_node2.right, target_node1.right

    return child1, child2


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
        baby_node=random_tree(1,num_features)
        baby = Individual(genome=baby_node)
        population.append(baby)
    for i in range(num_peop-num_ones):
        baby_node=random_tree(depth,num_features)
        baby = Individual(genome=baby_node)
        population.append(baby)
    return population

def cost(genome,x,y):
    predictions = np.array([genome.evaluate(x[:, i]) for i in range(x.shape[1])])
    mse = np.mean((predictions - y) ** 2)
    return mse

def assign_population_fitness(population, x, y):
    """
    Calculate and assign fitness values to the population.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    - x (np.ndarray): Input data.
    - y (np.ndarray): Target data.
    """
    removed_indices = []
    for i, individual in enumerate(population):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if individual.fitness is None:
                ind_cost = cost(individual.genome, x, y)
                if len(w) == 0:
                    individual.fitness = ind_cost
                else:
                    removed_indices.append(i)
                    
    # Remove invalid individuals
    for idx in sorted(removed_indices, reverse=True):
        del population[idx]

def age_population(population):
    """
    Increment the age of all individuals in the population.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    """
    for individual in population:
        individual.age += 1

def kill_eldest(population, max_age):
    """
    Remove the eldest individuals from the population.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    - max_age (int): The maximum age an individual can reach before being removed.
    """
    population[:] = [ind for ind in population if ind.age <= max_age]


def top_n_individuals(population, n):
    return sorted(population, key=lambda x: x.fitness)[:n]

# TODO requires update to use assign_population_fitness
def migration(population_1,population_2,num_peop,num_rand_peop):
    best_1 = tournament_selection(population_1,num_rand_peop,num_peop)
    best_2 = tournament_selection(population_2,num_rand_peop,num_peop)
    for i in best_1:
        population_1.remove(i)
        population_2.append(i)
    for i in best_2:
        population_2.remove(i)
        population_1.append(i)

def mutation_w_sa(individual, feature_count,x,y,alpha=0.95):
    child = mutation(individual, feature_count)
    with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ind_cost = cost(child.genome,x,y)
            if len(w) == 0:
                child.fitness = ind_cost
                if child.fitness > individual.fitness:
                    individual.T *=alpha
                    return child, True
                else: 
                    p= np.exp((child.fitness-individual.fitness)/(alpha*individual.T))
                    if np.random.random() < p:
                        return None, False
                    else:
                        individual.T *=alpha
                        return child, True
            else: 
                return None, False