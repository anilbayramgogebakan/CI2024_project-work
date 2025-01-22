import numpy as np
import random
import copy
import warnings
from tqdm import tqdm
from node import Node
from Individual import Individual

unary_operators=[np.sin, np.cos, np.exp, np.abs, np.log, np.tan]
binary_operators=[np.add, np.subtract, np.multiply, np.divide]


# unary_operators=[]
# binary_operators=[np.add, np.subtract, np.multiply, np.divide]
operators = unary_operators + binary_operators

def import_prova():
    print("Pass")


def split_dataset(x,y,split_ratio=0.8):
    train_len = int(x.shape[1]*split_ratio)
    train_x = x[:,:train_len]
    val_x = x[:,train_len:]
    train_y = y[:train_len]
    val_y = y[train_len:]
    return train_x, train_y, val_x, val_y

def tournament_selection(population, n, k, ELITISM=False, elite_count=3):
    """
    Perform tournament selection on a population of individuals.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    - n (int): Number of individuals to randomly select for each comparison.
    - k (int): Number of winners to select.

    Returns:
    - list of indexes: The indexes of selected individuals.
    """
    # TODO update might be required for elitism option because of new configuration of fitness
    # Ensure all individuals have their fitness calculated
    if any(ind.fitness is None for ind in population):
        raise ValueError("All individuals must have a fitness value assigned before tournament selection.")

    # Create an index list for the population
    indices = list(range(len(population)))

    if not ELITISM:
        elite_count=0
        
    while len(indices) >= k + elite_count:
        # Randomly select `n` indices for the tournament
        selected_indices = np.random.choice(indices, n, replace=False)
        
        # Find the index of the individual with the best (lowest) fitness in the selected group
        best_idx = selected_indices[np.argmin([population[i].fitness*(population[i].genome.complexity*0 +1)for i in selected_indices])]

        # Remove all other indices except the winner of the tournament
        for idx in sorted(selected_indices, reverse=True):
            if idx != best_idx:
                indices.remove(idx)

    if ELITISM:
        fitness_list = [population[i].fitness for i in indices]
        best_indices = np.argsort(fitness_list)[:elite_count]
        for idx in sorted(best_indices, reverse=True):
            indices.remove(indices[idx])

    # Return the selected individuals
    return indices


# Collect all nodes in the tree
def collect_nodes(n, nodes):
    if n is None:
        return
    nodes.append(n)
    collect_nodes(n.left, nodes)
    collect_nodes(n.right, nodes)

def mutation(individual, feature_count, ONLY_CONSTANT=False): # TODO: p values should be configurable
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
    if target_node.feature_index is not None and ONLY_CONSTANT==False: # Node is Xn
        if random.random() < 0.5 and feature_count>1:
        # Modify the feature index
            target_node.feature_index = np.random.choice([i for i in range(feature_count) if i != target_node.feature_index])
            # target_node.feature_index = random.randint(0, feature_count - 1) #TODO: find a better way to exclude existing feature index
        else:
            # Assign constant value to feature node
            target_node.feature_index = None
            target_node.value = np.random.normal(0,1,1)
    else:
    # Modify the operator or constant
        if target_node.value in operators: # If the node is an operator
            if ONLY_CONSTANT==False:
                if random.random() < 0.6: # Replace the operator with a constant or feature
                    if random.random() < 0.8: # Replace the operator with a constant
                        target_node.value = np.random.normal(0,1,1)
                        target_node.left = None
                        target_node.right = None
                    else: # Replace the operator with a feature
                        target_node.value = None
                        target_node.left = None
                        target_node.right = None
                        target_node.feature_index = random.randint(0, feature_count - 1)
                else: # Replace the operator with another operator
                    if target_node.value in unary_operators: # If the operator is one-argument, pick another one-argument operator
                        target_node.value = np.random.choice([op for op in unary_operators if op != target_node.value])
                    else: # If the operator is two-argument, pick another two-argument operator
                        target_node.value = np.random.choice([op for op in set(operators)-set(unary_operators) if op != target_node.value])
        else: # If the node is a constant, assign a new constant value
            if random.random() < 0.8 or ONLY_CONSTANT==True:
                # Replace the constant value with constant value
                target_node.value = np.random.normal(0,1,1)
            else:
                # Replace the constant value with a feature
                target_node.value = None
                target_node.feature_index = random.randint(0, feature_count - 1)
    child.genome.update_complexity()
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
    
    child1.genome.update_complexity()
    child2.genome.update_complexity()
    parent1.genome.update_complexity()
    parent2.genome.update_complexity()
    return child1, child2



def random_tree(depth, num_features, init_unary=unary_operators, init_binary=binary_operators):
    if depth == 0:
        if random.random() < 0.5:
            return Node(feature_index=random.randint(0, num_features - 1))
        else:
            return Node(value=np.random.normal(0,1,1))

    operators = init_binary + init_unary
    operator = random.choice(operators)
    node = Node(value=operator)

    if operator in init_unary:
        node.left = random_tree(depth - 1, num_features, init_unary, init_binary)
        node.right = None
    else:
        node.left = random_tree(depth - 1, num_features, init_unary, init_binary)
        node.right = random_tree(depth - 1, num_features, init_unary, init_binary)

    return node

def create_population(num_peop,depth,num_features,init_unary=unary_operators, init_binary=binary_operators):
    population = []
    num_ones = num_peop//2
    for i in range(num_ones):
        baby_node=random_tree(1, num_features, init_unary, init_binary)
        baby = Individual(genome=baby_node)
        population.append(baby)
    for i in range(num_peop-num_ones):
        baby_node=random_tree(depth, num_features, init_unary, init_binary)
        baby = Individual(genome=baby_node)
        population.append(baby)
    return population

def cost(genome,x,y):
    predictions = np.array([genome.evaluate(x[:, i]) for i in range(x.shape[1])])
    mse = np.mean((predictions - y) ** 2)
    return mse

def assign_population_fitness_train(population, x, y):
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
        
def assign_population_fitness_val(population, x, y):
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

            if individual.fitness_val is None:
                ind_cost = cost(individual.genome, x, y)
                if len(w) == 0:
                    individual.fitness_val = ind_cost
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

def kill_constant(population):
    """
    Remove the constant individuals from the population.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    """
    
    population[:] = [ind for ind in population if not (ind.genome.complexity==1 and ind.genome.feature_index==None)]

def kill_complex(population, max_complexity):
    """
    Remove the complex individuals from the population.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    - max_complexity (int): The maximum complexity an individual can reach before being removed.
    """
    population[:] = [ind for ind in population if ind.genome.complexity <= max_complexity]

# def top_n_individuals(population, n):
#     return sorted(population, key=lambda x: x.fitness_val)[:n]

def top_n_individuals(population, n):
    """
    Select the top n unique individuals from the population based on fitness value.

    Parameters:
    - population (list of Individual): The population of individuals.
    - n (int): The number of unique top individuals to return.

    Returns:
    - list of Individual: The top n unique individuals based on fitness.
    """
    # Use a dictionary to keep the first occurrence of each unique fitness value
    unique_best_individuals = []
    best_inv_fitness_val = []

    for ind in sorted(population, key=lambda x: x.fitness_val):
        if ind.fitness_val not in best_inv_fitness_val:
            unique_best_individuals.append(ind)
            best_inv_fitness_val.append(ind.fitness_val)
        if len(unique_best_individuals) == n:
            break

    # Return the top n unique individuals
    return unique_best_individuals


def calculate_mean_fitness(population):
    return np.mean([ind.fitness for ind in population])

def calculate_mean_complexity(population):
    return np.mean([ind.genome.complexity for ind in population])

def deduplicate_population(population):
    """
    Remove duplicate individuals from the population based on fitness and age.
    For duplicate fitness values, keep the individual with the minimum age.
    
    Parameters:
    - population (list of Individual): The population of individuals.
    
    Returns:
    - deduplicated (list of Individual): The deduplicated population.
    """
    # Dictionary to track unique fitness values and the individual with the minimum age
    fitness_to_individual = {}

    for ind in population:
        if ind.fitness not in fitness_to_individual:
            # Add if the fitness value is not already seen
            fitness_to_individual[ind.fitness] = ind
        else:
            # Replace if the current individual has a lower age
            if ind.age < fitness_to_individual[ind.fitness].age:
                fitness_to_individual[ind.fitness] = ind

    # Return the deduplicated list of individuals
    return list(fitness_to_individual.values())

# TODO requires update to use assign_population_fitness
def migration(population_1,population_2,num_peop=20):
    best_1_ind = tournament_selection(population_1,3,num_peop,ELITISM=False, elite_count=None)
    best_2_ind = tournament_selection(population_2,3,num_peop,ELITISM=False, elite_count=None)

    elites_1 = [population_1[idx] for idx in best_1_ind]
    elites_2 = [population_2[idx] for idx in best_2_ind]

    population_1 = [normal for normal in population_1 if normal not in elites_1]
    population_2 = [normal for normal in population_2 if normal not in elites_2]

    new_pop_1 = elites_2 + population_1
    new_pop_2 = elites_1 + population_2

    return new_pop_1, new_pop_2

def mutation_w_sa(individual, feature_count,x,y, ONLY_CONSTANT=False, alpha=0.95):
    child = mutation(individual, feature_count, ONLY_CONSTANT)
    with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ind_cost = cost(child.genome,x,y)
            if len(w) == 0:
                child.fitness = ind_cost
                if child.fitness < individual.fitness:
                    individual.T *=alpha
                    return child, True
                else: 
                    p= np.exp((individual.fitness-child.fitness)/(alpha*individual.T))
                    if np.random.random() < p:
                        return None, False
                    else:
                        individual.T *=alpha
                        return child, True
            else: 
                return None, False

def fit_constants(individual, iter,x, y):
    """
    Fit the constant value of a node to minimize the cost function.
    
    Args:
        node (Node): The node to fit.
        iter (int): The number of iterations to run the optimization.
        x (np+.ndarray): Input data.
        y (np.ndarray): Target data.
    """

    for _ in range(iter):
        if individual.genome.complexity != 1:
            child, success = mutation_w_sa(individual, x.shape[1],x,y, ONLY_CONSTANT=True)
            if success:
                child.T = individual.T
                child.age = individual.age//2
                individual = child                
    return individual

def simplify_constant_population(population):
    for i in range(len(population)):
        gen = population[i].genome
        simplify_constant(gen)
        gen.update_complexity()
               
def simplify_constant(gen):
    try:
        if gen.left:
            simplify_constant(gen.left)
        if gen.right:
            simplify_constant(gen.right)

        if gen.right!=None:
            if isinstance(gen.left.value, np.ndarray) and isinstance(gen.right.value, np.ndarray):
                gen.value=gen.evaluate()
                gen.right=None
                gen.left=None
    except:
        print("gen: ", gen)
        print("gen.left: ", gen.left)
        print("gen.right: ", gen.right)
        print("gen.feature_index: ", gen.feature_index )

def simplify_operation_population(population):
    for i in range(len(population)):
        gen = population[i].genome
        simplify_operation(gen)
        gen.update_complexity()
        
def simplify_operation(gen):
    try: 
        if gen.left:
            simplify_operation(gen.left)
        if gen.right:
            simplify_operation(gen.right)

        if gen.right!=None:
            if isinstance(gen.left.value, np.ndarray) and isinstance(gen.right.value, np.ndarray):
                gen.value=gen.evaluate()
                gen.right=None
                gen.left=None
        elif gen.left!=None:    # unary operator
            if isinstance(gen.left.value, np.ndarray):
                gen.value=gen.evaluate()
                gen.left=None
            elif gen.left.value==np.abs and gen.value==np.abs:
                gen.left = gen.left.left
    except:
        print("gen: ", gen)
        print("gen.left: ", gen.left)
        print("gen.right: ", gen.right)
        print("gen.feature_index: ", gen.feature_index )

def simplify_population(population):
    for individual in population:
        # Attempt to simplify the genome
        existing_genome = copy.deepcopy(individual.genome)
        simplified_genome = simplify_tree_with_multiplication_and_division(individual.genome)
        if simplified_genome is None:
            # Retain the original genome if simplification fails
            print(f"No simplification applied for genome: {individual.genome}")
            simplified_genome = individual.genome
        else:
            print(f"Simplified genome: {simplified_genome}")
            print(f"Original genome: {existing_genome}")
        # Update the individual's genome
        simplified_genome.update_complexity()
        individual.genome = simplified_genome

def simplify_tree_with_multiplication_and_division(node):
    try:
        # Simplify arithmetic terms
        terms = collect_terms_with_multiplication_division(node)
        simplified_tree = rebuild_tree_with_multiplication_division(terms)
        return simplified_tree
    except ValueError as e:
        print(f"Error during simplification: {e}")
        return node  # Return original node if simplification fails
    
    
def collect_terms_with_multiplication_division(node, terms=None, coefficient=1):
    if terms is None:
        terms = {}

    if node is None:
        return terms

    # If the node represents a variable (e.g., x[1])
    if node.feature_index is not None:
        key = f"x[{node.feature_index}]"
        terms[key] = terms.get(key, 0) + coefficient
        return terms

    # If the node represents a constant
    if node.value is not None and not isinstance(node.value, np.ufunc):
        terms["constant"] = terms.get("constant", 0) + coefficient * node.value
        return terms

    # Handle unary operations
    if node.value in Node.unary_operators:
        # First simplify the inner expression
        inner_terms = collect_terms_with_multiplication_division(node.left, {}, 1)
        inner_node = rebuild_tree_with_multiplication_division(inner_terms)
        if inner_node:
            # Create a new key for the simplified unary operation
            key = f"{Node.op_list[node.value]}({inner_node})"
            terms[key] = terms.get(key, 0) + coefficient
        return terms

    # Handle addition and subtraction
    if node.value == np.add:
        collect_terms_with_multiplication_division(node.left, terms, coefficient)
        collect_terms_with_multiplication_division(node.right, terms, coefficient)
    elif node.value == np.subtract:
        collect_terms_with_multiplication_division(node.left, terms, coefficient)
        collect_terms_with_multiplication_division(node.right, terms, -coefficient)

    # Handle multiplication
    elif node.value == np.multiply:
        if node.left and node.right:
            if node.left.value is not None and not node.left.is_operator(node.left.value):
                # Left is a constant
                new_coefficient = coefficient * node.left.value
                collect_terms_with_multiplication_division(node.right, terms, new_coefficient)
            elif node.right.value is not None and not node.right.is_operator(node.right.value):
                # Right is a constant
                new_coefficient = coefficient * node.right.value
                collect_terms_with_multiplication_division(node.left, terms, new_coefficient)
            else:
                # Cannot simplify further, leave as is
                key = f"({node.left} * {node.right})"
                terms[key] = terms.get(key, 0) + coefficient

    # Handle division
    elif node.value == np.divide:
        if node.left and node.right:
            # Case 1: division by a constant
            if node.right.value is not None and not node.right.is_operator(node.right.value) and node.right.value != 0:
                new_coefficient = coefficient / node.right.value
                collect_terms_with_multiplication_division(node.left, terms, new_coefficient)
            
            # Case 2: cancellation of variables (e.g., (x[1] * x[2]) / x[1] -> x[2])
            elif node.left.value == np.multiply and node.right.feature_index is not None:
                left_mult = node.left
                denominator_index = node.right.feature_index
                
                # Check if either the left or right part of multiplication matches the denominator
                if left_mult.left.feature_index == denominator_index:
                    # Cancel out the matching term and keep the other term
                    collect_terms_with_multiplication_division(left_mult.right, terms, coefficient)
                elif left_mult.right.feature_index == denominator_index:
                    # Cancel out the matching term and keep the other term
                    collect_terms_with_multiplication_division(left_mult.left, terms, coefficient)
                else:
                    # No cancellation possible
                    key = f"({node.left} / {node.right})"
                    terms[key] = terms.get(key, 0) + coefficient
            else:
                # Cannot simplify further
                key = f"({node.left} / {node.right})"
                terms[key] = terms.get(key, 0) + coefficient

    return terms

def rebuild_tree_with_multiplication_division(terms):
    if not terms:
        return None

    root = None
    
    # Handle constant terms
    if 'constant' in terms and terms['constant'] != 0:
        root = Node(value=terms.pop('constant'))

    # Handle variable terms (x[i])
    var_terms = {k: v for k, v in terms.items() if k.startswith('x[')}
    for var_key, coefficient in sorted(var_terms.items()):
        if coefficient == 0:
            continue
        
        feature_index = int(var_key[2:-1])
        if coefficient == 1:
            term_node = Node(feature_index=feature_index)
        else:
            term_node = Node(
                value=np.multiply,
                left=Node(value=coefficient),
                right=Node(feature_index=feature_index)
            )
            
        if root is None:
            root = term_node
        else:
            root = Node(value=np.add, left=root, right=term_node)

    # Handle unary operation terms
    unary_terms = {k: v for k, v in terms.items() if any(op in k for op in ['sin', 'cos', 'exp', 'log', 'tan', 'abs'])}
    for unary_key, coefficient in unary_terms.items():
        if coefficient == 0:
            continue

        # Extract the operator and inner expression
        op_name = unary_key[:unary_key.find('(')]
        inner_expr = unary_key[unary_key.find('(')+1:-1]
        
        # Find the corresponding numpy operator
        op_func = next(op for op, symbol in Node.op_list.items() if symbol == op_name)
        
        # Parse the inner expression using our custom parser
        inner_node = parse_inner_expression(inner_expr)
        if inner_node:
            unary_node = Node(value=op_func, left=inner_node)
            
            if coefficient != 1:
                term_node = Node(
                    value=np.multiply,
                    left=Node(value=coefficient),
                    right=unary_node
                )
            else:
                term_node = unary_node
                
            if root is None:
                root = term_node
            else:
                root = Node(value=np.add, left=root, right=term_node)

    return root


# Simplification functions
def parse_inner_expression(expr_str):
    """Helper function to parse inner expressions without using eval()"""
    if expr_str.startswith('x[') and expr_str.endswith(']'):
        # Handle variable terms like x[1]
        feature_index = int(expr_str[2:-1])
        return Node(feature_index=feature_index)
    elif expr_str.startswith('(') and expr_str.endswith(')'):
        # Handle composite expressions like (2 * x[1])
        inner = expr_str[1:-1].split(' * ')
        if len(inner) == 2:
            coefficient = float(inner[0])
            var_node = Node(feature_index=int(inner[1][2:-1]))
            return Node(value=np.multiply, left=Node(value=coefficient), right=var_node)
    # Add more parsing cases as needed
    return None