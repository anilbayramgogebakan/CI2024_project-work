import numpy as np

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

def mutation():
    pass

def crossover():
    pass