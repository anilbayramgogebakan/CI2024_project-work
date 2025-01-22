import numpy as np
from dataclasses import dataclass
from node import Node

@dataclass
class Individual:
    genome: Node
    fitness: float=None
    fitness_val: float=None
    age: int=0
    T: float=1