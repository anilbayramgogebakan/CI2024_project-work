import numpy as np
from dataclasses import dataclass
from src.Node import Node

@dataclass
class Individual:
    genome: Node
    fitness: float=None
    age: int=0
    T: float=1