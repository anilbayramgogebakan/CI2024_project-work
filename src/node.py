import numpy as np
import random
import math
operators=[np.add, np.subtract, np.multiply, np.sin, np.cos, np.exp]
one_arg_op=[np.sin, np.cos, np.exp]

def is_operator(val):
        return val in operators

class Node:
    op_list = {
        np.add: '+',
        np.subtract: '-',
        np.multiply: '*',
        np.sin: 'sin',
        np.cos: 'cos',
        np.exp: 'exp',  
    }
    def __init__(self, value=None, feature_index=None, left=None, right=None):
        self.value = value 
        self.feature_index = feature_index
        self.left = left
        self.right = right

    def evaluate(self,x=None):
        # first check if it is an operator 
        # if not proceed
        
        if not is_operator(self.value):
            #print("index",self.feature_index)
            if self.feature_index!= None:
                #print("ben inputun indexiyim")
                return x[self.feature_index]
            else:
                #print("ben bir sayıyım")
                return self.value
        # if it is an operator
        if self.value in one_arg_op:
            operand_value = self.left.evaluate(x)
            #print("tek",operand_value)
            return self.value(operand_value)
        
        left_value = self.left.evaluate(x)
        right_value = self.right.evaluate(x)
        #print(left_value)
        #print(right_value)
        return self.value(left_value, right_value)
    def __str__(self):
        if not is_operator(self.value):
            if self.feature_index != None:
                return f"x[{self.feature_index}]"
            return str(self.value)

        operator_symbol = self.op_list[self.value]

        if self.value in {np.sin, np.cos, np.log, np.exp}:
            return f"{operator_symbol}({self.left})"

        return f"({self.left} {operator_symbol} {self.right})"

def random_tree(depth, num_features):
    if depth == 0:
        if random.random() < 0.5:
            return Node(feature_index=random.randint(0, num_features - 1))
        else:
            return Node(value=random.randint(1, 10))

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

def cost(genome,x, y):
    predictions = np.array([genome.evaluate(x[:, i]) for i in range(x.shape[1])])
    mse = np.mean((predictions - y) ** 2)
    return mse