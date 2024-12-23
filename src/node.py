import numpy as np


class Node:
    op_list = {
        np.add: '+',
        np.subtract: '-',
        np.multiply: '*',
        np.sin: 'sin',
        np.cos: 'cos',
        np.exp: 'exp',  
    }
    operators=[np.add, np.subtract, np.multiply, np.sin, np.cos, np.exp]
    one_arg_op=[np.sin, np.cos, np.exp]

    def __init__(self, value=None, feature_index=None, left=None, right=None):
        self.value = value 
        self.feature_index = feature_index
        self.left = left
        self.right = right
    
    def is_operator(self, val):
        return val in self.operators

    def evaluate(self,x=None):
        # first check if it is an operator 
        # if not proceed
        
        if not self.is_operator(self.value):
            #print("index",self.feature_index)
            if self.feature_index!= None:
                #print("ben inputun indexiyim")
                return x[self.feature_index]
            else:
                #print("ben bir sayıyım")
                return self.value
        # if it is an operator
        if self.value in self.one_arg_op:
            operand_value = self.left.evaluate(x)
            #print("tek",operand_value)
            return self.value(operand_value)
        
        left_value = self.left.evaluate(x)
        right_value = self.right.evaluate(x)
        #print(left_value)
        #print(right_value)
        return self.value(left_value, right_value)
    
    def __str__(self):
        if not self.is_operator(self.value):
            if self.feature_index != None:
                return f"x[{self.feature_index}]"
            return str(self.value)

        operator_symbol = self.op_list[self.value]

        if self.value in {np.sin, np.cos, np.log, np.exp}:
            return f"{operator_symbol}({self.left})"

        return f"({self.left} {operator_symbol} {self.right})"
    
    def __repr__(self):
        if not self.is_operator(self.value):
            if self.feature_index != None:
                return f"x[{self.feature_index}]"
            return str(self.value)

        operator_symbol = self.op_list[self.value]

        if self.value in {np.sin, np.cos, np.log, np.exp}:
            return f"{operator_symbol}({self.left})"

        return f"({self.left} {operator_symbol} {self.right})"
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (
            np.array_equal(self.value, other.value) and
            self.feature_index == other.feature_index and
            self.left == other.left and
            self.right == other.right
        )

    def __hash__(self):
        def hashable(value):
            if isinstance(value, np.ndarray):
                return tuple(value.flatten())  # Convert array to a hashable tuple
            return value

        return hash((
            hashable(self.value),
            self.feature_index,
            self.left,
            self.right,
        ))