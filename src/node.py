import numpy as np


class Node:
    op_list = {
        np.add: '+',
        np.subtract: '-',
        np.multiply: '*',
        np.sin: 'sin',
        np.cos: 'cos',
        np.exp: 'exp',
        np.abs: 'abs',
        np.divide: '/',
        np.log: 'log',
        np.tan: 'tan'  
    }
    comp_list = {
        np.add: 1,
        np.subtract: 1,
        np.multiply: 1,
        np.sin: 4,
        np.cos: 4,
        np.exp: 5,
        np.abs: 1,
        np.divide: 1,
        np.log: 5,
        np.tan: 4
    }
    unary_operators=[np.sin, np.cos, np.exp, np.abs, np.log, np.tan]
    binary_operators=[np.add, np.subtract, np.multiply, np.divide]
    operators = unary_operators + binary_operators


    def __init__(self, value=None, feature_index=None, left=None, right=None):
        self._value = value 
        self.feature_index = feature_index
        self._left = left
        self._right = right
        self._complexity = self.calculate_complexity()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        self.update_complexity()

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, new_left):
        self._left = new_left
        self.update_complexity()

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, new_right):
        self._right = new_right
        self.update_complexity()

    @property
    def complexity(self):
        return self._complexity

    def update_complexity(self):
        self._complexity = self.calculate_complexity()

    def calculate_complexity(self):
        if not self.is_operator(self.value):
            return 1
        if self.value in self.unary_operators:
            if self.left:
                return self.comp_list[self.value]+(self.comp_list[self.value] * self.left.calculate_complexity())
        if self.left and self.right: 
            left_complexity = self.left.calculate_complexity()
            right_complexity = self.right.calculate_complexity()
            return self.comp_list[self.value]+(self.comp_list[self.value] * (left_complexity + right_complexity))
    
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
        if self.value in self.unary_operators:
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

        if self.value in self.unary_operators:
            return f"{operator_symbol}({self.left})"

        return f"({self.left} {operator_symbol} {self.right})"
    
    def __repr__(self):
        if not self.is_operator(self.value):
            if self.feature_index != None:
                return f"x[{self.feature_index}]"
            return str(self.value)

        operator_symbol = self.op_list[self.value]

        if self.value in self.unary_operators:
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