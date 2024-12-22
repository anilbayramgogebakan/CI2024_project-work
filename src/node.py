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