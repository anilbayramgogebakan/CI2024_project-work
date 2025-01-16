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
    unary_operators = [np.sin, np.cos, np.exp, np.abs, np.log, np.tan]
    binary_operators = [np.add, np.subtract, np.multiply, np.divide]
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
                return self.comp_list[self.value] + (self.comp_list[self.value] * self.left.calculate_complexity())
        if self.left and self.right:
            left_complexity = self.left.calculate_complexity()
            right_complexity = self.right.calculate_complexity()
            return self.comp_list[self.value] + (self.comp_list[self.value] * (left_complexity + right_complexity))

    def is_operator(self, val):
        return val in self.operators

    def evaluate(self, x=None):
        if not self.is_operator(self.value):
            if self.feature_index is not None:
                return x[self.feature_index]
            else:
                return self.value
        if self.value in self.unary_operators:
            operand_value = self.left.evaluate(x)
            return self.value(operand_value)

        left_value = self.left.evaluate(x)
        right_value = self.right.evaluate(x)
        return self.value(left_value, right_value)

    def __str__(self):
        if not self.is_operator(self.value):
            if self.feature_index is not None:
                return f"x[{self.feature_index}]"
            return str(self.value)

        operator_symbol = self.op_list[self.value]

        if self.value in self.unary_operators:
            return f"{operator_symbol}({self.left})"

        return f"({self.left} {operator_symbol} {self.right})"

    def __repr__(self):
        return self.__str__()

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

def simplify_tree_with_multiplication_and_division(node):
    try:
        # Simplify arithmetic terms
        terms = collect_terms_with_multiplication_division(node)
        simplified_tree = rebuild_tree_with_multiplication_division(terms)
        return simplified_tree
    except ValueError as e:
        print(f"Error during simplification: {e}")
        return node  # Return original node if simplification fails
