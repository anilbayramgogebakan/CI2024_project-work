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
            # Simplify if denominator is a constant
            if node.right.value is not None and not node.right.is_operator(node.right.value) and node.right.value != 0:
                new_coefficient = coefficient / node.right.value
                collect_terms_with_multiplication_division(node.left, terms, new_coefficient)
            elif node.right.feature_index is not None:
                # Simplify division if numerator contains the denominator
                if node.left.value == np.multiply and (
                    node.left.left.feature_index == node.right.feature_index
                    or node.left.right.feature_index == node.right.feature_index
                ):
                    other_term = (
                        node.left.right
                        if node.left.left.feature_index == node.right.feature_index
                        else node.left.left
                    )
                    collect_terms_with_multiplication_division(other_term, terms, coefficient)
                else:
                    # Cannot simplify further, leave as is
                    key = f"({node.left} / {node.right})"
                    terms[key] = terms.get(key, 0) + coefficient
            else:
                # Unsupported denominator
                raise ValueError("Cannot simplify division with unsupported denominator.")
        else:
            raise ValueError("Invalid division operation: missing left or right node.")

    return terms



def rebuild_tree_with_multiplication_division(terms):
    root = None

    # Handle constant terms
    if 'constant' in terms and terms['constant'] != 0:
        root = Node(value=terms.pop('constant'))

    # Handle feature terms
    for feature_key, coefficient in sorted(terms.items()):
        if coefficient == 0:
            continue

        if coefficient == 1:  # Coefficient of 1, no need to multiply
            term_node = Node(feature_index=int(feature_key[2:-1]))
        else:
            term_node = Node(
                value=np.multiply,
                left=Node(value=coefficient),
                right=Node(feature_index=int(feature_key[2:-1])),
            )
        
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
        return node
    
# Arithmetic Simplification (unchanged)
def simplify_tree_with_multiplication_and_division(node):
    try:
        terms = collect_terms_with_multiplication_division(node)
        simplified_tree = rebuild_tree_with_multiplication_division(terms)
        return simplified_tree
    except ValueError as e:
        print(f"Error during simplification: {e}")
        return node

# Simplify Unary Operations
def simplify_unary_operations(node):
    if node is None:
        return None

    if node.feature_index is not None:
        print(f"Preserving feature_index node: {node}")
        return Node(feature_index=node.feature_index)

    simplified_left = simplify_unary_operations(node.left)
    simplified_right = simplify_unary_operations(node.right)

    if node.value in Node.unary_operators:
        print(f"Attempting to simplify unary operator: {node.value} with operand: {simplified_left}")
        if simplified_left and simplified_left.value is not None and not simplified_left.is_operator(simplified_left.value):
            try:
                evaluated_value = node.value(simplified_left.value)
                print(f"Unary operator {node.value} evaluated to {evaluated_value}")
                return Node(value=evaluated_value)
            except Exception as e:
                print(f"Error simplifying unary operation: {e}")
        return Node(value=node.value, left=simplified_left)

    return Node(value=node.value, left=simplified_left, right=simplified_right)



# Combined Simplification
def simplify_tree(node):
    try:
        print("Starting arithmetic simplification...")
        simplified_arithmetic = simplify_tree_with_multiplication_and_division(node)
        print(f"Arithmetic Simplified Tree: {simplified_arithmetic}")

        if simplified_arithmetic is None:
            return None

        print("Starting unary simplification...")
        fully_simplified_tree = simplify_unary_operations(simplified_arithmetic)
        print(f"Fully Simplified Tree: {fully_simplified_tree}")

        return fully_simplified_tree
    except ValueError as e:
        print(f"Error during simplification: {e}")
        return node

