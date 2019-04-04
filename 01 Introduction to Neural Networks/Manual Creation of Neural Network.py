# criação de uma RN sem uso de frameworks
# vamos usar o conceito de grafos uma vez que TF usa o msm conceito
import numpy as np


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        # self aqui representa a opeção que será passada, ou seja, para cada
        # input, append a operação descrita na chamada da func
        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

        # esse método será sobrescrito nas chamadas posteriores
        def compute(self):
            pass


# classe adição
class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


# classe multiplicação
class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


# classe multiplicação de matrizes
class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


# placeholder
class Placeholder():
    def __init__(self):

        self.output_nodes = []

        # append alguns placeholders ao grafo default
        _default_graph.placeholders.append(self)


# placeholder
class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


# classe grafo
class Graph():

    def __init__(self):

        self.operations = []
        self.variables = []
        self.placeholders = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


# testando para z=ax+b, com
# a = 10
# b = 1
# z = 10x + 1

g = Graph()

g.set_as_default()

A = Variable(10)

b = Variable(1)

x = Placeholder()

y = multiply(A, x)

z = add(y, b)


# PostOrder Tree Traversal
# garante que executaremos as operação na ordem correta
def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


# classe seção
class Session():

    def run(self, operation, feed_dict={}):

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:
                # é uma Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


# criando uma sessão
sess = Session()

result = sess.run(operation=z, feed_dict={x: 10})

print(result)

# matmul
g = Graph()

g.set_as_default()

A = Variable([[10, 20], [30, 40]])

b = Variable([1, 2])

x = Placeholder()

y = matmul(A, x)

z = add(y, b)

sess = Session()

result = sess.run(operation=z, feed_dict={x: 10})

print(result)
