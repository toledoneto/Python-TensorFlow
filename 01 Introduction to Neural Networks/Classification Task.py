import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


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


# função de ativação sigmóide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


class Sigmoid(Operation):

    def __init__(self, z):
        # a is the input node
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


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


# vendo a curva da sigmoide
sample_z = np.linspace(-10,10,100)
sample_a = sigmoid(sample_z)

plt.figure()
plt.plot(sample_z, sample_a)

# gerando dados aleatórios para classificar
data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

# setando os dados
features = data[0]
plt.figure()
plt.scatter(features[:, 0], features[:, 1])

# setando as labels
labels = data[1]
plt.figure()
# colorindo a figura de acordo com as labels
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')

# fazendo uma linha que separa as features
x = np.linspace(0, 11, 10)
y = -x + 5
plt.figure()
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)

# aplicando a RN para prever um ponto
g = Graph()

g.set_as_default()

x = Placeholder()

# convertemos a equação y = -x + 5 em notação de matrix: [1, 1] * w - 5 = 0
# se o resultado for
# * >0, estaremos na área vermelha;
# * <0, estaremos na área azul;
w = Variable([1, 1])
b = Variable(-5)

z = add(matmul(w, x), b)

a = Sigmoid(z)

sess = Session()

# ponto claramente na área vermelha -> sigmóide devolve valor próx a 1
result = sess.run(operation=a, feed_dict={x: [8, 10]})

print(result)

# ponto claramente na área azul -> sigmóide devolve valor próx a 0
result = sess.run(operation=a, feed_dict={x: [0, -10]})

print(result)

plt.show()
