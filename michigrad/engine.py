# Calcado de Micrograd (https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)
import networkx as nx
import pyvis
from pyvis.network import Network



class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', name=''):
        self.data = data
        self.grad = 0
        self.name = name
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, name={self.name})"
    

    def visualize(self, filename="graph.html"):
        nodes = {} # Diccionario para almacenar nodos por ID
        edges = []

        def build(v):
            if id(v) not in nodes: #Usamos el id del objeto como key
                nodes[id(v)] = {"label": f"{v.name} | data={v.data:.2f} | grad={v.grad:.2f}", "shape": "box"}
                if v._op: # Si tiene una operación, crea el nodo de operación
                    op_node_id = f"op_{id(v)}" #ID unico para el nodo de operacion
                    nodes[op_node_id] = {"label": v._op, "shape": "circle", "color": "lightblue", "size": 20} #Nodo de operacion
                    for child in v._prev:
                        edges.append((id(child), op_node_id)) #Arista desde los hijos a la operacion
                    edges.append((op_node_id, id(v))) #Arista desde la operacion al resultado
                    for child in v._prev:
                        build(child) #Llamada recursiva para los hijos
        
        build(self)

        graph = nx.DiGraph()
        for node_id, node_data in nodes.items():
            graph.add_node(node_id, **node_data) #Agrega los nodos al grafo con sus atributos
        for edge in edges:
            graph.add_edge(edge[0], edge[1], arrows={'to': True, 'from': False})

        net = Network(notebook=True, cdn_resources='in_line', directed=True)

        net.from_nx(graph)
        net.show(filename)


        