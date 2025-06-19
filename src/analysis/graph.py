import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def slam_graph(self):
        # NetworkX implementation
        G = nx.Graph()
        for node in self.graph:
            G.add_node(node)
        for node in self.graph:
            for edge in self.graph[node].world:
                G.add_edge(node, edge)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold")
        plt.show()


        # Graphviz implementation
        # dot = Digraph()
        # for node in self.slam.graph.nodes:
        #     dot.node(str(node))
        # for edge in self.slam.graph.edges:
        #     dot.edge(str(edge[0]), str(edge[1]))
        # dot.render('slam_graph', view=True)
