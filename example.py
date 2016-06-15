import igraph
import pandas as pd

node_names = ['A', 'B', 'C']
a = pd.DataFrame([[1,2,3],[3,1,1],[4,0,2]], index=node_names, columns=node_names)

# Get the values as np.array, it's more convenenient.
A = a.values
print A

def show_graph(A):
	# Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
	g = igraph.Graph.Adjacency((A > 0).tolist())
	print g
	# Add edge weights and node labels.
	g.es['weight'] = A[A.nonzero()]
	g.vs['label'] = node_names  # or a.index/a.columns

	igraph.drawing.plot(g)