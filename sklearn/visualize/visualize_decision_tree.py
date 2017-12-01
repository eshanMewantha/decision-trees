from sklearn.datasets import load_iris
from sklearn import tree

# load data
iris = load_iris()

# define model
model = tree.DecisionTreeClassifier()

# train model
model.fit(iris.data, iris.target)

# export decision rules
tree.export_graphviz(model, out_file='tree.dot')
