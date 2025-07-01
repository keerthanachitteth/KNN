import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Load data
df = pd.read_csv("heart.csv")
print(df.head())

# Split data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize Decision Tree
export_graphviz(dt, out_file="tree.dot", feature_names=X.columns,
                class_names=["No Disease", "Disease"], filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).render("tree_visual", format="png", cleanup=True)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance
importances = rf.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importances - Random Forest")
plt.show()

# Cross-validation
scores = cross_val_score(rf, X, y, cv=5)
print("Cross-Validation Accuracy:", scores.mean())
