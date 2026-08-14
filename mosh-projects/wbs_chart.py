import matplotlib.pyplot as plt
import networkx as nx

# Create WBS structure
wbs = nx.DiGraph()
wbs.add_edges_from([
    ("Project", "Phase 1: Planning"),
    ("Project", "Phase 2: Development"),
    ("Project", "Phase 3: Testing"),
    ("Phase 1: Planning", "1.1 Requirements"),
    ("Phase 1: Planning", "1.2 Design"),
    ("Phase 2: Development", "2.1 Coding"),
    ("Phase 2: Development", "2.2 Review"),
    ("Phase 3: Testing", "3.1 Unit Testing"),
    ("Phase 3: Testing", "3.2 Integration Testing"),
])

# Better positioning for tree layout
pos = nx.spring_layout(wbs, k=2, seed=42, iterations=50)

# Draw with nicer formatting
plt.figure(figsize=(12, 8))
nx.draw(wbs, pos, 
        with_labels=True, 
        node_color='lightblue',
        node_size=4000,
        font_size=9,
        font_weight='bold',
        arrows=False,
        edge_color='gray',
        linewidths=2,
        edgecolors='darkblue')

plt.title("Work Breakdown Structure (WBS)", fontsize=14, fontweight='bold')
plt.show()