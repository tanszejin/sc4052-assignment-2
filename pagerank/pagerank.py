import networkx as nx
import numpy as np



def pagerank(G, c=0.5, max_iter=10):
    if len(G) == 0:
        return {}

    pages = list(G.nodes())
    N = len(pages)
    adj = dict(G.adjacency())
    out_degree = {page : 1.0 if len(adj[page]) == 0 else len(adj[page]) for page in pages}


    pr = {page : 1.0 / N for page in pages}
    for _ in range(max_iter):
        pr_old = pr.copy()
        for page in pages:
            pr[page] = c * sum(pr_old[neighbor] / out_degree[neighbor] for neighbor in G.predecessors(page)) + (1 - c) / N


        print("Change:", np.linalg.norm(np.array(list(pr.values())) - np.array(list(pr_old.values())), 1))
    
    total = sum(pr.values())
    pr = {page : value / total for page, value in pr.items()}

    return pr

def pagerank_matrix(G, c=0.5, max_iter=10):
    if len(G) == 0:
        return {}

    pages = list(G.nodes())
    N = len(pages)
    M = nx.to_numpy_array(G, nodelist=pages, dtype=float)
    out_degree = M.sum(axis=1)
    for i in range(N):
        if out_degree[i] != 0:
            M[i, :] /= out_degree[i]
    M = M.T

    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr_old = pr.copy()
        pr = c * M.dot(pr_old) + (1 - c) / N
        print("Change:", np.linalg.norm(pr - pr_old, 1))

    pr = pr / pr.sum()
    return {pages[i] : pr[i] for i in range(N)}


def pagerank_closed_form(G, c=0.5):
    if len(G) == 0:
        return {}

    pages = list(G.nodes())
    N = len(pages)
    try:
        M = nx.to_numpy_array(G, nodelist=pages, dtype=float)
    except Exception as e:
        print(f"Error occurred while converting graph to numpy array: {e}")
        return {}
    out_degree = M.sum(axis=1)
    for i in range(N):
        if out_degree[i] != 0:
            M[i, :] /= out_degree[i]
    M = M.T

    I = np.eye(N)
    A = I - c * M
    b = (1 - c) / N * np.ones(N).T

    pr = np.linalg.inv(A).dot(b)
    pr = pr / pr.sum()  
    return {pages[i] : pr[i] for i in range(N)}

if __name__ == "__main__":
    dataset = "toy_dataset.txt"
	
    G = nx.DiGraph()

    with open(dataset, "r") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue  
            
            parts = line.strip().split()
            G.add_edge(parts[0], parts[1])

    pr = pagerank_matrix(G)
    print("PageRank:", pr)
    # print("Sum of PageRank values:", sum(pr.values())) 

    # the closed form solution requires the use of matrix operations
    # the 800K dataset is too large to be converted to matrix form
    pr_closed_form = pagerank_closed_form(G)
    print("Closed-form PageRank:", pr_closed_form)
    # print("Sum of closed-form PageRank values:", sum(pr_closed_form.values()))








