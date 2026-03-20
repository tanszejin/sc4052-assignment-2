from pagerank import pagerank_closed_form
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p_values = [0.1, 0.5, 0.9, 1]
    dataset = "toy_dataset.txt"
	
    G = nx.DiGraph()

    with open(dataset, "r") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue  
            
            parts = line.strip().split()
            G.add_edge(parts[0], parts[1])
    
    # plot pagerank values for different p
    plt.figure(figsize=(10, 6))
    pages = list(G.nodes())

    for p in p_values:
        c = 1 - p
        pr_closed_form = pagerank_closed_form(G, c=c)
        pr_list = [pr_closed_form[page] for page in pages]
        plt.plot(pr_list, label=f'p={p}')

    plt.xlabel('Pages')
    plt.ylabel('PageRank')
    plt.title('PageRank Values for Different p')
    plt.legend()
    plt.show()
    plt.savefig('pagerank_values.png', dpi=300)

