from collections import defaultdict

class GraphRAG:
    def __init__(self, graph=None):
        if graph:
            self.nodes = set(graph.entities)
            self.graph = {n: [] for n in self.nodes}
            for o, _, i in graph.relations:
                self.graph[o].append(i)
        else:
            self.graph = defaultdict(list)  
            self.nodes = set()
            

    def personalised_pagerank(self, query_nodes, alpha=0.85, max_iter=100, tolerance=1e-6):
        print("Calculating PageRank...")
        # query nodes are the entities in the query
        nodes = list(self.nodes)
        N = len(nodes)

        pr = {n: 1.0 / N for n in nodes}

        personalisation = {n: 0.0 for n in nodes}
        for q in query_nodes:
            if q in personalisation:
                personalisation[q] = 1.0 / len(query_nodes)

        for _ in range(max_iter):
            pr_old = pr.copy()
            pr = {n: 0.0 for n in nodes}

            for n in nodes:
                out_degree = len(self.graph[n])
                if out_degree > 0:
                    for i in self.graph[n]:
                        if i in nodes:
                            pr[i] += alpha * pr_old[n] / out_degree
                pr[n] += (1 - alpha) * personalisation[n]

            total = sum(pr.values())
            pr = {page : value / total for page, value in pr.items()}

            # Convergence check
            diff = sum(abs(pr[n] - pr_old[n]) for n in nodes)
            if diff < tolerance:
                break

        return pr

    def top_k(self, query_nodes, k=5):
        scores = self.personalised_pagerank(query_nodes)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    

