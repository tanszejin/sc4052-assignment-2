from collections import defaultdict

class GraphRAG:
    def __init__(self):
        self.graph = defaultdict(list)   # adjacency list
        self.nodes = set()

    def add_edge(self, u, v):
        # knowledge graph should have ALL nodes (global)
        self.graph[u].append(v)
        self.nodes.add(u)
        self.nodes.add(v)

    def personalized_pagerank(self, query_nodes, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Compute personalized PageRank scores for the graph with respect to the query nodes.
         - query_nodes: list of nodes to bias the PageRank towards - i think this is the nodes that are in the query
         - alpha: damping factor
         - max_iter: maximum number of iterations
         - tol: convergence tolerance
        """
        nodes = list(self.nodes)
        N = len(nodes)

        # Initialize PR scores
        pr = {node: 1.0 / N for node in nodes}

        # Personalization vector (bias to query nodes)
        personalization = {node: 0.0 for node in nodes}
        for q in query_nodes:
            if q in personalization:
                personalization[q] = 1.0 / len(query_nodes)

        for _ in range(max_iter):
            new_pr = {node: 0.0 for node in nodes}

            # Distribute scores
            for u in nodes:
                out_links = self.graph[u]
                if out_links:
                    share = pr[u] / len(out_links)
                    for v in out_links:
                        new_pr[v] += alpha * share
                else:
                    # dangling node distributes evenly
                    for v in nodes:
                        new_pr[v] += alpha * (pr[u] / N)

            # Add personalization (teleportation)
            for v in nodes:
                new_pr[v] += (1 - alpha) * personalization[v]

            # Convergence check
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            pr = new_pr
            if diff < tol:
                break

        return pr

    def top_k(self, query_nodes, k=5):
        scores = self.personalized_pagerank(query_nodes)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]