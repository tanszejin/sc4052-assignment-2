from graphrag import GraphRAG
from kg_gen import KGGen
import os

if __name__ == "__main__":
    API_KEY = os.environ.get("GOOGLE_API_KEY")

    kg = KGGen(
    model="gemini/gemini-2.5-flash", 
    temperature=0.0,        # Default temperature
    api_key=API_KEY  # Optional if set in environment or using a local model
    )

    with open("graphrag/text.txt", "r") as f:
        text = f.read()

    graph = kg.generate(
        input_data=text,
    )
    print("Knowledge Graph generated from the text")
    # print("Nodes:", graph.entities)
    # print("Generated Graph:", graph)

    query = "What discoveries by Marie Curie led to later advances in medical imaging?"

    query_nodes = kg.generate(
        input_data=query,
    ).entities

    grag = GraphRAG(graph)
    top_k = grag.top_k(query_nodes, k=10)
    print("Top-k relevant nodes for the query:")
    for node, score in top_k:
        print(f"Node: {node}, Score: {score:.4f}")
    