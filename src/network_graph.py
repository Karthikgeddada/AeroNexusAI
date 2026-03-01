import pandas as pd
import networkx as nx


class AirlineNetworkGraph:
    """
    AeroNexus AI — Network Intelligence Engine

    Builds airport network graph and computes:
    - Degree Centrality
    - Betweenness Centrality
    - Revenue Weighted Connectivity
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.graph = None

    # -----------------------------------------------------
    # 1️⃣ Load and Prepare Route Revenue
    # -----------------------------------------------------
    def load_data(self):
        print("Loading data for network graph...")

        self.df = pd.read_csv(self.data_path)

        self.df = self.df.dropna(
            subset=["origin_city", "destination_city", "passengers", "avg_fare"]
        )

        self.df["revenue"] = self.df["passengers"] * self.df["avg_fare"]

        # Aggregate total revenue per route
        self.df = (
            self.df.groupby(["origin_city", "destination_city"])["revenue"]
            .sum()
            .reset_index()
        )

        print(f"Routes prepared: {len(self.df):,}")

    # -----------------------------------------------------
    # 2️⃣ Build Graph
    # -----------------------------------------------------
    def build_graph(self):
        print("Building network graph...")

        G = nx.Graph()

        for _, row in self.df.iterrows():
            G.add_edge(
                row["origin_city"],
                row["destination_city"],
                weight=row["revenue"],
            )

        self.graph = G

        print(f"Graph built with {G.number_of_nodes()} airports")
        print(f"Graph built with {G.number_of_edges()} routes")

    # -----------------------------------------------------
    # 3️⃣ Compute Centrality Metrics
    # -----------------------------------------------------
    def compute_centrality(self):

        print("Computing centrality metrics...")

        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)

        centrality_df = pd.DataFrame({
            "airport": list(degree_centrality.keys()),
            "degree_centrality": list(degree_centrality.values()),
            "betweenness_centrality": [
                betweenness_centrality[node]
                for node in degree_centrality.keys()
            ],
        })

        centrality_df = centrality_df.sort_values(
            by="betweenness_centrality",
            ascending=False
        )

        return centrality_df

    # -----------------------------------------------------
    # 4️⃣ Top Strategic Hubs
    # -----------------------------------------------------
    def get_top_hubs(self, top_n=10):
        centrality_df = self.compute_centrality()
        return centrality_df.head(top_n)