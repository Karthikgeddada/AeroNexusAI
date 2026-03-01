from src.route_scoring import RouteScoringEngine
from src.demand_model import DemandPredictionEngine
from src.network_graph import AirlineNetworkGraph
from src.optimization_engine import RouteExpansionSimulator


def main():

    data_path = "data/US_DOT_Airfare_Historical_2008_2025.csv"

    # --------------------------------------------------
    # Route Intelligence
    # --------------------------------------------------
    route_engine = RouteScoringEngine(data_path)
    route_engine.load_data()
    route_engine.compute_route_metrics()
    route_engine.generate_route_score()

    print("\n🔥 Top 5 Route Expansion Recommendations:\n")
    print(
        route_engine.get_top_routes(5)[
            ["origin_city", "destination_city", "route_score"]
        ]
    )

    # --------------------------------------------------
    # Demand Prediction
    # --------------------------------------------------
    demand_engine = DemandPredictionEngine(data_path)
    demand_engine.load_data()
    demand_engine.train_models()
    demand_engine.print_results()

    # --------------------------------------------------
    # Network Intelligence
    # --------------------------------------------------
    network_engine = AirlineNetworkGraph(data_path)
    network_engine.load_data()
    network_engine.build_graph()

    print("\n🌍 Top 5 Strategic Hub Airports:\n")
    print(network_engine.get_top_hubs(5))

    # --------------------------------------------------
    # Route Expansion Simulation
    # --------------------------------------------------
    print("\n🚀 Simulating New Route Expansion Opportunities...\n")

    simulator = RouteExpansionSimulator(
        graph=network_engine.graph,
        demand_model=demand_engine,
        route_df=route_engine.df
    )

    expansion_results = simulator.simulate_new_routes(top_n=5)

    print("\n💡 Top 5 Potential New Routes:\n")
    print(expansion_results)


if __name__ == "__main__":
    main()