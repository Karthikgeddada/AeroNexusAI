import pandas as pd

def load_airport_coordinates():
    """
    Load airport coordinates from CSV
    Returns dictionary:
    {
        "Airport Name": (lat, lon),
        ...
    }
    """
    df = pd.read_csv("data/airport_coordinates.csv")

    airport_dict = {
        row["airport"]: (row["latitude"], row["longitude"])
        for _, row in df.iterrows()
    }

    return airport_dict