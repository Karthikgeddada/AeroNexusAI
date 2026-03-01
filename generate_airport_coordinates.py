import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load dataset
df = pd.read_csv("data/US_DOT_Airfare_Historical_2008_2025.csv")

# Extract unique airports
airports = pd.unique(
    df[["origin_city", "destination_city"]].values.ravel()
)

print(f"Total unique airports: {len(airports)}")

geolocator = Nominatim(user_agent="aeronexus_ai")

coordinates = []

for airport in airports:
    try:
        location = geolocator.geocode(airport)

        if location:
            coordinates.append({
                "airport": airport,
                "latitude": location.latitude,
                "longitude": location.longitude
            })
            print(f"✔ {airport}")
        else:
            print(f"❌ Not found: {airport}")

        time.sleep(1)

    except Exception as e:
        print(f"⚠ Error for {airport}: {e}")

coords_df = pd.DataFrame(coordinates)
coords_df.to_csv("data/airport_coordinates.csv", index=False)

print("\n✅ airport_coordinates.csv created successfully")