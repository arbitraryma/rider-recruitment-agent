import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ==========================================
# 1. Load site database
# ==========================================

sites_df = pd.read_excel(
    "sites_geocoded.xlsx"
)


# ==========================================
# 2. Haversine distance
# ==========================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate geographical distance
    between two latitude/longitude points.

    Returns:
        Distance in kilometers.
    """

    R = 6371.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# ==========================================
# 3. Find nearest recruitment sites
# ==========================================

def find_nearest_sites(
    latitude,
    longitude,
    top_k=5
):
    """
    Find nearest recruitment sites.

    Args:
        latitude: user latitude
        longitude: user longitude
        top_k: number of results

    Returns:
        list of nearest sites
    """

    results = []

    for _, row in sites_df.iterrows():

        if (
            pd.isna(row["Latitude"])
            or pd.isna(row["Longitude"])
        ):
            continue

        try:
            site_lat = float(
                row["Latitude"]
            )

            site_lon = float(
                row["Longitude"]
            )

        except (
            ValueError,
            TypeError
        ):
            continue


        distance = haversine_distance(
            latitude,
            longitude,
            site_lat,
            site_lon
        )


        results.append({
            "City":
                row["City"],

            "District":
                row["District"],

            "Site":
                row["Site"],

            "Address":
                row["Address"],

            "Latitude":
                site_lat,

            "Longitude":
                site_lon,

            "Distance_km":
                round(
                    distance,
                    2
                )
        })


    results.sort(
        key=lambda x:
        x["Distance_km"]
    )

    return results[:top_k]


# ==========================================
# 4. Optional standalone test
# ==========================================

if __name__ == "__main__":

    test_results = find_nearest_sites(
        latitude=39.9087,
        longitude=116.3975,
        top_k=5
    )

    for site in test_results:
        print(
            site["Site"],
            site["Distance_km"],
            "km"
        )