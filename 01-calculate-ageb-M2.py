import pandas as pd
import geopandas as gpd

from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=40)

def calculate_polygon(point, gdf_ageb):
    for i, pol in enumerate(gdf_ageb["geometry"]):
        if pol.contains(point):
            return i
    return -1

fileno = 2
filename = f"./Data/phone_mobility_database/M{fileno}.csv"
chunksize = 20000
gdf_ageb = gpd.read_file("./Data/shape_files/26a.shp")
gdf_ageb.to_crs("EPSG:3857", inplace=True)
gdf_ageb.sort_values(by="CVE_AGEB", inplace=True)
result = pd.DataFrame([], dtype=int)
for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize, sep=";", decimal=",")):
    chunk["timestamp"] = chunk["timestamp"].astype("datetime64[ns, UTC]")
    chunk.drop_duplicates(subset=["id_adv", "timestamp"], keep="first", inplace=True)
    gdf = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"], crs="EPSG:4326"))
    gdf.to_crs("EPSG:3857", inplace=True)
    gdf["polygon"] = gdf["geometry"].parallel_apply(calculate_polygon, args=(gdf_ageb,))
    result = pd.concat([result, gdf[["id_adv", "timestamp", "lat", "lon", "polygon"]]])
result.to_csv(f"./Data/phone_mobility_database_sorted/ageb_M{fileno}_sorted.csv.zip", index=False, compression="zip", header=["id", "timestamp", "lon", "lat", "polygon"])
