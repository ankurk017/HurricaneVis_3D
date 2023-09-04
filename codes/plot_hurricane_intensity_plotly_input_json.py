import os
import json
import argparse
import plotly.io as pio
import tropycal.tracks as tracks
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

DEFAULT_JSON_FILE = "input.json"
DEFAULT_OUTPUT_DIR = "./"

parser = argparse.ArgumentParser(description="Process JSON file and output directory")
parser.add_argument(
    "--json_file", default=DEFAULT_JSON_FILE, help="Path to the input JSON file"
)
parser.add_argument(
    "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory path"
)
args = parser.parse_args()

json_file_path = args.json_file
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

storm_name = data["hurricane_name"]
storm_year = int(data["hurricane_year"])

url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2022-042723.txt"

basin = tracks.TrackDataset(basin="north_atlantic", atlantic_url=url)
storm = basin.get_storm((storm_name, storm_year))
dates = storm["date"]
ws = storm["vmax"]
mslp = storm["mslp"]

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=dates, y=mslp, mode="lines+markers", name="MSLP"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=dates, y=ws, mode="lines+markers", name="WS"), secondary_y=True
)

fig.update_layout(
    title=f"Time Series of MSLP and WS of {storm_name} ({storm_year})",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Mean Sea Level Pressure (hPa)"),
    yaxis2=dict(title="Wind Speed (kt)", overlaying="y", side="right"),
    font=dict(size=18), 
    width=900,
    height=500,
)

output_file = f"{output_dir}/{storm_year}_{storm_name}.html"
pio.write_html(fig, output_file)
plt.show()
