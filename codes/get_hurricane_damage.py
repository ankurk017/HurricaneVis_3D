import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
def remove_references(text):
    return re.sub(r'\[.*?\]', '', text)

def convert_damage(damage):
    if damage is not None:
        damage = remove_references(damage)
        damage = re.sub(r'^.*?\$', '$', damage)
        if 'billion' in damage:
            return float(damage.replace('$', '').replace(' billion', '')) * 1000
        elif 'million' in damage:
            return float(damage.replace('$', '').replace(' million', ''))
    return None


def extract_deaths(deaths):
    if deaths is not None:
        deaths = remove_references(deaths)
        match = re.search(r'(\d+)(?!\()', deaths)
        if match:
            return int(match.group(0))
    return None


for year in np.arange(2011, 2023):
	print(year)
	url = f"https://en.wikipedia.org/wiki/{year}_Atlantic_hurricane_season"
	response = requests.get(url)
	soup = BeautifulSoup(response.content, "html.parser")
	table = soup.find("table", {"class": "wikitable"})
	table_data = []
	for row in table.find_all("tr"):
	    cells = row.find_all(["th", "td"])
	    row_data = [cell.get_text(strip=True) for cell in cells]
	    table_data.append(row_data)

	df = pd.DataFrame(table_data[1:], columns=table_data[0])[['Stormname', 'Damage(USD)', 'Deaths']]
	df['Damage(USD)'] = df['Damage(USD)'].apply(convert_damage)
	df['Deaths'] = df['Deaths'].apply(extract_deaths)
	df[['Stormname', 'Damage(USD)', 'Deaths']].to_csv(f'Damage_{year}.csv', index=False)



