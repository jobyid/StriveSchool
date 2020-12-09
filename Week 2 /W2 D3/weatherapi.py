import weathercom
import requests
import pandas as pd
import matplotlib as plt

weatherDetails = weathercom.getCityWeatherDetails("san fransisco", queryType="ten-days-data")
print(weatherDetails)
df = pd.read_json(weatherDetails,convert_axes=True)
print(df)

#plt.show()
