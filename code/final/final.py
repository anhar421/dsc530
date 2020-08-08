import pandas as pd
import random
import thinkstats2
import thinkplot
import numpy as np
import scipy
from scipy.stats import norm
import statistics
import matplotlib.pyplot as plt

#Import datasets
planets = pd.read_csv("datasets_239296_540867_planets.csv")
species = pd.read_csv("datasets_239296_540867_species.csv")

#Remove irrelevant variables from dataframes and format variables
species['name'] = species['name'].str.replace(" ","_")
species['name'] = species['name'].str.replace("'","")
species['homeworld'] = species['homeworld'].str.replace(" ","_")
species['homeworld'] = species['homeworld'].str.replace("'","")
species.drop(["designation", "skin_colors", "hair_colors", "eye_colors", "average_lifespan",
                            "language"], axis=1, inplace=True)
species = species.set_index("homeworld")

planets['climate'] = planets['climate'].str.replace(",", "")
planets['climate'] = planets['climate'].str.replace(" ", "_")
planets['name'] = planets['name'].str.replace(" ", "_")
planets.drop(["population", "gravity", "terrain"], axis=1, inplace=True)
planets = planets.set_index("name")

# #Create new dataframe with all relevant variables
master_df = pd.concat([planets, species], axis=1)
master_df.dropna(inplace=True)


pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(master_df)
#
# #Histograms of each quantitative variable
# master_df.hist("diameter", grid=False)
# plt.title('Planet diameter')
# plt.xlabel('Diameter (km)')
# plt.ylabel('# of planets')
# plt.show()
#
# master_df.hist("rotation_period", grid=False, bins = 20)
# plt.title('Planet rotation period')
# plt.xlabel('Rotation (hours)')
# plt.ylabel('# of planets')
# plt.show()
#
# master_df.hist("orbital_period", grid=False)
# plt.title('Planet orbital period')
# plt.xlabel('Orbit (standard days)')
# plt.ylabel('# of planets')
# plt.show()
#
# master_df.hist("surface_water", grid=False, bins = 5)
# plt.title('Planet surface water')
# plt.xlabel('% surface water')
# plt.ylabel('# of planets')
# plt.show()
#
# master_df.hist("average_height", grid=False)
# plt.title('Species average height')
# plt.xlabel('Average height (cm)')
# plt.ylabel('# of species')
# plt.show()

#
# #Bar graphs of qualitative data
# master_df.climate.value_counts().plot.bar()
# plt.title('Planet climate')
# plt.xlabel('Climate type')
# plt.ylabel('# of planets')
# plt.show()
#
# master_df.classification.value_counts().plot.bar()
# plt.title('Species classification')
# plt.xlabel('Classification')
# plt.ylabel('# of species')
# plt.show()

#Calculate mean, mode, spread
#Mean
print(master_df.mean())

#Mode
print(master_df.mode())

#Spread
print(master_df.var())

#PMF
#Create new dataframes: Large planets and Small planets and compare their species heights
pmf_df = master_df[["name", "average_height", "diameter"]]
pmf_df = pmf_df.sort_values(by=["diameter"])
pmf_df.dropna(inplace=True)

#Split dataframe into two sets: large planets and small planets
small = pmf_df[pmf_df["diameter"] < 12250]
large = pmf_df[pmf_df["diameter"] > 12250]

#Create PMF
pmf1 = thinkstats2.Pmf(small.average_height, label='Small planets (< 12250 km)')
pmf2 = thinkstats2.Pmf(large.average_height, label='Large planets (> 12250 km)')

#Plot PMF
thinkplot.Hist(pmf1, align='right', width=5)
thinkplot.Hist(pmf2, align='left', width=5)
thinkplot.Config(title= 'Species height probability vs Planet size', xlabel='Average height (cm)',
                 ylabel='Probability', axis=[85, 305, 0, 0.35])
plt.show()
#CDF
#

