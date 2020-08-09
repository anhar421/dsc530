#Anna Harvey
#DSC 530 - Final project
#08/08/20

import pandas as pd
import random
import thinkstats2
import thinkplot
import numpy as np
import scipy
from scipy.stats import norm
import statistics
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

pd.set_option("display.max_rows", None, "display.max_columns", None)

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

#Histograms of each quantitative variable
master_df.hist("diameter", grid=False)
plt.title('Planet diameter')
plt.xlabel('Diameter (km)')
plt.ylabel('# of planets')

master_df.hist("rotation_period", grid=False, bins = 20)
plt.title('Planet rotation period')
plt.xlabel('Rotation (hours)')
plt.ylabel('# of planets')

master_df.hist("orbital_period", grid=False)
plt.title('Planet orbital period')
plt.xlabel('Orbit (standard days)')
plt.ylabel('# of planets')

master_df.hist("surface_water", grid=False, bins = 5)
plt.title('Planet surface water')
plt.xlabel('% surface water')
plt.ylabel('# of planets')

master_df.hist("average_height", grid=False)
plt.title('Species average height')
plt.xlabel('Average height (cm)')
plt.ylabel('# of species')

#Bar graphs of qualitative data
master_df.climate.value_counts().plot.bar()
plt.title('Planet climate')
plt.xlabel('Climate type')
plt.ylabel('# of planets')

master_df.classification.value_counts().plot.bar()
plt.title('Species classification')
plt.xlabel('Classification')
plt.ylabel('# of species')

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

#CDF
#Average species height
cdf = thinkstats2.Cdf(master_df.average_height, label='average_height')
thinkplot.Cdf(cdf)
thinkplot.Show(title='Average height CDF', xlabel='Avg height (cm)', ylabel='CDF')

#Create a normal probability plot
mean = master_df.average_height.mean()
std = master_df.average_height.std()

xs = [-2, 2]
fxs, fys = thinkstats2.FitLine(xs, inter=mean, slope=std)
thinkplot.Plot(fxs, fys, color='gray', label='model')

xs, ys = thinkstats2.NormalProbability(master_df.average_height)
thinkplot.Plot(xs, ys, label='heights')
thinkplot.show(title='Average Height Normal Probability Plot', ylabel='Avg height (cm)')

#Convert categorical data of species to numerical
#Create a copy of the master dataframe
master_df2 = master_df

#Create dicts for mapping
climate = {'temperate': 1,'arid': 2, 'tropical': 3, 'subtropical': 4, 'hot': 5}
classification = {'mammal': 1, 'amphibian': 2, 'reptilian': 3, 'insectoid': 4, 'gastropod': 5, 'unknown': 6}

#Map new values to dataframe
master_df2.climate = [climate[item] for item in master_df2.climate]
master_df2.classification = [classification[item] for item in master_df2.classification]

#Create scatter plots
#Climate vs average height
scatter_climate = thinkstats2.Jitter(master_df2.climate, 0.2)
scatter_height = thinkstats2.Jitter(master_df2.average_height)

thinkplot.Scatter(scatter_climate, scatter_height)
thinkplot.Show(title = 'Average Height Distribution by Climate', xlabel='Climate', ylabel='Height (cm)')

#Diameter vs classification
scatter_diameter = thinkstats2.Jitter(master_df2.diameter)
scatter_classification = thinkstats2.Jitter(master_df2.classification, 0.2)

thinkplot.Scatter(scatter_diameter, scatter_classification)
thinkplot.Show(title = 'Species Classification by Planet Diameter', xlabel='Diameter (km)', ylabel='Classification')

#Correlation tests
#Spearman correlations
print(thinkstats2.SpearmanCorr(master_df2.climate, master_df2.average_height))
print(thinkstats2.SpearmanCorr(master_df2.diameter, master_df2.classification))

#Hypothesis test
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

def CorrelationTest(variable1, variable2):
    data = variable1.values, variable2.values
    ht = CorrelationPermute(data)
    pvalue = ht.PValue()
    print('p-value =', pvalue)
    return

#Correlation tests for average_height vs planet variables
CorrelationTest(master_df2.orbital_period, master_df2.average_height)
CorrelationTest(master_df2.diameter, master_df2.average_height)
CorrelationTest(master_df2.climate, master_df2.average_height)
CorrelationTest(master_df2.rotation_period, master_df2.average_height)
CorrelationTest(master_df2.surface_water, master_df2.average_height)

#Correlation tests for classification vs planet variables
CorrelationTest(master_df2.orbital_period, master_df2.classification)
CorrelationTest(master_df2.diameter, master_df2.classification)
CorrelationTest(master_df2.climate, master_df2.classification)
CorrelationTest(master_df2.rotation_period, master_df2.classification)
CorrelationTest(master_df2.surface_water, master_df2.classification)

#Multiple regression
formula = 'average_height ~ diameter + rotation_period + orbital_period + climate + surface_water'
model = smf.ols(formula, data=master_df2)
results = model.fit()
print(results.summary())

