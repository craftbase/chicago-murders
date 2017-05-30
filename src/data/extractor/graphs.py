

import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("victim_info_2012_2017.csv")

def plot_murder_by_date(df):
    return
     
def plot_murder_by_time(df):
    return


def plot_murder_by_race(df):

    race = df.iloc[:, 7:8].values
    data = race
    #plt.hist(data, normed=True, bins = ['Black','Hispanic','Other','White'])
    #print race

plot_murder_by_race(df)