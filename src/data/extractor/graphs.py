


import pandas as pd
import matplotlib.pyplot as plt

# Make the graphs a bit prettier, and bigger

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
plt.interactive(False)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

plt.rcParams['figure.figsize'] = (15, 5)


df = pd.read_csv("victim_info_2012_2017.csv")

def plot_murder_by_date(df):
    return
     
def plot_murder_by_time(df):

    time = df['time'].value_counts()
    print time
    time.plot(kind='bar')
    plt.show()



def plot_murder_by_race(df):

    #race = df.iloc[:, 7:8].values
    race = df['race'].value_counts()
    race.plot(kind='bar')
    #plt.hist(data, normed=True, bins = ['Black','Hispanic','Other','White'])
    plt.show()

plot_murder_by_time(df)
