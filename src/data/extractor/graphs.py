import matplotlib.pyplot as plt
import pandas as pd
import analysis
import seaborn as sns
import geopy

# Make the graphs a bit prettier, and bigger

# This is necessary to show lots of columns in pandas 0.12.
# Not necessary in pandas 0.13.
plt.interactive(False)
plt.ylabel('Count')
plt.xlabel('Time')
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
    # race = df.iloc[:, 7:8].values
    race = df['race'].value_counts()
    race.plot(kind='bar')
    # plt.hist(data, normed=True, bins = ['Black','Hispanic','Other','White'])
    plt.show()


def plot_murder_by_type_race(df):
    type = df['cause'].value_counts()
    type.plot(kind='bar')
    plt.show()

def plot_murder_by_neighbourhood(df):

    type = df['neighbourhood'].value_counts()
    type.plot(kind='bar')
    plt.show()

def plot_murder_by_date(df):
    df = df.tail(100)
    date = df.groupby['']


def get_dist_plot(df):
    df = analysis.clean_data(df)
    print "plotting"
    sns.distplot(df['time'])
    plt.show()

def get_count_plot(df):
    df = analysis.clean_data(df)
    print "plotting"
    sns.countplot(x="age", hue="race", data=df)
    plt.show()

def get_joint_plot(df):
    df = analysis.clean_data(df)
    print "plotting"
    sns.jointplot(x="age", y="time", data=df)
    plt.show()

def get_pair_plot(df):
    df = analysis.clean_data(df)
    print "plotting"
    sns.pairplot(df,hue="cause")
    plt.show()

get_pair_plot(df)