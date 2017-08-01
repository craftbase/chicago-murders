import datetime
import logging
import os
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup


df = pd.read_csv("victim_info_2012_2017.csv")

def get_parameters(df):
    age_race = df[['race']].values
    X = np.unique(age_race)
    print X
    return age_race

def remove_non_numeric_age(df):

    #df = df(df['age'].str.contains("year")).loc['age'].apply(extract_numeric_age)
    df['age'] = df['age'].apply(extract_numeric_age)
    return df

def clean_data(df):
    df.drop(df.columns[0], axis=1,inplace=True)
    df.dropna(axis=0, inplace=True)
    df = remove_non_numeric_age(df)
    df['time'] = df['time'].apply(scale_time)
    df['cause'] = df['cause'].apply(clean_cause)
    df['month'] = df['date'].apply(extract_month)

    #df['age'].fillna(df['age'].median(),inplace=True)
    #df.head()
    #print df
    return df

def extract_numeric_age(string):
    #for char in string:
    if has_string(string):
        sub_strings = string.split()
        string = sub_strings[0];
    return int(string);

def scale_time(string):
    scaled_time = -100
    am_pm = "something"
    try:
        if ":" in string:
            sub_strings = string.split()
            time = sub_strings[0]
            if len(string)>5:
                am_pm = sub_strings[1]
            else:
                am_pm = "a"

            hour = time.split(":")[0]
            minute = time.split(":")[1]
            hour = int(hour)
            minute = int(minute)
            if am_pm[0] == 'p' and hour != 12:
                hour = 12 + hour
            scaled_time = hour*60 + minute
        else:
            sub_strings = string.split()
            time = sub_strings[0]
            am_pm = sub_strings[1]
            hour = int(time)
            if am_pm[0] == 'p' and hour != 12:
                hour = 12 + hour
            scaled_time = hour*60
    except IndexError:
        print(string)
        print(am_pm)
    return scaled_time

def clean_cause(string):
    if string == "null" or string == "0":
        string = "Shooting"
    return string


def has_string(string):
    return bool(re.search(r'\D', string))

def extract_month(string):
    sub_strings = string.split()
    return sub_strings[0]

def get_dummies(df):
    df = get_cause_dummies(df)
    df = get_race_dummies(df)
    return df

def get_cause_dummies(df):
    df = pd.get_dummies(df, prefix='Cause_', columns=['cause'])
    return df

def get_race_dummies(df):
    df = pd.get_dummies(df, prefix='Race_', columns=['race'])
    return df
#new_df = clean_data(df)
#print new_df['age'].describe()

df = clean_data(df)
df = get_dummies(df)
#print type(df['age'][1])
#get_parameters(df)[1]

print df.head()