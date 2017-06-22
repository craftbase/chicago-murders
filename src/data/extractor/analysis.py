import datetime
import logging
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

df = pd.read_csv("victim_info_2012_2017.csv")

def get_parameters(df):
    age_race = df[['race']].values
    X = np.unique(age_race)
    print X
    return age_race

def remove_non_numeric_age(df):

    mask = df['age'].str.contains("years")
    non_numeric_age = df[mask]
    df = df[df['age'].str.contains("year"),'age'].apply(extract_numeric_age)


def clean_data(df):
    remove_non_numeric_age(df)
    df['age'].fillna(df['age'].median(),inplace=True)
    return df

def extract_numeric_age(string):
    #for char in string:
    sub_strings = string.split()

    return sub_strings[0];

#new_df = clean_data(df)
#print new_df['age'].describe()

#get_parameters(df)