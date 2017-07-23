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
    df.dropna(axis=0,inplace=True)
    #df = df(df['age'].str.contains("year")).loc['age'].apply(extract_numeric_age)
    df['age'] = df['age'].apply(extract_numeric_age)
    return df

def clean_data(df):
    df = remove_non_numeric_age(df)
    #df['age'].fillna(df['age'].median(),inplace=True)
    #df.head()
    #print df
    return df

def extract_numeric_age(string):
    #for char in string:
    if has_string(string):
        sub_strings = string.split()
        string = sub_strings[0];
    return string;

def has_string(string):
    return bool(re.search(r'\D', string))

#new_df = clean_data(df)
#print new_df['age'].describe()
clean_data(df)

#get_parameters(df)