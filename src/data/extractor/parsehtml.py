import datetime
import logging
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import parse
from selenium import webdriver

logging.basicConfig(filename='scrape.log', level=20)
logging.info('Initialized logger')

CSV_FILE = "victim_info_2012_2017.csv"


def get_soup_object(html):
    return BeautifulSoup(html)


def get_dynamic_content(link):
    try:
        driver = webdriver.PhantomJS()
        logging.info("Webdriver initiated {0}".format(driver))
    except Exception as exp:
        logging.fatal("Webdriver exception: {exception}".format(exception=exp))
        raise exp

    driver.get(link)
    r = requests.get(link)
    if r.status_code != 200:
        raise Exception("Status code is not 200. Exiting!")
    html = driver.page_source
    soup = get_soup_object(html)
    # Dump the page source to log for debugging purpose, this will work only if level is set to 10 in line number 5
    logging.debug(soup.prettify())
    parsedata(soup)


def parsedata(soup):
    victims = soup.find_all("div", {"class": "homicide"})
    for victim in victims:
        # Set default values as NA for each victim
        date = name = age = race = cause = neighbourhood = murder_time = addr = "NA"
        try:
            date = victim.find("div", {"class": "date"}).get_text().strip()
        except Exception as exp:
            pass
        try:
            name = victim.find("h2", {"class": "name"}).get_text().strip()
        except Exception as exp:
            pass
        try:
            age = victim.find("div", {"class": "age"}).contents[1].strip()
        except Exception as exp:
            pass
        try:
            race = victim.find("div", {"class": "race"}).contents[1].strip()
        except Exception as exp:
            pass
        try:
            cause = victim.find("div", {"class": "cause"}).contents[1].strip()
        except Exception as exp:
            pass
        try:
            neighbourhood = victim.find("div", {"class": "neighborhood"}).contents[1].strip()
        except Exception as exp:
            pass
        try:
            murder_time = victim.find("span", {"class": "murder_time"}).get_text().strip()
        except Exception as exp:
            pass
        try:
            addr = victim.find("div", {"class": "address"}).contents[2].strip()
        except Exception as exp:
            pass

        victim_info = {'name': name,
                       'date': date,
                       'age': age,
                       'race': race,
                       'cause': cause,
                       'neighbourhood': neighbourhood,
                       'time': murder_time,
                       'address': addr
                       }

        victim_info_df = pd.DataFrame(victim_info, index=[0])

        # if file does not exist write header
        if not os.path.isfile(CSV_FILE):
            victim_info_df.to_csv(CSV_FILE, header=True, encoding='utf-8')
        else:  # else it exists so append without writing the header
            victim_info_df.to_csv(CSV_FILE, mode='a', header=False, encoding='utf-8')

        print "Date " + date + " Name " + name + " Age " + age + " Race " + race + " Cause " + cause + " Neigbourhood " + neighbourhood + " Time " + murder_time + " Address " + addr


def get_all_data():
    for year in range(2012, 2018):
        for month in range(1, 13):
            link = "https://www.dnainfo.com/chicago/{}-chicago-murders/timeline?mon={}".format(str(year), str(month))
            get_dynamic_content(link)


def get_last_entry_date():
    df = pd.read_csv(CSV_FILE)
    total_rows = len(df.index)
    date = df['date'].values[total_rows - 1]
    date_str = parse(date)
    date_str = date_str.strftime('%Y%m%d')
    return date_str


def update_data():
    last_date = get_last_entry_date()
    print "Last updated entry was on {}".format(last_date)
    entry_year = int(last_date[:4])
    entry_month = int(last_date[4:6])
    current_date = datetime.date.today()
    current_month = int(current_date.month)
    current_year = int(current_date.year)

    for year in range(entry_year, current_year + 1):
        for month in range(1, 13):
            if year == entry_year and month <= entry_month:
                continue
            link = "https://www.dnainfo.com/chicago/{}-chicago-murders/timeline?mon={}".format(str(year), str(month))
            get_dynamic_content(link)


def get_data():
    if not os.path.isfile(CSV_FILE):
        print "CSV file not found. Creating CSV file and appending data to it"
        get_all_data()
    else:
        print "CSV file found."
        update_data()


get_data()
# print get_last_entry_date()
