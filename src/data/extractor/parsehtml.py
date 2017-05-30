import logging
import os

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

logging.basicConfig(filename='scrape.log', level=20)
logging.info('Initialized logger')


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
        if not os.path.isfile('victim_info_2012_2018.csv'):
            victim_info_df.to_csv('victim_info_2012_2018.csv', header=True, encoding='utf-8')
        else:  # else it exists so append without writing the header
            victim_info_df.to_csv('victim_info_2012_2018.csv', mode='a', header=False, encoding='utf-8')

        print "Date " + date + " Name " + name + " Age " + age + " Race " + race + " Cause " + cause + " Neigbourhood " + neighbourhood + " Time " + murder_time + " Address " + addr


for year in range(2012, 2018):
    for month in range(1, 13):
        link = "https://www.dnainfo.com/chicago/{}-chicago-murders/timeline?mon={}".format(str(year), str(month))
        get_dynamic_content(link)
