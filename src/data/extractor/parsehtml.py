import logging

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
        date = victim.find("div", {"class": "date"}).get_text().strip()
        name = victim.find("h2", {"class": "name"}).get_text().strip()
        age = victim.find("div", {"class": "age"}).contents[1].strip()
        race = victim.find("div", {"class": "race"}).contents[1].strip()
        cause = victim.find("div", {"class": "cause"}).contents[1].strip()
        neighbourhood = victim.find("div", {"class": "neighborhood"}).contents[1].strip()
        time = victim.find("span", {"class": "murder_time"}).get_text().strip()
        addr = victim.find("div", {"class": "address"}).contents[2].strip()
        print "Name " + name + " Age " + age + " Race " + race + " Cause " + cause + " Neigbourhood " + neighbourhood + " Time " + time + " Address " + addr


link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

get_dynamic_content(link)
