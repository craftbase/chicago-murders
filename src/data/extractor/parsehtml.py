from bs4 import BeautifulSoup
from selenium import webdriver

import logging
logging.basicConfig(filename='scrape.log',level=20)
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
    #Dump the page source to log for debugging purpose, this will work only if level is set to 10 in line number 5
    logging.debug(soup.prettify())
    parsedata(soup)

def parsedata(soup):
    pass

link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

get_dynamic_content(link)