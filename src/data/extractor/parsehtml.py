import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import logging
logging.basicConfig(filename='scrape.log',level=20)
logging.info('Initialized logger')

def getBeautifulSoupObject(link):
    html = requests.get(link)
    return BeautifulSoup(html)

def getDynamicContent(link):
    try:
        driver = webdriver.PhantomJS()
        logging.info("Webdriver initiated {0}".format(driver))
    except Exception as exp:
        logging.fatal("Webdriver exception: {exception}".format(exception=exp))
        raise exp
    driver.get(link)
    #REMOVELATER: test if we are able to get the title
    print driver.title

link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

getDynamicContent(link)