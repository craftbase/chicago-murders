from bs4 import BeautifulSoup
from selenium import webdriver

import logging
logging.basicConfig(filename='scrape.log',level=20)
logging.info('Initialized logger')

def getBeautifulSoupObject(html):
    return BeautifulSoup(html)

def getDynamicContent(link):
    try:
        driver = webdriver.PhantomJS()
        logging.info("Webdriver initiated {0}".format(driver))
    except Exception as exp:
        logging.fatal("Webdriver exception: {exception}".format(exception=exp))
        raise exp
    driver.get(link)
    html = driver.page_source
    soup = getBeautifulSoupObject(html)
    #REMOVELATER: Check the HTML for testing purpose
    print soup.prettify()

link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

getDynamicContent(link)