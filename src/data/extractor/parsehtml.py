from urllib import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

def getBeautifulSoupObject(link):
    html = urlopen(link)
    return BeautifulSoup(html)

def getDynamicContent(link):
    driver = webdriver.PhantomJS()
    driver.get(link)


link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

getDynamicContent(link)