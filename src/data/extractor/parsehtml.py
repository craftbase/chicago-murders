from urllib import urlopen
from bs4 import BeautifulSoup

def getBeautifulSoupObject(link):
    html = urlopen(link)
    return BeautifulSoup(html)



link = "https://www.dnainfo.com/chicago/2017-chicago-murders"

print getBeautifulSoupObject(link)