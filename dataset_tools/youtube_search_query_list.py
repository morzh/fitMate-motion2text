import os
import pickle
import urllib.request
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch
from youtubesearchpython.__future__ import VideosSearch


folder_dataset = '/home/anton/work/fitMate/datasets/LegPressDataset'
filename_links = 'leg_press_urls.txt'
textToSearch = 'leg press'


videosSearch = VideosSearch(textToSearch, limit = 1)

print(videosSearch.result())