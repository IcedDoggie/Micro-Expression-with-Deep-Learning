import requests
from pyquery import PyQuery as pq
from lxml import etree
import urllib

d = pq("<html></html>")
d = pq(etree.fromstring("<html></html>"))
d = pq(url = "https://archive.org/details/FOXNEWSW_20170207_040300_The_OReilly_Factor/start/60/end/120")
# d = pq(filename = path_to_html_file)
# metas = d('meta').attr(property='og:video').eq(23).attr('content')
videos = d('meta').attr(property='og:video')
print(videos)

response = requests.get('https://archive.org/details/FOXNEWSW_20170207_040300_The_OReilly_Factor')
response.content[:40]


# start_index = 0
# jumper = 60
# end_index = 960

# counter = 0

# while counter < end_index:
# 	getLink = urllib.FancyURLopener()
# 	save_name = "trump_" + str(counter) + ".mp4"
# 	url = 'https://archive.org/download/FOXNEWSW_20170207_040300_The_OReilly_Factor/FOXNEWSW_20170207_040300_The_OReilly_Factor.mp4?' + "start=" + str(counter) + "&end=" + str(counter + jumper)
# 	getLink.retrieve(url, save_name)
# 	counter += jumper
# 	print("Downloaded " + url)

# https://archive.org/details/KQED_20161020_010000_PBS_NewsHour_Debates_2016_A_Special_Report


start_index = 0
jumper = 60
end_index = 5880

counter = 0

while counter < end_index:
	getLink = urllib.FancyURLopener()
	save_name = "debate_" + str(counter) + ".mp4"
	url = 'https://archive.org/details/KQED_20161020_010000_PBS_NewsHour_Debates_2016_A_Special_Report.mp4?' + "start=" + str(counter) + "&end=" + str(counter + jumper)
	getLink.retrieve(url, save_name)
	counter += jumper
	print("Downloaded " + url)