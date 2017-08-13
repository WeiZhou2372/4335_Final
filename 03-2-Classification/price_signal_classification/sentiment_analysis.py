from pycorenlp import StanfordCoreNLP
import csv
import re
#from collections import defaultdict

raw_tweets = []
output = open('tweets_data_sentiment_scores.csv','w')
print('date,retweet,favorite,score,type',file=output)

with open('tweet_data_01.csv','r',encoding='mac_roman') as csvfile:
	
	f_reader = csv.reader(csvfile,delimiter=',')
	for row in f_reader:
		tweet_tuple = (row[0],row[1],row[2],row[3]) #date,text,retweet,favorite
		raw_tweets.append(tweet_tuple)
csvfile.close()
#print(len(raw_tweets))
#text = 'Learning a winning stock trading strategy is EASY with Tim Sykes http://smq.tc/1BFjMXK  $MGPI $CORN $CAMP'
# text = '14  Top impressive #ETF $DOD $RWX $IFGL $BSJG $ERO $SCHC $VEGI $HGI $UYG $URA $GRU $JJG $CORN https://twitter.com/search?f=tweets&vertical=default&q=%24DOD%20OR%20%24RWX%20OR%20%24IFGL%20OR%20%24BSJG%20OR%20%24ERO%20OR%20%24SCHC%20OR%20%24VEGI%20OR%20%24HGI%20OR%20%24UYG%20OR%20%24URA%20OR%20%24GRU%20OR%20%24JJG%20OR%20%24CORN&src=typd …'
# text = re.sub('http\S+\s+','',text)
# print(text)

nlp = StanfordCoreNLP('http://localhost:9000')
cnt = 0
for tweet in raw_tweets:
	cnt += 1
	if cnt == 1:
		continue

	text = tweet[1].strip() #get raw tweet text
	text = re.sub('http\S+\s+','',text) #get rid of urls in the tweet
	res = nlp.annotate(text,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 10000,
                   })
	if len(res) > 1 or len(res) < 0:
		for s in res['sentences']:
			print(tweet[0],s['sentimentValue'],s['sentiment'])
	else:
		#only one long sentence
		for s in res['sentences']:
			print('{},{},{},{},{}'.format(tweet[0],tweet[2],tweet[3],s['sentimentValue'],s['sentiment']),file=output)
			break

output.close()