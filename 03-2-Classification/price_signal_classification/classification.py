import csv
import numpy as np
from sklearn import preprocessing,svm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
import keras

def data_transformation(filename,newfile):
	raw_data = []
	with open(filename,'r') as f:
		reader = csv.reader(f)
		next(reader) #skip header
		for row in reader:
			raw_data.append((row[0],float(row[2])))
	f.close()

	newfile = open(newfile,'w')
	step_size = 5

	for i in range(0,len(raw_data)-1):
		if i + step_size < len(raw_data) - 1:
			mean = np.mean(np.array([x[1] for x in raw_data[i+1:i+step_size]]))
			if (abs(raw_data[i][1] - mean) / raw_data[i][1]) < 0.01: 
				# print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'0'),file=newfile)
				if raw_data[i][1] > mean:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'-1'),file=newfile)
				else:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'1'),file=newfile)
			else:
				if raw_data[i][1] > mean:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'-1'),file=newfile)
				else:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'1'),file=newfile)
		else:
			mean = np.mean(np.array([x[1] for x in raw_data[i+1:]]))
			if (abs(raw_data[i][1] - mean) / raw_data[i][1]) < 0.01: 
				#print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'0'),file=newfile)
				if raw_data[i][1] > mean:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'-1'),file=newfile)
				else:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'1'),file=newfile)
			else:
				if raw_data[i][1] > mean:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'-1'),file=newfile)
				else:
					print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'1'),file=newfile)

	if raw_data[-1][1] > raw_data[-2][1]:
		print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'1'),file=newfile)
	else:
		print('{},{},{}'.format(raw_data[i][0],raw_data[i][1],'-1'),file=newfile)
	newfile.close()

def date_transformation(s):
	year,month,date = s.split('-')
	return str(int(month)) + '/' + str(int(date)) + '/' + year

def data_split(price_file,tweet_file,N):
	
	dates,price,signal = [],[],[]
	with open(price_file,'r') as f1:
		reader = csv.reader(f1)
		for line in reader:
			dates.append(line[0])
			price.append(float(line[1]))
			signal.append(int(line[2]))
	f1.close()

	tweet_dict = defaultdict(list)
	with open(tweet_file,'r') as f2:
		reader = csv.reader(f2)
		next(reader) #skip header
		for line in reader:
			tweet_dict[date_transformation(line[0])].append((line[1],line[2],int(line[3])))
	f2.close()

	
	#prepare data for training and testing
	full_x,full_y = [],[]
	total_cnt = 0
	empty_cnt = 0

	start,end = 0,0
	for i in range(0,len(dates)):
		total_cnt += 1
		end = i
		try:
			x_i = preprocessing.scale(price[i:i+N])
			tweet_dates = dates[i:i+10]
			# y_i = signal[i+N]
			if signal[i+N] == 1:
				# y_i = [0,1]
				y_i=1
			elif signal[i+N] == -1:
				# y_i = [1,0]
				y_i=-1
			else:
				# y_i = [0,0]
				y_i=0

			if int(tweet_dates[0].split('/')[2]) < 2010:
				continue

			for date in tweet_dates:
				if len(tweet_dict[date]) == 0:
					# print(date)
					x_i = np.append(x_i,2)
				else:
					sentiments = [item[2] for item in tweet_dict[date]]
					x_i = np.append(x_i,np.mean(sentiments))


			# date = str(dates[i+N])
			# if len(tweet_dict[date]) == 0:
			# 	empty_cnt += 1
			# 	continue
			# else:
			# 	sentiments = [item[2] for item in tweet_dict[date]]
			# 	x_i = np.append(x_i,np.mean(sentiments))
				#print(date,np.mean(sentiments))
			#impute for missing value
			#x_i = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(x_i).transform(x_i)
			#print(x_i,y_i)
		except:
			#print(Imputer(missing_values='NaN', strategy='mean', axis=0).fit(x_i).transform(x_i))
			break
		full_x.append(preprocessing.scale(x_i))
		full_y.append(y_i)
	print('{}/{}'.format(empty_cnt,total_cnt))
	print('start date:{} end date:{}'.format(dates[start],dates[end]))
	return np.array(full_x),np.array(full_y)

def auprc(true,pred,flag):
	# For each class
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(3):
		precision[i], recall[i], _ = precision_recall_curve(true[:, i],pred[:, i])
		average_precision[i] = average_precision_score(true[:, i], pred[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(true.ravel(),pred.ravel())
	average_precision["micro"] = average_precision_score(true, pred,average="micro")
	print(flag+' : Average precision score, micro-averaged over all classes: {0:0.8f}'.format(average_precision["micro"]))

def logistic(X,Y):
	#print(X,Y)
	f1 = open('logistic-prediction-binary-train.csv','w')
	f2 = open('logistic-prediction-binary-test.csv','w')
	#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
	x_train,x_test = np.split(X,[int(0.8*len(X))])
	y_train,y_test = np.split(Y,[int(0.8*len(Y))])
	
	model = LogisticRegression(C=0.01).fit(x_train,y_train)
	#model =  OneVsRestClassifier(LogisticRegression(C=0.01)).fit(x_train,y_train)
	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)

	# y_train = label_binarize(y_train, classes=[-1, 0, 1])
	# y_test = label_binarize(y_test, classes=[-1, 0, 1])
	# train_pred = label_binarize(train_pred, classes=[-1, 0, 1])
	# test_pred = label_binarize(test_pred, classes=[-1, 0, 1])

	# auprc(y_test,test_pred,'test')
	# auprc(y_train,train_pred,'train')

	test_auprc = average_precision_score(y_test, test_pred)
	train_auprc = average_precision_score(y_train, train_pred)

	for item in train_pred:
		print(item,file=f1)
	for item in test_pred:
		print(item,file=f2)
	f1.close()
	f2.close()

	print('Test precision-recall score: {0:0.4f}'.format(
      test_auprc))
	print('Train precision-recall score: {0:0.4f}'.format(
      train_auprc))

def LSVM(X,Y):
	#print(X,Y)
	f1 = open('LSVM-prediction-binary-train.csv','w')
	f2 = open('LSVM-prediction-binary-test.csv','w')
	#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
	x_train,x_test = np.split(X,[int(0.8*len(X))])
	y_train,y_test = np.split(Y,[int(0.8*len(Y))])
	# y_train = label_binarize(y_train, classes=[-1, 0, 1])
	# y_test = label_binarize(y_test, classes=[-1, 0, 1])


	#model = OneVsRestClassifier(svm.LinearSVC(random_state=np.random.RandomState(0))).fit(x_train, y_train)
	model = svm.LinearSVC(random_state=np.random.RandomState(0)).fit(x_train, y_train)
	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)

	for item in train_pred:
		print(item,file=f1)
	for item in test_pred:
		print(item,file=f2)

	f1.close()
	f2.close()
	# y_train = label_binarize(y_train, classes=[-1, 0, 1])
	# y_test = label_binarize(y_test, classes=[-1, 0, 1])
	# train_pred = label_binarize(train_pred, classes=[-1, 0, 1])
	# test_pred = label_binarize(test_pred, classes=[-1, 0, 1])

	# for item in zip(test_pred,y_test):
	# 	print(item)

	# auprc(y_test,test_pred,'test')
	# auprc(y_train,train_pred,'train')
	test_auprc = average_precision_score(y_test, test_pred)
	train_auprc = average_precision_score(y_train, train_pred)

	print('Test precision-recall score: {0:0.4f}'.format(
      test_auprc))
	print('Train precision-recall score: {0:0.4f}'.format(
      train_auprc))

def KSVM(X,Y):
	#print(X,Y)
	f1 = open('KSVM-prediction-binary-train.csv','w')
	f2 = open('KSVM-prediction-binary-test.csv','w')
	#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
	x_train,x_test = np.split(X,[int(0.8*len(X))])
	y_train,y_test = np.split(Y,[int(0.8*len(Y))])
	

	C_range = np.logspace(-2, 10, 10)
	gamma_range = np.logspace(-9, 3, 10)
	param_grid = dict(gamma=gamma_range, C=C_range)

	#model = OneVsRestClassifier(GridSearchCV(SVC(), param_grid=param_grid)).fit(x_train, y_train)

	model = GridSearchCV(SVC(), param_grid=param_grid).fit(x_train, y_train)

	print("The best parameters are %s with a score of %0.2f"% (model.best_params_, model.best_score_))

	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)

	for item in train_pred:
		print(item,file=f1)
	for item in test_pred:
		print(item,file=f2)
	f1.close()
	f2.close()

	# y_train = label_binarize(y_train, classes=[-1, 0, 1])
	# y_test = label_binarize(y_test, classes=[-1, 0, 1])
	# train_pred = label_binarize(train_pred, classes=[-1, 0, 1])
	# test_pred = label_binarize(test_pred, classes=[-1, 0, 1])

	# auprc(y_test,test_pred,'test')
	# auprc(y_train,train_pred,'train')

	test_auprc = average_precision_score(y_test, test_pred)
	train_auprc = average_precision_score(y_train, train_pred)

	print('Test precision-recall score: {0:0.4f}'.format(
      test_auprc))
	print('Train precision-recall score: {0:0.4f}'.format(
      train_auprc))

def MLP(X,Y,N):

	f1 = open('MLP-prediction-binary-train.csv','w')
	f2 = open('MLP-prediction-binary-test.csv','w')
	#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
	x_train,x_test = np.split(X,[int(0.8*len(X))])
	y_train,y_test = np.split(Y,[int(0.8*len(Y))])
	# y_train = keras.utils.to_categorical(y_train, num_classes=3)
	# y_test = keras.utils.to_categorical(y_test, num_classes=3)

	model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
	model.add(Dense(128, activation='relu', input_dim=N+10))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

	model.fit(x_train, y_train,epochs=20,batch_size=128)

	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)

	for item in train_pred:
		print(int(item[0]),file=f1)
	for item in test_pred:
		print(int(item[0]),file=f2)
	f1.close()
	f2.close()

	print("training:", model.evaluate(x_train, y_train, batch_size=128))
	print("test:", model.evaluate(x_test, y_test, batch_size=128))
	#score = model.evaluate(x_train, y_train, batch_size=128)

def annulized_return(price_singal,pred_train,pred_test):
	days = 0
	price_dict = {}
	dates = []
	with open(price_singal,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			price_dict[row[0]] = (float(row[1]),int(row[2]))
			dates.append(row[0])
	f.close()

	label_train,label_test = [],[]
	with open(pred_train,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			label_train.append(int(row[0]))
	f.close()
	with open(pred_test,'r') as f:
		reader = csv.reader(f)
		for row in reader:
			label_test.append(int(row[0]))

	print(len(label_train),len(label_test))

	# money = 10000
	# profit = 0
	# N = 120

	# print(len(label_train),int(0.8*(len(dates)-N)))

	# #for train
	# for i in range(0,len(label_train)):
	# 	price_now = price_dict[dates[i+N+2]][0]
	# 	price_next = price_dict[dates[i+N+3]][0]
	# 	if label_train[i] == 1:
	# 		#print(money,price_now,money/price_now)
	# 		profit = (money/price_now) * (price_next - price_now)
	# 		money += profit
	# 		#print(money)
	# print(money)

	# money_test = 10000
	# profit_test = 0
	# #for test
	# for i in range(0,len(label_test)):
	# 	price_now = price_dict[dates[i+len(label_train)+N+2]][0]
	# 	price_next = price_dict[dates[i+len(label_train)+N+3]][0]
	# 	if label_test[i] == 1:
	# 		profit_test = (money_test/price_now) * (price_next - price_now)
	# 		money_test += profit_test
	# 		#print(money_test)
	# print(money_test)



if __name__ == '__main__':
	# data_transformation('corn_price_data.csv','price_signal_binary.csv')
	N = 120
	#X,Y = data_split('price_signal_binary.csv','tweets_data_sentiment_scores.csv',N)
	#logistic(X,Y)
	#LSVM(X,Y)
	#KSVM(X,Y)
	#MLP(X,Y,N)
	annulized_return('price_signal_binary.csv',
		'KSVM-prediction-binary-train.csv','KSVM-prediction-binary-test.csv')
	
