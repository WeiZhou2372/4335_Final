library(pROC)
library(forecast)
setwd("~/Desktop/summer session/4335 ML/final project/")
tweets<-read.csv("~/Desktop/summer session/4335 ML/final project/tweet_rev.csv")
corn<-read.csv("~/Desktop/summer session/4335 ML/final project/corn_price_data.csv")
tweets<-as.data.frame(tweets)
corn<-as.data.frame(corn)
tweets_corn<-merge(tweets,corn,sort=FALSE)
tweets_corn<-tweets_corn[-1:-48,]
data.ts<-ts(tweets_corn,start=c(2011,6),frequency=365)
train<-window(data.ts,end=2015)
test<-window(data.ts,start=2015)
fit1<-tslm(settle~score+retweet+favorite,data=train)
summary(fit1)
forecast(fit1,newdata = as.data.frame(test))
pre1<-forecast(fit1,newdata = as.data.frame(test))
mse1=mean((pre1$lower[,1]-as.data.frame(test)$settle)^2)
pre1.0<-forecast(fit1,newdata = as.data.frame(train))
mse1=mean((pre1.0$lower[,1]-as.data.frame(train)$settle)^2)

###fiveday
fiveday<-matrix(NA, nrow=nrow(tweets_corn)-5,ncol=6)
colnames(fiveday)<-c('date','retweet','favorite','score','type','settle')
for( i in 1:(nrow(tweets_corn)-5)){
  fiveday[i,1]<-tweets_corn[i,1]
  fiveday[i,2]<-mean(tweets_corn[i:i+5,2])
  fiveday[i,3]<-mean(tweets_corn[i:i+5,3])
  fiveday[i,4]<-mean(tweets_corn[i:i+5,4])
  fiveday[i,5]<-tweets_corn[i,5]
  fiveday[i,6]<-tweets_corn[i+5,6]
}
five.ts<-ts(fiveday,start=c(2011,6),frequency=365)
train_five<-window(five.ts,end=2015)
test_five<-window(five.ts,start=2015)
fit_five<-tslm(settle~score+retweet+favorite,data=train_five)
summary(fit_five)
pre_five<-forecast(fit_five,newdata = as.data.frame(test_five))
mse_five=mean((pre_five$lower[,1]-as.data.frame(test_five)$settle)^2)

###tenday
tenday<-matrix(NA, nrow=nrow(tweets_corn)-10,ncol=6)
colnames(tenday)<-c('date','retweet','favorite','score','type','settle')
for( i in 1:(nrow(tweets_corn)-10)){
  tenday[i,1]<-tweets_corn[i,1]
  tenday[i,2]<-mean(tweets_corn[i:i+10,2])
  tenday[i,3]<-mean(tweets_corn[i:i+10,3])
  tenday[i,4]<-mean(tweets_corn[i:i+10,4])
  tenday[i,5]<-tweets_corn[i,5]
  tenday[i,6]<-tweets_corn[i+10,6]
}
ten.ts<-ts(tenday,start=c(2011,6),frequency=365)
train_ten<-window(ten.ts,end=2015)
test_ten<-window(ten.ts,start=2015)
fit_ten<-tslm(settle~score+retweet+favorite,data=train_ten)
summary(fit_ten)
pre_ten<-forecast(fit_ten,newdata = as.data.frame(test_ten))
mse_ten=mean((pre_ten$lower[,1]-as.data.frame(test_ten)$settle)^2)

#nonlinear
fit2<-tslm(log(as.data.frame(train)$settle)~score+retweet+favorite+score*retweet*favorite,data=train)
summary(fit2)
pre2<-forecast(fit2,newdata = as.data.frame(test))
mse2=mean((exp(1)^pre2$lower[,1]-as.data.frame(test)$settle)^2)
pre2.0<-forecast(fit2,newdata = as.data.frame(train))
mse2.0=mean((exp(1)^pre2.0$lower[,1]-as.data.frame(train)$settle)^2)


###fiveday
fit_five2<-tslm(log(as.data.frame(train_five)$settle)~score+retweet+favorite+score*retweet*favorite,data=train_five)
summary(fit_five2)
pre_five2<-forecast(fit_five2,newdata = as.data.frame(test_five))
mse_five2=mean((exp(1)^pre_five2$lower[,1]-as.data.frame(test_five)$settle)^2)
pre_five2.0<-forecast(fit_five2,newdata = as.data.frame(train_five))
mse_five2.0=mean((exp(1)^pre_five2.0$lower[,1]-as.data.frame(train_five)$settle)^2)

###tenday
fit_ten2<-tslm(log(as.data.frame(train_ten)$settle)~score+retweet+favorite+score*retweet*favorite,data=train_ten)
summary(fit_ten2)
pre_ten2<-forecast(fit_ten2,newdata = as.data.frame(test_ten))
mse_ten2=mean((exp(1)^pre_ten2$lower[,1]-as.data.frame(test_ten)$settle)^2)
pre_ten2.0<-forecast(fit_ten2,newdata = as.data.frame(train_ten))
mse_ten2.0=mean((exp(1)^pre_ten2.0$lower[,1]-as.data.frame(train_ten)$settle)^2)



pre_cur_linear<-forecast(fit1,newdata = as.data.frame(data.ts))
pre_five_linear<-forecast(fit_five,newdata = as.data.frame(data.ts))
pre_ten_linear<-forecast(fit_ten,newdata = as.data.frame(data.ts))
pre_cur_nlinear<-forecast(fit2,newdata = as.data.frame(data.ts))
pre_five_nlinear<-forecast(fit_five2,newdata = as.data.frame(data.ts))
pre_ten_nlinear<-forecast(fit_ten2,newdata = as.data.frame(data.ts))

