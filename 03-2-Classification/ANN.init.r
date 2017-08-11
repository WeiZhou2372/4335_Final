needs(TStools, forecast, xts, zoo, caret)
data = source("./03-2-Classification/data.getReady.r", local = T)$value

ttt = data %>%
  dplyr::select(date, y, settle, score) %>%
  na.omit() %>%
  mutate(y = (y >= 0))
ttt = zoo(ttt, order.by = ttt$date)
nrow(ttt)

# trail ver-------------------------------
temp.ttt= ttt %>%
  `[`( ,-1) %>%
  head(500)

aa = train(y~., data = temp.ttt, method = "nnet")
prediction = predict(aa, temp.ttt[,-1])
sum((as.numeric(prediction) == 1) == !(as.character(temp.ttt$y) == "FALSE"))

# prediction = data.frame(pred = prediction, 
#                         date = as.Date.character(as.character(index(temp.ttt))))
# #rownames(prediction) = index(temp.ttt)
# merged = cbind(as.data.frame(temp.ttt), prediction)
# merged = zoo(merged, order.by = merged$date)

# train  -------------------------


# test ---------------------------
test.ttt = ttt %>%
  `[`( ,-1) %>%
  `[`(501,)
prediction = predict(aa, test.ttt[,-1])
sum((as.numeric(prediction) == 1) == !(as.character(test.ttt$y) == "FALSE"))



