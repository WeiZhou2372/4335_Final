a = notebook %>%
  mutate(relative_change = (settle - lag(settle)) / lag(settle)) %>%
  rowwise() %>%
  mutate(cum = 1 + if(y == y.predict){abs(relative_change)}else{-abs(relative_change)}) %>%
  `[`(-1,) %>%
  mutate(cumd = cumprod(cum)) 

a = notebook %>%
  mutate(relative_change = (settle - lag(settle)) / lag(settle)) %>%
  rowwise() %>%
  mutate(cum = y.predict * relative_change +1) %>%
  `[`(-1,) %>%
  mutate(cumd = cumprod(cum)) 

a = train.data %>%
  mutate(relative_change = (settle - lag(settle)) / lag(settle)) %>%
  rowwise() %>%
  mutate(cum = (-1) * relative_change +1) %>%
  `[`(-1,) %>%
  mutate(cumd = cumprod(cum)) 


# adaline.2 -------------------
needs(pROC)
if(TRUE){
  # TEST 
cum.ada.2 = test.data %>%
  select(settle, y) %>%
  cbind(data.frame(y.predict = prediction.test$class)) %>%
  mutate(relative_change = (settle - lag(settle)) / lag(settle)) %>%
  rowwise() %>%
  mutate(cum = 1 + if(y == y.predict){abs(relative_change)}else{-abs(relative_change)}) %>%
  `[`(-1,) %>%
  mutate(cumd = cumprod(cum)) 
rev(cum.ada.2$cumd)[1]^(365/nrow(cum.ada.2))

roc(test.data$y%>%as.character()%>%as.numeric,prediction.test$class%>%as.character()%>%as.numeric,plot=F)

  # TRAIN
cum.ada.2 = train.data %>%
  select(settle, y) %>%
  cbind(data.frame(y.predict = prediction.train$class)) %>%
  mutate(relative_change = (settle - lag(settle)) / lag(settle)) %>%
  rowwise() %>%
  mutate(cum = 1 + if(y == y.predict){abs(relative_change)}else{-abs(relative_change)}) %>%
  `[`(-1,) %>%
  mutate(cumd = cumprod(cum)) 
rev(cum.ada.2$cumd)[1]^(365/nrow(cum.ada.2))
roc(train.data$y%>%as.character()%>%as.numeric,prediction.train$class%>%as.character()%>%as.numeric,plot=F)

}