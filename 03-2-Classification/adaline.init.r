data = source("./03-2-Classification/data.getReady2.r", local = T)$value

if(!TRUE){
  needs(fastAdaboost)
  # config ==================
  cacheSize = 1000
  startRow = 1942
  endRow = 2077
  
  notebook = data %>%
    select(date, settle, score, y) %>%
    mutate(y.predict = NA)
  
  # train & test ================
  data.wor = data %>%
    mutate(date = as.numeric(date))
  
  for(i in startRow:endRow){
    train.data = data.wor[(i - cacheSize + 1):i,] 
    test.data = data.wor[i + 1,]
    
    model = dbn.dnn.train(train.data %>% select(-y) %>% as.matrix(), y.matrix(train.data$y),
                          hidden = c(10,10,10), numepochs = 30, batchsize = 20)
    
    prediction = nn.predict(model, test.data %>% select(-y) %>% as.matrix()) %>% 
      as.numeric() 
    
    if(prediction[1] == max(prediction)){
      notebook[i, "y.predict"] = 1
    } else if(prediction[3] == max(prediction)){
      notebook[i, "y.predict"] = -1
    } else {
      notebook[i, "y.predict"] = 0
    }
    
  }
  
  notebook = notebook[startRow:endRow,]
  beepr::beep()
}


if(TRUE){
  needs(fastAdaboost)
  notebook = data %>%
    select(date, settle, score, y) %>%
    mutate(y.predict = NA)
  
  # train ======
  data.wor = data %>%
    mutate(date = as.numeric(date) %% 365,
           y = as.factor(y)) %>%
    as.data.frame()
  
  train.data = data.wor[1:1900,] 
  test.data = data.wor[1901:2077,]
  
  model = adaboost(y~., data = train.data, nIter = 10)
  
  prediction.train = predict(model, train.data[,-4])
  prediction.test = predict(model, test.data[,-4])
  
}