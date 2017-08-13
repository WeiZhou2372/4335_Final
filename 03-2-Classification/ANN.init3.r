data = source("./03-2-Classification/data.getReady2.r", local = T)$value
# deepnet -------------------------------
if(!TRUE){
needs(deepnet)
dim(data) #2088 74
data.wor = data %>%
  mutate(date = as.numeric(date))
  #select(-date)
train.data = data.wor[10:500,] 
test.data = data.wor[501:510,]

model = dbn.dnn.train(train.data %>% select(-y) %>% as.matrix(), train.data$y,
                 hidden = c(10,10,10), numepochs = 30, batchsize = 20)

nn.test(model, test.data %>% select(-y) %>% as.matrix(),test.data$y)

prediction = nn.predict(model, test.data %>% select(-y) %>% as.matrix())
#prediction
}
if(!TRUE){
  needs(nnet)
  data.wor = data %>%
    select(-date)
  train.data = data.wor[10:400,] 
  test.data = data.wor[401:410,]
  
  model = nnet(y~., data = train.data, size = 10)
  
}

if(TRUE){
  needs(deepnet)
  # config ==================
  cacheSize = 300
  startRow = 1942
  endRow = 2000
  
  notebook = data %>%
    select(date, settle, score, y) %>%
    mutate(y.predict = NA)
    
  # train & test ================
  data.wor = data %>%
    mutate(date = as.numeric(date))
  
  for(i in startRow:endRow){
    train.data = data.wor[(i - cacheSize + 1):i,] 
    test.data = data.wor[i + 1,]
    
    model = dbn.dnn.train(train.data %>% select(-y) %>% as.matrix(), train.data$y,
                          hidden = c(10,10,10), numepochs = 30, batchsize = 20)
    
    prediction = nn.predict(model, test.data %>% select(-y) %>% as.matrix()) %>% 
      as.numeric() %>%
      `[`(1)
    
    if(prediction > 0.5){
      notebook[i, "y.predict"] = 1
    } else {
      notebook[i, "y.predict"] = 0
    }
    
  }
  
  notebook = notebook[startRow:endRow,]
  beepr::beep()
}
sum(notebook$y == notebook$y.predict) / nrow(notebook)
# nnet --------------------------------
