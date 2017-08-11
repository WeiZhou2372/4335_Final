needs(TStools, forecast, xts, zoo, caret)
data = source("./03-2-Classification/data.getReady.r", local = T)$value

ttt = data %>%
  dplyr::select(date, y, settle, score) %>%
  na.omit() %>%
  mutate(y = (y >= 0))
# y: tomorrow's settle

start_row = floor(nrow(ttt) * 0.9) 

notes = ttt %>%
  select(date, y) %>%
  mutate(pred = NA) %>%
  `[`(start_row, )
# create a notebook to take down prediction
  
ttt = zoo(ttt, order.by = ttt$date)

