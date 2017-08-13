needs(dplyr, zoo)

# vector -> data_frame : rollmean
rollmean.matrix = function(x, naming = "p", start = 5, end = 120, by = 5){
  xx = data_frame(origin = x)
  
  for(ii in seq(start, end, by = by)){
    column.name = paste0(naming, ".", ii)
    xx = xx %>%
      mutate(!!column.name := rollmean(x = x, k = ii, align = "right", fill = NA))
  }
  message(paste("Expanded", ncol(xx) - 1, "columns from", start, "to", end, "by", by))
  xx[,-1]
}

df.matrix = function(x, naming = "p", start = 1, end = 120, by = 5){
  xx = data_frame(origin = x)
  
  for(ii in seq(start, end, by = by)){
    column.name = paste0(naming, ".", ii)
    xx = xx %>%
      mutate(!!column.name := x - lag(x, ii))
  }
  message(paste("Expanded", ncol(xx) - 1, "columns from", start, "to", end, "by", by))
  xx[,-1]
}

move.matrix = function(x, naming = "p", start = 1, end = 120, by = 5){
  xx = data_frame(origin = x)
  
  for(ii in seq(start, end, by = by)){
    column.name = paste0(naming, ".", ii)
    xx = xx %>%
      mutate(!!column.name := lag(x, ii))
  }
  message(paste("Expanded", ncol(xx) - 1, "columns from", start, "to", end, "by", by))
  xx[,-1]
}

y.matrix = function(y){
  #input : vector with 1,0,-1
  aa = data.frame(y = y)
  aa = aa %>%
    mutate(buy = as.numeric(y==1),
           hold = as.numeric(y==0),
           sell = as.numeric(y==-1)) %>%
    select(buy, hold, sell) %>%
    as.matrix()
  
  aa
  

}

