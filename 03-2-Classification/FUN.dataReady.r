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


