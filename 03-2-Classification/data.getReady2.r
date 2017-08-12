needs(readr, dplyr, zoo, forecast)
sentiment.raw <- read_csv("./03-2-Classification/sentiment_scores.csv", 
                          col_types = cols(date = col_date(format = "%Y-%m-%d")))
cornPrice.raw <- read_csv("./03-2-Classification/corn_price.csv", 
                          col_types = cols(date = col_date(format = "%m/%d/%Y"))) 

a = full_join(cornPrice.raw, sentiment.raw, by = "date") %>%
  dplyr::select(date, settle, score, retweet, favorite) %>%
  filter(date > as.Date.character("2011-01-01") & date < as.Date.character("2017-06-01")) %>%
  mutate(score = score - 2) %>%
  arrange(date) %>%
  group_by(date) %>%
  summarise(settle = mean(settle),
            score = mean(score * (retweet + favorite + 10)) ) %>% # this 10 may be changed later
  mutate(score = score - mean(score, na.rm = T))

a$settle = na.approx(object = a$settle, x = a$date)
a$score = na.approx(object = a$score, x = a$date)