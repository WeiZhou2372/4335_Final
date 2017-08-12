# config ------------------
N = 10           #balance number, to balance sentiment score with retweet and favorite
days.lookInTo = 5     # number of days to predict


# main -------------------
source("./03-2-Classification/FUN.dataReady.r")
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
            score = mean(score * (retweet + favorite + N)) ) %>% # this 10 may be changed later
  mutate(score = score - mean(score, na.rm = T)) 

a$settle = na.approx(object = a$settle, x = a$date)
a$score = na.approx(object = a$score, x = a$date)

a = a %>%
  mutate(y = lead(settle, days.lookInTo) - settle) 

settle_expand = rollmean.matrix(a$settle, naming = "p.roll", 1, 120, 1)
score_expand = df.matrix(a$score, naming = "s.roll", 1, 120, 1)

data = cbind(a, settle_expand, score_expand) %>%
 mutate(y = as.numeric(y >= 0)) %>%
  na.omit()
