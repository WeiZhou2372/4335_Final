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

b = a %>% 
  mutate(y = lead(settle, 5) - settle,
         p.df1 = settle - lag(settle),
         p.df5 = settle - lag(settle, 5),
         p.df10 = settle - lag(settle, 10),
         p.df20 = settle - lag(settle, 20),
         p.df60 = settle - lag(settle, 60),
         p.df120 = settle - lag(settle, 120),
         p.ma5 = rollmean(x = settle, k = 5, align = "right", fill = NA),
         p.ma10 = rollmean(x = settle, k = 10, align = "right", fill = NA),
         p.ma20 = rollmean(x = settle, k = 20, align = "right", fill = NA),
         p.ma60 = rollmean(x = settle, k = 60, align = "right", fill = NA),
         p.ma120 = rollmean(x = settle, k = 120, align = "right", fill = NA),
         s.ma5 = rollmean(x = score, k = 5, align = "right", fill = NA),
         s.ma10 = rollmean(x = score, k = 10, align = "right", fill = NA),
         s.ma20 = rollmean(x = score, k = 20, align = "right", fill = NA),
         s.ma60 = rollmean(x = score, k = 60, align = "right", fill = NA),
         s.ma120 = rollmean(x = score, k = 120, align = "right", fill = NA)
         )
rm(a, cornPrice.raw, sentiment.raw)

data = b %>%
  na.omit() %>%
  mutate(y = (y >= 0) %>% as.numeric())

data

