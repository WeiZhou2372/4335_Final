needs(readr, dplyr, stringr)
regex.html = "(https?|ftp|https)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
regex.rept = "[.><$,-]{2,}"

tweets_data <- read_csv("./01-Data Cleaning/tweets_data.csv", 
                        col_types = cols(date = col_datetime(format = "%m/%d/%Y %H:%M"))) %>%
  select(-location) %>%
  mutate(Date = format(date, "%Y-%m-%d") %>% as.Date.character()) %>%
  select(-date) %>%
  filter(Date >= as.Date.character("2010-09-01")) %>% # 有的地方中间间隔太多天
  filter(language == "en") %>%
  select(-language) %>%
  mutate(texted = str_replace_all(text, regex.html, "")) %>%
  mutate(texted = str_replace_all(texted, "\u00a0", " ")) %>%
  mutate(texted = str_replace_all(texted, "\\$\\w+", "")) %>%
  mutate(texted = str_replace_all(texted, "\\#\\w+", "")) %>%
  mutate(texted = str_replace_all(texted, "@\\w+", "")) %>%
  mutate(texted = str_replace_all(texted, "…", "")) %>%
  mutate(texted = str_replace_all(texted, regex.rept, "")) %>%
  group_by(Date, texted) %>%
  summarise(retweets = sum(retweets), favorites = sum(favorites)) %>%
  filter(texted != "")

#write.csv(tweets_data, file = "./01-Data Cleaning/tweet_data_01.csv", row.names = F)
  

# a = tweets_data %>% 
#   mutate(date1 = format(date, "%Y-%m-01") %>% as.Date.character()) %>%
#   group_by(date1) %>%
#   summarise(n = n())

b = tweets_data %>%
  group_by(Date) %>%
  summarise(n = n()) %>%
  mutate(day.diff = Date - lag(Date)) %>%
  filter(day.diff > 1)
