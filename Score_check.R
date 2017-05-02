library(dplyr)

load("data_type_abcd_allwindow.Rdata")
scores = read.csv("Fraud Scores.csv")

all_abcd = all_abcd[all_abcd$date > "2010-01-28",]
all_abcd$fraud[is.na(all_abcd$fraud)] = 0

scores$final = scores$RF + 0.8 * scores$NN + 0.6 * scores$SVM + 0.4 * scores$LR + 0.2 * scores$NB

score_comparison = data.frame(fraud = all_abcd$fraud, score = scores$final)
score_final = data.frame(all_abcd[,1:10], score = scores$final)
write.csv(score_final, file = "Fraud Scores Final.csv")

top_score_record = scores %>%
    arrange(-final) %>%
    top_n(100, final) %>%
    select(X)

top_records = all_abcd %>%
    filter(recordnum %in% top_score_record$X)

sum(top_records$fraud)

top_100 = score_comparison %>%
    arrange(-score) %>%
    top_n(300, score)

sum(top_100$fraud)