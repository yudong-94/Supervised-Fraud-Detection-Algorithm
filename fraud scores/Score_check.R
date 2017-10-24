library(dplyr)

setwd("~/Desktop/DSO 562/Project3")

load("all_abcd_updated.rda")
scores = read.csv("Fraud Scores.csv")

all_abcd = all_abcd[all_abcd$date > "2010-01-28",]
all_abcd$fraud[is.na(all_abcd$fraud)] = 0

#scores$final = scores$RF + 0.8 * scores$NN + 0.6 * scores$SVM + 0.4 * scores$LR + 0.2 * scores$NB

score_comparison = cbind(all_abcd[,c(1,10)], scores[,2:6])
#score_final = data.frame(all_abcd[,1:10], score = scores$final)
#write.csv(score_final, file = "Fraud Scores Final.csv")
score_comparison$avg_score = (score_comparison$RF + score_comparison$NN)/2
score_comparison$max_score = ifelse(score_comparison$RF>score_comparison$NN, score_comparison$RF, score_comparison$NN)

top_score_record = score_comparison %>%
    arrange(-avg_score) %>%
    top_n(100, avg_score)

sum(top_score_record$fraud)

top_score_record = score_comparison %>%
    arrange(-avg_score) %>%
    top_n(300, avg_score)

sum(top_score_record$fraud)


top_score_record = score_comparison %>%
    arrange(-max_score) %>%
    top_n(100, max_score)

sum(top_score_record$fraud)

top_score_record = score_comparison %>%
    arrange(-max_score) %>%
    top_n(300, max_score)

sum(top_score_record$fraud)

top_score_record = score_comparison %>%
    arrange(-max_score) %>%
    top_n(889, max_score)

sum(top_score_record$fraud)