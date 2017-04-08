setwd("~/Desktop/DSO 562/Project3")
#payments = read.csv("Card payments.csv")
load("Card Payments_Cleaned.rda")

library(dplyr)


#############################  Type a var     ######################
# historical count with 3 days time window

##############   Subset trial     #############
data_Jan = filter(data, date < "2010-02-01")
current = data_Jan[6748,]
row = 6748

###### time window subsetting method
ptm <- proc.time()
#subset = filter(data_Jan, date >= current$date - 3, recordnum < row)
for (i in 1:nrow(data_Jan)) {
    current_date = data_Jan[i,"date"]
    current_key = data_Jan[i,"cardnum"]
    subset = data_Jan %>% 
        filter(date >= current_date - 3, recordnum < i, cardnum == current_key) %>%
        select(cardnum, amount)
}
proc.time() - ptm
# 20s

ptm <- proc.time()
#subset = data_Jan[data_Jan$date >= current$date - 3 & data_Jan$recordnum < row,]
for (i in 1:nrow(data_Jan)) {
    current_date = data_Jan[i,"date"]
    current_key = data_Jan[i,"cardnum"]
    subset = data_Jan[data_Jan$date >= current_date - 3 & data_Jan$recordnum < i & data_Jan$cardnum == current_key, c("cardnum", "amount")]
}
proc.time() - ptm
# 11s


###### 3-day trial on the subset
ptm <- proc.time()
for (i in 1:nrow(data_Jan)) {
    current_date = data_Jan[i,"date"]
    current_key = data_Jan[i,"cardnum"]
    current_amount = data_Jan[i,"amount"]
    subset = data_Jan[data_Jan$date >= current_date - 3 & data_Jan$recordnum < i & data_Jan$cardnum == current_key, c("cardnum", "amount")]
    if (nrow(subset) != 0) {
        avg = mean(subset$amount)
        max = max(subset$amount)
        median = median(subset$amount)
        total = sum(subset$amount)
        data_Jan[i, paste0("amount_hist_avg_","3")] = current_amount / avg
        data_Jan[i, paste0("amount_hist_max_","3")] = current_amount / max
        data_Jan[i, paste0("amount_hist_median_","3")] = current_amount / median
        data_Jan[i, paste0("amount_hist_total_","3")] = current_amount / total
    } else {
        data_Jan[i, paste0("amount_hist_avg_","3")] = 1
        data_Jan[i, paste0("amount_hist_max_","3")] = 0
        data_Jan[i, paste0("amount_hist_median_","3")] = 1
        data_Jan[i, paste0("amount_hist_total_","3")] = 0
    }
}
proc.time() - ptm
# 15s

##############   Function packaging     #############
build_var <- function(df, time_window, key) {
    
    ###########################
    # df: the name of the cleaned data frame 
    # time_window: 3 or 7 or other
    # key: "card" or "merchant"
    ###########################
    
    df[, paste0(key, "_", "amount_to_avg_", time_window)] = 1
    df[, paste0(key, "_", "amount_to_max_", time_window)] = 0
    df[, paste0(key, "_", "amount_to_median_", time_window)] = 1
    df[, paste0(key, "_", "amount_to_total_", time_window)] = 0
    for (i in 1:nrow(df)) {
        #print(i)
        current_date = df[i,"date"]
        current_amount = df[i,"amount"]
        if (key == "card") {
            current_key = df[i,"cardnum"]
            subset = df[df$date >= current_date - time_window & df$recordnum < i & df$cardnum == current_key, c("cardnum", "amount")]
        } else if (key == "merchant") {
            current_key = df[i,"merchnum"]
            subset = df[df$date >= current_date - time_window & df$recordnum < i & df$merchnum == current_key, c("merchnum", "amount")]
        }
        #print(nrow(subset))
        if (nrow(subset) != 0) {
            avg = mean(subset$amount)
            max = max(subset$amount)
            median = median(subset$amount)
            total = sum(subset$amount)
            df[i, paste0(key, "_", "amount_to_avg_",time_window)] = current_amount / avg
            df[i, paste0(key, "_", "amount_to_max_", time_window)] = current_amount / max
            df[i, paste0(key, "_", "amount_to_median_", time_window)] = current_amount / median
            df[i, paste0(key, "_", "amount_to_total_", time_window)] = current_amount / total
        }
    }
    return(df)
}

# ptm <- proc.time()
# data_Jan = build_var(data_Jan, 3, "card")
# proc.time() - ptm
# # 14s
# 
# ptm <- proc.time()
# data_Jan = build_var(data_Jan, 7, "card")
# proc.time() - ptm
# # 14s
# 
# ptm <- proc.time()
# data_Jan = build_var(data_Jan, 3, "merchant")
# proc.time() - ptm
# # 5s
# 
# ptm <- proc.time()
# data_Jan = build_var(data_Jan, 7, "merchant")
# proc.time() - ptm
# # 5s

##############   Run on the whole dataset     #############
ptm <- proc.time()
data = build_var(data, 3, "card")
proc.time() - ptm
# 1016 s = 17 min

ptm <- proc.time()
data = build_var(data, 3, "merchant")
proc.time() - ptm
# 774 s = 13 min

ptm <- proc.time()
data = build_var(data, 7, "card")
proc.time() - ptm
# 1265 s = 21 min

ptm <- proc.time()
data = build_var(data, 7, "merchant")
proc.time() - ptm
# 866 s = 14 min

save(data, file = "data_type_a.Rda")
write.csv(data, file = "data_type_a.csv")
