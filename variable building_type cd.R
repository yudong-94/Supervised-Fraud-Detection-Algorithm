# load cleaned data
load("Card_Payments_Cleaned.rda")
library(dplyr)
library(lubridate)


#############################   Function packaging     ###############################
build_var <- function(df, time_window, key) {
  
  ###########################
  # df: the name of the cleaned data frame 
  # time_window: 3 or 7 or other
  # key: "zip" or "state"
  ###########################
  
  df[, paste0(key, "_with_merchnum_", time_window)] = 1
  
  for (i in 1:nrow(df)) {
    
    current_date = df[i,"date"]
    current_merch = df[i, "merchnum"]
    if (key == "zip") {
      current_key = df[i,"merch.zip"]
      subset = df[df$date >= current_date - time_window & df$recordnum < i & df$merchnum == current_merch, c("merch.zip", "merchnum")]
      
      if (nrow(subset)!=0) {
        df[i, paste0("zip_with_merchnum_", time_window)] = length(unique(subset$merch.zip))
      }
      
    } else if (key == "state") {
      current_key = df[i,"merch.state"]
      subset = df[df$date >= current_date - time_window & df$recordnum < i & df$merchnum == current_merch, c("merch.state", "merchnum")]
      
      if (nrow(subset)!=0) {
        df[i, paste0("state_with_merchnum_", time_window)] = length(unique(subset$merch.state))
      }
      
    }
    
  }
  return(df)
}


########################## Building Variable #############################

############ Trial #############
# try = data[1:1000,]
# try = build_var(try3, 3, "zip")


# Number of zip per merchant - 3 day
ptm <- proc.time()
data = build_var(data, 3, "zip")
proc.time() - ptm
# 8 min
#summary(data$zip_with_merchnum_3)

# Number of state per merchant - 3 day
ptm <- proc.time()
data = build_var(data, 3, "state")
proc.time() - ptm
# 8 min
#summary(data$state_with_merchnum_3)

# Number of zip per merchant - 7 day
ptm <- proc.time()
data = build_var(data, 7, "zip")
proc.time() - ptm
# 8 min
summary(data$zip_with_merchnum_7)

# Number of state per merchant - 7 day
ptm <- proc.time()
data = build_var(data, 7, "state")
proc.time() - ptm
# 8 min


#############################   Function packaging     ###############################
build_vard <- function(df, time_window, key) {
  
  ###########################
  # df: the name of the cleaned data frame 
  # time_window: 3 or 7 or other
  # key: "zip" or "state"
  ###########################
  if (key == "cardnum") {
    df[, paste0(key, "_per_merch_", time_window)] = 1
  } else if (key == "merchnum") {
    df[, paste0(key, "_per_card_", time_window)] = 1
  }
  
  for (i in 1:nrow(df)) {
    #print(i)
    current_date = df[i,"date"]
    current_merch = df[i, "merchnum"]
    current_card = df[i, "cardnum"]
    
    
    if (key == "cardnum") {
     
      current_key = df[i,"cardnum"]
      subset = df[df$date >= current_date - time_window & 
                    df$recordnum < i & 
                    df$merchnum == current_merch, c("cardnum", "merchnum")]
      
      if (nrow(subset)!=0) {
        df[i, paste0(key, "_per_merch_", time_window)] = length(unique(subset$cardnum))
      }
      
    } else if (key == "merchnum") {
      
      current_key = df[i,"merchnum"]
      subset = df[df$date >= current_date - time_window & 
                    df$recordnum < i & 
                    df$cardnum == current_card, c("cardnum", "merchnum")]
      
      if (nrow(subset)!=0) {
        df[i, paste0(key, "_per_card_", time_window)] = length(unique(subset$merchnum))
      }
      
    }
    
  }
  return(df)
}

# Number of card per merchant - 3 day
ptm <- proc.time()
data = build_vard(data, 3, "cardnum")
proc.time() - ptm
# 10 min
summary(data$cardnum_per_merch_3)

# Number of card per merchant - 7 day
ptm <- proc.time()
data = build_vard(data, 7, "cardnum")
proc.time() - ptm
# 10 min

# Number of merchant per card - 3 day
ptm <- proc.time()
data = build_vard(data, 3, "merchnum")
proc.time() - ptm
# 14 min
# summary(data$merchnum_per_card_3)

# Number of merchant per card - 7 day
ptm <- proc.time()
data = build_vard(data, 7, "merchnum")
proc.time() - ptm
# 15 min
# summary(data$merchnum_per_card_7)


save(data, file = "data_type_cd.Rdata")


