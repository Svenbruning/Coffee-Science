# VISITOR.CSV DESIGN
visitor <- read.csv("visitor_sample.csv", sep = ";", header = FALSE)
colnames(visitor) <- c("ID Ticket", "Type of ticket", "Date", "Time", "Nr")

visitor$HolidayNetherlands <- rep(0, 431)
visitor$HolidayGermany <- rep(0, 431)
visitor$Event <- rep(0, 431)
visitor$Promotion <- rep(0, 431)
visitor$Temperature <- rep(0, 431)
visitor$Precipitation <- rep(0, 431)

# Date in visitor.csv was char, not date type
visitor$Date <- as.Date(visitor$Date, format = "%Y-%m-%d")

View(visitor)


# WEATHER.CSV DESIGN
weather <- read.csv("weather.csv")

# convert the Date column to a proper Date object
weather$Date <- as.Date(weather$Date, format = "%Y-%m-%d")

# keep only rows between 2025-01-02 and 2025-01-07 like in visitor.csv
weather <- subset(weather,
                  Date >= as.Date("2025-01-02") &
                    Date <= as.Date("2025-01-07"))

# remove column ID_time I don't need
weather$ID_Time <- NULL

View(weather)


# COMBINE VISITOR.CSV AND WEATHER.CSV IN A NEW CSV: JOINED.CSV
library(dplyr)

# join the two csv files based on Date and Time
joined <- left_join(visitor, weather, by = c("Date", "Time"))

# %>% is called the pipe operator in R
# passes the output of one function as the first argument to the next function
# lets you write a sequence of operations clearly, from left to right, instead of nesting them inside each other
joined <- joined %>%
  mutate(
    Temperature = ifelse(Temperature.x == 0, Temperature.y, Temperature.x),
    Precipitation = ifelse(Precipitation.x == 0, Precipitation.y, Precipitation.x)
  ) %>%
  select(-ends_with(".x"), -ends_with(".y"))   # remove the temporary columns

write.csv(joined, "joined.csv", row.names = FALSE)


# INSERT HOLIDAYS GERMANY - LOWER SAXONY TO JOINED.CSV
holidays_lower_saxony_2025 <- read.csv("holidays_lower_saxony_2025.csv")
colnames(holidays_lower_saxony_2025) <- c("Date", "HolidayGermany")
View(holidays_lower_saxony_2025)

joined <- read.csv("joined.csv")
# join the HolidayGermany columns from both csv's on joined. csv
# based on Date
# join the two csv files based on Date and Time
joined <- left_join(joined, holidays_lower_saxony_2025, by = "Date")

# Replace the HolidayGermany column with holiday names where available, keep 0 otherwise
joined <- joined %>%

  mutate(
    HolidayGermany = ifelse(!is.na(HolidayGermany.y), HolidayGermany.y, HolidayGermany.x)
  ) %>%
  select(-HolidayGermany.x, -HolidayGermany.y)  # remove the temporary columns

View(joined)


# TO DEBUG WHY MY HOLIDAYS DID NOT APPEAR ON DATA FRAME => NO HOLIDAYS MATCH THE DATES ATM
# see what holidays are in my CSV
print(holidays_lower_saxony_2025)

# check if any dates in joined data match holidays
any(joined$Date %in% holidays_lower_saxony_2025$Date)

# see specific matches
matched_dates <- joined$Date[joined$Date %in% holidays_lower_saxony_2025$Date]
print(matched_dates)
