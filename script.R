# VISITOR.CSV DESIGN
visitor <- read.csv("visitordaily_sample.csv", sep = ";", header = TRUE)
colnames(visitor) <- c("Access", "Type of ticket", "Date", "Nr")

visitor$HolidayNetherlands <- rep(0, 386)
visitor$HolidayGermany <- rep(0, 386)
visitor$Event <- rep(0, 386)
visitor$Promotion <- rep(0, 386)
visitor$Temperature <- rep(0, 386)
visitor$Precipitation <- rep(0, 386)

# Date in visitor.csv was char, not date type
visitor$Date <- as.Date(visitor$Date, format = "%Y-%m-%d")

#View(visitor)



# WEATHER.CSV DESIGN
library(dplyr)
weather <- read.csv("weather.csv")
weather$Date <- as.Date(weather$Date, format = "%Y-%m-%d")

# make avg of weather daily to join with visitor sample v2 csv file
# Aggregate to daily statistics (assuming you have precipitation and temperature columns)
daily_weather <- weather %>%
  group_by(Date) %>%
  summarize(
    Avg_Temperature = round(mean(Temperature, na.rm = TRUE), 1),
    Max_Temperature = round(max(Temperature, na.rm = TRUE), 1),
    Min_Temperature = round(min(Temperature, na.rm = TRUE), 1),
    Total_Precipitation = round(sum(Precipitation, na.rm = TRUE), 1),
    Rainfall_Hours = round(sum(Precipitation > 0, na.rm = TRUE), 1)
  ) %>%
  filter(Date >= as.Date("2025-01-02") & Date <= as.Date("2025-01-30"))

# change col names
colnames(daily_weather) <- c("Date", "Temperature", "Max_Temperature", "Min_Temperature", "Precipitation", "Rainfall_Hours") 
# Now you can join with your other CSV by Date
#View(daily_weather)



# COMBINE visitordaily_sample.csv AND WEATHER.CSV IN A NEW CSV: JOINED.CSV
library(dplyr)

# join the two csv files based on Date and Time
joined <- left_join(visitor, daily_weather, by = "Date")

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
# View(holidays_lower_saxony_2025)

joined <- read.csv("joined.csv")
# join the HolidayGermany columns from both csv's on joined. csv
# based on Date
# join the two csv files based on Date
joined <- left_join(joined, holidays_lower_saxony_2025, by = "Date")

# Replace the HolidayGermany column with holiday names where available, keep 0 otherwise
joined <- joined %>%

  mutate(
    HolidayGermany = ifelse(!is.na(HolidayGermany.y), HolidayGermany.y, HolidayGermany.x)
  ) %>%
  select(-HolidayGermany.x, -HolidayGermany.y)  # remove the temporary columns

# TO DEBUG WHY MY HOLIDAYS DID NOT APPEAR ON DATA FRAME => NO HOLIDAYS MATCH THE DATES ATM
# see what holidays are in my CSV
print(holidays_lower_saxony_2025)

# check if any dates in joined data match holidays
any(joined$Date %in% holidays_lower_saxony_2025$Date)

# see specific matches
matched_dates <- joined$Date[joined$Date %in% holidays_lower_saxony_2025$Date]
print(matched_dates)



# ADD csv file holidays_north_rhine_westphalia_2025.csv
# read csv file
library(tidyr) # for separate function
library(lubridate) # for mutate

holidays_north_rhine_westphalia_2025 <- read.csv("holidays_north_rhine_westphalia_2025.csv")


# join the HolidayGermany columns from both csv's on joined.csv
# based on Date
# join the two csv files based on Date
colnames(holidays_north_rhine_westphalia_2025) <- c("date_range", "HolidayGermany")
#View(holidays_north_rhine_westphalia_2025)

# SIMPLIFIED: Process North Rhine Westphalia dates
holidays_north_rhine_westphalia_expanded <- data.frame()

for(i in 1:nrow(holidays_north_rhine_westphalia_2025)) {
  date_range <- as.character(holidays_north_rhine_westphalia_2025$date_range[i])
  holiday_name <- as.character(holidays_north_rhine_westphalia_2025$HolidayGermany[i])
  
  if(grepl("-", date_range)) {
    # Date range: split and expand
    dates <- strsplit(date_range, "-")[[1]]
    start_date <- as.Date(trimws(dates[1]), format = "%d.%m.%Y")
    end_date <- as.Date(trimws(dates[2]), format = "%d.%m.%Y")
    
    all_dates <- as.character(seq(start_date, end_date, by = "1 day"))
    
    for(d in all_dates) {
      holidays_north_rhine_westphalia_expanded <- rbind(
        holidays_north_rhine_westphalia_expanded,
        data.frame(Date = d, HolidayGermany = holiday_name)
      )
    }
  } else {
    # Single date
    holidays_north_rhine_westphalia_expanded <- rbind(
      holidays_north_rhine_westphalia_expanded,
      data.frame(Date = date_range, HolidayGermany = holiday_name)
    )
  }
}

# Join North Rhine Westphalia holidays
joined <- left_join(joined, holidays_north_rhine_westphalia_expanded, by = "Date")

# North Rhine Westphalia fills only where Lower Saxony has no holiday
joined <- joined %>%
  mutate(
    HolidayGermany = ifelse(!is.na(HolidayGermany.y) & HolidayGermany.x == 0, 
                            HolidayGermany.y, HolidayGermany.x)
  ) %>%
  select(-HolidayGermany.x, -HolidayGermany.y)

joined$Rainfall_Hours <- NULL
#View(joined)

# First, let's see what's actually in the North Rhine Westphalia file
print("Contents of North Rhine Westphalia holidays:")
print(holidays_north_rhine_westphalia_2025)

# Check which rows have date ranges vs single dates
holidays_north_rhine_westphalia_2025$has_hyphen <- grepl("-", holidays_north_rhine_westphalia_2025$date_range)
print("Rows with hyphens (date ranges):")
print(holidays_north_rhine_westphalia_2025[holidays_north_rhine_westphalia_2025$has_hyphen, ])
print("Rows without hyphens (single dates):")
print(holidays_north_rhine_westphalia_2025[!holidays_north_rhine_westphalia_2025$has_hyphen, ])

# TO DEBUG WHY MY HOLIDAYS DID NOT APPEAR ON DATA FRAME => NO HOLIDAYS MATCH THE DATES ATM
# see what holidays are in my CSV
print(holidays_north_rhine_westphalia_2025)

# check if any dates in joined data match holidays
any(joined$Date %in% holidays_north_rhine_westphalia_2025$Date)

# see specific matches
matched_dates <- joined$Date[joined$Date %in% holidays_north_rhine_westphalia_2025$Date]
print(matched_dates)



# JOIN EVENTS.CSV FOR 2025
events <- read.csv("events.csv")
#View(events)

joined <- left_join(joined, events, by = "Date")
joined <- joined %>%
  
  mutate(
    Event = ifelse(!is.na(Event.y), Event.y, Event.x)
  ) %>%
  select(-Event.x, -Event.y)  # remove the temporary columns

#View(joined)
# check if any dates in joined data match holidays
any(joined$Date %in% events$Date)



# ADD HOLIDAYS NETHERLANDS CSV
netherlands_holidays <- read.csv("nl_holidays_2025.csv") 
colnames(netherlands_holidays) <- c("Date", "HolidayNetherlands")
View(netherlands_holidays)

joined <- left_join(joined, netherlands_holidays, by = "Date")
joined <- joined %>%
  
  mutate(
    HolidayNetherlands = ifelse(!is.na(HolidayNetherlands.x), HolidayNetherlands.y, HolidayNetherlands.x)
  ) %>%
  select(-HolidayNetherlands.x, -HolidayNetherlands.y)  # remove the temporary columns

joined$HolidayNetherlands[is.na(joined$HolidayNetherlands)] <- 0
View(joined)

# check if any dates in joined data match holidays
any(joined$Date %in% holidays_lower_saxony_2025$Date)

# see specific matches
matched_dates <- joined$Date[joined$Date %in% netherlands_holidays$Date]
print(matched_dates)
