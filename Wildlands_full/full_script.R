# FUNCTION TO TRANSFORM DATE IN ACTUAL DATES
# FILTER THEM BASED ON WHAT DATES WE NEED
library(dplyr)

filter_dates <- function(data, date_col = Date, start_date, end_date, date_format = "%Y-%m-%d") {
  date_col <- enquo(date_col)
  
  data %>%
    mutate({{date_col}} := as.Date(!!date_col, format = date_format)) %>%
    filter({{date_col}} >= as.Date("2023-01-01") & {{date_col}} <= as.Date("2026-12-31"))
}



# READ FIRST CSV FILE FOR VISITORS
visitor <- read.csv("visitordaily.csv", sep = ";")

# add new columns for what we need 
visitor$HolidayNetherlands <- rep(0, 48557)
visitor$HolidayGermany <- rep(0, 48557)
visitor$Event <- rep(0, 48557)


# filter dates for 2023-2025 (what we have until now)
visitor <- filter_dates(visitor, Date, "2023-01-01", "2025-09-07")
View(visitor)


# IMPLEMENT OPENMETEO API TO GET METEO DATA SINCE 2023 UNTIL NOW
library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)

add_weather <- function(data) {
  # Add city coordinates = Emmen
  weather <- fromJSON(content(GET("https://archive-api.open-meteo.com/v1/archive", 
                                  query = list(
    latitude = 52.7826141,      
    longitude = 6.8908096,     # Wildlands Emmen coordinates
    start_date = min(visitor$Date),
    end_date = max(visitor$Date),
    daily = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
    timezone = "auto"
  )), "text"))$daily
  
  left_join(data, data.frame(
    Date = as.Date(weather$time),
    temp_mean = weather$temperature_2m_mean,
    precipitation = weather$precipitation_sum
  ), by = "Date")
}

visitor_filtered <- filter_dates(visitor)
visitor <- add_weather(visitor_filtered)

colnames(visitor) <- c("Ticket ID","Type of Ticket", "Date", "Nr Used Entrances", "Holiday_Netherlands", "Holiday_Germany", "Event", "Temperature (mean)", "Precipitation (sum)")
#View(visitor)



# JOIN HOLIDAYS 2024 NL AND DE
# GERMANY
# read csv files
holidays_germany_2024 <- read.csv("holidays_germany_2024.txt")
colnames(holidays_germany_2024) <- c("Date", "Holiday_Germany")
holidays_germany_2024 <- filter_dates(holidays_germany_2024, Date, "2024-01-01", "2025-01-01-")
#View(holidays_germany_2024)

# join csv file with visitor csv file
visitor <- left_join(visitor, holidays_germany_2024, by = "Date")
visitor <- visitor %>%
  
  mutate(
    Holiday_Germany = ifelse(!is.na(Holiday_Germany.y), Holiday_Germany.y, Holiday_Germany.x)
  ) %>%
  select(-Holiday_Germany.x, -Holiday_Germany.y)  # remove the temporary columns

#View(visitor)



# NETHERLANDS
holidays_netherlands_2024 <- read.csv("holidays_netherlands_2024.txt")
colnames(holidays_netherlands_2024) <- c("Date", "Holiday_Netherlands")
holidays_netherlands_2024 <- filter_dates(holidays_netherlands_2024, Date, "2024-01-01", "2025-01-01")
#View(holidays_netherlands_2024)

# join csv file with visitor csv file
visitor <- left_join(visitor, holidays_netherlands_2024, by = "Date")
visitor <- visitor %>%
  
  mutate(
    Holiday_Netherlands = ifelse(!is.na(Holiday_Netherlands.y), Holiday_Netherlands.y, Holiday_Netherlands.x)
  ) %>%
  select(-Holiday_Netherlands.x, -Holiday_Netherlands.y)  # remove the temporary columns

#View(visitor)


# JOIN HOLIDAYS 2025 NL AND DE
# GERMANY 
holidays_germany_2025 <- read.csv("holidays_germany_2025.txt", sep = ",")
#View(holidays_germany_2025)
colnames(holidays_germany_2025) <- c("Date", "Holiday_Germany")
holidays_germany_2025 <- filter_dates(holidays_germany_2025, Date, "2025-01-01", "2026-01-01")
  
# join csv file with visitor csv file
visitor <- left_join(visitor, holidays_germany_2025, by = "Date")
visitor <- visitor %>%
  
  mutate(
    Holiday_Germany = ifelse(!is.na(Holiday_Germany.y), Holiday_Germany.y, Holiday_Germany.x)
  ) %>%
  select(-Holiday_Germany.x, -Holiday_Germany.y)  # remove the temporary columns

# View(visitor)



# NETHERLANDS
holidays_netherlands_2025 <- read.csv("holidays_netherlands_2025.txt", sep = ",")
colnames(holidays_netherlands_2025) <- c("Date", "Holiday_Netherlands")
holidays_netherlands_2025 <- filter_dates(holidays_netherlands_2025, Date, "2025-01-06", "2026-01-01")
#View(holidays_netherlands_2025)

# Aggregate duplicate dates by combining holiday names
holidays_netherlands_2025 <- holidays_netherlands_2025 %>%
  group_by(Date) %>%
  summarize(Holiday_Netherlands = paste(unique(Holiday_Netherlands), collapse = "; ")) %>%
  ungroup()

# Join and coalesce properly
visitor <- left_join(visitor, holidays_netherlands_2025, by = "Date") %>%
  mutate(
    Holiday_Netherlands = coalesce(Holiday_Netherlands.y, Holiday_Netherlands.x)
  ) %>%
  select(-Holiday_Netherlands.x, -Holiday_Netherlands.y)

#View(visitor)


# HOLIDAYS 2023 DE AND NL
# GERMANY
holidays_germany_2023 <- read.csv("holidays_germany_2023.txt", sep = ",")
#View(holidays_germany_2023)
colnames(holidays_germany_2023) <- c("Date", "Holiday_Germany")
holidays_germany_2023 <- filter_dates(holidays_germany_2023, Date, "2023-01-02", "2023-12-31")

# Aggregate duplicate dates by combining holiday names
holidays_germany_2023 <- holidays_germany_2023 %>%
  group_by(Date) %>%
  summarize(Holiday_Germany = paste(unique(Holiday_Germany), collapse = "; ")) %>%
  ungroup()

# Join and coalesce properly
visitor <- left_join(visitor, holidays_germany_2023, by = "Date") %>%
  mutate(
    Holiday_Germany = coalesce(Holiday_Germany.y, Holiday_Germany.x)
  ) %>%
  select(-Holiday_Germany.x, -Holiday_Germany.y)

# View(visitor)



# NETHERLANDS 2023
holiday_netherlands_2023 <- read.csv("holiday_netherlands_2023.txt", sep = ",")
colnames(holiday_netherlands_2023) <- c("Date", "Holiday_Netherlands")
holiday_netherlands_2023 <- filter_dates(holiday_netherlands_2023, Date, "2023-01-01", "2023-12-31")
#View(holidays_netherlands_2025)

# Aggregate duplicate dates by combining holiday names
holiday_netherlands_2023 <- holiday_netherlands_2023 %>%
  group_by(Date) %>%
  summarize(Holiday_Netherlands = paste(unique(Holiday_Netherlands), collapse = "; ")) %>%
  ungroup()

# Join and coalesce properly
visitor <- left_join(visitor, holiday_netherlands_2023, by = "Date") %>%
  mutate(
    Holiday_Netherlands = coalesce(Holiday_Netherlands.y, Holiday_Netherlands.x)
  ) %>%
  select(-Holiday_Netherlands.x, -Holiday_Netherlands.y)

#View(visitor)



# ADD EVENTS FROM 2023, 2024, 2025
events_2023_2024_2025 <- read.csv("events_2023_2024_2025.txt", sep = ",")
events_2023_2024_2025 <- filter_dates(events_2023_2024_2025, Date, "2023-01-01", "2025-12-31")
#View(events_2023_2024_2025)

# Aggregate duplicate dates by combining holiday names
events_2023_2024_2025 <- events_2023_2024_2025 %>%
  group_by(Date) %>%
  summarize(Event = paste(unique(Event), collapse = "; ")) %>%
  ungroup()

visitor <- visitor %>%
  left_join(events_2023_2024_2025, by = "Date") %>%
  mutate(
    # Keep the event description where available, otherwise keep original
    Event = ifelse(!is.na(Event.y), Event.y, as.character(Event.x))
  ) %>%
  select(-Event.x, -Event.y)  # Remove the temporary columns

#View(visitor)



# ADD CAMPAIGNS
# clean csv that contains campaigns
campaign <- read.csv("all promosl.csv", sep = ";")
colnames(campaign) <- c("Week", "Date", "Campaign_Netherlands", "Campaign_Germany")

# remove column for week
campaign$Week <- NULL

# Convert the date column as the dates in full data frame
campaign <- campaign %>%
  mutate(
    Date = format(dmy(Date), "%Y-%m-%d")
  )
campaign <- filter_dates(campaign, Date, "2023-01-01", "2026-12-31")


# convert + to 0 (char type)
# For a specific column
campaign$Campaign_Netherlands[campaign$Campaign_Netherlands == "+"] <- "Yes"
campaign$Campaign_Netherlands[is.na(campaign$Campaign_Netherlands) | campaign$Campaign_Netherlands == ""] <- "No"
View(campaign)

# For a specific column
campaign$Campaign_Germany[campaign$Campaign_Germany == "+"] <- "Yes"
campaign$Campaign_Germany[is.na(campaign$Campaign_Germany) | campaign$Campaign_Germany == ""] <- "No"
#View(campaign)


# add campaigns to main data frame
# Select only the Date and the two columns you want to add
columns_to_add <- campaign %>%
  select(Date, Campaign_Netherlands, Campaign_Germany)

# Join with your main dataframe
visitor <- visitor %>%
  left_join(columns_to_add, by = "Date")


library(tidyr)
# for 2023, only N/A s appeared, since we don't have the campaigns for that year
visitor <- visitor %>%
  mutate(
    Campaign_Netherlands = replace_na(Campaign_Netherlands, "0"),
    Campaign_Germany = replace_na(Campaign_Germany, "0")
  )
  
View(visitor)

write.csv(visitor, file = "visitor.csv", row.names = TRUE)


# add budget column
budget <- read.csv("budget.csv")
colnames(budget) <- c("Date", "Type of Ticket", "Budget")
budget <- budget %>%
  mutate(
    Date = format(mdy(Date), "%Y-%m-%d")
  )
budget <- filter_dates(budget, Date, "2023-01-01", "2025-12-31")

# each type of ticket was written in at least 3 ways
# cleaning part
library(stringr)
budget <- budget %>%
  mutate('Type of Ticket' = case_when(
    str_detect(`Type of Ticket`, "Group|Groep") ~ "Groepen",
    str_detect(`Type of Ticket`, "Gratis") ~ "Gratis", 
    str_detect(`Type of Ticket`, "Actie") ~ "Actie",
    str_detect(`Type of Ticket`, "Vol | Betalend") ~ "Vol betalend",
    str_detect(`Type of Ticket`, "Accomodatie|Accomodatiehouder") ~ "Accomodatie", 
    str_detect(`Type of Ticket`, "Abonnement") ~ "Abonnement",
    str_detect(`Type of Ticket`, "Evenementen") ~ "Evenementen",
    TRUE ~ `Type of Ticket`
  ))

visitor <- visitor %>%
  mutate('Type of Ticket' = case_when(
    str_detect(`Type of Ticket`, "Group|Groep") ~ "Groepen",
    str_detect(`Type of Ticket`, "Gratis") ~ "Gratis", 
    str_detect(`Type of Ticket`, "Actie") ~ "Actie",
    str_detect(`Type of Ticket`, "Vol betalend | Betalend") ~ "Vol betalend",
    str_detect(`Type of Ticket`, "Accommodatie|Accommodatiehouder e-tickets") ~ "Accommodatie", 
    str_detect(`Type of Ticket`, "Abonnement") ~ "Abonnement",
    str_detect(`Type of Ticket`, "Evenementen") ~ "Evenementen",
    str_detect(`Type of Ticket`, "Inkoop") ~ "Inkoop",
    TRUE ~ `Type of Ticket`
  ))


View(budget)


# Aggregate visitor data - sum Nr Used Entrances and keep one row per date+ticket
visitor <- visitor %>%
  group_by(Date, `Type of Ticket`) %>%
  summarise(
    `Nr Used Entrances` = sum(`Nr Used Entrances`, na.rm = TRUE),
    # For non-numeric columns, take the first value (or most appropriate)
    across(c(Holiday_Germany, Holiday_Netherlands, Event, 
             Campaign_Netherlands, Campaign_Germany, `Precipitation (sum)`, `Temperature (mean)`), first),
    .groups = "drop"
  )
visitor <- left_join(visitor, budget, by = c("Date", "Type of Ticket"))


# add nr used entrances for inkoop tickets to budget
visitor <- visitor %>%
  mutate(Budget = ifelse(is.na(Budget), `Nr Used Entrances`, Budget))

View(visitor)
