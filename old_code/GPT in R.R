install.packages("httr")
install.packages("jsonlite")

install.packages("openssl")
install.packages("chattr")
install.packages("shiny")


library(shiny)
library(chattr)
library(jsonlite)

Sys.setenv("OPENAI_API_KEY" = " your_openai_api_key_here ")

chattr_use("gpt35")

chattr_app(as_job = TRUE)

input<-(bloomberg_clean_29042024)

install.packages("dplyr")
install.packages("tidyverse")
install.packages("tidymodels")

data<-input

library(readr)
library(dplyr)

library(tidyverse)
library(tidymodels)



# Data preprocessing
data_processed <- data %>%
  gather() %>%
  group_by(COUNTRY_FULL_NAME) %>%
summarize(mean = mean(value))

# Model building
set.seed(123)
data_split <- initial_split(data, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)

# Create a linear model
lm_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Train the model
lm_fit <- lm_model %>%
  fit(mean ~ key, data = data_train)

# Make predictions
predictions <- lm_fit %>%
  predict(new_data = data_test)

# Evaluate the model
metrics <- predictions %>%
  yardstick::rmse(truth = data_test$mean, estimate = .pred)

# Print RMSE
metrics



