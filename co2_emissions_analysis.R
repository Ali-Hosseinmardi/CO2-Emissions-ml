#===============================================================================
# Name: Ali Hosseinmardi
# Student Number: 24109150
# Lecturer: Safa Olia
# Final Assignment
# Course: Machine Learning
#===============================================================================

# Set working directory (update as needed)
setwd("E:/Machine learning/week0")

# Install required packages (only once)
# install.packages(c("tidyverse", "caret", "e1071", "rpart", "randomForest", "cluster", "factoextra", "reshape2", "corrplot"))

# Load libraries
library(tidyverse)
library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(cluster)
library(factoextra)
library(reshape2)
library(corrplot)

# ---------------------------
# LOAD AND CLEAN DATA
# ---------------------------

# Load the CO2 emissions dataset
data <- read.csv("co2-wide.csv")

# View summary statistics
summary(data)

# Check for missing values
sum(is.na(data))

# Save the 'country' column for future labeling, then remove from model input
countries <- data$country

# Fill missing numeric values with median
data <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Check that no missing values remain
sum(is.na(data))

# Normalize numeric features (z-score scaling)
numeric_cols <- sapply(data, is.numeric)

data[numeric_cols] <- scale(data[numeric_cols])

# Set the target variable
data$CO2EMISSIONS <- data$X2021

# Remove 'country' column from modeling dataset
data <- data %>% select(-country)

# Save and view a sample for reporting (first 10 rows with country names)
display_sample <- data.frame(Country = countries, data)
write.csv(head(display_sample, 10), "sample_display_data.csv", row.names = FALSE)
head(display_sample, 10)

# ---------------------------
# DATA PARTITIONING
# ---------------------------

# Split data into training (80%) and testing (20%) sets
set.seed(123)
train_index <- createDataPartition(data$CO2EMISSIONS, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# ---------------------------
# VISUALIZATIONS
# ---------------------------

# 1. Histogram: 2021 Emissions Distribution
ggplot(data, aes(x = CO2EMISSIONS)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "black") +
  labs(title = "Distribution of CO₂ Emissions in 2021",
       x = "CO₂ Emissions (scaled)", y = "Frequency")

# 2. Time Series Trend Plot: Emissions Over Time
data_long <- melt(data.frame(country = countries, data), id.vars = "country",
                  measure.vars = c("X1975", "X1985", "X2005", "X2010", "X2015", "X2019", "X2020", "X2021"))

ggplot(data_long, aes(x = variable, y = value, group = country)) +
  geom_line(alpha = 0.3, color = "grey") +
  stat_summary(fun = mean, geom = "line", aes(group = 1), color = "red", size = 1.5) +
  labs(title = "CO₂ Emissions Trend Over Time", x = "Year", y = "CO₂ Emissions (scaled)") +
  theme_minimal()

# 3. Correlation Heatmap
cor_matrix <- cor(data %>% select(starts_with("X")))
corrplot(cor_matrix, method = "color", tl.col = "black", tl.srt = 45)

# ---------------------------
# MODEL 1: DECISION TREE
# ---------------------------
model_dt <- rpart(CO2EMISSIONS ~ ., data = train_data)
pred_dt <- predict(model_dt, test_data)
rmse_dt <- RMSE(pred_dt, test_data$CO2EMISSIONS)
r2_dt <- R2(pred_dt, test_data$CO2EMISSIONS)

# ---------------------------
# MODEL 2: K-NEAREST NEIGHBORS
# ---------------------------
ctrl <- trainControl(method = "cv", number = 5)
model_knn <- train(CO2EMISSIONS ~ ., data = train_data, method = "knn", trControl = ctrl)
pred_knn <- predict(model_knn, test_data)
rmse_knn <- RMSE(pred_knn, test_data$CO2EMISSIONS)
r2_knn <- R2(pred_knn, test_data$CO2EMISSIONS)

# ---------------------------
# MODEL 3: SUPPORT VECTOR REGRESSION (SVR)
# ---------------------------
model_svr <- svm(CO2EMISSIONS ~ ., data = train_data)
pred_svr <- predict(model_svr, test_data)

rmse_svr <- RMSE(pred_svr, test_data$CO2EMISSIONS)
r2_svr <- R2(pred_svr, test_data$CO2EMISSIONS)

# ---------------------------
# MODEL 4: RANDOM FOREST
# ---------------------------
model_rf <- randomForest(CO2EMISSIONS ~ ., data = train_data, ntree = 100)
pred_rf <- predict(model_rf, test_data)
rmse_rf <- RMSE(pred_rf, test_data$CO2EMISSIONS)
r2_rf <- R2(pred_rf, test_data$CO2EMISSIONS)

# ---------------------------
# MODEL 5: K-MEANS CLUSTERING (UNSUPERVISED)
# ---------------------------
cluster_data <- data %>% select(-CO2EMISSIONS)
set.seed(123)
kmeans_result <- kmeans(cluster_data, centers = 3)

# Visualize clusters
fviz_cluster(kmeans_result, data = cluster_data)

# ---------------------------
# MODEL EVALUATION OUTPUT
# ---------------------------
cat("Decision Tree: RMSE =", rmse_dt, ", R2 =", r2_dt, "\n")
cat("KNN: RMSE =", rmse_knn, ", R2 =", r2_knn, "\n")
cat("SVR: RMSE =", rmse_svr, ", R2 =", r2_svr, "\n")
cat("Random Forest: RMSE =", rmse_rf, ", R2 =", r2_rf, "\n")

# ---------------------------
# RANDOM FOREST INTERPRETABILITY
# ---------------------------
importance(model_rf)
varImpPlot(model_rf)
