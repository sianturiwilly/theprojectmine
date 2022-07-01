# Prediction of Health Insurance Costs

# Installing Packages
install.packages("tidyverse")
install.packages("psych")
install.packages("ggridges")
install.packages("broom.mixed")
install.packages("viridis")
install.packages("modelr")
install.packages("lme4")
install.packages("patchwork")

# Load the Libraries
library(tidyverse)
library(psych)
library(ggridges)
library(broom.mixed)
library(viridis)
library(modelr)
library(lme4)
library(patchwork)

# 1. Data Extraction
insurance_df <- read.csv("data/insurance.csv")
insurance_df

## 1.1. Check for Missing Values
insurance_df %>%
  summarise_all(
    ~ sum(is.na(.))
  )
summary(insurance_df)

## 1.2. Change Target to Factor
insurance_df <- insurance_df %>%
  mutate(sex = factor(sex),
         smoker = factor(smoker),
         region = factor(region),
         children = factor(children)
  )
insurance_df

# 2. Exploratory Data Analysis (EDA)
## 2.1. Charges Variables
insurance_df %>%
  ggplot(aes(charges, ..density.., fill = I("#d45087"))) +
  geom_histogram(bins = 50) +
  geom_density()+
  theme(legend.position = "none")

tibble(low_charges = nrow(filter(insurance_df,charges<15000 )),
       middle_charges =  nrow(filter(insurance_df,charges>=15000 & charges<50000)),
       high_charges = nrow(filter(insurance_df, charges>=50000)),
)

## 2.2. Age Variables
insurance_df %>%
  ggplot(aes(age, fill = I("#d45087")))+
  geom_bar()+ 
  theme(legend.position = "none")

p1 <- insurance_df %>% 
  ggplot(aes(age, charges))+
  geom_point(color = I("#d45087"))

p2 <- insurance_df %>% 
  ggplot(aes(age, charges, color = smoker))+
  geom_point()

p1+p2

## 2.3. Sex Variables
p1 <- insurance_df %>%
  count(sex) %>% 
  ggplot(aes(sex, n, fill = I("#d45087")))+
  geom_col()+
  theme(legend.position = "none")

p2 <- insurance_df %>% 
  ggplot(aes(charges, sex, fill = sex)) +
  ggridges::geom_density_ridges()

p3 <- insurance_df %>% 
  ggplot(aes(sex, charges, fill = sex))+
  geom_boxplot()

(p1+p2)/p3

## 2.4. BMI Variables
p1 <- insurance_df %>%
  ggplot(aes(x = bmi, y = stat(density))) +
  geom_histogram( aes(fill = I("#d45087")), bins = 50) +
  theme(legend.position = "none") +
  geom_density()

p2 <- insurance_df %>% 
  ggplot(aes(x = bmi, charges, color = charges)) +
  geom_point() +
  scale_color_viridis(option = "E")

p3 <- insurance_df %>% 
  ggplot(aes(factor(age), bmi, color = I("#d45087"))) +
  geom_point() +
  geom_jitter() +
  stat_summary(fun = mean, colour = "blue", geom = "point", size = 5)

(p1+p2)/p3

## 2.5. The Number of Children
p1 <- insurance_df %>%
  count(children) %>% 
  ggplot(aes(children, n))+
  geom_col( aes(fill = I("#d45087")))+
  theme(legend.position = "none")

p2 <- insurance_df %>% 
  ggplot(aes(children, charges, fill = children))+
  geom_boxplot()

p1+p2

## 2.6. Smoker Variables
p1 <- insurance_df %>%
  ggplot(aes(smoker,charges, fill = smoker))+
  geom_boxplot()

p2 <- insurance_df %>%
  ggplot(aes(charges, fill = smoker, alpha = 0.5))+
  geom_density()

p1/p2

## 2.7. Region Variables
p1 <- insurance_df %>%
  count(region) %>% 
  ggplot(aes(region, n, width = 0.7, fill = I("#d45087")))+
  geom_col()+
  coord_flip()+
  theme(legend.position = "none")

p2 <- insurance_df %>% 
  ggplot(aes(charges, region, fill = region)) +
  ggridges::geom_density_ridges()

p3 <- insurance_df %>% 
  ggplot(aes(region, charges, fill = region))+
  geom_boxplot()+
  coord_flip()

p1+(p2/p3)

# 3. Remove Outliers
## 3.1. Get Outliers Values
outliers_charges <- boxplot.stats(insurance_df$charges)$out
outliers_charges

## 3.2. Get Index of Outliers
outliers_ind <- which(insurance_df$charges %in% c(outliers_charges))
outliers_ind

## 3.3. Data With Outliers
outliers_data <- insurance_df[outliers_ind, ]
insurance_df[outliers_ind, ]

## 3.4. Data Without Outliers
insurance_df[-outliers_ind, ]
insurance_df_no <- insurance_df[-outliers_ind,]

## 3.5 One Hot Encoder
library(caret)
dmy = dummyVars(" ~ .", data = insurance_df_no[, -7])
encode_df = data.frame(predict(dmy, newdata = insurance_df_no[, -7]))
encode_df$charges <- insurance_df_no$charges

## 3.6. Divide Data Into Train and Test
m <- nrow(insurance_df_no)
set.seed(2021)
train_idx <- sample(m, m * 0.8)

train_df <- insurance_df_no[train_idx, ]
test_df <- insurance_df_no[-train_idx, ]

# 4. Modelling
## 4.1. Simple Linear Regression
fit.linreg <- lm(charges ~ smoker+age+bmi+children+region, data = train_df)
summary(fit.linreg)

## 4.2. Polynomial Regression
fit.polyreg <- lm(charges ~ smoker+age+bmi+children+region+I(age^2)+I(bmi^2), data = train_df)
summary(fit.polyreg)

## 4.3. Multiple Linear Regression
fit.multireg <- lm(formula = charges ~ ., data = train_df)
summary(fit.multireg)

## 4.4. Multiple Linear Regression with Interaction
fit.inmultireg <- lm(charges ~ smoker+age+bmi+children+region+I(age^2)+I(bmi^2)+age:bmi, data = train_df)
summary(fit.inmultireg)

# 5. Evaluate the Model
performance = function(test_df, model, model_name) {
  
  pred <- predict(model, test_df)
  error <- pred - test_df$charges
  mse <- mean(error ^ 2)
  rmse <- sqrt(mse)
  
  result <- paste("RMSE of", model_name, ":",
                  round(rmse, 2))
  result
}

performance(test_df, fit.linreg, "Simple Linear Regression")
performance(test_df, fit.polyreg, "Polynomial Regression")
performance(test_df, fit.multireg, "Multiple Linear Regression")
performance(test_df, fit.inmultireg, "MLR with Interaction")

### 5.1. Scatterplot
library(car)
scatterplot(x = test_df$charges, y = predict(fit.linreg, test_df), 
            xlim = c(0,40000), ylim = c(0,40000))
scatterplot(x = test_df$charges, y = predict(fit.polyreg, test_df), 
            xlim = c(0,40000), ylim = c(0,40000))
scatterplot(x = test_df$charges, y = predict(fit.multireg, test_df), 
            xlim = c(0,40000), ylim = c(0,40000))
scatterplot(x = test_df$charges, y = predict(fit.inmultireg, test_df), 
            xlim = c(0,40000), ylim = c(0,40000))