#----------------------------------------------------------------------------- #
# ----------- STAT-642 Data Mining for Business Analytics Final Project ------ #
# -------------------- Telco Customer Churn Analysis ------------------------- #
# ------------------------------- Group 2 ------------------------------------ #
# ---------- Sunita Barik, Aida Karimu, Jingxin Yao, Zihan Huang ------------- #
#----------------------------------------------------------------------------- #

#-------------------------------------------------------------------------------
## Preliminary Code
#-------------------------------------------------------------------------------
#Clear the workspace
rm(list=ls())

# Setting working directory
setwd("D:/Courses/STAT-642/Final Project")

# Loading packages
library(caret)
library(rpart)
library(rpart.plot)
library(cluster)
library(factoextra)
library(fpc)
library(MLeval)

# Loading the data
churn <- read.csv("CustomerChurn.csv",
                  stringsAsFactors = FALSE)

# Load Clus_Plot_Fns.RData file for clustering external validation  
load("Clus_Plot_Fns.RData")

#-------------------------------------------------------------------------------
# Part 1 Initial Data EXploration
#-------------------------------------------------------------------------------

## 1.1 View the structure and summary of the data
# Structure of data
str(churn)

# summary of data
summary(churn)

## 1.2 Identify variables type and converting
# Target variable
churn$Churn <- factor(churn$Churn)

# Numerical variables
nums <- c("tenure","MonthlyCharges","TotalCharges")

# Ordinal variables
ords <- c("Contract")

# Nominal variables
facs <- c("gender","SeniorCitizen","Partner","Dependents","PhoneService",
          "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
          "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
          "PaperlessBilling","PaymentMethod")

# Convert the variables
# Nominal variables
churn[ ,facs] <- lapply(X = churn[ ,facs],
                              FUN = factor)

# Ordinal variable
churn$Contract = factor(x = churn$Contract,
                        levels = c("Month-to-month","One year", "Two year"),
                        ordered = TRUE) 

# Create a vector for predictor variables
vars <- c(nums,ords,facs)

## 1.3 Descriptive Statistics Analysis

# Summary of data after converting the variables
summary(churn)
# There are 11 missing values in Total Charges

# Standard deviation for numerical variables 
# (Rerun after missing data imputation)
lapply(X = churn[,nums], FUN = sd)

# Modes
modefun <- function(x){
  if(any(tabulate(match(x, unique(x))) > 1)){
    outp <- unique(x)[which(tabulate(match(x, unique(x))) == max(tabulate(match(x, unique(x)))))]
  } else {
    outp <- "No mode exists!"}
  return(outp)
}

lapply(X = churn, FUN =  modefun)

# Create a correlation matrix for numerical variables 
# (Rerun after missing data imputation)
cor(churn[,nums])

#-------------------------------------------------------------------------------
# Part 2  Data Preprocessing
#-------------------------------------------------------------------------------

## 2.1 Missing Data Detection and Processing
any(is.na(churn))

# Missing rows
na_rows <- rownames(churn)[!complete.cases(churn)]
na_rows

# Check the distribution of TotalCharges
boxplot(churn$TotalCharges,
        main = "TotalCharges",
        horizontal = TRUE) 

# TotalCharges is skewed and we impute missing value with median.
churn$TotalCharges[is.na(churn$TotalCharges)] <- median(churn$TotalCharges, 
                                            na.rm = TRUE)

## 2.2 Data Binarization and Recoding for SVM

# Prepare another dataset for SVM
churn_svm <- churn

# Nominal varibles with 2 levels - Binarization
churn_svm[,c("gender","SeniorCitizen","Partner","Dependents",
             "PhoneService","PaperlessBilling")] <- 
  lapply(X = churn[,c("gender","SeniorCitizen","Partner","Dependents",
                      "PhoneService","PaperlessBilling")],
         FUN = class2ind,
         drop2nd = TRUE)

# Nominal variables with more than 2 levels - Create Dummy Variables
cats <- dummyVars(formula =  ~ MultipleLines + InternetService + OnlineSecurity
                  +OnlineBackup + DeviceProtection + TechSupport + StreamingTV
                  +StreamingMovies + PaymentMethod,
                  data = churn_svm)
cats_dums <- predict(object = cats, 
                     newdata = churn_svm)

#Create a new dataframe with dummy variables
churn_dums <-  data.frame(churn_svm[ ,!names(churn_svm) %in% c("MultipleLines","InternetService"
                                                               ,"OnlineSecurity","OnlineBackup",
                                                               "DeviceProtection","TechSupport",
                                                               "StreamingTV","StreamingMovies",
                                                               "PaymentMethod")],
                          cats_dums)

# Convert ordinal variable to numeric form
churn_dums$Contract <- as.numeric(churn_dums$Contract)

## 2.3 Discretization
churn$tenure_disc <- cut(x = churn$tenure,
                         breaks = 8,
                         labels = c(1,2,3,4,5,6,7,8),
                         ordered_result = TRUE)

## 2.4 Data Transformation

# Use YeoJohnson to normalize the data
cen_yj <- preProcess(x = churn[,nums],
                     method = c("YeoJohnson"))
churn_yj <- predict(object = cen_yj,
                   newdata = churn)

# Standardize transformed data
cen_cs <- preProcess(x = churn_yj[nums],
                     method = c("center", "scale"))
churn_yjcs <- predict(object = cen_cs,
                   newdata = churn_yj)

## 2.3 Outliers Detection and Handling

# Draw boxplots for numerical variables
par(mfrow = c(1,3))
boxplot(churn_yjcs$tenure,
        main = "tenure")
boxplot(churn_yjcs$MonthlyCharges,
        main = "MonthlyCharges")
boxplot(churn_yjcs$TotalCharges,
        main = "TotalCharges")
par(mfrow = c(1,1))

# Z-score method
outs <- sapply(churn_yjcs[,nums], function(x) which(abs(scale(x)) > 3))
outs
## There are no identified outliers.

## 2.4 Prepare Training and Testing data

# Initial random seed
set.seed(210917)

# Create list of training indices
sub <- createDataPartition(y = churn$Churn,
                           p = 0.80, # 80% in training
                           list = FALSE)

# Subset the transformed data 
# to create the training (train)
# and testing (test) datasets
train <- churn[sub, ] 
test <- churn[-sub, ]

# Create train and test dataset for SVM
train_svm <- churn_dums[sub, ]
test_svm <- churn_dums[-sub, ]

#-------------------------------------------------------------------------------
## Part 3 Data Visualization and Exploratory Analysis
#-------------------------------------------------------------------------------

## 3.1 Distribution of tenure
crosstab1 <- data.frame(table(Churn = churn$Churn,tunure = churn$tenure_disc))

ggplot(crosstab1,aes(x = tunure, y = Freq, fill = Churn)) +
  geom_bar(stat = 'identity',position = position_dodge(),alpha = 0.9)+
  geom_text(aes(label = Freq), vjust = 1.5,
            position = position_dodge(.9), size = 3) +
  labs(x = "\nTenure in Months", y = "Count of Customers\n", title = "\nDistribution of Customer Tenure in Months\n") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_text(face="bold", colour="#0072B2", size = 12),
        axis.title.y = element_text(face="bold", colour="#0072B2", size = 12),
        legend.title = element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.background = element_rect(fill = "transparent"))

## 3.2 Distribution of Contract
crosstab2 <- data.frame(table(Churn = churn$Churn,Contract = churn$Contract))

ggplot(crosstab2,aes(x = Contract, y = Freq, fill = Churn)) +
  geom_bar(stat = 'identity',position = position_dodge(),alpha = 0.9)+
  geom_text(aes(label = Freq), vjust = 1.5,
            position = position_dodge(.9), size = 3) +
  labs(x = "\nContract Types", y = "Count of Customers\n", title = "\nDistribution of Contract Types\n") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_text(face="bold", colour="#0072B2", size = 12),
        axis.title.y = element_text(face="bold", colour="#0072B2", size = 12),
        legend.title = element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.background = element_rect(fill = "transparent"))

## 3.3 Distribution of PaymentMethod
crosstab3 <- data.frame(table(Churn = churn$Churn,PaymentMethods = churn$PaymentMethod))

ggplot(crosstab3,aes(x = PaymentMethods, y = Freq, fill = Churn)) +
  geom_bar(stat = 'identity',position = position_dodge(),alpha = 0.9)+
  geom_text(aes(label = Freq), vjust = 1.5,
            position = position_dodge(.9), size = 3) +
  labs(x = "\nPayment Methods", y = "Count of Customers\n", title = "\nDistribution of Payment Methods\n") +
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title.x = element_text(face="bold", colour="#0072B2", size = 12),
        axis.title.y = element_text(face="bold", colour="#0072B2", size = 12),
        legend.title = element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.background = element_rect(fill = "transparent"))

## ggplot parameter reference: 
## https://dk81.github.io/dkmathstats_site/rvisual-sidebyside-bar.html

#-------------------------------------------------------------------------------
## Part 4 k-Means Cluster Analysis
#-------------------------------------------------------------------------------

## 4.1 Make Silhouette plot to find optimal number of clusters
sil_plot(scaled_data = churn_yjcs[ ,nums], 
         method = "kmeans", 
         max.k = 15, 
         seed_no = 210917) 

# 4.2 kMC clustering
# Set random seed
set.seed(210917)

# k-Means clustering
churn_kmeans <- kmeans(x = churn_yjcs[ ,nums], 
                  centers = 4, # # of clusters
                  trace = FALSE, 
                  nstart = 30)

# Cluster results
churn_kmeans

# Number of customers in clusters
churn_kmeans$size

# Visualize clusters
fviz_cluster(object = churn_kmeans,
             data = churn_yjcs[ ,nums])

# Cluster centers
churn_kmeans$centers

# Visualize cluster centers
matplot(t(churn_kmeans$centers), # cluster centroids
        type = "l", 
        ylab = "", 
        xlim = c(1, 3.1), 
        xaxt = "n", 
        col = 1:4, 
        lty = 1:4, 
        lwd = 2,
        main = "Cluster Centers") 

# Add custom x-axis labels
axis(side = 1, 
     at = 1:3, 
     labels = nums, 
     las = 1) 

# Add legend
legend("left", 
       legend = 1:4, 
       col = 1:4,
       lty = 1:4, 
       cex = 0.8)

## 4.3 Cluster Validation 
# External validation
table(Churn = churn$Churn, 
      Clusters = churn_kmeans$cluster)

# Internal validation
wss_plot(scaled_data = churn_yjcs[ ,nums], 
         method = "kmeans", 
         max.k = 15, 
         seed_no = 210917) 

#-------------------------------------------------------------------------------
## Part 5 Decision Tree Model Training and Performance Analysis
#-------------------------------------------------------------------------------

## 5.1 Basic Decision Tree Model
churn.rpart <- rpart(formula = Churn ~ .,
                  data = train[ ,c(vars, "Churn")], 
                  method = "class")

# Model results
churn.rpart

# Visualize decision tree model
rpart.plot(churn.rpart,
           extra = 2)

## Basic Model Performance
# Generate class prediction for training set
base.trpreds <- predict(object = churn.rpart, 
                        newdata = train, 
                        type = "class")

# Obtain performance information
churn_base_train_conf <- confusionMatrix(data = base.trpreds, 
                                 reference = train$Churn,
                                 positive = "Yes",
                                 mode = "everything")

# Generate class prediction for testing set
base.tepreds <- predict(object = churn.rpart, 
                        newdata = test,
                        type = "class")

# Obtain performance information
churn_base_test_conf <- confusionMatrix(data = base.tepreds, 
                                reference = test$Churn,
                                positive = "Yes",
                                mode = "everything")

## Goodness of Fit
# Overall
cbind(Training = churn_base_train_conf$overall,
      Testing = churn_base_test_conf$overall)

# Class-Level
cbind(Training = churn_base_train_conf$byClass,
      Testing = churn_base_test_conf$byClass)

## 5.2 Hyperparameter Model Tuning 
# Set up a grid for a grid search
grids <- expand.grid(cp = seq(from = 0,
                              to = 0.2, # Search values from 0 to 0.2
                              by = 0.005))

# Set a 10-fold cross validation grid search 
DT_ctrl <- trainControl(method = "repeatedcv",
                     number = 10, 
                     repeats = 3, 
                     search = "grid")

# Set randomm seed
set.seed(510610)

# Tune the model
churnDTFit <- train(form = Churn ~ ., 
               data = train[ ,c(vars,"Churn")], 
               method = "rpart", 
               trControl = DT_ctrl, 
               tuneGrid = grids)

## Hyperparameter Tuned Model Performance
# Generate class prediction for training set
DT.trpreds <- predict(object = churnDTFit, 
                        newdata = train)

# Obtain performance information
DT_train_conf <- confusionMatrix(data = DT.trpreds, 
                                         reference = train$Churn,
                                         positive = "Yes",
                                         mode = "everything")

# Generate class prediction for testing set
DT.tepreds <- predict(object = churnDTFit, 
                        newdata = test)

# Obtain performance information
DT_test_conf <- confusionMatrix(data = DT.tepreds, 
                                        reference = test$Churn,
                                        positive = "Yes",
                                        mode = "everything")

## Goodness of Fit
# Overall
cbind(Training = DT_train_conf$overall,
      Testing = DT_test_conf$overall)

# Class-Level
cbind(Training = DT_train_conf$byClass,
      Testing = DT_test_conf$byClass)

## 5.3 Class Imbalance Detection and Remodeling Using Class Weighting

# Identify class imbalance
plot(churn$Churn,
     main = 'Churn')
## There are far more not churned customers

# Identify target variable
target_var <- train$Churn

# Calculate the weights
weights <- c(sum(table(target_var))/(nlevels(target_var)*table(target_var)))
weights # Weightage is 0.6807/1.8837

# Set the weight vector
wghts <- weights[match(x = target_var, 
                       table = names(weights))]

# Set random seed
set.seed(210610)

# Retune the model
churnFit_CW <- train(form = Churn ~ ., 
                     data = train[ ,c(vars,"Churn")], 
                     method = "rpart", 
                     trControl = DT_ctrl, 
                     tuneGrid = grids,
                     weights = wghts)

# Model results
churnFit_CW

# Optimal result
churnFit_CW$results[churnFit_CW$results$cp %in% churnFit_CW$bestTune,] 

## Class Weighted Model Performance
# Generate class prediction for training set
CW.trpreds <- predict(object = churnFit_CW, 
                      newdata = train)

# Obtain performance information
CW_train_conf <- confusionMatrix(data = CW.trpreds, 
                                 reference = train$Churn, 
                                 positive = "Yes",
                                 mode = "everything")

# Test performance
CW.tepreds <- predict(object = churnFit_CW,
                    newdata = test)
CW_test_conf <- confusionMatrix(data = CW.tepreds, 
                           reference = test$Churn, 
                           positive = "Yes",
                           mode = "everything")

## Goodness of Fit
# Overall
cbind(Training = CW_train_conf$overall,
      Testing = CW_test_conf$overall)

# Class-Level
cbind(Training = CW_train_conf$byClass,
      Testing = CW_test_conf$byClass)

# Variables of importance
churnFit_CW$finalModel$variable.importance

# Model comparison
# Overall
cbind(Basic = churn_base_test_conf$overall,
      Hyperparameter = DT_test_conf$overall,
      Weighted = CW_test_conf$overall)

# Class-Level
cbind(Basic = churn_base_test_conf$byClass,
      Hyperparameter = DT_test_conf$byClass,
      Weighted = CW_test_conf$byClass)

#-------------------------------------------------------------------------------
## Part 6 Support Vector Machine
#-------------------------------------------------------------------------------

# Set up grids for the cost and sigma hyperparameters
SVM_grids <-  expand.grid(C = seq(from = 1, 
                              to = 5, 
                              by = 1),
                      sigma = seq(from = 0.01,
                                  to = 0.11,
                                  by = 0.02))

# Set up control object
SVM_ctrl <- trainControl(method = "repeatedcv",
                     number = 5, 
                     repeats = 3, 
                     search = 'grid',
                     classProbs = TRUE, # needed for AUC
                     savePredictions = TRUE, # save the predictions to plot
                     summaryFunction = twoClassSummary) 

# Set random seed
set.seed(917610)

# Train the model
churn_SVMFit <- train(form = Churn ~ .,
                data = train_svm[,-1],  
                method = "svmRadial",                   
                preProcess = c("center", "scale"),      
                trControl = SVM_ctrl,                       
                tuneGrid = SVM_grids,
                metric = "ROC",
                weights = wghts) 

# Model Results
churn_SVMFit

# Make ROC plot
evalm(churn_SVMFit)$roc

# Support Vector Machines Model Performance
# Generate predict information on the training set
svm.tr.preds <- predict(object = churn_SVMFit,
                        newdata = train_svm)

# Get performance information
SVM_trtune_conf <- confusionMatrix(data = svm.tr.preds, 
                                   reference = train_svm$Churn, 
                                   positive = "Yes",
                                   mode = "everything")
SVM_trtune_conf

# Generate predict information on the testing set
svm.te.preds <- predict(object = churn_SVMFit,
                        newdata = test_svm)

# Get performance information
SVM_tetune_conf <- confusionMatrix(data = svm.te.preds, 
                                   reference = test_svm$Churn, 
                                   positive = "Yes",
                                   mode = "everything")
SVM_tetune_conf

## Goodness of Fit
# Overall
cbind(Training = SVM_trtune_conf$overall,
      Testing = SVM_tetune_conf$overall)

# Class-Level
cbind(Training = SVM_trtune_conf$byClass,
      Testing = SVM_tetune_conf$byClass)

#-------------------------------------------------------------------------------
## Save the Workspace
save.image("Final_Group2.RData")
