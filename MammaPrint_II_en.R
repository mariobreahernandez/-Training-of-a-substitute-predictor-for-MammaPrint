#MammaPrint parte II
#Mario Brea
setwd('C:/Users/224D2000/Documents/MammaPrint_II')
getwd()

#Parallelization-ON (one time)
install.packages("doParallel")
library(doParallel)
ncores <- parallel::detectCores(logical = TRUE)  # detection
cl <- parallel::makePSOCKcluster(ncores)
registerDoParallel(cl)

# Parallelization-OFF (at the end of the trainning methods)
#stopCluster(cl)
#registerDoSEQ()

#libraries
install.packages(c("readr", "caret", "pROC", "ranger", "glmnet", "gbm"), dependencies = TRUE)

library(caret)
library(pROC)
library(readr)
library(ranger)
library(glmnet)
library(gbm)


#Load X (predictor varible (gene matrix) and Y (target variable: risk)
x<- readRDS("X_filtrada.rds")
y<- readRDS("y_riesgo.rds")
class(y)
class(x)
head(y)
head(x, c(2, 4))

# 80/20 stratified by class partition
set.seed(123)  # reproducibilidad
idx_train <- caret::createDataPartition(y, p = 0.80, list = FALSE)

x_tr <- x[idx_train, ]
x_te <- x[-idx_train, ]
y_tr <- y[idx_train]
y_te <- y[-idx_train]

#dimensiones
dim(x_tr); dim(x_te)
prop.table(table(y_tr))
prop.table(table(y_te))


#Preprocess with only training data 
pp <- caret::preProcess(x_tr, method = c("center", "scale"))

#We apply that pre-preocess to our data set
x_tr_pp <- predict(pp, x_tr)
x_te_pp <- predict(pp, x_te)

#RANDOM FOREST
# Cross validation (CV)
ctrl <- trainControl(
  method = "repeatedcv",          # repeaed cross validation
  number = 5,                     # nº of folds = 5, division of the dataset
  repeats = 2,                    # repetition of CV 2 times
  classProbs = TRUE,              # calcule probabilities (not only classes)
  summaryFunction = twoClassSummary, # use metrics ROC, Sens, Spec
  savePredictions = "final" ,      # save predictions of the cross validation CV
  allowParallel = TRUE            # parallelization
)

#training
set.seed(123)  # reproducibility
fit_rf <- train(
  x = x_tr_pp, y = y_tr,          # preprocessed training data
  method = "ranger",              # this is an optimized RF, faster
  trControl = ctrl,               # defined control
  metric = "ROC",                 # optimize by AUC ROC
  tuneLength = 3                  # model tries 3 mtry values (candidate gene in each split)
)

# results
print(fit_rf)     # mean AUC per metry
plot(fit_rf)      # AUC curve vs mtry

#visualización
class(fit_rf)                  # "train"
names(fit_rf)                  # available fields
fit_rf$bestTune                # optimal hiperparameters (f.ex., mtry and splitrule)
fit_rf$results                 # cross-validation table (ROC, Sens, Spec per each grid)
getTrainPerf(fit_rf)           # benchmark of the best in cross validation
fit_rf$finalModel              # final model (ranger object)
varImp(fit_rf)                 # importance of variables
plot(fit_rf)                   # ROC (CV) vs hiperparameters

#evaluation with the test subset
prob_rf <- predict(fit_rf, x_te_pp, type = "prob")[,"Alto"]  # "High" prob
pred_rf <- factor(ifelse(prob_rf >= 0.5, "Alto", "Bajo"),
                  levels = levels(y_te))  # from numbers to levels (high/low)

#Confusion matrix
cm_rf <- caret::confusionMatrix(pred_rf, y_te, positive = "Alto")

#ROC curve and AUC
roc_rf <- pROC::roc(response = y_te, predictor = prob_rf, levels = rev(levels(y_te)))
auc_rf <- as.numeric(pROC::auc(roc_rf))

#graphix
plot(roc_rf, col="blue", main="Curva ROC - Random Forest (test)")


#GLMNET
# cross validation
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

# training
set.seed(123)
fit_glmnet <- train(
  x = x_tr_pp, y = y_tr,
  method = "glmnet",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10   # nº of lambda/alpha combinations to try
)

# results
print(fit_glmnet)
plot(fit_glmnet)

#evaluation in the test subset
prob_glmnet <- predict(fit_glmnet, x_te_pp, type = "prob")[,"Alto"]  # "High" prob
pred_glmnet <- factor(ifelse(prob_glmnet >= 0.5, "Alto", "Bajo"),
                      levels = levels(y_te))  # from numbers to levels

#Confusion matrix
cm_glmnet <- caret::confusionMatrix(pred_glmnet, y_te, positive = "Alto")

#ROC curve and AUC
roc_glmnet <- pROC::roc(response = y_te, predictor = prob_glmnet, levels = rev(levels(y_te)))
auc_glmnet <- as.numeric(pROC::auc(roc_glmnet))

#ROC graphix
plot(roc_glmnet, col="blue", main="Curva ROC - GLMNet (test)")


library(caret)
library(pROC)

#SVM
#cross validation
ctrl <- trainControl(
  method = "repeatedcv", number = 5, repeats = 2,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final", allowParallel = TRUE
)

#training
#if grid is not fixed, caret will calculate sigma automatically (kernlab)
set.seed(123)
fit_svm <- train(
  x = x_tr_pp, y = y_tr,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10   # explore ~10 combinatios of (C, sigma)
)

#results
print(fit_svm)      # best C-sigma combination and the ROC mean curve
plot(fit_svm)       # heatmap

# evaluation in test subset
p_svm <- predict(fit_svm, x_te_pp, type = "prob")[,"Alto"]   # prob of "High"
pred_svm <- factor(ifelse(p_svm >= 0.5, "Alto", "Bajo"),
                   levels = levels(y_te))

#Confusion matrix
cm_svm  <- confusionMatrix(pred_svm, y_te, positive = "Alto")

#ROC curve and AUC
roc_svm <- roc(response = y_te, predictor = p_svm, levels = c("Bajo","Alto"),
               direction = "<")
auc_svm <- as.numeric(auc(roc_svm))

#ROC graphix
plot(roc_svm, main = "Curva ROC - SVM radial (test)")


# GMB
# corss validation in training set
ctrl <- trainControl(
  method = "repeatedcv", number = 5, repeats = 2,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final", allowParallel = TRUE
)

# training
set.seed(123)
fit_gbm <- train(
  x = x_tr_pp, y = y_tr,
  method = "gbm",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5,              # try different hiperparameters combinations
  verbose = FALSE
)

#resultados
print(fit_gbm)   # best hiperparameters combinations
plot(fit_gbm)    # ROC evolution VS hiperparameters

# evaluation in test subset
prob_gbm <- predict(fit_gbm, x_te_pp, type = "prob")[,"Alto"]
pred_gbm <- factor(ifelse(prob_gbm >= 0.5, "Alto", "Bajo"), levels = levels(y_te))

#Confusion matrix
cm_gbm  <- confusionMatrix(pred_gbm, y_te, positive = "Alto")

# AUC curve and ROC
roc_gbm <- roc(response = y_te, predictor = prob_gbm, levels = c("Bajo","Alto"), direction = "<")
auc_gbm <- as.numeric(auc(roc_gbm))

#graphix
plot(roc_gbm, main = "Curva ROC - GBM (test)")




save.image("myenv_MammaPrint_II")
load("myenv_MammaPrint_II")
