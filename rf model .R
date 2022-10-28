##################################
#   -   DAV-R Course             #
#   -   Graduation project       #
#   -   2022-9-27                #
#   -   copyright : hager mohsen #
#                   amr mohamed  #
##################################
#R version 4.2.1 (2022-06-23 ucrt)
#Platform: x86_64-w64-mingw32/x64 (64-bit)
#Running under: Windows 10 x64 (build 19042)

#Matrix products: default

#locale:
#[1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8
#[3] LC_MONETARY=English_United States.utf8 LC_NUMERIC=C
#[5] LC_TIME=English_United States.utf8


#load the required libraries
library(shiny)
library(shinydashboard)
library(nortest)
library(mvnormtest)
library(MASS)
library(shinyLP)
library(GGally)
library(class)
library(gmodels)
library(caret)
library(rattle)
library(ranger)
library(klaR)
library(kernlab)
library(e1071)
library(NeuralNetTools)
library(neuralnet)
library(nnet)
library(mclust)
library(psych)
library(ggplot2)
library(caret)
library(survival)
library(class)
library(Hmisc)
library(viridisLite)
library(randomForestExplainer)
library(viridis)
library(caTools)
library(randomForest)
library(cvms)
library(yardstick)
library(reprtree)
library(Rcpp)
library(hrbrthemes)
library(Hmisc)
library(e1071)
library(RcppZiggurat)
library(Rfast)
library(tree)
library(party)
library(plotrix)
library(tidyverse)
library(reprtree)
library(ROCR)
library(rpart)
library(loon)
library(tcltk)
library(rfviz)
library(pROC)
library(rpart.plot)
library(Rfast)
library(stats)


#load raw dataset
df = read.csv('diabetes.csv')

#check and explore dataset
summary(df)
length(df)
str(df)
dim(df)
class(df)
typeof(df)

#check the missing values
sum(is.na(df))

#check correlation between all features
cor.ci(df , method = 'spearman')

#dealing with zeros values
colSums(df==0)
df[df == 0]=NA
df$Glucose[is.na(df$Glucose)] = mean(df$Glucose , na.rm=TRUE)
df$BloodPressure[is.na(df$BloodPressure)] = mean(df$BloodPressure , na.rm=TRUE)
df$BMI[is.na(df$BMI)] = mean(df$BMI , na.rm=TRUE)
df[is.na(df)] = 0

#check outlier
boxplot(df)

# create detect outlier function
detect_outlier = function(x) {

# calculate first quantile
Quantile1 = quantile(x, probs=0.25)

# calculate third quantile
Quantile3 = quantile(x, probs=0.75)

# calculate inter quartile range
IQR = Quantile3-Quantile1

# return true or false
x > Quantile3 + (IQR*.5) | x < Quantile1 - (IQR*1.5)
}

# create remove outlier function
remove_outlier = function(dataframe, columns=names(dataframe)) {

# for loop to traverse in columns vector
  for (col in columns) {

# remove observation if it satisfies outlier function
    dataframe = dataframe[!detect_outlier(dataframe[[col]]), ]
  }

# return dataframe
  print("Remove outliers")
  print(dataframe)
}

df = remove_outlier(df, c('Pregnancies' , 'Glucose' , 'BloodPressure' ,
                     'SkinThickness' , 'Insulin' , 'BMI' ,
                     'DiabetesPedigreeFunction' , 'Age'))

boxplot(df)

#replace 0,1 in outcome feature with 'diabetic' , 'indiabetic'
df$Outcome = factor(df$Outcome , levels = c(0,1) , labels=c('Indiabetic' , 'diabetic'))

#data splitting
split = sample.split(df$Outcome , SplitRatio = 0.9 )
train_set = subset(df , split==TRUE)
test_set = subset(df , split==FALSE)

write.csv(train_set, "training.csv")
write.csv(test_set, "testing.csv")

#feature scaling
#train_set[,1:8] =scale(train_set[,1:8])
#test_set[,1:8] =scale(test_set[,1:8])
#absolute the data
#train_set[,2:8] =abs(train_set[,2:8])
#test_set[,2:8] =abs(test_set[,2:8])
################################################################################

#fitting the random forest model on the training data
rf_model = randomForest(Outcome ~ ., data=train_set, importance=TRUE, ntree=500, mtry = 2, do.trace=100)
plot(rf_model)

#prediction
y_pred = predict(rf_model , newdata = test_set[-9])
#check model accuracy using confusion matrix
confmatrix = confusionMatrix(test_set[,9] , y_pred)
confmatrix
fourfoldplot(confmatrix$table, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

#check features importance using gini index
varImpPlot(rf_model)

#hyperparameter tuning using grid search method
tuneRF(train_set[-9] ,train_set$Outcome , plot = TRUE , stepFactor = 0.5 , trace = TRUE ,
       improve = 0.05)


#splits = rf_prep(x =train_set[-9] ,y=train_set$Outcome)
#rf_viz(splits , input = TRUE ,imp = TRUE , cmd = TRUE, hl_color = 'orange')

#k-fold cross validation
folds = createFolds(train_set$Outcome, k = 10)
cv = lapply(folds , function(x){
  train_fold = train_set[-x , ]
  test_fold = test_set[-x , ]
  rf_model = randomForest(Outcome ~ ., data=train_set, importance=TRUE, ntree=500, mtry = 2, do.trace=100)
  y_pred = predict(rf_model , newdata = test_set[-9])
  confmatrix = table(test_set[,9] , y_pred)
  acc = (confmatrix[1,1] + confmatrix[2,2]) / (confmatrix[1,1] + confmatrix[2,2] + confmatrix[1,2] + confmatrix[2,1])
  return(acc)
})
acc = mean(as.numeric(cv))
acc # 89% , Sensitivity : 0.88 , Specificity : 1.0000

# Roc , AUC curve
predictionwithclass = predict(rf_model , test_set , type = 'class')
predictionwithprob = predict(rf_model , test_set ,type = 'prob')
predictionwithprob
auc = auc(test_set$Outcome , predictionwithprob[,2])
plot(roc(test_set$Outcome , predictionwithprob[,2]))
auc
#visualize one tree from all forest
windows(60,60)
reprtree:::plot.getTree(rf_model)
################################################################################
#obtain the distribution of minimal depth
min_depth_frame <- min_depth_distribution(rf_model)
save(min_depth_frame, file = "min_depth_frame.rda")
load("min_depth_frame.rda")
head(min_depth_frame, n = 10)

#obtain a plot of the distribution of minimal depth for variables according to mean minimal depth
plot_min_depth_distribution(min_depth_frame)


#explore variable importance measures
importance_frame <- measure_importance(rf_model)
save(importance_frame, file = "importance_frame.rda")
load("importance_frame.rda")
importance_frame
plot_multi_way_importance(importance_frame, size_measure = "no_of_nodes")
plot_importance_rankings(importance_frame)


#After selecting a set of most important variables we can investigate interactions with respect
#to them, i.e. splits appearing in maximal subtrees with respect to one of the variables selected.
#To extract the names of 5 most important variables according to both the mean minimal depth and number of
#trees in which a variable appeared, we pass our importance_frame to the function important_variables as follows:

(vars <- important_variables(importance_frame, k = 5, measures = c("mean_min_depth", "no_of_trees")))
interactions_frame <- min_depth_interactions(rf_model, vars)
save(interactions_frame, file = "interactions_frame.rda")
load("interactions_frame.rda")
head(interactions_frame[order(interactions_frame$occurrences, decreasing = TRUE), ])

# plot_min_depth_interactions(forest)
plot_min_depth_interactions(interactions_frame)


#saving the model
saveRDS(rf_model , 'randomforest model.rds')
################################################################################

