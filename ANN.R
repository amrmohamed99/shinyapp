library(neuralnet)
library(dplyr)
library(ROCR)
library(caret)

#load raw dataset
df = read.csv('diabetes.csv')

#check and explore dataset
summary(df)
str(df)

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
  x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
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

#feature scaling
#train_set[,1:8] =scale(train_set[,1:8])
#test_set[,1:8] =scale(test_set[,1:8])

#fitting the Artificial Neural Network model on the training data
ann_model = neuralnet(Outcome~. , train_set , hidden = c(5,3),
                      act.fct = "logistic",
                      linear.output = FALSE)
plot(ann_model)

#prob_pred = neuralnet::compute(ann_model , test_set[-9])

#prediction
pred= compute(ann_model , test_set[-9])
pred
head(pred$net.result)
head(test_set[1,])
colnames(pred$net.result)=c('Indiabetic' ,'diabetic')

#confusion matrix
pre = predict(ann_model, test_set, type = "raw") %>%
  as.data.frame() %>%
  mutate(prediction = if_else(0 < 1, 0, 1)) %>%
  pull(prediction)

ann_table = table(pre, test_set$Outcome)
ann_table
accuracy_test_ann = sum(diag(ann_table)) / sum(ann_table)
list('predict matrix' = ann_table, 'accuracy' = accuracy_test_ann) #68%

# Roc , AUC curve
predictionwithclass = predict(ann_model , test_set , type = 'class')
predictionwithprob = predict(ann_model , test_set ,type = 'prob')
colnames(predictionwithprob)=c('Indiabetic','diabetic')
predictionwithprob

auc = auc(test_set$Outcome , predictionwithprob[,2])
plot(roc(test_set$Outcome , predictionwithprob[,2]))

#saving the model
saveRDS(ann_model , 'ANN model.rds')

