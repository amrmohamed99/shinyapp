#SVM model
svm_model = svm(formula= Outcome~. ,
                data = train_set,
                type = 'C-classification' ,
                kernel= 'radial' ,
                C = 0.25 ,
                sigma=0.105)
y_predict = predict(svm_model , newdata = test_set[-9])
cm = confusionMatrix(test_set[,9] , y_predict)
cm
fourfoldplot(cm$table, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

#svm hyperparameter tuning
svm_model = train(form=Outcome~. ,
                  data=train_set ,
                  method='svmRadial')
svm_model
svm_model$bestTune

#k-fold cross validation
folds = createFolds(train_set$Outcome, k = 10)
cv = lapply(folds , function(x){
  train_fold = train_set[-x , ]
  test_fold = test_set[-x , ]
  svm_model = svm(formula= Outcome~. ,
                  data = train_set,
                  type = 'C-classification' ,
                  kernel= 'radial',
                  C = 0.25 ,
                  sigma=0.105)
  y_predict = predict(svm_model , newdata = test_set[-9])
  cm = table(test_set[,9] , y_predict)
  acc = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(acc)
})
acc = mean(as.numeric(cv))
acc #77%

saveRDS(svm_model , 'SVM model.rds')
