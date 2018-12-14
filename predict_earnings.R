
library(caret)
library(ROCR)

library(pROC)


library(AppliedPredictiveModeling)
install.packages("DMwR")
library(corrplot)
install.packages("DMwR")
library(DMwR)
install.packages("psych")
library(psych)
install.packages("prediction")
library(prediction)
install.packages("")
library("irr")
library(xlsx)

fraud<-read.xlsx("C:/Users/Sarvagna/Desktop/Sem 2/predictivecase-data.xlsx",sheetName = "Complete Data")
fraud <- read.csv("Earnings_model.csv")
str(fraud)
fraud <-as.data.frame(fraud[,2:10]) # Remove Company ID
fraud$Manipulator<-ifelse(fraud$Manipulater=="Yes",1,0)
View(fraud)
fraud$Manipulator<-as.factor(fraud$Manipulater)
str(fraud$Manipulator)
#levels(fraud$Manipulator) <- make.names(levels(factor(fraud$Manipulator)))



transparentTheme(trans = .4)
featurePlot(x = fraud[, 1:8], 
            y = fraud$Manipulator, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(2,1), 
            auto.key = list(columns = 2))
str(fraud)




par(mfrow=c(1,4))
for(i in 1:8) {
  hist(fraud[,i], main=names(fraud)[i])
}



corVariables <- cor(fraud[,1:8])
corrplot(corVariables)


highlyCorVariables <- findCorrelation(corVariables, cutoff = .75)
highlyCorVariables


#split data


set.seed(1408)
splitIndex <- createDataPartition(fraud$Manipulator, p = .7,list = FALSE, times = 1)
trainData <- fraud[splitIndex,]
testData <- fraud[-splitIndex,]
View(trainData)

set.seed(1408)
smote_train <- SMOTE(Manipulator ~.,trainData,perc.over = 100,perc.under = 200)
View(smote_train)
seed   <- 1408   # Keeping seed constant for all models
metric <- "ROC"  # Evaluation metric for Model selection

#glm model

#lmfit<-glm(formula = Manipulator~DSRI+GMI+AQI+SGI+ACCR,family = binomial,data = smote_train)

lmfit<-train(Manipulator~DSRI+GMI+AQI+SGI+ACCR,data = trainData,method='glm',metric='Accuracy')
summary(lmfit)
print(lmfit)

pred1<-predict(lmfit,testData,type='raw')
pred1
pred3<-ifelse(pred1>0.5,1,0)

nrow(testData$Manipulator)
dm = as.matrix(table(pred1,testData$Manipulator))
n1 = sum(dm)
diag1 = diag(dm)
nc1 = nrow(dm) # number of classes
rowsums1 = apply(dm, 1, sum) # number of instances per class
colsums1 = apply(dm, 2, sum) # number of predictions per class
p1 = rowsums1 / n1 # distribution of instances over the actual classes
q1 = colsums1 / n1 

accuracy1 = sum(diag1) / n1 
accuracy1 
precision1 = diag1 / colsums1 
recall1 = diag1 / rowsums1 
f11 = 2 * precision1 * recall1 / (precision1 + recall1) 
data.frame(precision1, recall1, f11) 
result.roc<-roc(testData$Manipulator,pred1,smooth=TRUE)
par(mfrow=c(1,1))
plot(result.roc)
result.roc$auc




library(randomForest)

#randomForest
rfmod=randomForest(formula = Manipulator~DSRI+GMI+AQI+SGI+ACCR,data = smote_train,ntree=10)
print(rfmod)
predrf=predict(rfmod,newdata = testData,type="prob")
cm = as.matrix(table(predrf[,2]>0.5,testData$Manipulator))
n = sum(cm)
diag = diag(cm)
nc = nrow(cm) # number of classes
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n 

accuracy = sum(diag) / n 
accuracy 
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 
#boosting

control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 1408
metric <- "Accuracy"
# C5.0
set.seed(seed)
fit.c50 <- train(Manipulator~., data=smote_train, method="C5.0", metric=metric, trControl=control)
# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Manipulator~., data=smote_train, method="gbm", metric=metric, trControl=control, verbose=FALSE)
# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)
print(boosting_results)

library(rpart)
tree <- rpart(Manipulator ~ ., data = smote_train, control = rpart.control(cp = 0.0001))
printcp(tree)
bestcp <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
tree.pruned <- prune(tree, cp = bestcp)
conf.matrix <- table(testData$Manipulator, predict(tree.pruned,type="class"))
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ":")
colnames(conf.matrix) <- paste("Pred", colnames(conf.matrix), sep = ":")
print(conf.matrix)

rpartmodel<-train(Manipulator~.,data=smote_train,method="rpart")
predictions<-predict(rpartmodel,testData,type="prob")
print(predictions)
confusionMatrix(predictions,testData$Manipulater)
result.roc<-roc(predictions[,2], testData$Manipulator)
plot(result.roc)
result.roc$auc
