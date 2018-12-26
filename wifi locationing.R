
#wifi locationing


library(readr)
setwd("C:/Users/Surya/Desktop/UJIndoorLoc")
indoortrain <- read.csv("trainingData.csv")
indoortrain

str(indoortrain)
summary(indoortrain)
tail(indoortrain)
validationset <- read.csv("validationData.csv")
validationset
tail(validationset)

library(caret)

#combine the building, floor, and specific location attributes into a 
#single unique identifier for each instance.
singleuniqueidentifier <- indoortrain[c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION")]

uniqueidentifier <- validationset[c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "LATITUDE","LONGITUDE")]

WAPVALIDSET <- validationset[ , 1:520]

Wifi <- validationset[ , 527:528]
wirelessAP <- cbind(WAPVALIDSET, Wifi)

uniqueidentifier$BUILDINGID <- as.factor(uniqueidentifier$BUILDINGID) 
uniqueidentifier$BUILDINGID
uniqueidentifier$FLOOR <- as.factor(uniqueidentifier$FLOOR) 
uniqueidentifier$FLOOR

uniqueidentifier$RELATIVEPOSITION <- as.factor(uniqueidentifier$RELATIVEPOSITION) 
uniqueidentifier$RELATIVEPOSITION
str(uniqueidentifier)

WAP1 <- indoortrain[ , 1:520]
tail(WAP1)
head(WAP1)
str(WAP1)
WAP2 <- indoortrain[ , 527:528]
head(WAP2)

wap <- cbind(WAP1, WAP2)
wap

singleuniqueidentifier$BUILDINGID <- as.factor(singleuniqueidentifier$BUILDINGID) 
singleuniqueidentifier$BUILDINGID
singleuniqueidentifier$FLOOR <- as.factor(singleuniqueidentifier$FLOOR) 
singleuniqueidentifier$FLOOR

singleuniqueidentifier$RELATIVEPOSITION <- as.factor(singleuniqueidentifier$RELATIVEPOSITION) 
singleuniqueidentifier$RELATIVEPOSITION


str(singleuniqueidentifier)

LATLON <- indoortrain[c("LATITUDE", "LONGITUDE")]
uniqueid <- cbind(singleuniqueidentifier, LATLON)
str(uniqueid)


newindooetrain1 <- cbind(uniqueidentifier, wirelessAP)
colnames(newindooetrain1)
dim(newindooetrain1)

newindooetrain <- cbind(uniqueid, wap)
colnames(newindooetrain)
str(newindooetrain)




trainingset <- createDataPartition(newindooetrain$RELATIVEPOSITION, p=0.75, list=FALSE, times = 1)


# Step 2: Create the training  dataset
trainData1 <- newindooetrain[trainingset,]
str(trainData1)

# Step 3: Create the test dataset
testData1 <- newindooetrain[-trainingset,]
summary(testData1)



library(C50)
library(caretEnsemble)
require(e1071) #Contains the SVM 
library(caret)
library(rpart)
library(mlbench)
library(randomForest)

Control <- trainControl(method = "repeatedcv",number = 10,repeats = 1)
#model_list <- caretList(RELATIVEPOSITION ~ ., data= trainData1, trControl=Control,preProc = c("center", "scale"), methodList = c("svmLinear2", "rf", "knn"), tuneLength=3)

set.seed(825)


x = nearZeroVar(trainData1, saveMetrics = TRUE)
head(x)
tail(x)
colnames(x)
summary(x)
str(x)

dim(trainData1)
x



x1 = nearZeroVar(testData1, saveMetrics = TRUE)
head(x1)
tail(x1)
colnames(x1)
summary(x1)
str(x1)

dim(testData1)
x1









x1 <- nearZeroVar(testData1)
filteredDescr1 <- testData1[, -x1]
filteredDescr1
dim(filteredDescr1)

str(filteredDescr1)




x <- nearZeroVar(trainData1)
filteredDescr <- trainData1[, -x]
dim(filteredDescr)

str(filteredDescr)


#control <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

svmFit <- train(RELATIVEPOSITION ~ ., data = filteredDescr, method = "svmLinear2", trControl = Control, preProc = c("center", "scale"),tuneLength = 2)
svmFit                

kNN1 <- train(RELATIVEPOSITION~., data = filteredDescr, method = "knn", maximize = TRUE, trControl = trainControl(method = "cv", number = 10), preProcess=c("center", "scale"), tuneGrid=data.frame(.k=1:10))
kNN1

rfFitted <- train(RELATIVEPOSITION ~ ., data = filteredDescr, method = "rf", trControl = Control, preProc = c("center", "scale"),tuneLength = 2)
rfFitted                


predrf1= predict(rfFitted,filteredDescr1)
predrf1

summary(predrf1)
plot(predrf1)


predknn1= predict(kNN1,filteredDescr1)
predknn1

summary(predknn1)
plot(predknn1)


predsvm1= predict(svmFit,filteredDescr1)
predsvm1

summary(predsvm1)
plot(predsvm1)
_______________________________________

predsvmfinal <- predict(svmFit, newindooetrain1)
predsvmfinal
summary(predsvmfinal)
svmfinal <- table(predsvmfinal, newindooetrain1$RELATIVEPOSITION)
svmfinal

predrffinal <- predict(rfFitted, newindooetrain1)
predrffinal
summary(predrffinal)


predknnfinal <- predict(kNN1, newindooetrain1)
predknnfinal
summary(predknnfinal)
knnconf <- confusionMatrix(kNN1)
knnconf
svmconf <- confusionMatrix(svmFit)
svmconf
rfconf <- confusionMatrix(rfFitted)
rfconf

plotting <- cbind(predknn1, predsvm1, predrf1)
dotchart(plotting, cex=.7,main="SVM,RF,KNN Models", xlab="A PERSON IS LOCATED INSIDE/OUTSIDE")

