setwd("~/Desktop/Work/Training/Coursera/Machine Learning/Project")
testdata = read.csv(file = "pml-testing.csv")
trainingdata = read.csv(file = "pml-training.csv")

library(caret)

colnames(trainingdata)
summary(trainingdata)

# Cross validation
# Use 70% of training set data to built a model, and use the rest to test the model

## Loading required package: lattice
## Loading required package: ggplot2

set.seed(1111)
train <- createDataPartition(y=trainingdata$classe,p=.70,list=F)
training <- trainingdata[train,]
testing <- trainingdata[-train,]

#Cleaning the training data

#exclude identifier, timestamp, and window data (they cannot be used for prediction)
Cl <- grep("name|timestamp|window|X", colnames(training), value=F) 
trainingCl <- training[,-Cl]
#select variables with high (over 95%) missing data --> exclude them from the analysis
trainingCl[trainingCl==""] <- NA
NArate <- apply(trainingCl, 2, function(x) sum(is.na(x)))/nrow(trainingCl)
trainingCl <- trainingCl[!(NArate>0.95)]
summary(trainingCl)

#PCA
#Since the number of variables are still over 50, PCA is applied

preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.8) #12 components are required
preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.9) #18 components are required
preProc <- preProcess(trainingCl[,1:52],method="pca",thresh=.95) #25 components are required

preProc <- preProcess(trainingCl[,1:52],method="pca",pcaComp=25) 
preProc$rotation
trainingPC <- predict(preProc,trainingCl[,1:52])

# Random forest
# Apply ramdom forest method (non-bionominal outcome & large sample size)

library(randomForest)

modFitRF <- randomForest(trainingCl$classe ~ .,   data=trainingPC, do.trace=F)
print(modFitRF) # view results 

# Check with test set

testingCl <- testing[,-Cl]
testingCl[testingCl==""] <- NA
NArate <- apply(testingCl, 2, function(x) sum(is.na(x)))/nrow(testingCl)
testingCl <- testingCl[!(NArate>0.95)]
testingPC <- predict(preProc,testingCl[,1:52])
confusionMatrix(testingCl$classe,predict(modFitRF,testingPC))

#Predict the test cases
testdataCl <- testdata[,-Cl]
testdataCl[testdataCl==""] <- NA
NArate <- apply(testdataCl, 2, function(x) sum(is.na(x)))/nrow(testdataCl)
testdataCl <- testdataCl[!(NArate>0.95)]
testdataPC <- predict(preProc,testdataCl[,1:52])
testdataCl$classe <- predict(modFitRF,testdataPC)