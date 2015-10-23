---
title: "Machine Learning"
author: "Nick Lim"
date: "22 October 2015"
output: html_document
---
# Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. Using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, we will train an algorithm to learn if barbell lifts were performed correctly. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Processing the data
We begin by loading the data. We omit the first 7 columns of the data, as these columns are information regarding the measurement and the participants and not information from the accelerometers. We use the function as.numeric on the entries of the data, to induce a NA on blank/null and DIV/#0 entries. We restore the label column 'classe' as the previous operation would have induced NAs in the labels. We then filter out the columns that are mostly NAs (>60% NAs) 

```{r}
knitr::opts_chunk$set(warning=FALSE, message=FALSE)
library(caret)
library(randomForest)
library(rpart)
pml.training <- read.csv("~/pml-training.csv",na.strings = c("NA",""))
clean= pml.training[,-c(1,2,3,4,5,6,7)]

clean=apply(clean, 2, function(x) suppressWarnings(as.numeric(x)))
clean=data.frame(clean)
clean$classe=pml.training$classe
idx=c(which(colSums(is.na(clean))<0.6*19622))
clean=clean[,idx]
```
## Preparing the learning algorithm 
We set a random seed to ensure reproducibility of our data, and create a 75%/25% split of the data between the training set and the test set. We then use the k-nearest neighbour algorithm to train our learner, and check the out-of sample performance of our learner

```{r}
set.seed(111111)
trainIdx = createDataPartition(clean$classe, p = 0.75, list = FALSE)  # Pick out the row indexes of a random 75% subset of the data as the training set
trainSet = clean[trainIdx, ]  
testSet = clean[ -trainIdx, ] ## Assign the remainder of the data as the testSet
## Do a K-nearest neighbour predictor
ctrlKNN = trainControl(method = "adaptive_cv")
modKNN = train(classe ~ ., data=trainSet, method = "knn", trControl = ctrlKNN)
#Test the model on the testSet 
predKNNIn = predict(modKNN,trainSet)
predKNNOut = predict(modKNN,testSet)
knnConfusionIn = confusionMatrix(predKNNIn, trainSet$classe)
knnConfusionOut = confusionMatrix(predKNNOut, testSet$classe)
```

K-nearest neighbour gives a `r knnConfusionOut$overall[1]` out of sample accuracy. We can probably do better :). We then explored other algorithms (ie. GBM, decision tree, random forest). The learner was not able to learn within a reasonable amount of time using the GBM algorithm.  

```{r}
# GBM killed my computer!
#modGBM=train(classe ~ ., data=trainSet, method = "gbm")
#predGBMIn = predict(modKNN,trainSet)
#predGBMOut = predict(modKNN,testSet)
#gbmConfusionIn = confusionMatrix(predGBMIn,trainSet$classe)
#gbmConfusionOut = confusionMatrix(predGBMOut,testSet$classe)
```


```{r}
modRP = train(classe ~ ., data=trainSet, method = "rpart")
predRPIn = predict(modRP,trainSet)
predRPOut = predict(modRP,testSet)
rpConfusionIn = confusionMatrix(predRPIn, trainSet$classe)
rpConfusionOut = confusionMatrix(predRPOut, testSet$classe)
```

R-Part (Decision Tree) gives a horrible out of sample accuracy of `r rpConfusionOut$overall[1]`. We then assume that the algorithm is unable to discriminate between the classes.

```{r}
modRF = randomForest(classe ~.,data=trainSet)
predRFIn = predict(modRF,trainSet)
predRFOut = predict(modRF,testSet)

rfConfusionIn = confusionMatrix(predRFIn, trainSet$classe)
rfConfusionOut = confusionMatrix(predRFOut, testSet$classe)
```
The random forest algorithm on the other hand gives significantly better out of sample accuracy of `r rfConfusionOut$overall[1]`. This learner is our primary model for the test cases in pml-testing.csv

## Running the learners on the unlabled testcases

```{r}
pml.testing <- read.csv("~/pml-testing.csv")
cleanTest= pml.testing[,-c(1,2,3,4,5,6,7)]
cleanTest=apply(cleanTest, 2, function(x) as.numeric(x))
cleanTest=data.frame(cleanTest)
cleanTest=cleanTest[,idx]
predRFTest = predict(modRF,cleanTest)
predRPTest = predict(modRP,cleanTest)
predKNNTest = predict(modKNN,cleanTest)
```

Below are the labels predicted by the Random Forest learner
```{r}
predRFTest
```

We then  compare this result to the prediction by the other two models
```{r}
testConfusion1=confusionMatrix(predRPTest,predRFTest)
testConfusion2=confusionMatrix(predKNNTest,predRFTest)
testConfusion3=confusionMatrix(predRPTest,predKNNTest)

testConfusion1
testConfusion2
#testConfusion3
```

From the feedback from the submission, we found that the random forest was able to label all 20 testcases correctly. Giving an accuracy of 100%. The confusion matrix above shows our error for the other "weaker" learners.
The accuracy of the KNN algorithm (`r testConfusion2$overall[1]`) and decision tree (`r testConfusion1$overall[1]`) appears to agree with our estimates above, (`r knnConfusionOut$overall[1]` and `r rpConfusionOut$overall[1]` respectively)

## Conclusion
The random forest algorithm has managed to learn the training data and provide excellent accuracy on our test data. By cleaning the dataframe to include only the relevant features, we created learner that can learn the training data efficiently, without overfitting the data.   
