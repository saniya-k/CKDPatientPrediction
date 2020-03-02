#Logistic Regression to find CKD high risk patients
#Code By @Saniya Khan
library("caret")
library("dplyr")
library("InformationValue")
library(pROC)
library(plotROC)
library(ggplot2)
library(data.table)
library(gridExtra)
library(tidyr)

## Step 0  - Read in Data
data=read.csv("Imputed_V2.csv")
names(data)
data=data[,-1]  ## remove ID


## Step 1 - Explore and relabel Data
y=data$CKD
class(data)
summary(data)
out_sample=which(is.na(data$CKD)==1) # Outsample is the one on which prediction is needed
data_out=data[out_sample,]   ## the ones without a disease status
data_in=data[-out_sample,]   ## the ones with a disease status
summary(data_in)
data_in <- data_in %>% mutate(
  CKD=as.factor(CKD)
)
summary(data_in)
#train test split, cross validating to find best combination of variables

Train <- createDataPartition(data_in$CKD, p=0.65, list=FALSE)
training <- data_in[ Train, ]
summary(training)
testing <- data_in[ -Train, ]
summary(testing)
set.seed(33343)
train_ctr <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
#mod_fit1 <- train(CKD ~ Age + HDL + LDL + 
#                    BMI+Activity + SBP +DBP  + Fam.Diabetes + Diabetes + 
#                    +CVD+PVD + CHF ,  data=data_in, method="glm", family="binomial",
#                 trControl = train_ctr, tuneLength = 5)

mod_fit1 <- train(CKD ~ Age + Waist + HDL + LDL + 
                    PVD + Activity +DBP+ SBP+ Fam.Hypertension + Diabetes + 
                    CVD + CHF + Anemia, 
                  data=data_in, method="glm", family="binomial",
                  trControl = train_ctr, tuneLength = 5)
pred = predict(mod_fit1, newdata=testing)
confusionMatrix(data=pred, testing$CKD)

summary(mod_fit1)

#Step 1a
# create training data
input_ones <- data_in[which(data_in$CKD == 1), ]  # all 1's
input_zeros <- data_in[which(data_in$CKD == 0), ]  # all 0's
set.seed(100)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.65*nrow(input_ones))  # 1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.65*nrow(input_zeros))  # 0's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)  # row bind the 1's and 0's 

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)

#Step 2 Build different models for training data

model=glm(CKD~ .,family="binomial",data=trainingData) #AIC: 1467
summary(model)
#check variables given by stepwise elimination
model3=step(model,direction="both") #Age + Female + Unmarried + Weight + BMI + Obese + Waist + HDL + LDL + Dyslipidemia + PVD + Activity + 
#  Smoker + Hypertension + Diabetes + CVD + Anemia + CHF
summary(model3)#AIC:1447.8
#Step 3a : Run for reduced set of vars:
model2=glm(CKD~Age + Racegrp + HDL + LDL + SBP + DBP +
             PVD + Activity + Fam.Hypertension + Diabetes +
             CVD + CHF + Anemia,family="binomial",data=trainingData)
summary(model2) #AIC 1471.3

model4=glm(CKD~Age+ Racegrp + HDL + LDL + 
             PVD + Activity +Hypertension+ Fam.Hypertension + Diabetes + 
             Stroke + CHF + Anemia,family="binomial",data=trainingData)
summary(model4) #AIC 1463.5

#Step 2a Check the predictions for test data.
pred = plogis(predict(model, newdata=testData)) #glm predicts log odds, use plogis to get prob
summary(pred)
pred2 = plogis(predict(model2, newdata=testData)) #glm predicts log odds, use plogis to get prob
summary(pred2)

pred3=plogis(predict(model3, newdata=testData))
summary(pred3)

pred4 = plogis(predict(model4, newdata=testData)) #glm predicts log odds, use plogis to get prob
summary(pred4)
#write.csv(pred,"pred.csv")
#write.csv(pred2,"pred2.csv")
#write.csv(pred3,"pred3.csv")
#write.csv(pred4,"pred4.csv")
#Step 2b : Calculate accuracy and costs for the 4 models.
model2_money_max =calc_thresh(model2,pred2,testData)
model4_money_max =calc_thresh(model4,pred4,testData)
model4_money_max-model2_money_max
g1 <- roc(CKD ~ pred, data = testData,percent=TRUE)
plot(g1)
auc(g1)
g3 <- roc(CKD ~ pred3, data = testData,percent=TRUE)
plot(g3)
auc(g3)


g2 <- roc(CKD ~ pred2, data = testData,percent=TRUE)
plot(g2)
auc(g2)


g4 <- roc(CKD ~ pred4, data = testData,percent=TRUE,print.auc = T,add = T)
plot(g4)
auc(g4,)

classify2=ifelse(pred2>0.057,1,0)  # this is a threshold, we say if probability >50% , then say "yes"
summary(classify2)

acc2=c_accuracy(testData$CKD,classify2)
acc2

classify4=ifelse(pred4>0.067,1,0)  # this is a threshold, we say if probability >50% , then say "yes"
summary(classify4)

acc4=c_accuracy(testData$CKD,classify4)
acc4



#Step 3  - Run the Logistic Regression on all data,

dim(data)
# model=glm(CKD~ .,family="binomial",data=data_in) #AIC: 2227 on test dataset
# summary(model)
# #check variables given by stepwise elimination
# model3=step(model,direction="both") #Age + Female + Racegrp + Weight + BMI + Obese + Waist + DBP + HDL + LDL + PVD + Activity + Hypertension + Fam.Hypertension + 
#                                     #Diabetes + CVD + Fam.CVD + CHF + Anemia
#Step 3a : Run for reduced set of vars:
model2_in=glm(CKD~Age + Racegrp + HDL + LDL + SBP + DBP +
                PVD + Activity + Fam.Hypertension + Diabetes +
                CVD + CHF + Anemia,family="binomial",data=data_in)
summary(model2_in) #AIC 2246.7

model4_in=glm(CKD~Age+ Racegrp + HDL + LDL + 
                PVD + Activity +Hypertension+ Fam.Hypertension + Diabetes + 
                Stroke + CHF + Anemia,family="binomial",data=data_in)
summary(model4_in) #AIC 2230.8

plot(model4_in)


#money from two models

pred2_in = plogis(predict(model2_in)) #glm predicts log odds, use plogis to get prob
pred4_in=plogis(predict(model4_in))

model2_money_max_in =calc_thresh(model2_in,pred2,testData)
model4_money_max_in =calc_thresh(model4_in,pred4,testData)

## odds ratios
pred4_in/(1-pred4_in)

#Step 4:Hypothesis test of model, Compare 2 models


##MODEL 2
with(model2_in, null.deviance - deviance)
##df
with(model2_in, df.null - df.residual)
## pvalue of difference
with(model2_in, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
# if <.05, then model is significant from a null model (model with no variables)
# note that you can do this incrementally by adding one variable at a time.

## Step 4b - Alternate. Ho:  Model Fits the Data, Ha: Model does not Fit, Definition 5-2
## deviance
-2*logLik(model2_in)
## test
with(model2_in, pchisq(deviance, df.residual, lower.tail = FALSE))

#MODEL 4
with(model4_in, null.deviance - deviance)
##df
with(model4_in, df.null - df.residual)
## pvalue of difference
with(model4_in, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
# if <.05, then model is significant from a null model (model with no variables)
# note that you can do this incrementally by adding one variable at a time.

## Step 4b - Alternate. Ho:  Model Fits the Data, Ha: Model does not Fit, Definition 5-2
## deviance
-2*logLik(model4_in)
## test
with(model4_in, pchisq(deviance, df.residual, lower.tail = FALSE))

#step 4c - Plot ROC for complete data

g2_in <- roc(CKD ~ pred2_in, data = data_in,percent=TRUE)
plot(g2_in)
auc(g2_in)


g4_in <- roc(CKD ~ pred4_in, data = data_in,percent=TRUE)
plot(g4_in)
auc(g4_in)

classify2_in=ifelse(pred2_in>0.057,1,0)  # this is a threshold, we say if probability >50% , then say "yes"
summary(classify2_in)

acc2_in=c_accuracy(data_in$CKD,classify2_in)
acc2_in

classify4_in=ifelse(pred4_in>0.067,1,0)  # this is a threshold, we say if probability >50% , then say "yes"
summary(classify4_in)

acc4_in=c_accuracy(data_in$CKD,classify4_in)
acc4_in

#Step 5 : Predict for out sample
pred2_out = plogis(predict(model2_in, newdata=data_out)) #glm predicts log odds, use plogis to get prob
summary(pred2_out)
length(pred2_out)
pred4_out = plogis(predict(model4_in, newdata=data_out)) #glm predicts log odds, use plogis to get prob
summary(pred4_out)
#money_total2=calc_thresh(model2,pred2,data_out)
#money_total4=calc_thresh(model4,pred4,data_out)
classify_model4_fin=ifelse(pred4_out>0.067,1,0)  # this is a threshold, we say if probability >20% , then say "yes"
summary(classify_model4_fin)
prediction_df=data.frame(pred4_out,classify_model4_fin)
colnames(prediction_df)<-c("Probability","CKD?")
write.csv(prediction_df,"Predictions.csv")

#acc=c_accuracy(data_out$CKD,classify)
#acc
calc_thresh=function(model,pred,testData){
  min_model_thresh=0
  max_model_money=-1
  money_vec=c()
  threshold_vec=c()
  for(i in seq(1,100,0.1)){
    thresh=i*0.01
    classify=ifelse(pred>thresh,1,0)  # this is a threshold, we say if probability >50% , then say "yes"
    summary(classify)
    
    acc=c_accuracy(testData$CKD,classify)
    acc
    c1=1300   # Reward me  $1300 for a true positive
    c2=-100  #  penalize me $100 for a false positive
    money=acc[7]*c1+acc[9]*c2
    money_vec<-c(money_vec,money)
    threshold_vec<-c(threshold_vec,thresh)
    if(money>max_model_money){
      max_model_money=money
      min_model_thresh=thresh
    }
  }
  df=data.frame(money_vec,threshold_vec)
  colnames(df)<-c("Money","Threshold")
  #plot(df)
  gg<- ggplot(data= df,mapping = aes(x=df$Money,y=df$Threshold)) + geom_point()+
    geom_text(aes(label=ifelse(df$Threshold==min_model_thresh,as.character(df$Threshold),' ')),hjust=0, vjust=0)
  print(gg)
  print(min_model_thresh)
  return(max_model_money)
}

# This Method is written by Prof. Matthew J Schnieder.
c_accuracy=function(actuals,classifications){
  df=data.frame(actuals,classifications);
  
  
  tp=nrow(df[df$classifications==1 & df$actuals==1,]);        
  fp=nrow(df[df$classifications==1 & df$actuals==0,]);
  fn=nrow(df[df$classifications==0 & df$actuals==1,]);
  tn=nrow(df[df$classifications==0 & df$actuals==0,]); 
  
  
  recall=tp/(tp+fn)
  precision=tp/(tp+fp)
  accuracy=(tp+tn)/(tp+fn+fp+tn)
  tpr=recall
  fpr=fp/(fp+tn)
  fmeasure=2*precision*recall/(precision+recall)
  scores=c(recall,precision,accuracy,tpr,fpr,fmeasure,tp,tn,fp,fn)
  names(scores)=c("recall","precision","accuracy","tpr","fpr","fmeasure","tp","tn","fp","fn")
  
  #print(scores)
  return(scores);
}


