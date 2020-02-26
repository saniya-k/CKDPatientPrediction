#install.packages("mice") 
library("mice")
library(VIM)
library("dplyr")
## Step 1  - Read in Data
setwd("C:\\Workarea 1.1\\General\\LEARN\\MSBA\\Terms\\Winter 2019\\STAT-630\\group project\\CKD\\CKD_Project")
data=read.csv("casestudydata.csv")
names(data)



class(data)
summary(data)
CKD = data$CKD #store pred var
data1 = data[-c(34)] # remove prediction variable
summary(data1)

head(data1)
str(data1)
sapply(data1, function(x) sum(is.na(x))) #check which have missing
#proportion of missing data
pMiss <- function(x){sum(is.na(x))/length(x)*100}
anyNA(data1)

#check for % missing per feature
datamissd<-apply(data1,1,pMiss)
datamissd<-datamissd[datamissd!=0]
length(datamissd)

md.pattern(data1)
#plot missing val distr
data_miss <- aggr(data1, col=c('darkgreen','red'), numbers=TRUE, sortVars=TRUE, labels=names(data1), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
data_miss

data1 <- data1 %>% mutate(
  Educ = as.factor(Educ),
  Unmarried = as.factor(Unmarried),
  Income = as.factor(Income),
  Insured = as.factor(Insured),
  Obese = as.factor(Obese),
  Activity=as.factor(Activity),
  PoorVision=as.factor(PoorVision),
  Hypertension=as.factor(Hypertension),
  Diabetes=as.factor(Diabetes),
  Fam.Hypertension=as.factor(Fam.Hypertension),
  Smoker=as.factor(Smoker),
  Fam.Diabetes=as.factor(Fam.Diabetes),
  Stroke=as.factor(Stroke),
  CVD=as.factor(CVD),
  Fam.CVD=as.factor(Fam.CVD),
  CHF=as.factor(CHF),
  Anemia=as.factor(Anemia),
  Dyslipidemia=as.factor(Dyslipidemia),
  PVD=as.factor(PVD)
  
)
?mice
str(data1)
init=mice(data1,maxit=0)
meth = init$method
predM = init$predictorMatrix
predM[, c("ID")]=0
meth[c("SBP","DBP","Total.Chol","HDL","LDL","CareSource")]="pmm"
meth[c("Weight","Height","BMI","Waist")]="norm" 
meth[c("Educ","Unmarried","Income","Obese","Insured","Diabetes","Fam.Diabetes","Smoker","Stroke","CVD","Fam.CVD","CHF","Anemia","PoorVision","Hypertension","Fam.Hypertension")]="logreg" 
meth[c("Activity")]="polyreg"
imp <- mice(data1, method = meth,  predictorMatrix=predM , m = 2) # Impute data
data_sto <- complete(imp)
densityplot(imp)#similar shape indicates good distribution
data_sto = data[c[34]]
data_sto["CKD"] = CKD
write.csv(data_sto, "Imputed.csv")
sapply(data_sto, function(x) sum(is.na(x)))