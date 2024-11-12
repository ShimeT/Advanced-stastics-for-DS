
#Pinheiro, J. C. and Bates, D. M. (2000), Mixed-Effects Models in S and S-PLUS, Springer, New York. (Appendix A.17)

# Potthoff, R. F. and Roy, S. N. (1964), A generalized multivariate analysis of variance model useful especially for growth curve problems, Biometrika, 51, 313-326.





## PSID
library(tigerstats)
library(faraway)
data(psid)
mypsid<-subset (psid, (subset=(person <4)))
mypsid1<-subset(mypsid, (subset=(year < 75)))
mypsid1
attached(mypsid1)
### profile plot
xyplot(income ~ year , psid, type="l", subset=( person < 20),strip=TRUE)
### profile plot by sex

# xyplot(income ~ year | sex, psid, type="l", subset=( person < 500),strip=TRUE)
 xyplot(income ~ age | sex, mypsid1, type="l", subset=( person < 500),strip=TRUE)

 ### mean profile plot
mean1<-tapply(income, age, mean)
interaction.plot(age, Sex, income, ylim=c(20, 30), lty=c(1, 2), fun=mean,
                 lwd=3,ylab="income",xlab="Age", trace.label="Sex")
title(main=" Profile plot of psi by sex")
##### mean profile

library(nlme)
data(Orthodont)
?Orthodont
library(lattice)
library(foreign)
#dat<-read.table("C:\\growth.txt", header = TRUE, sep = "" )
#growth<-dat
growth<-Orthodont
attach(growth)
age<-as.factor(age)
Sex<-as.factor(Sex)
head(growth)
write.csv(wcgs,file="growth.csv",row.names=FALSE)
# xyplot(distance ~ age|Sex, growth, type="l", xlab= " age ",
#     ylab=" distance from pituitary to pterygomaxillary fissure (mm) ")
mean1<-tapply(distance, age, mean)
age1<-as.numeric(unique(age))
plot(age1, mean1, type= "l",ylim=c(20,30), xlab="age",
     ylab=" The mean distance", lwd=3, main=" The mean profile of the growth data set")
interaction.plot(age, Sex, distance, ylim=c(20, 30), lty=c(1, 2), fun=mean,
                 lwd=3,ylab="distance from pituitary to pterygomaxillary fissure (mm) ",
                 xlab="Age", trace.label="Sex")
title(main=" Profile plot of growth data set by sex")

####
library(lattice)
library(foreign)
#dat<-read.table("C:\\growth.txt", header = TRUE, sep = "" )
#attach(dat)
varg<- tapply( distance,age, var)
names(varg)
age1<-as.numeric(unique(age))
varg<-as.vector(varg)
plot(age1, varg, type='l',main =" Observed variance ", xlab=' age', ylab='The variance of distance', lwd=3)
interaction.plot(age, Sex, distance, lty=c(1, 2), fun=var,
                 ylab="distance from pituitary to pterygomaxillary fissure (mm) ",
                 xlab="Age", trace.label="Sex")
title(main=" The variance of the growth data set by sex")

library(nlme)
Ortho.fit1 <- lme(fixed = distance ~ Sex+Sex*age,data = Orthodont,
                  random = ~ 1 | Subject)
print(Ortho.fit1) ## It prints the estimates of the standard deviations and the correlations of the random
                   ## effects, the within-group standard error, and the fixed effects.
coef(Ortho.fit1) ##  estimates of the parameters of the model.

fixef(Ortho.fit1) ## exract fixed effects

summary(Ortho.fit1)

###### ML method
library(nlme)
Ortho.fit2 <- lme(fixed = distance ~ Sex+Sex*age,method= " ML",
                  data = Orthodont, random = ~ 1 | Subject)
intervals(Ortho.fit2) 
summary(Ortho.fit2)


### Survival Analysis
## Loading packages needed for analysis
library(survival)
library(lattice)

## Read in and attach the Schizonphrenia patient data
setwd("c://")
schizo = read.csv("C:/Users/Administrator/Desktop/Notes-7-8-11-12/Schizophrenia.csv", header=T, sep=";", as.is=T) 
attach(schizo)
head(schizo)
## Look at data summaries to understand the data

dim(schizo)
schizo[1:15,]

summary(schizo)

table(Gender)
table(Gender, Censor)
table(Marital)
table(Gender, Marital)
hist(Time)

#histogram( ~ Time | Censor)
#histogram( ~ Time | Censor + Gender)

plot(Time, Onset, pch=Censor+2)

## Kaplan-Meier (Product Limit Estimator) Analysis

KM_schizo_l = survfit(Surv(Time, Censor)~1, data=schizo,
                      type='kaplan-meier', conf.type = 'log-log')
KM_schizo_l
summary(KM_schizo_l)
plot(KM_schizo_l, conf.int=T)

## Kaplan-Meier Analysis by Gender

KM_schizo_l_Gen = survfit(Surv(Time, Censor)~ Gender, data=schizo,
                          type='kaplan-meier', conf.type = 'log-log')
KM_schizo_l_Gen
summary(KM_schizo_l_Gen)
plot(KM_schizo_l_Gen, conf.int=T)


#### coxph
phm.age = coxph(Surv(Time,Censor)~Age)
phm.sex = coxph(Surv(Time,Censor)~Gender)
summary(phm.age)
summary(phm.sex)
