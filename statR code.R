## Two-sample test
x=c(7.85, 12.03, 21.84, 13.94, 16.68, 41.78, 14.97, 12.072)
y=c(9.59, 34.50, 4.55, 20.78, 11.69, 32.51, 5.46, 12.95)
    t.test(x,y, paired = TRUE) # or
  d=x-y 
  d
  mean(d)
  var(d)
  sqrt(var(d))
  sd(d)
  qqnorm(x-y)
  qqline(x-y)
  
  x=c(2.3,3.4,1.2,4.4)
  y=c(3.2,1.5,2.6,3.3,4.5)
  t.test(x,y,var.equal=T)
  x=c(1,1,1,1,2,2,2,2,2)
  y=c(2.3,3.4,1.2,4.4,3.2,1.5,2.6,3.3,4.5)
  t.test(y~x,var.equal=T) # the same results
  
  treated=c(18,43,28,50,16,32,13,35,38,33,6,7)
  Untreated=c(40, 54, 26, 63, 21, 37, 39, 23, 48, 58, 28, 39)
  #dev.off()
  t.test(treated,Untreated)
  par(mfrow=c(2,2))
  qqnorm(treated)
  qqline(treated)
  qqnorm(Untreated)
  qqline(Untreated)
  library(car)
  qqPlot(treated)
  qqnorm(treated, pch = 1, frame = FALSE)
  qqline(my_data$len, col = "steelblue", lwd = 2)
  boxplot(treated,main="Box plot for treated")
  boxplot(Untreated,main="Box plot for Untreated")
  
  BR<-c(20.8, 4.1, 30.0, 24.7, 13.8, 7.5, 7.5,
        11.9, 4.5, 3.1, 8.0, 4.7, 28.1, 10.3, 10.0,
        5.1, 2.2, 9.2, 2.0, 2.5, 6.1, 7.5)
  LF<-c(rep("A",5),rep("B",12),rep("C",5))
  Group<-factor(LF)
  fit<-aov(BR~Group)
  anova(fit)
  
  
  
  
  
  y=c(2.3,3.4,1.2,4.4,3.2,1.5,2.6,3.3,4.5)
  t.test(y~x,var.equal=T)
    
    t.test(x-y)
    
  ?t.test()  
##
    qqnorm(x-y)
    qqline(x-y)
### independent two sample
    
    x=c(18, 43, 28, 50, 16, 32, 13, 35, 38, 33, 6, 7)
    y=c(40, 54, 26, 63, 21, 37, 39, 23, 48, 58, 28, 39)
    t.test(x,y,var.equal=T,alternative = "less")
    
    #par(mfrow=c(#rows,#columns))
    png("p3_sa_para.png", 640, 480)
    # help(png)
    treated=c(18, 43, 28, 50, 16, 32, 13, 35, 38, 33, 6, 7)
    Untreated=c(40, 54, 26, 63, 21, 37, 39, 23, 48, 58, 28, 39)
    #par(mfrow=c(2,1))
    qqnorm(treated)
    qqline(treated)
    qqnorm(Untreated)
    qqline(Untreated)
    boxplot(treated,main="Box plot for treated")
    boxplot(Untreated,main="Box plot for Untreated")
    dev.off()
    #png("p3_sa_para.png", 640, 480)
    ###Two-way ANOVA
    activity<-c(2.283,2.396,2.838,2.956,4.216,3.620,2.889,3.550,
                3.105,4.556,3.087,4.939,3.486,3.079,2.649,1.943,4.198,2.473,
                2.033,2.200,2.157,2.801,3.421,1.811,4.281,4.772,3.586,3.944,
                2.669,3.050,4.275,2.963,3.236,3.673,3.110)
    genotype<-factor(c("ff","fs","ff","fs","ff","ss","ff","fs","fs",
                       "fs","fs","ff","ff","ss","fs","fs","ff","ff","ff","fs","fs",
                       "ss","ss","ff","fs","fs","ss","ff",
                       "ss","ss","ss","ss","ss","ss","ss"))
    sex<-factor(c("m","m","f","m","f","f","f","f","m","f","f",
                  "m","m","f","m","f","f","f","f","f","f","m","m","f",
                  "f","f","f","f","f","f","m","f","f","f","m"))
    res<-aov(activity~genotype*sex)
    res
    summary(res)
    
    
    
    
    ### regression
    
bw<-c(25,25,25,27,27,27,24,30,30,31,30,31,30,28,32,
      32,32,32,34,34,34,35,35,34,35,36,37,38,40,39,43)
estriol<-c(7,9,9,12,14,16,16,14,16,16,17,19,21,24,15,
            16,17,25,27,15,15,15,16,19,18,17,18,20,22,25,24)
plot(bw~estriol,xlab="Estriol",ylab="Birth Weight")
rf<-lm(bw~estriol) # model fit
summar(rf)
anova(rf)
lines(fitted(rf)~estriol) # add reg. line
text(12,40,expression(paste("y = 21.52+.608x"))) #

###par(mfrow=c(#rows,#columns))
par(mfrow=c(1,3))
plot(bw~estriol,xlab="Estriol",ylab="Birth Weight")
boxplot(bw,xlab="pred. value",
        ylab="studentized residual",
        main="Box plot", type="n")
qqplot(bw,estriol)
#### CI
  pre<-data.frame(estriol=c(10))
 predict(rf,newdata=pre,interval="confidence",level=0.95)
predict(rf,newdata=pre,interval="prediction",level=0.95)

SBP=c(89,90,83,77,92,98,82,85,96,95,80,79,86,97,92,88)
bw=c(135,120,100,105,130,125,125,105,120,90,120,95,120,150,160,125)
age=c(3,4,3,2,4,5,2,3,5,4,2,3,3,4,3,3)
data=data.frame(SBP,bw,age)
attach(data)
res<-lm(SBP~bw+age,data=data)
plot(rstudent(res) ~ hatvalues(res),xlab="pred. value",
      ylab="studentized residual",
      main="Stud. res. vs pred. val.", type="n")
text(hatvalues(res),rstudent(res),1:16)
influence.measures(res) #


res1<-lm(SBP~bw+age,data=data[-10,])
plot(rstudent(res1) ~ hatvalues(res1),xlab="pred. value",
     ylab="studentized residual",
     main="Stud. res. vs pred. val. w/o [10th]",type="n")
text(hatvalues(res1),rstudent(res1),seq(16)[-10])

 influence.measures(res) # identify influential points
library(car)
crPlots(res) # partial residual plots, need package {car}
crPlots(res1) 
#multiple regression
res<-lm(SBP~bw+age,data=data)
summary(res)

res<-lm(SBP~bw+age,data=data)
plot(rstudent(res) ~ hatvalues(res),xlab="pred. value",
      ylab="studentized residual",
      main="Stud. res. vs pred. val.", type="n")
text(hatvalues(res),rstudent(res),1:16)
influence.measures(res) # evaluation of influential points

res1<-lm(SBP~bw+age,data=data[-10,])
plot(rstudent(res1) ~ hatvalues(res1),xlab="pred. value",
     ylab="studentized residual",
     main="Stud. res. vs pred. val. w/o [10th]",type="n")
text(hatvalues(res1),rstudent(res1),seq(16)[-10])


###Homework
data=read.csv("ESTRADL.csv") # input data
data=read.csv("C:/Users/Administrator/Desktop/Notes-7-8-11-12/ESTRADL.csv")
attach(data)
model=lm(Estradl~Entage)
summary(model)


#### Binary data
x=c(683,1498); n=c(3220, 10245)
prop.test(x, n, alternative = c("two.sided"),
           conf.level = 0.95, correct = TRUE)

t22<-matrix(c(13,4987,7,9993),nrow = 2, ncol = 2, byrow =TRUE)
  chisq.test(t22)
  
  ### chi-square test
  freq<-matrix(c(320,1206,1011,463,220,1422,4432,
                 2893,1092,406),nrow = 2, ncol = 5, byrow = TRUE)
  HT<-chisq.test(x=freq)
  HT
  HT$expected
### Logistic
wcgs<-read.csv("wcgs.csv",header=T)
lrf1<-glm(chd69~smoke,family=binomial(link = "logit"),data=wcgs)
summary(lrf1)

### count data
log.fit<-glm(satell~width, family=poisson(link=log),data=crab)
summary(log.fit)

