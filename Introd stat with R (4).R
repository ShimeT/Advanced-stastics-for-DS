## at a basic level we can use R much as you would use a calculator.
## The arithmetic operators are +, -, *, /
a=2 + 2 ## plus 
a
b=2-2 ## minus
b
c=2*2 ## product
c
d=2/2 ## ratio or devide
d
aa=log(1)              # logarithm to base e
aa
bb=log10(1)            # logarithm to base 10
bb
cc=exp(1)              # natural antilog  ## [1] 2.718282
cc
dd=sqrt(4)             # square root
dd
ee=4^(3*4)                  # 4 to the power of 2
ee
ff=ee/dd
ff
## The c() function is short for concatenate and we use it to join together a series of values and store them in a data structure called a vector 
mydata <- c(2, 3,1, 6, 4, 3, 3, 7)
sort(mydata)
n=length(mydata)  # returns the number of elements in mydata
n
sum=sum(mydata,na.rm = TRUE)
print(sum)
sum
barplot(mydata)
sm2=sum(mydata^2)
sm2
smm2=mydata^2
smm2
sm2=sum(smm2)
sm2
help(sum)
my_seq <- 1:10     # create regular sequence
my_seq
my_seq2 <- 10:1    # in decending order
my_seq2
ratio=my_seq2/my_seq
ratio
product=my_seq2*my_seq
product
## Other useful functions for generating vectors of sequences include the seq() and rep() functions. For example, to generate a sequence from 1 to 5 in steps of 0.5
## Here we've used the arguments from = and to = to define the limits of the sequence and the by = argument to specify the increment of the sequence
my_seq2 <- seq(from = 1, to = 5, by = 1)
my_seq2

## The rep() function allows you to replicate (repeat) values a specified number of times. To repeat the value 2, 10 times
my_seq3 <- rep(2, times = 10)   # repeats 2, 10 times
my_seq3
help(rep)
rep(1:4, 4)
rep(1:4, each = 2)
rep(1:4, c(2,2,2,2))
rep(1:4, c(2,1,2,1))

## You can also repeat non-numeric values
my_seq4 <- rep("abc", times = 3)    # repeats 'abc' 3 times 
my_seq4

## or each element of a series
my_seq5 <- rep(1:5, times = 3)  # repeats the series 1 to 5, 3 times
my_seq5

## or elements of a series
my_seq6 <- rep(1:5, each = 3)   # repeats each element of the series 3 times
my_seq6

## We can also repeat a non-sequential series
## Note in the code above how we've used the c() function inside the rep() function. 
## Nesting functions allows us to build quite complex commands within a single line of code and is a very common practice when using R.
bb=c(3, 1, 10, 7)
my_seq7 <- rep(bb, each = 3) # repeats each element of the  series 3 times
my_seq7


## Positional index
## To extract elements based on their position we simply write the position inside the [ ]. For example, to extract the 3rd value of my_vec
my_vec <- c(2, 3, 1, 6, 4, 3, 3, 7)
my_vec[3]     # extract the 3rd value
my_vec[4]/3
## if you want to store this value in another object
val_3 <- my_vec[3]
val_3
## We can also extract more than one value by using the c() function inside the square brackets. Here we extract the 1st, 5th, 6th and 8th element from the my_vec object
my_vec[c(1, 5, 6, 8)]
## Or we can extract a range of values using the : notation. To extract the values from the 3rd to the 8th elements
my_vec[3:8]

help(matrix)
?matrix()
mtrx=matrix(my_vec,4,2)
mtrx1=matrix(c(2,4,6,8,9,6,4,8),4,2)
t(mtrx)
mtrx%*%t(mtrx1)
## the expression for 'greater or equal to' is >=. 
## To test whether a value is equal to a value we need to use a double equals symbol == and 
## for 'not equal to' we use != (the ! symbol means 'not').

my_vec[my_vec >= 4]        # values greater or equal to 4

my_vec[my_vec < 4]         # values less than 4

my_vec[my_vec <= 4]        # values less than or equal to 4

my_vec[my_vec == 4]        # values equal to 4

my_vec[my_vec != 4]        # values not equal to 4

## We can also combine multiple logical expressions using Boolean expressions. In R the & symbol means AND and the | symbol means OR. 
## For example, to extract values in my_vec which are less than 6 AND greater than 2

val26 <- my_vec[my_vec < 6 & my_vec > 2]
val26
## or extract values in my_vec that are greater than 6 OR less than 3
val63 <- my_vec[my_vec > 6 | my_vec < 3]
val63

## Replacing elements
my_vec[4] <- 500
my_vec
# replace the 6th and 7th element with 100
my_vec[c(6, 7)] <- 100
my_vec
## replace element that are less than or equal to 4 with 1000
my_vec[my_vec <= 4] <- 1000
my_vec

## Ordering elements
vec_sort <- sort(my_vec)
vec_sort

## To reverse the sort, from highest to lowest, we can either include the decreasing = TRUE argument when using the sort() function
vec_sort2 <- sort(my_vec, decreasing = TRUE)
vec_sort2

height <- c(180, 155, 160, 167, 181)
height
p.names <- c("Joanna", "Charlotte", "Helen", "Karen", "Amy")
p.names
height_ord <- order(height)
height_ord
names_ord <- p.names[height_ord]
names_ord
## Vectorisation
# create a vector
my_vec2 <- c(3, 5, 7, 1, 9, 20)
# multiply each element by 5
my_vec2 * 5

##      Data Presentation in R
# Create Data
data<-c('G','E','E','T','A','N','S','H','S','A','H','N','I')

# Use table() to get the frequency table
table <- table(data)
table
# Printing table
print(table)
table<-table(data)
print("Simple Frequency Table")
print(table)

# Use cumsum function to 
# Create cumulative frequency table

cumsum_table <- cumsum(table)   
print("cumulative Frequency Table")
print(cumsum_table)

# Create data frame
employee = c('A', 'B', 'A','A', 'B', 'C','A','B','C')
sales = round(runif(9, 2000, 5000), 0)
complaints = c('Yes','No','Yes','Yes','Yes','Yes','No','No','Yes')
data1=cbind(employee,sales,complaints)
data <- data.frame(
  employee = c('A', 'B', 'A','A', 'B', 'C','A','B','C'),
  sales = round(runif(9, 2000, 5000), 0),
  complaints = c('Yes','No','Yes','Yes','Yes','Yes','No','No','Yes') )
# print data
print(data)

table(data$employee,data$complaints)


##      Types of R - Charts
##      Bar Plot or Bar Chart
##      Pie Diagram or Pie Chart
##      Histogram
##      Scatter Plot
##      Box Plot

# defining vector
x <- c(7, 15, 23, 12, 44, 56, 32)
# plotting vector
#barplot(x, xlab = "website",ylab = "Frequency", col = "white",col.axis = "darkgreen",col.lab = "darkgreen")

barplot(x, xlab = "Media Audience",ylab = "Count", col = "gree",col.axis = "darkgreen",col.lab = "darkgreen")

## Histograme
mid.age <- c(2.5,7.5,13,16.5,17.5,19,22.5,44.5,70.5)
acc.count <- c(28,46,58,20,31,64,149,316,103)
age.acc <- rep(mid.age,acc.count)
brk <- c(0,5,10,16,17,18,20,25,60,80)
hist(age.acc,breaks=brk)
par(mfrow=c(1,2))

barplot(x, xlab = "Media Audience",ylab = "Count", col = "white",col.axis = "darkgreen",col.lab = "darkgreen")

hist(age.acc,breaks=brk)

caff.marital <- matrix(c(652,1537,598,242,36,46,38,21,218,327,106,67),nrow=3,byrow=T)
colnames(caff.marital) <- c("0","1-150","151-300",">300")
rownames(caff.marital) <- c("Married","Prev.married","Single")
caff.marital

par(mfrow=c(2,2))
 barplot(caff.marital, col="white")
 barplot(t(caff.marital), col="white")
 barplot(t(caff.marital), col="white", beside=T)
 barplot(prop.table(t(caff.marital),2), col="white", beside=T)
 par(mfrow=c(1,1))

## As usual, there are a multitude of ways to "prettify" the plots. Here is one possibility
 barplot(prop.table(t(caff.marital),2),beside=T,legend.text=colnames(caff.marital),col=c("white","grey80","grey50","black"))
 
#hist(age.acc,breaks=brk)
#hist(x,                             # Add count labels
 #    labels = TRUE)
#hist(x,                             # Add percentage labels
 #    labels = paste0(round(hist(x, plot = FALSE)$counts / length(x) * 100, 1), "%"))

## combining plots
x <- rnorm(100)
hist(x,freq=F)
curve(dnorm(x),add=T)

## pie chart
count_3 <- c(20, 50, 30)

# sum(count_3) # 100
pie(count_3, labels = paste0(count_3, "%"))
?pie()
legend("top", legend = c("Theatre", "Series", "Movies"),
       fill =  c("white", "lightblue", "mistyrose"))
# defining vector with ages of employees
x <- c(42, 21, 22, 24, 25, 30, 29, 22,
       23, 23, 24, 28, 32, 45, 39, 40)


# box plotting
boxplot(x, xlab = "Box Plot", ylab = "Age",col.axis = "darkgreen", col.lab = "darkgreen")

## Measure fo Central Tendency and Dispersion

## mean, variance, standard deviation and number of elements in our vector by using the mean(), var(), sd() and length() functions
mydata <- c(2, 3, 1, 6, 4, 3, 3, 7)
mean=mean(mydata)    # returns the mean of mydata
mean
mean=sum(mydata)/length(mydata)
mean
var=var(mydata)     # returns the variance of mydata
var
var=sum((mydata-mean(mydata))^2)/(length(mydata)-1)
sd=sd(mydata)      # returns the standard deviation of mydata
sd
sd=sqrt(var)
sd
#length(mydata)  # returns the number of elements in mydata
summary(mydata)
mydata
# Import the data using read.csv()
data=read.csv("C:/Users/HP/Documents/CardioGoodFitness.csv")
#data$Product
attach(data)
head(data)
#myData = read.csv("data",stringsAsFactors=F)
library(readr)
library(haven)
library("foreign")
#data1 <-read.dta("C:/Users/Administrator/Desktop/IntroStatLec/Feb8BHH.dta")
#data1 <-file.choose("C:/Users/Administrator/Desktop/IntroStatLec/Feb8BHH.dta")
#data2 <-read.sav("C:/Users/Administrator/Desktop/IntroStatLec/file.sav")
print(head(data))

# table 2by2 table
## mtGender <- table(data$Gender,data$MaritalStatus)
mtGender <- table(Gender,MaritalStatus)
mtGender

#  
margin.table(mtGender)
margin.table(mtGender,1)
margin.table(mtGender,2)
mtGender/margin.table(mtGender)
prop.table(mtGender)
prop.table(mtGender,1)
prop.table(mtGender,2)


## mean
mean=mean(Miles)
mean
head(data)
median = median(Age)
print(median)

# Import the library
library(modeest)
head(data)
# Compute the mode value
mode = mfv(Age)
print(mode)
sort(table(Age))

## Probability

## sample range lies between 1 to 5
x<- sample(1:5)
#prints the samples
x
## samples range is 1 to 5 and number of samples is 3
x<- sample(1:5, 4)
## prints the samples (3 samples)
x
#sample range is 1 to 5 and the number of samples is 6
x<- sample(1:5, 6)
## shows error as the range should include only 5 numbers (1:5)
## specifing replace=TRUE or T will allow repetition of values so that the function will generate 6 samples in the range 1 to 5. Here 2 is repeated.
x<- sample(1:5, 6, replace=T)
x
#samples without replacement 
x<-sample(1:8, 7, replace=F)
### when you take the samples, they will be random and change each time. 
### In order to avoid that or if you don't want different samples each time, you can make use of set.seed() function.
## set.seed() - set.seed function will produce the same sequence when you run it.
#set the index 
set.seed(5)
#takes the random samples with replacement
sample(1:5, 4, replace=F)
set.seed(5)
sample(1:5, 4, replace=T)

# generate the vector of probabilities 
## We can use it to simulate the random outcome of a dice roll. Let's roll the dice!
sample(1:6, size=1) 
probability <- rep(1/6, 6) 
probability
# plot the probabilities 
plot(probability,xlab = "Outcomes",ylab="Probability",main = "Probability Distribution",pch=20)

# generate the vector of cumulative probabilities 
cum_probability <- cumsum(probability) 

# plot the probabilites 
plot(cum_probability, xlab = "Outcomes", ylab="Cumulative Probability",main = "Cumulative Probability Distribution",pch=20) 

###  Bernoulli Trials

# We might as well simulate coin tossing with outcomes  H(heads) and  T(tails).
# The result of a single coin toss is a Bernoulli distributed random variable, i.e., a variable with two possible distinct outcomes.

sample(c("H", "T"), 1)

## It is a well known result that the number of successes  k in a Bernoulli experiment follows a binomial distribution. We denote this as
## Let us compute  Bin(5,10,0.5)
dbinom(x = 5, size = 10, prob = 0.8) 
## Now assume we are interested in  P(4???k???7)
## i.e., the probability of observing  
##  This may be computed by providing a vector as the argument x in our call of dbinom() and summing up using sum().
# compute P(4 <= k <= 7) using 'dbinom()'
sum(dbinom(x = 4:7, size = 10, prob = 0.5))
#Alternative compute P(4 <= k <= 7) using 'pbinom()'
pbinom(size = 10, prob = 0.5, q = 7) - pbinom(size = 10, prob = 0.5, q = 3) 
# set up vector of possible outcomes
sum(dbinom(0:6,10,0.5))
#k <- 0:10
## To visualize the probability distribution function of  k
## we may therefore do the following:
    # assign the probabilities
k <- 0:10
  probability <- dbinom(x = k,size = 10, prob = 0.5)
  # plot the outcomes against their probabilities
  plot(x = k,y = probability,ylab="Prob",main = "PDF",pch=17) 

  ## Normal Density 
## dnorm() function in R programming measures density function of distribution. In statistics, 
## it is measured by below formula-creating a sequence of values  
## between -15 to 15 with a difference of 0.1 
x = seq(-15, 15, by=0.1) 
data
mean(x)
sd(x)
y = dnorm(x, mean(x), sd(x)) 

# Plot the graph. 
plot(x, y) 

##  pnorm() function is the cumulative distribution function which measures the probability that 
##  a random number X takes a value less than or equal to x i.e., in statistics it is given by-

## qnorm() function is the inverse of pnorm() function. It takes the probability value and gives output which corresponds to the probability value. 
## It is useful in finding the percentiles of a normal distribution.

## Syntax: qnorm(p, mean, sd)
?pnorm()
# Create a sequence of probability values  
# incrementing by 0.02. 
x <- seq(0, 1, by = 0.02) 

y <- qnorm(x, mean(x), sd(x)) 
# Plot the graph. 
plot(x, y) 
## rnorm() function in R programming is used to generate a vector of random numbers which are normally distributed.
rnorm(10)
rbinom(10,size=20,prob=.5)
##  Syntax: rnorm(x, mean, sd)
# Create a vector of 1000 random numbers 
# with mean=90 and sd=5 
x <- rnorm(10000, mean=90, sd=5) 

# Create the histogram with 50 bars 
hist(x, breaks=50) 

# Save the file. 
# dev.off() 

## dbinom() Function: This function is used to find probability at a particular value for a data that follows binomial distribution i.e. it finds:

##  P(X = k)
## Syntax: dbinom(k, n, p)

dbinom(3, size = 13, prob = 1 / 6) 
probabilities <- dbinom(x = c(0:10), size = 10, prob = 1 / 6) 
#data.frame(x, probs) 
plot(0:10, probabilities, type = "l")

## pbinom() Function: The function pbinom() is used to find the cumulative probability of a data following binomial distribution till a given value ie it finds

# P(X <= k)
## Syntax:pbinom(k, n, p)
pbinom(3, size = 13, prob = 1 / 6) 
plot(0:10, pbinom(0:10, size = 10, prob = 1 / 6), type = "l") 

## qbinom() Function
## This function is used to find the nth quantile, that is if P(x <= k) is given, it finds k.

## Syntax: qbinom(P, n, p)

qbinom(0.8419226, size = 13, prob = 1 / 6) 
x <- seq(0, 1, by = 0.1) 
y <- qbinom(x, size = 13, prob = 1 / 6) 
plot(x, y, type = 'l') 

## rbinom():  Function  This function generates n random variables of a particular probability.

## Syntax:: rbinom(n, N, p)
rbinom(8, size = 13, prob = 1 / 6) 
hist(rbinom(8, size = 13, prob = 1 / 6)) 

## In R, we can conveniently obtain densities of normal distributions using the function dnorm(). 
## Let us draw a plot of the standard normal density function using curve() together with dnorm().
# draw a plot of the N(0,1) PDF
curve(dnorm(x),
      xlim = c(-3.5, 3.5),
      ylab = "Density", 
      main = "Standard Normal Density Function") 

## We can obtain the density at different positions by passing a vector to dnorm().
## compute density at x=-1.96, x=0 and x=1.96
dnorm(x = c(-1.96, 0, 1.96))

# plot the standard normal CDF
curve(pnorm(x),xlim = c(-3.5, 3.5),ylab = "Probability",main = "Standard Normal Cumulative Distribution Function")
# define the standard normal PDF as an R function
f <- function(x) {
  1/(sqrt(2 * pi)) * exp(-0.5 * x^2)
}

## Let us check if this function computes standard normal densities by passing a vector.
# define a vector of reals
quants <- c(-1.96, 0, 1.96)

# compute densities
f(quants)

# plot the standard normal density
curve(dnorm(x),xlim = c(-4, 4),xlab = "x",lty = 2,ylab = "Density",main = "Densities of t Distributions")

# plot the t density for M=2
curve(dt(x, df = 2),xlim = c(-4, 4),col = 2,add = T)

# plot the t density for M=4
curve(dt(x, df = 4),xlim = c(-4, 4),col = 3,add = T)

# plot the t density for M=25
curve(dt(x, df = 25),xlim = c(-4, 4),col = 4,add = T)

# add a legend
legend("topright",c("N(0, 1)", "M=2", "M=4", "M=25"),col = 1:4,lty = c(2, 1, 1, 1))
## This section reviews important statistical concepts:
##  Estimation of unknown population parameters
## Hypothesis testing
## Confidence intervals

# packages which are not part of the base version of R:
# readxl - allows to import data from Excel to R.
# dplyr - provides a flexible grammar for data manipulation.
# MASS - a collection of functions for applied statistics.

library(dplyr)
library(MASS)
library(readxl)
# plot the chi_12^2 distribution
curve(dchisq(x, df=12), 
      from = 0, 
      to = 40, 
      ylab = "Density", 
      xlab = "Hourly earnings in Euro")
## We now draw a sample of  n=100 observations and take the first observation  Y1as an estimate for  ??Y
# set seed for reproducibility
set.seed(1)

# sample from the chi_12^2 distribution, use only the first observation
rsamp <- rchisq(n = 100, df = 12)
rsamp[1]

# plot the standard normal density on the interval [-4,4]
curve(dnorm(x),
      xlim = c(-4, 4),
      main = "Calculating a p-Value",
      yaxs = "i",
      xlab = "z",
      ylab = "",
      lwd = 2,
      axes = "F")
# add x-axis
axis(1, 
     at = c(-1.5, 0, 1.5), 
     padj = 0.75,
     labels = c(expression(-frac(bar(Y)^"act"~-~bar(mu)["Y,0"], sigma[bar(Y)])),
                0,
                expression(frac(bar(Y)^"act"~-~bar(mu)["Y,0"], sigma[bar(Y)]))))
# shade p-value/2 region in left tail
polygon(x = c(-6, seq(-6, -1.5, 0.01), -1.5),
        y = c(0, dnorm(seq(-6, -1.5, 0.01)),0), 
        col = "steelblue")
# shade p-value/2 region in right tail
polygon(x = c(1.5, seq(1.5, 6, 0.01), 6),
        y = c(0, dnorm(seq(1.5, 6, 0.01)), 0), 
        col = "steelblue")
## p-value when std is unknown
## observations of the Bernoulli distributed variable Y with success probability
# sample and estimate, compute standard error
samplemean_act <- mean(
  sample(0:1, 
         prob = c(0.9, 0.1), 
         replace = T, 
         size = 100))

SE_samplemean <- sqrt(samplemean_act * (1 - samplemean_act) / 100)

# null hypothesis
mean_h0 <- 0.1

# compute the p-value
pvalue <- 2 * pnorm(- abs(samplemean_act - mean_h0) / SE_samplemean)
pvalue

## The t-statistic
# compute a t-statistic for the sample mean
tstatistic <- (samplemean_act - mean_h0) / SE_samplemean
tstatistic
# prepare empty vector for t-statistics
tstatistics <- numeric(10000)
# plot density and compare to N(0,1) density
plot(density(tstatistics),
     xlab = "t-statistic",
     main = "Estimated Distribution of the t-statistic when n=300",
     lwd = 2,
     xlim = c(-4, 4),
     col = "steelblue")

# N(0,1) density (dashed)
curve(dnorm(x), 
      add = T, 
      lty = 2, 
      lwd = 2)
# set sample size
n <- 300

# simulate 10000 t-statistics
for (i in 1:10000) {
  
  s <- sample(0:1, 
              size = n,  
              prob = c(0.9, 0.1),
              replace = T)
  
  tstatistics[i] <- (mean(s)-0.1)/sqrt(var(s)/n)
  
}
## Hypothesis Testing
# check whether p-value < 0.05
pvalue < 0.05
## The condition is not fulfilled so we do not reject the null hypothesis correctly.
## When working with a  t-statistic instead, it is equivalent to apply the following rule:
## We reject the null hypothesis at the significance level of  5% if the computed  t-statistic lies beyond the critical value of 1.96 in absolute value terms.  
## 1.96 is the  0.975 -quantile of the standard normal distribution.
# check the critical value
## Reject H0 if |tcal|> 1.96
qnorm(p = 0.975)
# check whether the null is rejected using the t-statistic computed further above
abs(tstatistic) > 1.96


### Confidence interval in R

data("mtcars")
mtcars
head(mtcars)
sample.mean <- mean(mtcars$mpg)

print(sample.mean)
sample.n <- length(mtcars$mpg)
sample.n
sample.sd <- sd(mtcars$mpg)
sample.sd
sample.se <- sample.sd/sqrt(sample.n)
print(sample.se)

alpha = 0.05
degrees.freedom = sample.n - 1
t.score = qt(p=alpha/2, df=degrees.freedom,lower.tail=F)
print(t.score)

margin.error <- t.score * sample.se

lower.bound <- sample.mean - margin.error
upper.bound <- sample.mean + margin.error
print(c(lower.bound,upper.bound))

####
data=read.csv("C:/Users/HP/Documents/CardioGoodFitness.csv")
#data$Mile
attach(data)
head(data)
mean=mean(Miles)
mean
cor(Age,Income)
sample.sd=sd(Miles)
sample.n=length(Miles)
sample.se=sample.sd/sqrt(sample.n)
t.score = qt(p=alpha/2, df=degrees.freedom,lower.tail=F)
margin.error <- t.score * sample.se

lower.bound <- mean - margin.error
upper.bound <- mean + margin.error
print(c(lower.bound,upper.bound))

data("mtcars")

sample.mean <- mean(mtcars$mpg)
print(sample.mean)

sample.n <- length(mtcars$mpg)
sample.sd <- sd(mtcars$mpg)
sample.se <- sample.sd/sqrt(sample.n)
print(sample.se)

alpha = 0.05
degrees.freedom = sample.n - 1
t.score = qt(p=alpha/2, df=degrees.freedom,lower.tail=F)
print(t.score)

margin.error <- t.score * sample.se
lower.bound <- sample.mean - margin.error
upper.bound <- sample.mean + margin.error
print(c(lower.bound,upper.bound))

# Calculate the mean and standard error
l.model <- lm(Miles ~ 1, data)

# Calculate the confidence interval
confint(l.model, level=0.99)

#### Normal distribution

# P(X <= 70) X ~ N(75,5)
pnorm(q=70, mean=75, sd=5, lower.tail=T)

# P(X >= 85) X ~ N(75,5)
pnorm(q=85, mean=75, sd=5, lower.tail=F)

# P(Z >= 1) where Z ~ N(0,1)
pnorm(q=1, mean=0, sd=1, lower.tail=F)

# Find the value of X such that P(X <= 70) = 25%  where X ~ N(75,5)
qnorm(p=0.25, mean=75, sd=5, lower.tail=T)


## 1, Find the value of probability that Z is greater than 1 where Z ~ N(0,1).
## 2, Find the value of X such that P(Z <= X) = 25% where X ~ N(75,5).
# Question 1
pnorm(q=1, mean=0, sd=1, lower.tail=F)

# Question 2
qnorm(p=0.25, mean=75, sd=5, lower.tail=T)


######### practical session
#########
data1 <- read.csv("https://stats.idre.ucla.edu/stat/data/hsbraw.csv")
head(data1)
data=read.csv("C:/Users/Administrator/Desktop/IntroStatLec/CardioGoodFitness.csv")
#data$Product
attach(data)
head(data)
### Creating categorical variables
agelessthan30 <- ifelse(Age < 30,1,0)
agelessthan30
Fitn <- ifelse(Fitness==4,1,0)
Fitn
### editing a data 
fix(agelxessthan30)

### Saving an R dataframe 
fit1 <-cbind(data,Fitn)
head(fit1)
### Finding means, medians and standard deviations
mean(Income)
summary(data)
### function calculates standard deviations
sd(Miles)
### proportion
table=table(Gender)
table
prop.table(table(Gender))
prop.table(76,104)
76/180
help(prop.table)
### The na.omit( ) function omits missing data from a calculation.
xx=c(2,3,4,NA,5,6,7,8,9,10)
xx
na.omit(xx)
mean(xx)
## We can calculate the mean for the non-missing values the 'na.omit( )' function:
mean(na.omit(xx))
mean(xx, na.rm=TRUE)
## Statistical tables in R
## The standard normal (z) distribution
## The pnorm( ) function gives the area, or probability, below a z-value:
pnorm(1.96)
## To find a two-tailed area (corresponding to a 2-tailed p-value) for a positive z-value:
2*(1-pnorm(1.96))
## The qnorm( ) function gives critical z-values corresponding to a given lower-tailed area:
qnorm(.05) 
## To find a critical value for a two-tailed 95% confidence interval:
qnorm(1-.05/2)
## Confidence interval for a mean
##  The t.test( ) function performs one-sample and two-sample t-tests. In performing a one-sample t-test, this function also gives a confidence interval for the population mean.
t.test(Age)
mean(Age)
## Here, the mean age at Age for the sample of n=180 infants 
## (degrees of freedom are n-1) was 28.78, with a 95% confidence interval of (27.76,29.81).
## R calculates a 95% confidence interval by default, 
t.test(Age,conf.level=.90)
## Confidence interval for a proportion
table(Gender)
prop.test(76,104)
prop.test(table(Gender))
##  t-tests for means of measurement outcomes
##  The one-sample t-test for a mean
## The one-sample t-test compares the mean from one sample to some hypothesized value. 
## The t.test( ) function performs a one-sample t-test. For input, we need to specify 
## the variable (vector) that we want to test, and the hypothesized mean value. 
## To test whether the mean age at walking is equal to 12 months for the infants 
## in our Age variables example:
t.test(Age,mu=28.78,two.tailed="False")
## z-tests for proportions, categorical outcomes
table(Gender)
## 76 out of 180
prop.test(76,180,p=0.5)
head(data)
table(Gender,MaritalStatus)
## The chisq.test() function applied to a table 
## object compares these two percentages through the chi-square test of independence:
chisq.test(table(Gender,MaritalStatus))
## Simple regression analysis
## Regression analysis is performed through the 'lm( )' function. 
## LM stands for Linear Models, and this function can be used to perform simple regression, 
## multiple regression, and Analysis of Variance. For simple regression (with just one independent or predictor variable), predicting FEV1 from height:
lm(Income ~ Age)
summary(lm(Income ~ Age+Miles+Age*Miles) )

cor(Income,Miles)
head(data)
## Diagnosis 
## x=c(18, 43, 28, 50, 16, 32, 13, 35, 38, 33, 6, 7)
qqnorm()
qqline()
