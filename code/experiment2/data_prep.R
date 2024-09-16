################################################################################################################################
## Apr 2024
## r version 4.3.2
## RStudio 2023.12.1+402 "Ocean Storm" Release (4da58325ffcff29d157d9264087d4b1ab27f7204, 2024-01-28) for windows
##
## Claire Little, University of Manchester
## Code to subset the Census data, create a training and holdout dataset and create a synthetic dataset using synthpop
################################################################################################################################


## Load original census (UK 1991) data
data <- read.delim('data\gb91ind.tab', sep = '\t')
## 1116181 observations, 67 variables

## select subset, those who live in South Yorkshire
sub <- subset(data, (DCOUNTY ==5))
## 25636 respondents

## Now subset on age >= 50
sub1 <- subset(sub, AGE>=50)
## 8312 respondents

## Select the variables that are mentioned in the paper
sub2 <- sub1[,c(1,3,25,11,16,40,36)]  # AREAP, AGE, SEX, ETHGROUP, LTILL, TENURE, CARS

## remove those who have NA for CARS
sub3 <- subset(sub2, !is.na(CARS))
## 8054 respondents

## Modify CARS such that having no car is 0 and 1 or more is 1
sub3$CARS_alt <- sub3$CARS
sub3$CARS_alt[sub3$CARS_alt >=1] <- 1  ## 1 or more set to 1

### since ethgroup has very few in the other groups (than white), change to white and other
## ethgroup: white = 1, others =2
sub3$ETHGROUP_alt <- sub3$ETHGROUP
sub3$ETHGROUP_alt[sub3$ETHGROUP_alt != 1] <- 2

## Just keep the alternative CARS and ETHGROUP and rename
sub3 <- sub3[,c(1:3,9,5:6,8)]
names(sub3)[4] <- 'ETHGROUP'; names(sub3)[7] <- 'CARS'


## Split the data into two randomly - two datasets of size 4027
## set the seed
set.seed(123)
ind <- sample(seq_len(nrow(sub3)),replace=FALSE, size = 4027)
## call them train and test for now
train <- sub3[ind, ]
test <- sub3[-ind, ]


## save the data
write.csv(train, 'syorkshire_traindata.csv', row.names = FALSE)
write.csv(test, 'syorkshire_testdata.csv', row.names = FALSE)




################################################################################################################################
## Generate synthetic data using the original (train) dataset
## This can be compared to the synthetic data that was generated without access to the original
train <- read.csv('syorkshire_traindata.csv')

## use synthpop, version 1.8-0
library(synthpop)

## set everything but AGE as factors
cols <- names(train[,c(1,3:7)])
train[cols] <- lapply(train[cols], factor);

## create a synthetic dataset
synthds <- syn(train, seed=123)  # using defaults
summary(synthds)
write.syn(synthds,'syorkshire_synthetic_from_orig', filetype='csv')


