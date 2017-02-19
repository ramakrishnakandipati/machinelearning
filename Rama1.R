
###############################################################################
############ K-Means algorithm using IRIS Data Set ############################
###############################################################################
##str(iris)
###  IRIS data set which removed labels colums
str(iris1)
summary(iris1)


## Build Model and verify
iris_clusters=kmeans(iris1,3)
iris_clusters$size

#Plot iris data set 
plot(iris1$Petal.Width,iris1$Petal.Length,col=iris_clusters$cluster)

#### Print table redicted vs original
table(iris$Species,iris_clusters$cluster)

plot(iris1[c("Petal.Width","Petal.Length")],col=iris_clusters$cluster)
plot(iris1[c("Sepal.Length","Sepal.Width")],col=iris_clusters$cluster)
## plot the points
points(iris_clusters$centers[ ,c("Sepal.Length","Sepal.Width")],col=1:3,pch=8,cex=2)


###########  New one 

getwd()
setwd("C:/Rama/Rama_Personal/DataScience/Rama_R_Scripts")

EnergyDS=read.csv("SampleDatasets/ENB2012_data.csv",header = T)
str(EnergyDS)
summary(EnergyDS)
EnergyDS$X=NULL
EnergyDS$X.1=NULL

##3 remove NA rows from Existing data set
EnergyDS1=na.omit(EnergyDS)
str(EnergyDS1)
summary(EnergyDS1)

EnergyDS_Cluster=kmeans(EnergyDS1,4)
EnergyDS_Cluster

plot(EnergyDS1$X7,EnergyDS1$Y2,col=EnergyDS_Cluster$cluster)

## ggplot examples------
qplot(EnergyDS1$Y2)
qplot(EnergyDS1$Y2,col="red")
qplot(EnergyDS1$Y2,fill="red")
qplot(EnergyDS1$X7,EnergyDS1$Y2)
qplot(EnergyDS1$X7,EnergyDS1$Y2,col=EnergyDS_Cluster$cluster)
#### qplot(EnergyDS1$X7,EnergyDS1$Y2,facets =EnergyDS1$X7~EnergyDS1$Y2)   -----  Not working, need exact sample

str(iris)
summary(iris)
qplot(iris$Petal.Length,iris$Sepal.Width)
qplot(iris$Petal.Length,iris$Sepal.Width,col=iris$Species)
?qplot()
