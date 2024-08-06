## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Trees"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----treefit------------------------------------------------------------------
calif <- read.table("http://www.stat.cmu.edu/~cshalizi/350/hw/06/cadata.dat",header=TRUE)
require(tree)
treefit <- tree(log(MedianHouseValue) ~ Longitude+Latitude,data=calif)


## ----calif-lat-long-tree,echo=FALSE-------------------------------------------
plot(treefit)
text(treefit,cex=0.75)



## ----calif-lat-long-partition,echo=FALSE--------------------------------------
price.deciles <- quantile(calif$MedianHouseValue,0:10/10)
cut.prices <- cut(calif$MedianHouseValue,price.deciles,include.lowest=TRUE)
plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.prices],pch=20,
     xlab="Longitude",ylab="Latitude")
partition.tree(treefit,ordvars=c("Longitude","Latitude"),add=TRUE)



## -----------------------------------------------------------------------------
summary(treefit)


## -----------------------------------------------------------------------------
treefit2 <- tree(log(MedianHouseValue) ~ Longitude+Latitude,data=calif, mindev=0.001)


## ----echo=FALSE---------------------------------------------------------------
plot(treefit2)
text(treefit2, cex=0.5, digits=3)


## ----calif-treefit2-partition,echo=FALSE--------------------------------------
plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.prices],pch=20,
     xlab="Longitude",ylab="Latitude")
partition.tree(treefit2,ordvars=c("Longitude","Latitude"),add=TRUE,cex=0.3)



## -----------------------------------------------------------------------------
treefit3 <- tree(log(MedianHouseValue) ~., data=calif)


## ----calif-treefit3-tree,echo=FALSE-------------------------------------------
plot(treefit3)
text(treefit3,cex=0.5,digits=3)



## ----calif-treefit3-map,echo=FALSE--------------------------------------------
cut.predictions <- cut(predict(treefit3),log(price.deciles),include.lowest=TRUE)
plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.predictions],pch=20,
     xlab="Longitude",ylab="Latitude")











## ----treefit2-cv,echo=FALSE---------------------------------------------------
treefit2.cv <- cv.tree(treefit2)
plot(treefit2.cv)



## ----calif-treefit2-pruned-tree,echo=FALSE------------------------------------
opt.trees <- which(treefit2.cv$dev == min(treefit2.cv$dev))
best.leaves <- min(treefit2.cv$size[opt.trees])
treefit2.pruned <- prune.tree(treefit2,best=best.leaves)
plot(treefit2.pruned)
text(treefit2.pruned,cex=0.75)



## ----calif-treefit2-pruned-map,echo=FALSE-------------------------------------
plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.prices],pch=20,
     xlab="Longitude",ylab="Latitude")
partition.tree(treefit2.pruned,ordvars=c("Longitude","Latitude"),
               add=TRUE,cex=0.3)

