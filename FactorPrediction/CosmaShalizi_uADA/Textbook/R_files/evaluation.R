## ----include=FALSE------------------------------------------------------------
##### _Advanced Data Analysis from an Elementary Point of View_ #####
# R code for the chapter "Model Evaluation"
# Please do not re-distribute or use without attribution
# http://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/




## ----error-surface, echo=FALSE------------------------------------------------
# Figure: compare in-sample and generalization error for a simple model

# Example: linear regression through the origin, with Gaussian noise
# fix n, true slope, generate data, with X uniform on [0,1]
n<-20; theta<-5
x<-runif(n); y<-x*theta+rnorm(n)
# empirial risk = in-sample mean-squred error
empirical.risk <- function(b) { mean((y-b*x)^2) }
# generalization error of a line through the origin with slope b
  # EXERCISE: derive this formula
true.risk <- function(b) { 1 + (theta-b)^2*(0.5^2+1/12) }
# Plot the in-sample risk
curve(Vectorize(empirical.risk)(x),from=0,to=2*theta,
  xlab="regression slope",ylab="MSE risk")
  # R trickery: the empirical.risk() function, as written, does not behave well
  # when given a vector of slopes, and curve() wants its first argument to be a
  # function which can take a vector.  Vectorize() turns its argument into a
  # function which can take a vector; writing the expression
    # Vectorize(empirical.risk)(x)
  # rather than just
    # Vectorize(empirical.risk)
  # helps curve() figure out where to pass its vector of points.
curve(true.risk,add=TRUE,col="grey")
  # by contrast the true.risk() function works nicely with vectors.



## ----x-and-quadratic-y, echo=FALSE--------------------------------------------
# Create training data for a running example and plot it with the true
# curve

# Examples of training data
# 20 standard-Gaussian X's
x = rnorm(20)
# Quadratic Y's
y = 7*x^2 - 0.5*x + rnorm(20)

# Initial plot of training data plus true regression curve
plot(x,y)
curve(7*x^2-0.5*x,col="grey",add=TRUE)



## ----x-and-quad-y-with-poly-fits-training, echo=FALSE-------------------------
plot(x,y)
poly.formulae <- c("y~1", paste("y ~ poly(x,", 1:9, ")", sep=""))
poly.formulae <- sapply(poly.formulae, as.formula)
df.plot <- data.frame(x=seq(min(x),max(x),length.out=200))
fitted.models <- list(length=length(poly.formulae))
for (model_index in 1:length(poly.formulae)) {
  fm <- lm(formula=poly.formulae[[model_index]])
  lines(df.plot$x, predict(fm,newdata=df.plot),lty=model_index)
  fitted.models[[model_index]] <- fm
}



## ----mse-in-sample-quadratic-log-scale, echo=FALSE----------------------------
mse.q <- sapply(fitted.models, function(mdl) { mean(residuals(mdl)^2) })
plot(0:9,mse.q,type="b",xlab="polynomial degree",ylab="mean squared error",
     log="y")



## ----mses-in-and-out-of-sample-log-scale, echo=FALSE--------------------------
x.new = rnorm(2e4); y.new = 7*x.new^2 - 0.5*x.new + rnorm(2e4)
gmse <- function(mdl) { mean((y.new - predict(mdl, data.frame(x=x.new)))^2) }
gmse.q <- sapply(fitted.models, gmse)
plot(0:9,mse.q,type="b",xlab="polynomial degree",
     ylab="mean squared error",log="y",ylim=c(min(mse.q),max(gmse.q)))
lines(0:9,gmse.q,lty=2,col="blue")
points(0:9,gmse.q,pch=24,col="blue")



## ----rsq-as-epic-fail, echo=FALSE---------------------------------------------
# Extract R^2 and adjusted R^2 from a fitted model
  # Inputs: the model object
  # Output: vector of length 2, R^2 first
extract.rsqd <- function(mdl) {
  c( summary(mdl)$r.squared, summary(mdl)$adj.r.squared)
}
# Create array of R^2 and adjusted R^2 for our polynomials
rsqd.q <- sapply(fitted.models, extract.rsqd)
# Plot the R^2
plot(0:9,rsqd.q[1,],type="b",xlab="polynomial degree",ylab=expression(R^2),
  ylim=c(0,1))
# Plot the adjusted R^2
lines(0:9,rsqd.q[2,],type="b",lty="dashed")
# Add a legend
legend("bottomright",legend=c(expression(R^2),expression(R[adj]^2)),
  lty=c("solid","dashed"))



## ----data-splitting-example---------------------------------------------------
# Load the data, get rid of incomplete rows
CAPA <- na.omit(read.csv("http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/01/calif_penn_2011.csv"))

# Divide the data randomly into two (nearly) equal halves
half_A <- sample(1:nrow(CAPA),size=nrow(CAPA)/2,replace=FALSE)
half_B <- setdiff(1:nrow(CAPA),half_A)
# Write out the formulas for our two linear model specifications just once
small_formula = "Median_house_value ~ Median_household_income"
large_formula = "Median_house_value ~ Median_household_income + Median_rooms"
small_formula <- as.formula(small_formula)
large_formula <- as.formula(large_formula)
# Fit each model specification to each half of the data
msmall <- lm(small_formula,data=CAPA,subset=half_A)
mlarge <- lm(large_formula,data=CAPA,subset=half_A)
  # EXERCISE: Extract the coefficients for all the models
# Calculating the in-sample MSE is a repeated task, so write a function for it
in.sample.mse <- function(model) { mean(residuals(model)^2) }
# Calculating the MSE of a model on new data also deserves a function
new.sample.mse <- function(model,half) {
   test <- CAPA[half,]
   predictions <- predict(model,newdata=test)
   return(mean((test$Median_house_value - predictions)^2))
}
  # EXERCISE: is in.sample.mse(msmall) == new.sample.mse(msmall,half_A) ?
  # EXERCISE: should they be equal?


## ----data-splitting-example-tables, echo=FALSE--------------------------------
display.columns <- c("Median_house_value", "Median_household_income",
                     "Median_rooms")
display.rows <- c(1:4,(nrow(CAPA)-1):nrow(CAPA))
kable(CAPA[display.rows, display.columns])
kable(CAPA[intersect(display.rows, half_A), display.columns])
kable(CAPA[intersect(display.rows, half_B), display.columns])


## ----kfold-cv-for-linear-models-----------------------------------------------
# General function to do k-fold CV for a bunch of linear models
  # Inputs: dataframe to fit all models on,
    # list or vector of model formulae,
    # number of folds of cross-validation
  # Output: vector of cross-validated MSEs for the models
cv.lm <- function(data, formulae, nfolds=5) {
  # Strip data of NA rows
    # ATTN: Better to check whether NAs are in variables used by the models
  data <- na.omit(data)
  # Make sure the formulae have type "formula"
  formulae <- sapply(formulae, as.formula)
  n <- nrow(data)
  # Assign each data point to a fold, at random
    # see ?sample for the effect of sample(x) on a vector x
  fold.labels <- sample(rep(1:nfolds, length.out=n))
  mses <- matrix(NA, nrow=nfolds, ncol=length(formulae))
  colnames <- as.character(formulae)
  # EXERCISE: Replace the double for() loop below by defining a new
  # function and then calling outer()
  for (fold in 1:nfolds) {
    test.rows <- which(fold.labels == fold)
    train <- data[-test.rows,]
    test <- data[test.rows,]
    for (form in 1:length(formulae)) {
       # Fit the model on the training data
       current.model <- lm(formula=formulae[[form]], data=train)
       # Generate predictions on the testing data
       predictions <- predict(current.model, newdata=test)
       # Get the responses on the testing data, using the formula and eval()
       # a formula is, internally, a list with attributes.  The first element of
        # the list is always "~", and then the second element of the list is
        # the response term, including the transformation
       # eval() takes an expression and then evaluates in an environment, such
        # as a data frame
       test.responses <- eval(formulae[[form]][[2]], envir=test)
       # Calculate errors
       test.errors <- test.responses - predictions
       # Calculate the MSE on that fold
       mses[fold, form] <- mean(test.errors^2)
    }
  }
  return(colMeans(mses))
}


## ----cv-vs-generalization-error-for-poly-fits, echo=FALSE---------------------
# How well does cross-validation work?
  # Remember that our original data had 20 points, and we're seeing how
  # we generalize to 20,000 points from the same distribution
# Make a little data frame out of our little data
little.df <- data.frame(x=x, y=y)
# CV for the polynomials (defaults to five-fold)
cv.q <- cv.lm(little.df, poly.formulae)
# Plot the in-sample error
plot(0:9,mse.q,type="b",xlab="polynomial degree",
     ylab="mean squared error",log="y",ylim=c(min(mse.q),max(gmse.q)))
# Add the generalization error
lines(0:9,gmse.q,lty=2,col="blue",type="b",pch=2)
# Add the CV error
lines(0:9,cv.q,lty=3,col="red",type="b",pch=3)
legend("topleft",legend=c("In-sample","Generalization","CV"),
       col=c("black","blue","red"),lty=1:3,pch=1:3)


