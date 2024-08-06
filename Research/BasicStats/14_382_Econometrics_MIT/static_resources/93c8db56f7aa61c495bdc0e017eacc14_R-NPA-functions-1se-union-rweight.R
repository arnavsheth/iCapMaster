#######################################################################################;
# Empirical Example for Nonparametric Policy Analysis,                                #;    
# by A. Belloni, V. Chernozhukov, I. Fernandez-Val, and C. Hansen                     #;
#######################################################################################;
 
#######################################################################################;
# Program:  Auxiliary functions    		                                                #;
# Source of data:  SIPP 1990 (Chris)					                                        #;
#######################################################################################;

# Cross validation uses 1se rule and union of selected variables for all the components;
# Bootstrap weights follow Mamen distribution (Wild bootstrap);

# Created on Sep-24, 2013;

############# Estimates of ATE  ########################################################;

ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( mean( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x ) );
}

############# Std Errors of ATE  ########################################################;

SE.ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( sd( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x )/sqrt(length(y)) );
}

############# Estimates of ATT  ########################################################;

ATT <- function(y, d, my_d0x, md_x, md)
{
  return( mean(d * (y - my_d0x) - (1 - d) * md_x * (y - my_d0x) / (1 - md_x) ) / md );
}

############# Std. Errors of ATT  ########################################################;

SE.ATT <- function(y, d, my_d0x, md_x, md)
{
  return( sd(d * (y - my_d0x) - (1 - d) * md_x * (y - my_d0x) / (1 - md_x) ) / (md * sqrt(length(y))) );
}


############# Estimates of LATT  ########################################################;

LATT <- function(y, d, z, my_z0x, mz_x, md_z0x)
{
  return( mean( (y - my_z0x) - (1 - z) * (y - my_z0x) / (1 - mz_x) ) /  mean( (d - md_z0x) - (1 - z) * (d - md_z0x) / (1 - mz_x) ));
}

############# Std Errors of LATT  ########################################################;

SE.LATT <- function(y, d, z, my_z0x, mz_x, md_z0x)
{
  return( sd( ((y - my_z0x) - (1 - z) * (y - my_z0x) / (1 - mz_x) ) /  mean( (d - md_z0x) - (1 - z) * (d - md_z0x) / (1 - mz_x) )) / sqrt(length(y)) );
}

############# Estimates of ASF  ########################################################;

ASF <- function(y, d, my_d1x, my_d0x, md_x, d0)
{
  return( mean( d0*(d * (y - my_d1x) / md_x) -  (d0-1)*((1 - d) * (y - my_d0x) / (1 - md_x)) + d0*my_d1x - (d0-1)*my_d0x )  );
}

############# Std Errors of ASF  ########################################################;

SE.ASF <- function(y, d, my_d1x, my_d0x, md_x, d0)
{
  return( sd( (d0*(d * (y - my_d1x) / md_x) -  (d0-1)*((1 - d) * (y - my_d0x) / (1 - md_x)) + d0*my_d1x - (d0-1)*my_d0x)   ) /sqrt(length(y)) );
}

############# Estimates of ASFT  ########################################################;

ASFT <- function(y, d, my_d0x, md_x, md, d0)
{
  return( mean(d0 * d * y - (d0 - 1) * d * my_d0x - (d0 - 1)*(1 - d) * md_x * (y - my_d0x) / (1 - md_x) ) / md );
}

############# Std. Errors of ASFT  ########################################################;

SE.ASFT <- function(y, d, my_d0x, md_x, md, d0)
{
  return( sd( d0 * d * y - (d0 - 1) * d * my_d0x - (d0 - 1)*(1 - d) * md_x * (y - my_d0x) / (1 - md_x)  ) / (md * sqrt(length(y))) );
}


############# Nonparametric Components ###################################################;

COND.COMP <- function(data, form_x, form_y, lasso = TRUE)
{
  if (lasso)
{
  ################################;
  # 1 - VARIABLE SELECTION;  
  ################################;
    
  ############################;
  # my_dx = E[Y | D, X];
  ############################;
    
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- c(FALSE, !as.vector(fit.lasso$beta == 0));
  selected.logit  <- c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  fit           <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  ############################;
  # md_x = E[D | X];
  ############################;
  
  form           <- as.formula(paste("p401", "~", form_x));
  
  fit           <- lm(form, x = TRUE, y = TRUE);
  
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  
  ############################;
  # mz_x = E[Z | X];
  ############################;
  
  form           <- as.formula(paste("e401", "~", form_x));
  
  fit           <- lm(form, x = TRUE, y = TRUE);
  
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  
  ############################;
  # md_zx = E[D | Z, X];
  ############################;
  
  form           <- as.formula(paste("p401", "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  
  ############################;
  # my_zx = E[Y | Z, X];
  ############################;
  
  # Z = 1;
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
    
  
  # Z = 0;
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
  cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
  fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
  selected.lpm  <- selected.lpm | c(FALSE, !as.vector(fit.lasso$beta == 0));
  selected.logit  <- selected.logit | c(FALSE, !as.vector(fit.lasso$beta == 0));
  
  
  ###############################;
  # 2 - POST-LASSO;
  ###############################;

  ############################;
  # my_dx = E[Y | D, X];
  ############################;
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ , selected.logit]);
  my_d1x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  fit           <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  my_d0x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  ############################;
  # md_x = E[D | X];
  ############################;
  
  form           <- as.formula(paste("p401", "~", form_x));
  
  fit           <- lm(form, x = TRUE, y = TRUE);
  
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.lpm]);
  md_x.lpm      <- predict(fit.plasso);
  
  fit.plasso    <- glm(fit$y ~ fit$x[ ,selected.logit], family = binomial(link = "logit"));
  md_x.logit    <- predict(fit.plasso, type = "response");
  
  
  ############################;
  # mz_x = E[Z | X];
  ############################;
  
  form           <- as.formula(paste("e401", "~", form_x));
  
  fit           <- lm(form, x = TRUE, y = TRUE);
  
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.lpm]);
  mz_x.lpm      <- predict(fit.plasso);
  
  fit.plasso    <- glm(fit$y ~ fit$x[ ,selected.logit], family = binomial(link = "logit"));
  mz_x.logit    <- predict(fit.plasso, type = "response");
  
  
  ############################;
  # md_zx = E[D | Z, X];
  ############################;
  
  form           <- as.formula(paste("p401", "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.lpm]);
  md_z1x.lpm    <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.lpm[-1])] %*% as.vector(coef(fit.plasso));
  md_z0x.lpm    <- rep(0,n);
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- glm(fit$y ~ fit$x[ ,selected.logit], family = binomial(link = "logit"));
  md_z1x.logit  <- plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso)));
  md_z0x.logit  <- rep(0,n);
  
  
  ############################;
  # my_zx = E[Y | Z, X];
  ############################;
  
  # Z = 1;
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  my_z1x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  # myd1_zx = E[YD | Z, X];
  
  form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  myd1_z1x      <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  
  # myd0_zx = E[Y(1-D) | Z, X];
  
  form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
  
  fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  myd0_z1x      <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  #    }
  
  
  # Z = 0;
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  my_z0x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  # myd1_zx = E[YD | Z, X];
  
  myd1_z0x      <- rep(0,length(p401));
  
  # myd0_zx = E[Y(1-D) | Z, X];
  
  form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
  
  fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
  fit.plasso    <- lm(fit$y ~ fit$x[ ,selected.logit]);
  myd0_z0x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected.logit[-1])] %*% as.vector(coef(fit.plasso));
  
  }
  else
  {
  ############################;
  # my_dx = E[Y | D, X];
  ############################;
  
  
  form           <- as.formula(paste(form_y, "~", form_x));
  
  fit.d1        <- lm(form, subset = (p401 == 1));
  fit.d0        <- lm(form, subset = (p401 == 0));
  my_d1x        <- predict(fit.d1, data);
  my_d0x        <- predict(fit.d0, data);
  
  
  ############################;
  # md_x = E[D | X];
  ############################;
  
  form          <- as.formula(paste("p401", "~", form_x));
  
  fit.lpm       <- lm(form);
  fit.logit     <- glm(form, family = binomial(link = "logit"));
  md_x.lpm      <- predict(fit.lpm);
  md_x.logit    <- predict(fit.logit, type = "response");
  
  
  ############################;
  # mz_x = E[Z | X];
  ############################;
  
  form           <- as.formula(paste("e401", "~", form_x));;
  
  fit.lpm       <- lm(form);
  fit.logit     <- glm(form, family = binomial(link = "logit"));
  mz_x.lpm      <- predict(fit.lpm);
  mz_x.logit    <- predict(fit.logit, type = "response");
  
  
  ############################;
  # md_zx = E[D | Z, X];
  ############################;
  
  form           <- as.formula(paste("p401", "~", form_x));
  
  fit.lpm.z1    <- lm(form, subset = (e401 == 1));
  fit.lpm.z0    <- lm(form, subset = (e401 == 0));
  fit.logit.z1  <- glm(form, family = binomial(link = "logit"), subset = (e401 == 1));
  fit.logit.z0  <- glm(form, family = binomial(link = "logit"), subset = (e401 == 0));
  md_z1x.lpm    <- predict(fit.lpm.z1, data);
  md_z0x.lpm    <- predict(fit.lpm.z0, data);
  md_z1x.logit  <- predict(fit.logit.z1, data, type = "response");
  md_z0x.logit  <- rep(0,n);
  
  
  ############################;
  # my_zx = E[Y | Z, X];
  ############################;
  
  form          <- as.formula(paste(form_y, "~", form_x));
  
  fit.z1        <- lm(form, subset = (e401 == 1));
  fit.z0        <- lm(form, subset = (e401 == 0));
  my_z1x        <- predict(fit.z1, data);
  my_z0x        <- predict(fit.z0, data);
  
  ############################;
  # myd1_zx = E[YD | Z, X];
  ############################;
  
  form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
  
  fit.z1        <- lm(form, subset = (e401 == 1));
  fit.z0        <- lm(form, subset = (e401 == 0));
  myd1_z1x      <- predict(fit.z1, data);
  myd1_z0x      <- predict(fit.z0, data);
  
  ############################;
  # myd0_zx = E[Y(1-D) | Z, X];
  ############################;
  
  form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
  
  fit.z1        <- lm(form, subset = (e401 == 1));
  fit.z0        <- lm(form, subset = (e401 == 0));
  myd0_z1x      <- predict(fit.z1, data);
  myd0_z0x      <- predict(fit.z0, data);

  
  selected.lpm    <- NULL;
  selected.logit  <- NULL;
  
  }
  
  ############################;
  # md = E[D];
  ############################;
  
  md   <- mean(p401);
  
  ############################;
  # mz = E[Z];
  ############################;
  
  mz   <- mean(e401);
  
  return(list(my_d1x = my_d1x, my_d0x = my_d0x, md_x.lpm = md_x.lpm, md_x.logit = md_x.logit, mz_x.lpm = mz_x.lpm, mz_x.logit = mz_x.logit, 
              md_z1x.lpm = md_z1x.lpm, md_z1x.logit = md_z1x.logit, md_z0x.lpm = md_z0x.lpm, md_z0x.logit = md_z0x.logit, my_z1x = my_z1x, my_z0x = my_z0x, 
              myd1_z1x = myd1_z1x, myd1_z0x = myd1_z0x, myd0_z1x = myd0_z1x, myd0_z0x = myd0_z0x, md = md, mz = mz, 
              selected.lpm = c(TRUE, selected.lpm[-1]), selected.logit = c(TRUE, selected.logit[-1]) ));
  
}

############# Nonparametric Components by linear distribution regression  ###################################################;

COND.COMP.LPM <- function(data, dys, form_x, form_y, lasso = TRUE)
{
  selected <- NULL;
  if (lasso)
  {
  ########################################;
  # 1 - VARIABLE SELECTION;
  ########################################;  
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- c(FALSE, !as.vector(fit.lasso$beta == 0));    
    
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form          <- as.formula(paste("e401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit.lasso$beta == 0));    
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "gaussian", nfolds = 500);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit.lasso$beta == 0));    
    
    ############################;
    # my_dx = E[Y | D, X];
    ############################;
    
    data$dy       <-  dys[,ceiling(ncol(dys)/2)];
    
    form          <- as.formula(paste(form_y, "~", form_x));
    
    fit1          <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit1$x[ ,-1], fit1$y, family = "gaussian", nfolds = 500);
    fit1.lasso    <- glmnet(fit1$x[ ,-1], fit1$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit1.lasso$beta == 0));    
    
    fit0          <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit0$x[ ,-1], fit0$y, family = "gaussian", nfolds = 500);
    fit0.lasso    <- glmnet(fit0$x[ ,-1], fit0$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit0.lasso$beta == 0));    
    
    ############################;
    # my_zx = E[Y | Z, X];
    ############################;
    
    data$dy       <-  dys[,ceiling(ncol(dys)/2)];
    
    form          <- as.formula(paste(form_y, "~", form_x));
    
    fit1          <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit1$x[ ,-1], fit1$y, family = "gaussian", nfolds = 500);
    fit1.lasso    <- glmnet(fit1$x[ ,-1], fit1$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit1.lasso$beta == 0));    
    
    fit0          <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit0$x[ ,-1], fit0$y, family = "gaussian", nfolds = 500);
    fit0.lasso    <- glmnet(fit0$x[ ,-1], fit0$y, family = "gaussian", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit0.lasso$beta == 0));    
    
      
  ########################################;
  # 2 - POST-LASSO;
  ########################################;  
    
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
    md_x          <- predict(fit.plasso);
    
    
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form          <- as.formula(paste("e401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
    mz_x          <- predict(fit.plasso);
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    md_z1x        <- lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso));
    md_z0x        <- rep(0,n);
    
  }
  else
  {
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form      <- as.formula(paste("p401", "~", form_x));
    
    fit       <- lm(form);
    md_x      <- predict(fit);
        
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form      <- as.formula(paste("e401", "~", form_x));;
    
    fit       <- lm(form);
    mz_x      <- predict(fit);
    
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form      <- as.formula(paste("p401", "~", form_x));
    
    fit.z1    <- lm(form, subset = (e401 == 1));
    fit.z0    <- lm(form, subset = (e401 == 0));
    md_z1x    <- predict(fit.z1, data);
    md_z0x    <- predict(fit.z0, data);
    
        
  }
  
  ############################;
  # md = E[D];
  ############################;
  
  md   <- mean(p401);
  
  ############################;
  # mz = E[Z];
  ############################;
  
  mz   <- mean(e401);
  
  return(list(md_x = md_x, mz_x = mz_x,  md_z1x = md_z1x, md_z0x = md_z0x, md = md, mz = mz, selected = selected ));
  
}



############# Nonparametric Components by logit distribution regression ###################################################;

COND.COMP.LOGIT <- function(data, dys, form_x, form_y, lasso = TRUE)
{
  selected <- NULL;
  
  if (lasso)
  {
  #####################################;
  # 1 - VARIABLE SELECTION;
  #####################################;    
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form           <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
        
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- c(FALSE, !as.vector(fit.lasso$beta == 0));    
    
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form           <- as.formula(paste("e401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
      
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit.lasso$beta == 0))    
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit$x[ ,-1], fit$y, family = "binomial", nfolds = 50);
    fit.lasso     <- glmnet(fit$x[ ,-1], fit$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit.lasso$beta == 0))    
    
    ############################;
    # my_dx = E[Y | D, X];
    ############################;
    
    data$dy       <-  dys[,ceiling(ncol(dys)/2)];
    
    form          <- as.formula(paste(form_y, "~", form_x));
    
    fit1          <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit1$x[ ,-1], fit1$y, family = "binomial", nfolds = 50);
    fit1.lasso    <- glmnet(fit1$x[ ,-1], fit1$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit1.lasso$beta == 0))    
    
    fit0          <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit0$x[ ,-1], fit0$y, family = "binomial", nfolds = 50);
    fit0.lasso    <- glmnet(fit0$x[ ,-1], fit0$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit0.lasso$beta == 0))    
    
    
    ############################;
    # my_zx = E[Y | Z, X];
    ############################;
    
    data$dy       <-  dys[,ceiling(ncol(dys)/2)];
    
    form          <- as.formula(paste(form_y, "~", form_x));
    
    fit1          <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit1$x[ ,-1], fit1$y, family = "binomial", nfolds = 50);
    fit1.lasso    <- glmnet(fit1$x[ ,-1], fit1$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit1.lasso$beta == 0))    
    
    fit0          <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
    cv.lasso      <- cv.glmnet(fit0$x[ ,-1], fit0$y, family = "binomial", nfolds = 50);
    fit0.lasso    <- glmnet(fit0$x[ ,-1], fit0$y, family = "binomial", lambda = cv.lasso$lambda.1se);
    selected      <- selected | c(FALSE, !as.vector(fit0.lasso$beta == 0))    
    
    
 #####################################;
 # 2 - POST-LASSO;
 #####################################;    
    
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form           <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
    md_x          <- predict(fit.plasso, type = "response");
    
    
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form           <- as.formula(paste("e401", "~", form_x));
    
    fit           <- lm(form, x = TRUE, y = TRUE);
    
    fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
    mz_x          <- predict(fit.plasso, type = "response");
    
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form          <- as.formula(paste("p401", "~", form_x));
    
    fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
    fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
    md_z1x        <- plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)));
    md_z0x        <- rep(0,n);
    
    
  }
  else
  {
    
    ############################;
    # md_x = E[D | X];
    ############################;
    
    form    <- as.formula(paste("p401", "~", form_x));
    
    fit     <- glm(form, family = binomial(link = "logit"));
    md_x    <- predict(fit, type = "response");
    
    
    ############################;
    # mz_x = E[Z | X];
    ############################;
    
    form    <- as.formula(paste("e401", "~", form_x));;
    
    fit     <- glm(form, family = binomial(link = "logit"));
    mz_x    <- predict(fit, type = "response");
    
    
    ############################;
    # md_zx = E[D | Z, X];
    ############################;
    
    form    <- as.formula(paste("p401", "~", form_x));
    
    fit.z1  <- glm(form, family = binomial(link = "logit"), subset = (e401 == 1));
    fit.z0  <- glm(form, family = binomial(link = "logit"), subset = (e401 == 0));
    md_z1x  <- predict(fit.z1, data, type = "response");
    md_z0x  <- rep(0,n);
    
  }
  
  ############################;
  # md = E[D];
  ############################;
  
  md   <- mean(p401);
  
  ############################;
  # mz = E[Z];
  ############################;
  
  mz   <- mean(e401);
  
  return(list(md_x = md_x, mz_x = mz_x, md_z1x = md_z1x, md_z0x = md_z0x, md = md, mz = mz, selected = selected ));
  
}


############# Conditional CDFs by linear distribution regression  ###################################################;

COND.CDF.LPM <- function(data, dys, form_x, form_y, lasso = TRUE, selected)
{
  if (lasso)
  {
    ############################;
    # my_dx = E[Y | D, X];
    ############################;
    
    form          <- as.formula(paste(form_y, "~", form_x));

    my_d1x        <- NULL;
    my_d0x        <- NULL;
    
    for (i in 1:ncol(dys))
    {   
      data$dy       <- dys[ ,i];
      
      fit           <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      my_d1x        <- cbind( my_d1x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
      
      fit           <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      my_d0x        <- cbind( my_d0x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
    }   
    
    ############################;
    # my_zx = E[Y | Z, X];
    ############################;
    
    my_z1x        <- NULL;
    my_z0x        <- NULL;
    myd1_z1x      <- NULL;
    myd0_z1x      <- NULL;
    myd1_z0x      <- NULL;
    myd0_z0x      <- NULL;
    
    for (i in 1:ncol(dys))
    {   
      data$dy       <- dys[ ,i];
      
      # Z = 1;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      my_z1x        <- cbind( my_z1x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
      
      # myd1_zx = E[YD | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      myd1_z1x      <- cbind( myd1_z1x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
      
      
      # myd0_zx = E[Y(1-D) | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      myd0_z1x      <- cbind( myd0_z1x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );

      # Z = 0;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      my_z0x        <- cbind( my_z0x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
      
      # myd1_zx = E[YD | Z, X];
      
      myd1_z0x      <- cbind( myd1_z0x, rep(0,length(p401)) );
      
      # myd0_zx = E[Y(1-D) | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- lm(fit$y ~ fit$x[ ,selected]);
      myd0_z0x      <- cbind( myd0_z0x, lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso)) );
      
    }   

    
  }
  else
  {
    my_d1x        <- NULL;
    my_d0x        <- NULL;
    my_z1x        <- NULL;
    my_z0x        <- NULL;
    myd1_z1x      <- NULL;
    myd0_z1x      <- NULL;
    myd1_z0x      <- NULL;
    myd0_z0x      <- NULL;
    
    for (i in 1:ncol(dys))
      {
      data$dy       <- dys[ ,i];
      
      ############################;
      # my_dx = E[Y | D, X];
      ############################;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      
      fit.d1        <- lm(form, subset = (p401 == 1));
      fit.d0        <- lm(form, subset = (p401 == 0));
      my_d1x        <- cbind( my_d1x, predict(fit.d1, data) );
      my_d0x        <- cbind( my_d0x, predict(fit.d0, data) );
      
      ############################;
      # my_zx = E[Y | Z, X];
      ############################;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      
      fit.z1        <- lm(form, subset = (e401 == 1));
      fit.z0        <- lm(form, subset = (e401 == 0));
      my_z1x        <- cbind( my_z1x, predict(fit.z1, data) );
      my_z0x        <- cbind( my_z0x, predict(fit.z0, data) );
      
      ############################;
      # myd1_zx = E[YD | Z, X];
      ############################;
      
      form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
      
      fit.z1        <- lm(form, subset = (e401 == 1));
      fit.z0        <- lm(form, subset = (e401 == 0));
      myd1_z1x      <- cbind( myd1_z1x, predict(fit.z1, data) );
      myd1_z0x      <- cbind( myd1_z0x, predict(fit.z0, data) );
      
      ############################;
      # myd0_zx = E[Y(1-D) | Z, X];
      ############################;
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      
      fit.z1        <- lm(form, subset = (e401 == 1));
      fit.z0        <- lm(form, subset = (e401 == 0));
      myd0_z1x      <- cbind( myd0_z1x, predict(fit.z1, data) );
      myd0_z0x      <- cbind( myd0_z0x, predict(fit.z0, data) );
    }
  }

  return(list(my_d1x = my_d1x, my_d0x = my_d0x, my_z1x = my_z1x, my_z0x = my_z0x, myd1_z1x = myd1_z1x, myd1_z0x = myd1_z0x, myd0_z1x = myd0_z1x, myd0_z0x = myd0_z0x ));
  
}

############# Conditional distributions by logit distribution regression ###################################################;

COND.CDF.LOGIT <- function(data, dys, form_x, form_y, lasso = TRUE, selected)
{
  if (lasso)
  {
 
    ############################;
    # my_dx = E[Y | D, X];
    ############################;
    
    form          <- as.formula(paste(form_y, "~", form_x));

    my_d1x        <- NULL;
    my_d0x        <- NULL;
    
    for (i in 1:ncol(dys))
    {   
      data$dy       <- dys[ ,i];
      
      fit           <- lm(form, subset = (p401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      my_d1x        <- cbind( my_d1x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
      fit           <- lm(form, subset = (p401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      my_d0x        <- cbind( my_d0x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
    }   
    
    ############################;
    # my_zx = E[Y | Z, X];
    ############################;
    
    my_z1x        <- NULL;
    my_z0x        <- NULL;
    myd1_z1x      <- NULL;
    myd0_z1x      <- NULL;
    myd1_z0x      <- NULL;
    myd0_z0x      <- NULL;
    
    for (i in 1:ncol(dys))
    {   
      data$dy       <- dys[ ,i];
      
      # Z = 1;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      my_z1x        <- cbind( my_z1x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
      # myd1_zx = E[YD | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      myd1_z1x      <- cbind( myd1_z1x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
      
      # myd0_zx = E[Y(1-D) | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      fit           <- lm(form, subset = (e401 == 1), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      myd0_z1x      <- cbind( myd0_z1x, plogis( lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
      # Z = 0;
      
      form          <- as.formula(paste(form_y, "~", form_x));
      fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      my_z0x        <- cbind( my_z0x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
      # myd1_zx = E[YD | Z, X];
      
      myd1_z0x      <- cbind( myd1_z0x, rep(0,length(p401)) );
      
      # myd0_zx = E[Y(1-D) | Z, X];
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      fit           <- lm(form, subset = (e401 == 0), x = TRUE, y = TRUE);
      fit.plasso    <- glm(fit$y ~ fit$x[ ,selected], family = binomial(link = "logit"));
      myd0_z0x      <- cbind( myd0_z0x, plogis(lm(form, x = TRUE)$x[ ,c(TRUE, selected[-1])] %*% as.vector(coef(fit.plasso))) );
      
    }   
    
  }
  else
  {
    my_d1x        <- NULL;
    my_d0x        <- NULL;
    my_z1x        <- NULL;
    my_z0x        <- NULL;
    myd1_z1x      <- NULL;
    myd0_z1x      <- NULL;
    myd1_z0x      <- NULL;
    myd0_z0x      <- NULL;
    
    for (i in 1:ncol(dys))
    {
      data$dy       <- dys[ ,i];
      
      ############################;
      # my_dx = E[Y | D, X];
      ############################;
      
      
      form          <- as.formula(paste(form_y, "~", form_x));
      
      fit.d1        <- glm(form, subset = (p401 == 1), family = binomial(link = "logit"));
      fit.d0        <- glm(form, subset = (p401 == 0), family = binomial(link = "logit"));
      my_d1x        <- cbind(my_d1x, predict(fit.d1, data, type = "response") );
      my_d0x        <- cbind(my_d0x, predict(fit.d0, data, type = "response") );
      
      
      ############################;
      # my_dx = E[Y | Z, X];
      ############################;
      
      
      form          <- as.formula(paste(form_y, "~", form_x));
      
      fit.z1        <- glm(form, subset = (e401 == 1), family = binomial(link = "logit"));
      fit.z0        <- glm(form, subset = (e401 == 0), family = binomial(link = "logit"));
      my_z1x        <- cbind(my_z1x, predict(fit.z1, data, type = "response") );
      my_z0x        <- cbind(my_z0x, predict(fit.z0, data, type = "response") );
      
      ############################;
      # myd1_zx = E[YD | Z, X];
      ############################;
      
      form          <- as.formula(paste("I(",form_y, " * p401) ~", form_x));
      
      fit.z1        <- glm(form, subset = (e401 == 1), family = binomial(link = "logit"));
      fit.z0        <- glm(form, subset = (e401 == 0), family = binomial(link = "logit"));
      myd1_z1x      <- cbind(myd1_z1x, predict(fit.z1, data, type = "response") );
      myd1_z0x      <- cbind(myd1_z0x, predict(fit.z0, data, type = "response") );
      
      ############################;
      # myd0_zx = E[Y(1-D) | Z, X];
      ############################;
      
      form          <- as.formula(paste("I(",form_y, " * (1 - p401)) ~", form_x));
      
      fit.z1        <- glm(form, subset = (e401 == 1), family = binomial(link = "logit"));
      fit.z0        <- glm(form, subset = (e401 == 0), family = binomial(link = "logit"));
      myd0_z1x      <- cbind(myd0_z1x, predict(fit.z1, data, type = "response") );
      myd0_z0x      <- cbind(myd0_z0x, predict(fit.z0, data, type = "response") );
    }
      
  }
  
  
  return(list(my_d1x = my_d1x, my_d0x = my_d0x, my_z1x = my_z1x, my_z0x = my_z0x, myd1_z1x = myd1_z1x, myd1_z0x = myd1_z0x, myd0_z1x = myd0_z1x, myd0_z0x = myd0_z0x ));
  
}

############# Weighted estimates of QSF for bootstrap ########################################################;

QSF <- function(y, d, my_d1x, my_d0x, md_x, d0, weights)
{
  return( weighted.mean( d0*(d * (y - my_d1x) / md_x) -  (d0-1)*((1 - d) * (y - my_d0x) / (1 - md_x)) + d0*my_d1x - (d0-1)*my_d0x,  weights) );
}


############# Weighted estimates of QSFT  for bootstrap ########################################################;

QSFT <- function(y, d, my_d0x, md_x, d0, weights)
{
  return( weighted.mean(d0 * d * y - (d0 - 1) * d * my_d0x - (d0 - 1)*(1 - d) * md_x * (y - my_d0x) / (1 - md_x), weights) / weighted.mean(d, weights) );
}




############## Obtains the joint distribution of quantile effects by bootstrap ##############################################

QE.bootstrap <- function(data=data, n = n, dys = dys, taus = taus, ys = ys, B=B, cond.comp, cond.cdf, qte, qtt, dsf1.0, dsf0.0, dsft1.0, dsft0.0, alpha = .05, rang)
    {

    qtes     <- NULL;
    qtts     <- NULL;
    dsf1s     <- NULL;
    dsf0s     <- NULL;
    dsft1s     <- NULL;
    dsft0s     <- NULL;    
    
    
    for (b in 1:B)
    {
      dsf1   <- NULL;
      dsft1  <- NULL;
      
      dsf0   <- NULL;
      dsft0  <- NULL;
      
      weights   <- 1 + rnorm(n)/sqrt(2) + (rnorm(n)^2 - 1)/2;
      weights   <- weights/sum(weights);

      for (i in 1:ncol(dys))
      {
        
        data$dy   <-  dys[,i];

        dsf1      <- c(dsf1, QSF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1, weights) );
        dsft1     <- c(dsft1, QSFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 1, weights) );
        
        dsf0      <- c(dsf0, QSF(data$dy, e401, cond.cdf$my_z1x[,i], cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0, weights) );
        dsft0     <- c(dsft0, QSFT(data$dy, e401, cond.cdf$my_z0x[,i], cond.comp$mz_x, d0 = 0, weights) );
        
      }
      qsf1    <- approxfun(dsf1, ys, yleft = -Inf, yright = Inf);
      qsft1   <- approxfun(dsft1, ys, yleft = -Inf, yright = Inf);
      
      qsf0    <- approxfun(dsf0, ys, yleft = -Inf, yright = Inf);
      qsft0   <- approxfun(dsft0, ys, yleft = -Inf, yright = Inf);
      
      qtes     <- cbind(qtes, qsf1(taus)   - qsf0(taus));
      qtts     <- cbind(qtts, qsft1(taus)  - qsft0(taus));
            
      dsf1s   <- cbind(dsf1s, dsf1);
      dsf0s   <- cbind(dsf0s, dsf0);     
      dsft1s   <- cbind(dsft1s, dsft1);
      dsft0s   <- cbind(dsft0s, dsft0);
      
      
    }

    V.qte     <- apply(qtes, 1, var, na.rm = T);
    V.qtt     <- apply(qtts, 1, var, na.rm = T);
    V.dsf1     <- apply(dsf1s, 1, var, na.rm = T);
    V.dsf0     <- apply(dsf0s, 1, var, na.rm = T);
    V.dsft1     <- apply(dsft1s, 1, var, na.rm = T);
    V.dsft0     <- apply(dsft0s, 1, var, na.rm = T);
    
    
    t.qte 		<- sqrt((qtes -  matrix(qte, length(taus), B, byrow = F))^2 /  matrix(V.qte, length(taus), B, byrow = F));
    t.qtt   	<- sqrt((qtts -  matrix(qtt, length(taus), B, byrow = F))^2 /  matrix(V.qtt, length(taus), B, byrow = F));
    t.dsf1   	<- sqrt((dsf1s -  matrix(dsf1.0, length(taus), B, byrow = F))^2 /  matrix(V.dsf1, length(taus), B, byrow = F));
    t.dsf0   	<- sqrt((dsf0s -  matrix(dsf0.0, length(taus), B, byrow = F))^2 /  matrix(V.dsf0, length(taus), B, byrow = F));
    t.dsft1     <- sqrt((dsft1s -  matrix(dsft1.0, length(taus), B, byrow = F))^2 /  matrix(V.dsft1, length(taus), B, byrow = F));
    t.dsft0   	<- sqrt((dsft0s -  matrix(dsft0.0, length(taus), B, byrow = F))^2 /  matrix(V.dsft0, length(taus), B, byrow = F));
    
    
    tmax.qte  	 <- apply(t.qte[rang, ],2, FUN=max, na.rm = T);
    tmax.qtt     <- apply(t.qtt[rang, ],2, FUN=max, na.rm = T);
    tmax.dsf1    <- apply(t.dsf1, 2, FUN=max, na.rm = T);
    tmax.dsf0    <- apply(t.dsf0, 2, FUN=max, na.rm = T);
    tmax.dsft1    <- apply(t.dsft1, 2, FUN=max, na.rm = T);
    tmax.dsft0    <- apply(t.dsft0, 2, FUN=max, na.rm = T);
    
    
    talpha.qte 	    <- quantile(tmax.qte, 1-alpha);
    talpha.qtt      <- quantile(tmax.qtt, 1-alpha);
    talpha.dsf1     <- quantile(tmax.dsf1, 1-alpha);
    talpha.dsf0     <- quantile(tmax.dsf0, 1-alpha);
    talpha.dsft1     <- quantile(tmax.dsft1, 1-alpha);
    talpha.dsft0     <- quantile(tmax.dsft0, 1-alpha);
    
    
    se.qte   	<- sqrt(V.qte);
    se.qtt    <- sqrt(V.qtt);
    se.dsf1   <- sqrt(V.dsf1);
    se.dsf0   <- sqrt(V.dsf0);
    se.dsft1   <- sqrt(V.dsft1);
    se.dsft0   <- sqrt(V.dsft0);
    
    
    
    return(list(talpha.qte = talpha.qte, se.qte = se.qte, talpha.qtt = talpha.qtt, se.qtt = se.qtt, talpha.dsf1 = talpha.dsf1, se.dsf1 = se.dsf1, talpha.dsf0 = talpha.dsf0, se.dsf0 = se.dsf0, 
                talpha.dsft1 = talpha.dsft1, se.dsft1 = se.dsft1, talpha.dsft0 = talpha.dsft0, se.dsft0 = se.dsft0));
    }


############# Weighted estimates of ATE for bootstrap ########################################################;

ATE.w <- function(y, d, my_d1x, my_d0x, md_x, weights)
{
  return( weighted.mean( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x, weights ) );
}

############# Weighted estimates of ATE for bootstrap ########################################################;

ATT.w <- function(y, d, my_d0x, md_x, weights)
{
  return( weighted.mean(d * (y - my_d0x) - (1 - d) * md_x * (y - my_d0x) / (1 - md_x), weights ) / weighted.mean(d, weights) );
}



# Functions to obtain step-function for distribution;
cdf<-  function(ys, Fs){;
                        ys<- sort(ys);
                        Fs<- sort(Fs);
                        F<- approxfun(ys, Fs, method="constant", f=0, rule=1:2);
                        return(F);
};


# Function to obtain step-function for left-inverse (quantile);
left.inv<- function(ys, Fs, rule) {;
                                   ys<- sort(ys);
                                   Fs<- sort(Fs);  
                                   iF<- approxfun(Fs, ys, method="constant", f=1, rule=rule);
                                   return(iF);
};

