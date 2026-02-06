#' Function to perform L1+L2 wSVM sparse learning algorithm
#'
#' @param x.train traing x
#' @param y.train traing y
#' @param x.tune tuning x
#' @param y.tune tuning y
#' @param x.test testing x
#' @param y.test testing y
#' @param kernel Kernel params list. default: 'linear' in sparse learning
#' @param osqp.option option to use osqp as solver
#' @param small.tune tune lambda only for L1
#' @param double.tune tune lambda for both L1 and L2
#' @param beta_precision round decimal points for beta
#' @param eps Small threshold
#'
#' @return Result matrix for downstream analysis
#' @export

L1wSVM <- function(x.train, y.train, x.tune, y.tune, x.test, y.test, kernel = list(type = 'linear', param1 = NULL, param2 = NULL),
                   osqp.option = TRUE, small.tune = TRUE, double.tune = FALSE, beta_precision, eps = 1e-10){

  ## tune lambda for both L1 and L2
  if(double.tune){
    # set lambda grid
    lambda1 <- rep(0, 8)
    count <- 1
    for (jj in -4:3){
      low <- 10^jj
      up <- 10^(jj+1)
      lambda1[count] <- seq(low+ (up-low)/2, up, length.out = 1)
      count <- count+1
    }

    lambda2 <- rep(0, 8)
    count <- 1
    for (jj in -4:3){
      low <- 10^jj
      up <- 10^(jj+1)
      lambda2[count] <- seq(low+ (up-low)/2, up, length.out = 1)
      count <- count+1
    }
    # create the Error Matrix
    Error_Linear <- matrix(0, nrow = length(lambda1)*length(lambda2), ncol = 4)

    # count the row in matrix Error_Radial
    count <- 1

    # calculate the training and test error for all the tuning parameters
    for(i in 1:length(lambda1)){
      for (j in 1:length(lambda2)){

        Error_Linear[count,1] <- lambda1[i]
        Error_Linear[count,2] <- lambda2[j]

        #fit linear svm
        prob_estimate_svm <-wsvm.prob (x.train, y.train, x.tune, y.tune, svm.type = 'wl1dsvm', lambda = lambda1[i], lambda2 = lambda2[j], osqp.option = osqp.option, beta_precision = beta_precision)
        est_prob <- prob_estimate_svm$estimate_prob

        #calculate egkl
        egkl <- wsvm.egkl(y.tune, est_prob)

        Error_Linear[count,3] <- egkl

        #testing error
        Error_Linear[count,4]  = prob_estimate_svm$test.error
        count <- count+1
      }
    }

    Error_Linear[Error_Linear[,3] == 0,][,3] <- Inf
    best.lambda1.linear <- Error_Linear[which.min(Error_Linear[,3]),1]
    best.lambda2.linear <- Error_Linear[which.min(Error_Linear[,3]),2]
    best.egkl.linear <- Error_Linear[which.min(Error_Linear[,3]),3]
    best.testerr.linear <- Error_Linear[which.min(Error_Linear[,3]),4]
    # cat("Best tuning parameters lambda in linear kernel:\n", c(best.lambda1.linear,best.lambda2.linear), "\n")
    # cat("Best EGKL in linear kernel:\n",best.egkl.linear,"\n")
    # cat("Best test error in linear kernel:\n",best.testerr.linear,"\n")

    colnames(Error_Linear) <- c("lambda1","lambda2","EGKL","test_error")
    # print(Error_Linear)

    # fit the wsvm with the best lambda
    prob_estimate_svm <- wsvm.prob (x.train, y.train, x.test, y.test, svm.type = 'wl1dsvm', plot.option = 'f_importance',
                                    lambda = best.lambda1.linear, lambda2 = best.lambda2.linear, osqp.option = osqp.option, beta_precision = beta_precision)
    est_prob <- prob_estimate_svm$estimate_prob
    egkl <- wsvm.egkl(y.test, est_prob)
    testerr <- prob_estimate_svm$test.error
    f_matrix <- prob_estimate_svm$f_matrix

    result <- list(best.lambda1= best.lambda1.linear, best.lambda2= best.lambda2.linear, egkl = egkl, TestErr =testerr, beta_precision = beta_precision, Error_Table=Error_Linear, f_matrix = f_matrix)
  }

  ## tune lambda only for L1
  if(!double.tune){
    if(!small.tune){
      # set lambda grid
      lambda <- rep(0, 60)
      count <- 1
      for (j in -4:1){
        low <- 10^j
        up <- 10^(j+1)
        lambda[count:(count+9)] <- seq(low+ (up-low)/10, up, length.out = 10)
        count <- count+10
      }

      # create the Error Matrix
      Error_Linear <- matrix(0, nrow = length(lambda), ncol = 3)

      # calculate the training and test error for all the tuning parameters
      for(i in 1:length(lambda)){

        Error_Linear[i,1] <- lambda[i]

        #fit linear svm
        prob_estimate_svm <- wsvm.prob (x.train, y.train, x.tune, y.tune, svm.type = 'wl1svm', lambda = lambda[i], osqp.option = osqp.option, beta_precision = beta_precision)
        est_prob <- prob_estimate_svm$estimate_prob

        #calculate egkl
        egkl <- wsvm.egkl(y.tune, est_prob)

        Error_Linear[i,2] <- egkl

        #testing error
        Error_Linear[i,3]  = prob_estimate_svm$test.error

      }

      Error_Linear[Error_Linear[,2] == 0,][,2] <- Inf
      best.lambda.linear <- Error_Linear[which.min(Error_Linear[,2]),1]
      best.egkl.linear <- Error_Linear[which.min(Error_Linear[,2]),2]
      best.testerr.linear <- Error_Linear[which.min(Error_Linear[,2]),3]
      # cat("Best tuning parameters lambda in linear kernel:\n",best.lambda.linear,"\n")
      # cat("Best EGKL in linear kernel:\n",best.egkl.linear,"\n")
      # cat("Best test error in linear kernel:\n",best.testerr.linear,"\n")

      colnames(Error_Linear) <- c("lambda","EGKL","test_error")
      # print(Error_Linear)

      # plot(test$x1, test$x2, xlim = c(-1,1), ylim = c(-1,1), col=c("red","blue")[as.factor(test$y)],  xlab="x1", ylab ="x2")


      # fit the wsvm with the best lambda
      prob_estimate_svm <- wsvm.prob (x.train, y.train, x.test, y.test, svm.type = 'wl1svm', plot.option = 'f_importance',
                                      lambda = best.lambda.linear, osqp.option = osqp.option, beta_precision = beta_precision)
      est_prob <- prob_estimate_svm$estimate_prob
      egkl <- wsvm.egkl(y.test, est_prob)
      testerr <- prob_estimate_svm$test.error
      f_matrix <- prob_estimate_svm$f_matrix

      result <- list(best.lambda= best.lambda.linear, egkl = egkl, TestErr =testerr, beta_precision = beta_precision, Error_Table=Error_Linear, f_matrix = f_matrix)
    }


    if(small.tune){
      # set lambda grid
      lambda <- rep(0, 10)
      count <- 1
      for (j in -4:1){
        low <- 10^j
        up <- 10^(j+1)
        lambda[count:(count+1)] <- seq(low+ (up-low)/2, up, length.out = 2)
        count <- count+2
      }


      # create the Error Matrix
      Error_Linear <- matrix(0, nrow = length(lambda), ncol = 3)

      # calculate the training and test error for all the tuning parameters
      for(i in 1:length(lambda)){

        Error_Linear[i,1] <- lambda[i]

        #fit linear svm
        prob_estimate_svm <- wsvm.prob (x.train, y.train, x.tune, y.tune, svm.type = 'wl1svm', lambda = lambda[i], osqp.option = osqp.option, beta_precision = beta_precision)
        est_prob <- prob_estimate_svm$estimate_prob

        #calculate egkl
        egkl <- wsvm.egkl(y.tune, est_prob)

        Error_Linear[i,2] <- egkl

        #testing error
        Error_Linear[i,3]  = prob_estimate_svm$test.error

      }
      Error_Linear[Error_Linear[,2] == 0,][,2] <- Inf
      best.lambda.linear <- Error_Linear[which.min(Error_Linear[,2]),1]
      best.egkl.linear <- Error_Linear[which.min(Error_Linear[,2]),2]
      best.testerr.linear <- Error_Linear[which.min(Error_Linear[,2]),3]
      # cat("Best tuning parameters lambda in linear kernel:\n",best.lambda.linear,"\n")
      # cat("Best EGKL in linear kernel:\n",best.egkl.linear,"\n")
      # cat("Best test error in linear kernel:\n",best.testerr.linear,"\n")

      colnames(Error_Linear) <- c("lambda","EGKL","test_error")
      # print(Error_Linear)

      # fit the wsvm with the best lambda
      prob_estimate_svm <- wsvm.prob (x.train, y.train, x.test, y.test, svm.type = 'wl1svm', plot.option = 'f_importance',
                                      lambda = best.lambda.linear, osqp.option = osqp.option, beta_precision = beta_precision)
      est_prob <- prob_estimate_svm$estimate_prob
      egkl <- wsvm.egkl(y.test, est_prob)
      testerr <- prob_estimate_svm$test.error
      f_matrix <- prob_estimate_svm$f_matrix

      result <- list(best.lambda= best.lambda.linear, egkl = egkl, TestErr =testerr, beta_precision = beta_precision, Error_Table=Error_Linear, f_matrix = f_matrix)
    }
  }

  return (result)
}
