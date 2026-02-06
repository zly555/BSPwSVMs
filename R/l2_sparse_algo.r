#' Function to perform L2 wSVM
#'
#' @param x.train traing x
#' @param y.train traing y
#' @param x.tune tuning x
#' @param y.tune tuning y
#' @param x.test testing x
#' @param y.test testing y
#' @param kernel Kernel params list. default: 'linear' in sparse learning
#' @param small.tune size of lambda to tune
#' @param eps Small threshold
#'
#' @return Result matrix for downstream analysis
#' @export

L2wSVM <- function(x.train, y.train, x.tune, y.tune, x.test, y.test, kernel = list(type = 'linear', param1 = NULL, param2 = NULL), small.tune = TRUE, eps = 1e-10){

  ## tuning with 60 lambdas
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
      prob_estimate_svm <- wsvm.prob (x.train, y.train, x.tune, y.tune, svm.type = 'wl2svm', lambda = lambda[i])
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
    prob_estimate_svm <- wsvm.prob (x.train, y.train, x.test, y.test, svm.type = 'wl2svm', lambda = best.lambda.linear)
    est_prob <- prob_estimate_svm$estimate_prob
    egkl <- wsvm.egkl(y.test, est_prob)
    testerr <- prob_estimate_svm$test.error

    result <- list(best.lambda = best.lambda.linear, egkl = egkl, TestErr =testerr, Error_Table=Error_Linear)
  }


  ## tuning with 10 lambdas
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
      prob_estimate_svm <- wsvm.prob (x.train, y.train, x.tune, y.tune, svm.type = 'wl2svm', lambda = lambda[i])
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
    prob_estimate_svm <- wsvm.prob (x.train, y.train, x.test, y.test, svm.type = 'wl2svm', lambda = best.lambda.linear)
    est_prob <- prob_estimate_svm$estimate_prob
    egkl <- wsvm.egkl(y.test, est_prob)
    testerr <- prob_estimate_svm$test.error

    result <- list(best.lambda = best.lambda.linear, egkl = egkl, TestErr =testerr, Error_Table=Error_Linear)
  }



  return (result)

}
