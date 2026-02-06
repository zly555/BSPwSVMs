#' Function to perform L1 weighted SVM using primal approach
#'
#' @param x.train Training data in matrix form
#' @param y.train Label as vector
#' @param PI Weight on negative class data
#' @param kernel Kernel to choose, refer to function wsvm.kernel, default is linear kernel
#' @param lambda Regulatory parameter for L1-norm penalty and for solving weighted SVMs, lambda > 0
#' @param beta_precision minimum value of beta before set to 0
#'
#' @return L1-norm regulated wSVM model object use primal optimization
#' @export

wl1p.svm<- function(x.train, y.train, PI = 0.5, kernel = list(type = 'linear', param1 = NULL, param2 = NULL), lambda = 0.01, beta_precision = 1e-10){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)

  # declare preliminary quantities
  eps <- 1e-12
  n.data <- nrow(x.train)
  dim.data <- ncol(x.train)
  p2 <- 2*dim.data
  In <- diag(n.data)
  Y <- I * y.train
  e <- rep(1, n.data)

  # generated L(y) based on y
  L <- y.train
  L[L==1] <- 1-PI
  L[L==-1] <- PI

  # Matrix in LP contstraint
  cmat <- Y%*%x.train
  c1 <- n.data*lambda

  nvar <- n.data+p2+1
  a1 <- nvar-1

  c <- c(L,c1*rep(1,p2),0)

  # W matrix in contraints

  W <- cbind(In, cmat, -cmat, Y%*%e)
  colnames(W)<-NULL

  # number of total constraints
  ncons <- 2*n.data+p2

  # coef of objective
  objf <- c

  ## Solve LP by lpSolveAPI

  # Set number of constraints and number of decision variables
  lprec <- make.lp(nrow = ncons, ncol = nvar)

  # Set the type of problem we are trying to solve
  lp.control(lprec, sense="min")

  # Set type of decision variables as real number
  set.type(lprec, 1:nvar, type=c("real"))

  # Set objective function coefficients vector C
  set.objfn(lprec, objf)

  # define the LP problem
  for (i in 1 : ncons) {
    if (i <= a1){
      set.row(lprec, row=i, xt=c(1), indices=c(i))
      set.constr.type(lprec, ">=", constraints=i)
      set.rhs(lprec, b = 0, constraints=i)
    }

    else {
      set.row(lprec, row=i, xt=W[i-a1,])
      set.constr.type(lprec, ">=", constraints=i)
      set.rhs(lprec, b = 1, constraints=i)
    }

  }

  # find alpha by LP
  solve(lprec)

  # Get the decison variables values
  x_optimal <- get.variables(lprec)


  u <- x_optimal[(n.data+1):(n.data+dim.data)]
  v <- x_optimal[(n.data+dim.data+1):a1]

  beta <- as.vector(u-v)
  beta0 <- x_optimal[nvar]

  # return the index vector of the 1 to p features will be kept
  beta_index <- 1:dim.data
  f_index <- beta_index[abs(beta) > beta_precision]

  # set beta below precision to 0
  beta[-f_index] <- 0

  # updated training data with selected features
  Xp.train <- as.matrix(x.train[,f_index])


  # prepare output
  wl1p.svm.model <- list(beta = beta, beta0 = beta0, Xp.train = Xp.train, f_ind = f_index, kernel = kernel, lambda = lambda)

  return(wl1p.svm.model)
}
