#' Fit an weighted svm model with given weight pi with L1-norm penalty
#'
#' @param x.train Training data in matrix form
#' @param y.train Label as vector
#' @param PI Weight on negative class data
#' @param kernel Kernel to choose, refer to function wsvm.kernel, default is linear kernel
#' @param lambda Regulatory parameter for L1-norm penalty and for solving weighted SVMs, lambda > 0
#' @param osqp.option choose to use lpSolveAPI or OSQP as solver
#' @param beta_precision round decimal points for beta
#'
#' @return L1-norm wSVM model object
#' @export

wl1svm <- function(x.train, y.train, PI = 0.5, kernel = list(type = 'linear', param1 = NULL, param2 = NULL), lambda = 0.01, osqp.option = TRUE, beta_precision){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)

  ## Solve by OSQP
  if(osqp.option) {
    # declare preliminary quantities
    eps <- 1e-12
    n.data <- nrow(x.train)
    dim.data <- ncol(x.train)
    I <- diag(n.data)
    Y <- I * y.train
    e <- rep(1, n.data)
    ep <- rep(1, dim.data)
    M0 <- matrix(0,n.data,n.data)

    # generated L(y) based on y
    L <- y.train
    L[L==1] <- 1-PI
    L[L==-1] <- PI

    # Matrix in LP contstraint
    cmat <- t(x.train)%*%Y
    c1 <- n.data*lambda*ep

    # Use OSQP to solve LP by setting Q as 0
    Dmat <- Matrix(M0, sparse = TRUE)
    dvec <- -1*e

    # # print(Amat)
    # compactMat <- QP.compact.matrix(Amat)
    # # print(compactMat)
    # Amat <- compactMat$Amat.compact
    # Aind <- compactMat$Aind

    # Make sparse matrix Amat for osqp
    Amat <- rbind(t(y.train),I,cmat)
    Amat <- Matrix(Amat, sparse=TRUE)

    low <- c(0,rep(0, n.data),(-1)*c1)
    high <- c(0,L,c1)

    # print(bvec)

    settings <- osqpSettings(verbose = FALSE, eps_abs=1e-3, eps_rel = 1e-3)
    alpha <- solve_osqp(Dmat, dvec, Amat, l=low, u=high, pars=settings)$x


    # print(alpha)
    iv <- cmat%*%alpha

    # compute the s, t
    s <- c1 - iv
    t <- c1 + iv

    # return a indicator matrix that:
    # nrow: num of dimension p
    # ncol: s, t
    # for each row if both s and t > 0, the the sum of that row = 0
    # the feature being selected is the row sum (fv) not equal to 0

    ind_mat <- 0+(round(cbind(s,t), beta_precision) == 0)

    # return the index vector of the 1 to p features will be kept
    f_ind <- which(as.vector(0 + (rowSums(ind_mat) != 0)) != 0)

    # updated training data with selected features
    Xp.train <- as.matrix(x.train[,f_ind])


    # prepare output
    wl1svm.model <- list(f_ind = f_ind, Xp.train = Xp.train, kernel = kernel, lambda = lambda)
  }


  ## Solve by lpSolveAPI
  if (!osqp.option){
    # declare preliminary quantities
    eps <- 1e-12
    n.data <- nrow(x.train)
    dim.data <- ncol(x.train)
    I <- diag(n.data)
    Y <- I * y.train
    e <- rep(1, n.data)
    ep <- rep(1, dim.data)

    # generated L(y) based on y
    L <- y.train
    L[L==1] <- 1-PI
    L[L==-1] <- PI

    # Matrix in LP contstraint
    cmat <- t(x.train)%*%Y
    c1 <- n.data*lambda*ep

    # number of total constraints
    ncons <- 2*n.data+1+dim.data+dim.data

    # coef of objective
    objf <- -1*e

    ## Solve LP by lpSolveAPI

    # Set number of constraints and number of decision variables
    lprec <- make.lp(nrow = ncons, ncol = n.data)

    # Set the type of problem we are trying to solve
    lp.control(lprec, sense="min")

    # Set type of decision variables as real number
    set.type(lprec, 1:n.data, type=c("real"))

    # Set objective function coefficients vector C
    set.objfn(lprec, objf)

    # define the LP problem
    for (i in 1 : ncons) {
      if (i <= n.data){
        set.row(lprec, row=i, xt=c(1), indices=c(i))
        set.constr.type(lprec, ">=", constraints=i)
        set.rhs(lprec, b = 0, constraints=i)
      }

      else if(i > n.data && i <= 2*n.data){
        set.row(lprec, row=i, xt=c(1), indices=c(i-n.data))
        set.constr.type(lprec, "<=", constraints=i)
        set.rhs(lprec, b = L[(i-n.data)], constraints=i)
      }

      else if(i == (2*n.data+1)){
        set.row(lprec, row=i, xt=y.train)
        set.constr.type(lprec, "=", constraints=i)
        set.rhs(lprec, b = 0, constraints=i)
      }

      else if(i > (2*n.data+1) && i <= (2*n.data+1+dim(cmat)[1])){
        set.row(lprec, row=i, xt=cmat[i-(2*n.data+1),])
        set.constr.type(lprec, "<=", constraints=i)
        set.rhs(lprec, b = c1[i-(2*n.data+1)], constraints=i)
      }

      else {
        set.row(lprec, row=i, xt=cmat[i-(2*n.data+1+dim(cmat)[1]),])
        set.constr.type(lprec, ">=", constraints=i)
        set.rhs(lprec, b = (-1)*c1[i-(2*n.data+1+dim(cmat)[1])], constraints=i)
      }

    }

    # find alpha by LP
    solve(lprec)

    # Get the decison variables values
    alpha <- get.variables(lprec)


    # print(alpha)
    iv <- cmat%*%alpha

    # compute the s, t
    s <- c1 - iv
    t <- c1 + iv

    # return a indicator matrix that:
    # nrow: num of dimension p
    # ncol: s, t
    # for each row if both s and t > 0, the the sum of that row = 0
    # the feature being selected is the row sum (fv) not equal to 0

    ind_mat <- 0+(round(cbind(s,t), beta_precision) == 0)

    # return the index vector of the 1 to p features will be kept
    f_ind <- which(as.vector(0 + (rowSums(ind_mat) != 0)) != 0)

    # updated training data with selected features
    Xp.train <- as.matrix(x.train[,f_ind])


    # prepare output
    wl1svm.model <- list(f_ind = f_ind, Xp.train = Xp.train, kernel = kernel, lambda = lambda)
  }

  return(wl1svm.model)
}
