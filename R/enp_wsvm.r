#' Function to perform Elastic Net weighted SVM using primal optimization
#'
#' @param x.train Training data in matrix form
#' @param y.train Label as vector
#' @param PI Weight on negative class data
#' @param kernel Kernel to choose, refer to function wsvm.kernel, default is linear kernel
#' @param lambda1 Regulatory parameter for L1-norm penalty and for solving elatic netweighted SVMs, lambda1 > 0
#' @param lambda2 Regulatory parameter for L2-norm penalty and for solving elatic netweighted SVMs, lambda2 > 0
#' @param beta_precision minimum value of beta before set to 0
#' @param eps Small threshold
#'
#' @return Elastic Net regulated wSVM model object use primal optimization
#' @export

wENp.svm<- function(x.train, y.train, PI = 0.5, kernel = list(type = 'linear', param1 = NULL, param2 = NULL),
                    lambda1 = 0.01, lambda2 = 0.01, beta_precision = 1e-3, eps = 1e-10){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)

  # declare preliminary quantities
  eps <- 1e-12
  n.data <- nrow(x.train)
  dim.data <- ncol(x.train)
  p2 <- 2*dim.data
  In <- diag(n.data)
  Ip <- diag(dim.data)

  # M(0)pxp
  M0p <- matrix(0,dim.data,dim.data)
  # M(0)pxn
  M0nxp <- matrix(0,dim.data,n.data)
  # M(0)px(p+1)
  M0p1xn <- matrix(0,dim.data,dim.data+1)
  # M(0)px(n+p)
  M0npxp <- matrix(0,dim.data,n.data+dim.data)
  # M(0)nx(2p+1)
  M02pxn <- matrix(0,n.data,p2+1)


  Y <- In * y.train
  e = rep(1, n.data)

  # generated L(y) based on y
  L <- y.train
  L[L==1] <- 1-PI
  L[L==-1] <- PI

  # coeffcients
  c1 <- n.data*lambda1
  c2 <- n.data*lambda2

  a1 <- n.data+p2+1


  cmat <- Y%*%x.train


  # Q and c in objective
  Q1 <- matrix(0,a1,n.data)
  Q2 <- rbind(t(M0nxp), Ip,M0p, t(rep(0,dim.data)))
  Q3 <- rbind(t(M0npxp), Ip, t(rep(0,dim.data)))

  Q <- (cbind(Q1, Q2, Q3, rep(0,a1)))*c2

  c <- c(L,c1*rep(1,p2),0)

  # W matrix in contraints

  wc1 <- cbind(In,M02pxn)
  wc2 <- cbind(M0nxp, Ip, M0p1xn)
  wc3 <- cbind(M0npxp, Ip, rep(0,dim.data))
  wc4 <- cbind(In, cmat, -cmat, Y%*%e)

  W <- rbind(wc1,wc2,wc3,wc4)
  colnames(W)<-NULL

  low <- c(rep(0,n.data+p2),rep(1,n.data))
  high <- rep(Inf, (2*n.data+2*dim.data))

  # Solve QP
  # min(−dTb + 1/2bTDb) with the constraints ATb >= b0

  # for numerical stability if the determinat of H is too small
  # pd_Dmat <- nearPD(H)
  # Dmat <- as.matrix(pd_Dmat$mat)
  # print('start QP')
  # print(PI)
  # H <- as.matrix(nearPD(H)$mat)
  # Dmat <- H
  # diag(Dmat) <- diag(Dmat) + eps
  # Dmat <- as.matrix(nearPD(Dmat)$mat)
  Dmat <- Matrix(Q, sparse = TRUE)

  dvec <- c
  Amat <- Matrix(W, sparse=TRUE)
  # print(Amat)
  # compactMat <- QP.compact.matrix(Amat)
  # # print(compactMat)
  # Amat <- compactMat$Amat.compact
  # Aind <- compactMat$Aind
  # bvec <- x0

  # find x optimal solution by QP
  settings <- osqpSettings(verbose = FALSE, eps_abs=1e-3, eps_rel = 1e-3)
  x_optimal <- solve_osqp(Dmat, dvec, Amat, l=low, u=high, pars=settings)$x

  # get alpha, u, v from x_optimal then get beta

  u <- x_optimal[(n.data+1):(n.data+dim.data)]
  v <- x_optimal[(n.data+dim.data+1):(n.data+p2)]

  beta <- as.vector(u-v)
  beta0 <- x_optimal[n.data+p2+1]

  # return the index vector of the 1 to p features will be kept
  beta_index <- 1:dim.data
  f_index <- beta_index[abs(beta) > beta_precision]

  # set beta below precision to 0
  beta[-f_index] <- 0

  # updated training data with selected features
  Xp.train <- as.matrix(x.train[,f_index])
  # prepare output
  wENp.svm.model <- list(beta = beta, beta0 = beta0,  Xp.train = Xp.train, f_ind = f_index, lambda1 = lambda1, lambda2 = lambda2)

  return(wENp.svm.model)

}
