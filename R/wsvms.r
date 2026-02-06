#' Fit an weighted svm model with given weight pi on negative class data
#'
#' @param x.train Training data in matrix form
#' @param y.train Label as vector
#' @param PI Weight on negative class data
#' @param kernel Kernel to choose, refer to function wsvm.kernel, default is linear kernel
#' @param lambda Regulatory parameter for solving weighted SVMs, lambda > 0
#' @param eps  Small threshold
#'
#' @export
#' @return  Model object as a list consists of model fit: c, d, alpha, and sv
wsvm <- function(x.train, y.train, PI = 0.5, kernel = list(type = 'linear', param1 = NULL, param2 = NULL), lambda = 0.01, eps = 1e-10){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)

  # declare preliminary quantities
  eps <- 1e-12
  n.data <- nrow(x.train)
  I <- diag(n.data)
  Y <- I * y.train
  e = rep(1, n.data)

  # generated L(y) based on y
  L <- y.train
  L[L==1] <- 1-PI
  L[L==-1] <- PI

  # compute kernel matrix
  K <- wsvm.kernel(x.train, x.train, kernel = kernel)
  H <- (1/(2*n.data*lambda))*Y%*%K%*%Y

  # print(H)

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
  Dmat <- H
  diag(Dmat) <- diag(Dmat) + eps


  dvec <- e
  Amat <- cbind(y.train, I, -I)
  # print(Amat)
  compactMat <- QP.compact.matrix(Amat)
  # print(compactMat)
  Amat <- compactMat$Amat.compact
  Aind <- compactMat$Aind
  bvec <- c(0, rep(0, n.data), -L)
  # print(bvec)

  # find alpha by QP
  alpha <- solve.QP.compact(Dmat, dvec, Amat, Aind, bvec, meq = 1, factorized=FALSE)$solution

  # print(alpha)

  # compute the index and the number of support vectors
  S <- 1:n.data
  sv.index <- S[alpha > eps]
  sv.number <- length(sv.index)
  sv <- list(index = sv.index, number = sv.number)

  # let alpha = 0 if it is too small
  alpha[-sv.index] <- 0

  # compute c, d
  c <- (1/(2*n.data*lambda))*(Y%*%as.matrix(alpha))
  d <- (t(e)%*%(I*alpha)%*%(I*(L-alpha))%*%(as.matrix(1/y.train)-K%*%c))/(t(as.matrix(alpha))%*%as.matrix(L-alpha))

  # prepare output
  wsvm.model <- list(alpha = alpha, c = c, d = d, sv = sv, kernel = kernel, lambda = lambda)

  return(wsvm.model)
}
