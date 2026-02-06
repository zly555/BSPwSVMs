#' Function to perform Elastic Net weighted SVM using dual optimization
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
#' @return Elastic Net regulated wSVM model object use dual optimization
#' @export

wENsvm <- function(x.train, y.train, PI = 0.5, kernel = list(type = 'linear', param1 = NULL, param2 = NULL),
                   lambda1 = 0.01, lambda2 = 0.01, beta_precision = 1e-5, eps = 1e-10){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)

  # declare preliminary quantities
  eps <- 1e-12
  n.data <- nrow(x.train)
  dim.data <- ncol(x.train)
  p3 <- 3*dim.data
  In <- diag(n.data)
  Ip <- diag(dim.data)

  # M(0)pxp
  M0p <- matrix(0, dim.data, dim.data)
  # M(0)3pxn
  M03pxn <- matrix(0, p3, n.data)
  # M(0)nxp
  M0nxp <- matrix(0, n.data, dim.data)
  # M(0)2pxp
  M02pxp <- matrix(0, 2*dim.data, dim.data)
  # M(0)(n+p)xp
  M0npxp <- matrix(0, (n.data+dim.data), dim.data)
  # M(0)(n+p)xp
  M0n2pxp <- matrix(0, (n.data+2*dim.data), dim.data)

  Y <- In * y.train
  e = rep(1, n.data)
  ep <- rep(1, dim.data)

  # generated L(y) based on y
  L <- y.train
  L[L==1] <- 1-PI
  L[L==-1] <- PI

  # coeffcients
  c0 <- n.data*lambda2
  c1 <- 1/c0
  c2 <- -1*(lambda1/lambda2)

  cmat <- t(x.train)%*%Y

  A1 <- c1*cmat
  A2 <- c1*Ip
  A3 <- M0p
  A4 <- c2*Ip

  B1 <- (-1*c1)*cmat
  B2 <- M0p
  B3 <- A2
  B4 <- A4

  A <- cbind(A1,A2,A3,A4)
  B <- cbind(B1,B2,B3,B4)

  At <- t(A)
  Bt <- t(B)

  # Q and c in objective
  c <- c(e, rep(0,p3))
  Q <- c0*(At%*%A+Bt%*%B)

  # W matrix in contraints
  # equality constraint
  wc1 <- c(y.train, rep(0,p3))
  wc2 <- rbind(M0n2pxp, Ip)
  # inequality constraint
  wc3 <- rbind(In,M03pxn)
  wc4 <- rbind(-In,M03pxn)
  wc5 <- rbind(M0nxp, Ip, M02pxp)
  wc6 <- rbind(M0npxp, Ip, M0p)


  W <- t(cbind(wc1,wc2,wc3,wc4,wc5,wc6,At,Bt))
  colnames(W)<-NULL

  low <- c(0,rep(1,dim.data),rep(0,n.data),-L,rep(0,4*dim.data))
  high <- c(0,rep(1,dim.data),rep(Inf, 2*n.data+4*dim.data))

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
  # Dmat <- Q
  # diag(Dmat) <- diag(Dmat) + eps
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
  x_optimal <- solve_osqp(Dmat, -dvec, Amat, l=low, u=high, pars=settings)$x


  # get alpha, u, v from x_optimal then get beta
  alpha <- x_optimal[1:n.data]
  u <- A%*%x_optimal
  v <- B%*%x_optimal
  beta <- as.vector(u-v)

  # return the index vector of the 1 to p features will be kept
  beta_index <- 1:dim.data
  f_index <- beta_index[abs(beta) > beta_precision]

  # set beta below precision to 0
  beta[-f_index] <- 0

  beta0 <- as.numeric((t(e)%*%(In*alpha)%*%(In*(L-alpha))%*%(as.matrix(1/y.train)-x.train%*%as.matrix(beta)))/(t(as.matrix(alpha))%*%as.matrix(L-alpha)))

  # compute the index and the number of support vectors
  S <- 1:n.data
  sv.index <- S[alpha > eps]
  sv.number <- length(sv.index)
  sv <- list(index = sv.index, number = sv.number)

  # let alpha = 0 if it is too small
  alpha[-sv.index] <- 0

  # updated training data with selected features
  Xp.train <- as.matrix(x.train[,f_index])
  # prepare output
  wENsvm.model <- list(alpha = alpha, beta = beta, beta0 = beta0,  Xp.train = Xp.train, sv = sv, f_ind = f_index, lambda1 = lambda1, lambda2 = lambda2)

  return(wENsvm.model)

}
