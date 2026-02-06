#' predict the label with given wsvm classifier
#'
#' @param x New data point
#' @param x.train Training data matrix
#' @param c Fitted classifier c
#' @param d Fitted classifier d
#' @param kernel Kernel to use, default 'linear'
#'
#' @return return Predicted label {+1,-1}
#' @export

wsvm.predict <- function(x, x.train, c, d, kernel = list(type = 'linear', param1 = NULL, param2 = NULL)){

  if(!is.matrix(x)) x <- as.matrix(x)
  if(!is.matrix(c)) c <- as.matrix(c)
  if(!is.numeric(d)) d <- as.numeric(d)
  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  # print('start predict')
  # print( dim(c))

  K <- wsvm.kernel(x, x.train, kernel)
  d <- as.numeric(d)

  return(as.vector(sign(K%*%c + d)))

}


#' predict the label with given elastic net wsvm classifier
#'
#' @param x new data points
#' @param beta parameter beta
#' @param beta0 parameter beta0
#' @param kernel Kernel to use, default 'linear'
#'
#' @return return Predicted label {+1,-1}
#' @export

wENsvm.predict <- function(x, beta, beta0, kernel = list(type = 'linear', param1 = NULL, param2 = NULL)){

  if(!is.matrix(x)) x <- as.matrix(x)
  if(!is.matrix(beta)) beta <- as.matrix(beta)
  if(!is.numeric(beta0)) beta0 <- as.numeric(beta0)

  return(as.vector(sign(x%*%beta + beta0)))

}


#' find Amat.compact and Aind for QP.solve.compact
#'
#' @param Amat Amat in solve.QP
#'
#' @return compact form of Amat
#' @export

QP.compact.matrix <- function(Amat){
  nr <- nrow(Amat)
  nc <- ncol(Amat)
  Amat.compact <- matrix(0, nr, nc)
  Aind <- matrix(0, nr+1, nc)

  for (j in 1:nc){
    index <- (1:nr)[Amat[, j] != 0]
    number <- length(index)
    Amat.compact[1:number, j] <- Amat[index, j]
    Aind[1, j] <- number
    Aind[2:(number+1), j] <- index
  }

  max.number <- max(Aind[1, ])
  Amat.compact <- Amat.compact[1:max.number, ]

  Aind <- Aind[1:(max.number+1), ]
  compact <- list(Amat.compact = Amat.compact, Aind = Aind)

  return(compact)
}


#' Calculate argmin function
#'
#' @param vec Input R vector
#'
#' @export
#' @return The minimum index with the minimum value of the given vector
argmin <- function(vec) {
  index <- 1:length(z)
  argmin <- min(index[z == min(z)])
  return(argmin)
}

#' Calculate argmax function
#'
#' @param vec Input R vector
#'
#' @export
#' @return The maximum index with the maximum value of the given vector
argmax <- function(vec){
  index <- 1:length(z)
  argmax <- max(index[z == max(z)])
  return(argmax)
}

#' Calculate the EGKL
#'
#' @param y Label vector
#' @param px Estimated probability
#'
#' @export
#' @return EGKL value
wsvm.egkl <- function(y, px) {
  n <- length(y)
  return(-0.5*mean((y+1)*log(px) + (1-y)*log(1-px)))
}


#' Estimate the binary class probability
#'
#' @param v Predicted Labels
#' @param p Vector of weights
#'
#' @export
#' @return Estimated probability for {+1} class
prob.estimate<- function(v, p){
  df <- as.data.frame(cbind(v,p))
  colnames(df) <- c("labels","Pi")

  L_max<-aggregate(Pi ~ labels, data = df, max)
  L_min <-aggregate(Pi ~ labels, data = df, min)

  Pi_ls <- L_max$Pi[L_max$labels==1]
  Pi_us <- L_min$Pi[L_max$labels==-1]

  return((Pi_ls+Pi_us)/2)
}
