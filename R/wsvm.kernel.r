#' Function to calculate the Kernel matrix
#'
#' @param U U matrix
#' @param V V matrix
#' @param kernel Kernel params list. default: 'linear'
#'               kernel$type = type of kernel,
#'               (eg: 'linear', 'poly', 'rbf').
#'               kernel$param = parameter of kernel,
#'               (eg: param1 = d degree for 'poly',
#'                   param1 = sigma scale for 'rbf',
#'                   param1,2 = kapa1 and kapa2 for 'nn')
#'
#' @export
#' @return A nrow(U) x nrow(V) kernel matrix

wsvm.kernel <- function(U, V, kernel = list(type = 'linear', param1 = NULL, param2 = NULL)) {

  if(!is.matrix(U)) U <- as.matrix(U)
  if(!is.matrix(V)) V <- as.matrix(V)

  if (kernel$type == 'linear') K <- (U %*% t(V))
  else if (kernel$type == 'poly') K <- (1 + kernel$param2 * U %*% t(V))^kernel$param1
  else if (kernel$type == 'rbf'){
    a = as.matrix(apply(U^2, 1, 'sum'))
    b = as.matrix(apply(V^2, 1, 'sum'))
    one.a = matrix(1, ncol = nrow(b))
    one.b = matrix(1, ncol = nrow(a))
    K1 = one.a %x% a
    K2 = U %*% t(V)
    K3 = t(one.b %x% b)
    K = exp(-(K1 - 2 * K2 + K3)/(2 * kernel$param1^2))
  }
  else if (kernel$type == 'nn') K <- tanh(kernel$param2 + (U %*% t(V))*kernel$param1)

  return(K)
}
