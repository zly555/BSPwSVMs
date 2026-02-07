<!-- README.md is generated from README.Rmd. Please edit that file -->

# BSPwSVMs

<!-- badges: start -->
<!-- badges: end -->

The goal of **BSPwSVMs** is to provide an efficient implementation of **Binary Sparse Probability estimation with Weighted Support Vector Machines (wSVMs) with automated feature selection**.  This package implements the methodology developed in:  **Zeng, L. and Zhang, H. H. (2023).** *Sparse Learning and Class Probability Estimation with Weighted Support Vector Machines.* [arXiv:2312.10618](https://arxiv.org/abs/2312.10618) 

## Overview 

While traditional $L_2$-norm regulated weighted SVMs, such as those described in **Wang, Zhang, and Wu (2019)** and **Zeng and Zhang (2022)**, provide a strong foundation for multiclass probability estimation, they face significant challenges in high-dimensional settings. When the number of features $p$ is much larger than the number of observations $n$, noise features can lead to overfitting and degrade the accuracy of the probability estimation.



**BSPwSVMs** (Zeng and Zhang, 2023) addresses these limitations by integrating sparsity with weighted SVM learning:

- **Automated Variable Selection:** By incorporating sparsity-inducing penalties (such as the $L_1$ norm or Elastic Net), the model automatically identifies and retains only the most informative features. This feature selection is critical for model interpretability and performance when $p \gg n$.
- **Calibrated Probability Estimation:** By combining automated feature selection with a weighted SVM framework, the package produces posterior probability estimates that are more robust to noise and better calibrated than standard wSVM implementations.
- **Variable Grouping and Importance:** The framework allows for the assessment of feature importance and supports variable grouping, providing deeper insights into the underlying data structure.

## Installation

You can install the development version of `BSPwSVMs`  from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("zly555/BSPwSVMs")
```

## Example

This example demonstrates the sparse learning and automated feature selection capabilities of the `BSPwSVMs` package using the simulation framework from **Example 3 in Zeng and Zhang (2023)**. This scenario specifically evaluates model performance in the presence of **unbalanced classes** and **highly correlated features** where $p \gg n$.

We compare four distinct algorithms to showcase the advantages of our proposed methods:

- **$L_2$ - wSVMs (Baseline):** The standard weighted SVM with no feature selection capability.
- **$L_1$ - $L_2$ - wSVMs (LOTWSVM):** Incorporates Lasso-style penalties for variable selection.
- **Elastic Net - wSVMs (ENPWSVM):** Combines $L_1$ and $L_2$ penalties to handle correlated predictors.
- **Elastic Net - $L_2$ - wSVMs (ENTPWSVM):** A specialized hybrid scheme for optimized probability estimation.

The example tracks four key performance metrics:

1. **Running Time:** Computational efficiency in high-dimensional spaces.
2. **EGKL:** Expected Generalized Kullback-Leibler divergence, used as a rigorous measure of class probability estimation accuracy.
3. **Test Error:** Overall classification accuracy on out-of-sample data.
4. **Feature Selection:** The model's ability to identify the true informative predictors from the noise.

### Install the Required R Library

``` r
library(BSPwSVMs)
library(lpSolveAPI)
library(quadprog)
library(osqp)
library(Matrix)
library(MASS)
library(ggplot2)
library(tictoc, warn.conflicts = FALSE)
```

### Generate the Simulated Data

``` r
#' Geneate the simulation data for example 3
#'
#' @param n sample size
#' @param p dimensionality
#' @param mu multivariate normal mean
#' @param sig_feature number of significant features
#' @param rho rho covariance matrix
#' @param unbalance.ratio class unbalance ratio
#' @param seed random seed
#'
#' @return Simulated data

generator_ex3 <- function(n, p, mu, sig_feature=5, rho = 0, unbalance.ratio =0.5, seed=1){

  mu1 <-  c(rep(mu,sig_feature),rep(0,p-sig_feature))
  mu2 <-  c(rep(-mu,sig_feature),rep(0,p-sig_feature))
  
  Sig_star <- matrix(0, nrow =sig_feature, ncol=sig_feature)
  v <- c(1, rho, rho^2, rho^3, rho^4)
  Sig_star[1,] <- v
  Sig_star[sig_feature,] <- rev(v)
  Sig_star[2,] <- c(v[2],v[1],v[2],v[3],v[4])
  Sig_star[3,] <- c(v[3],v[2],v[1],v[2],v[3])
  Sig_star[sig_feature-1,] <- c(v[4],v[3],v[2],v[1],v[2])
  Sigma1 <-  diag(p)
  Sigma1[1:sig_feature, 1:sig_feature] <- Sig_star
  Sigma2 <- Sigma1

  set.seed(seed)
  X <- rbind(MASS::mvrnorm(floor(n*unbalance.ratio),mu1,Sigma1), mvrnorm(floor(n*(1-unbalance.ratio)),mu2,Sigma2))
  y <- c(rep(1,floor(n*unbalance.ratio)),rep(-1,floor(n*(1-unbalance.ratio))))

  return(list(X= as.matrix(X), y = y, n=n, p=p, sig_feature=sig_feature))
}
```

### Running Setup

``` r
############################
###### Sample Setting ######
############################
n <- 200 # tuning plus training size
train.tune.ratio <- 0.5
p_dim <- c(100, 200, 400, 1000)
n_train <- floor(n*train.tune.ratio)


nsim <- 1 # number of simulatiom

# number of PIs 
m = floor(sqrt(n_train))-1

########################
###### Algorithms ######
########################
# We illustrate 4 algorithms, the L2 baseline, and 3 proposed sparse learning algorithms
# L2 wSVM use quadprog solver (Dense)
# L1+L2 wSVM use osqp solver (Sparse)
# ElasticNet_Primal use osqp solver (Sparse)
# ElasticNet_L2_Primal use osqp solver (Sparse)
baseline_algo <- c("L2_wSVM")

# algorithm with feature selection
algs <- c("L1_L2_wSVM", "ElasticNet_wSVM", "ElasticNet_L2_wSVM")

n_alg <- length(algs)
algo_index <- 1:n_alg

# set thresholds
max_freq <- floor(sqrt(n_train))-1
threshold <- c(0.2, 0.3, 0.5)
threshold_freq <- max_freq*threshold

###################################################
###### Change Seed for each simulation Files ######
###################################################

set.seed (42)
rand_seed <- sample.int(10000, 3*nsim)

##################################
######Prepare result Matrix ######
##################################

# the matrix to hold all the results with all the siumulation results
Result_Matrix <- as.data.frame(matrix(0, nrow = length(p_dim)*nsim*(n_alg+2), ncol = 20+2*length(threshold)))
col_names <- c('SimNum', 'Algorithm','Optim', 'Solver', 'N_train','Dims','ElapsedTime(mins)', 
  'LOG_ElapsedTime', 'TrueF','lambda1','lambda2','beta_prec', 'EGKL', 'TestErr', 'p_signal_union','p_noisy_union',
  'p_signal_intersec','p_noisy_intersec', 'p_signal_byrow','p_noisy_byrow')

for (fq in threshold_freq){
    col_names <- c(col_names, paste("p_signal_thres_", fq ,sep = ""))
    col_names <- c(col_names, paste("p_noisy_thres_", fq ,sep = ""))
}

colnames(Result_Matrix) <- col_names

# create a list of feature matrix based on differnt dimensions, and different algorithms
# last five row are the aggregation for different algorithms
F_matrix <- list()

for (p in p_dim){
  F_matrix[[paste("F_Matrix_p", p, sep = "")]] <- matrix(c(rep(algo_index, each = m), algo_index, rep(0,(m+1)*n_alg*p)), nrow = (m+1)*n_alg, ncol = p+1, byrow=FALSE)
}

# set inital count
n_records <- 1
```

### MCMC Simulations

``` r
tictoc::tic() # time measurement

# simulated with different dimensions
for (p in p_dim){
  seed.cnt <- 1
    # loop for number of simulation (here use 1 simulation as illustration)
    for (sim in 1:nsim){
      # generate training, tuning and testing data
      train.data <- generator_ex3(n=n_train, p = p, mu = 1, sig_feature=5, rho = 0.8, unbalance.ratio =0.6, seed = rand_seed[seed.cnt])
      tune.data <- generator_ex3(n=n-n_train, p = p, mu = 1, sig_feature=5, rho = 0.8, unbalance.ratio =0.6, seed = rand_seed[seed.cnt+1])
      test.data <- generator_ex3(n=25*n, p = p, mu = 1, sig_feature=5, rho = 0.8, unbalance.ratio =0.6, seed = rand_seed[seed.cnt+2])

      x.train <- train.data$X
      y.train <- train.data$y
      x.tune <- tune.data$X
      y.tune <- tune.data$y
      x.test <- test.data$X
      y.test <- test.data$y
      
      # significant feature list
      n_sig_features <- train.data$sig_feature
      sig_features <- 1:n_sig_features

      seed.cnt <- seed.cnt + 3


      #############
      ## L2 wSVM ##
      #############
      tictoc::tic()
      result <- L2wSVM(x.train, y.train, x.tune, y.tune, x.test, y.test, small.tune = FALSE)
      
      T.diff <- toc()
      elapsed_time <- round(as.numeric(T.diff$toc - T.diff$tic)/60, 3)
      log_elapsed_time <- round(log(elapsed_time),3)
      
      Result_Matrix[n_records, 1] <- sim
      Result_Matrix[n_records, 2] <- baseline_algo[1]
      Result_Matrix[n_records, 3] <- "Dual"
      Result_Matrix[n_records, 4] <- "quadprog"
      Result_Matrix[n_records, 5] <- n_train 
      Result_Matrix[n_records, 6] <- p
      Result_Matrix[n_records, 7] <- elapsed_time
      Result_Matrix[n_records, 8] <- log_elapsed_time
      Result_Matrix[n_records, 9] <- "x1-x5"
      Result_Matrix[n_records, 10] <- result$best.lambda
      Result_Matrix[n_records, 11] <- NA
      Result_Matrix[n_records, 12] <- NA
      Result_Matrix[n_records, 13] <- result$egkl
      Result_Matrix[n_records, 14] <- result$TestErr
      Result_Matrix[n_records, 15:length(col_names)] <- NA
      
      n_records <- n_records + 1

      ################
      ## L1+L2 wSVM ##
      ################
        
      algo_cnt <- 1
        
      tictoc::tic()
      result <- L1wSVM(x.train, y.train, x.tune, y.tune, x.test, y.test, osqp.option = TRUE, small.tune = FALSE, double.tune = FALSE, beta_precision = 2)
      
      T.diff <- toc()
      elapsed_time <- round(as.numeric(T.diff$toc - T.diff$tic)/60, 3)
      log_elapsed_time <- round(log(elapsed_time),3)
      
      Result_Matrix[n_records, 1] <- sim
      Result_Matrix[n_records, 2] <- algs[1]
      Result_Matrix[n_records, 3] <- "Dual"
      Result_Matrix[n_records, 4] <- "OSQP"
      Result_Matrix[n_records, 5] <- n_train 
      Result_Matrix[n_records, 6] <- p
      Result_Matrix[n_records, 7] <- elapsed_time
      Result_Matrix[n_records, 8] <- log_elapsed_time
      Result_Matrix[n_records, 9] <- "x1-x5"
      Result_Matrix[n_records, 10] <- result$best.lambda
      Result_Matrix[n_records, 11] <- NA
      Result_Matrix[n_records, 12] <- 10^((-1)*result$beta_precision)
      Result_Matrix[n_records, 13] <- result$egkl
      Result_Matrix[n_records, 14] <- result$TestErr
      
      # Union features
      aggr_f_sum <- colSums(result$f_matrix)
      Result_Matrix[n_records, 15] <- sum(aggr_f_sum[1:n_sig_features]!=0)
      Result_Matrix[n_records, 16] <- sum(aggr_f_sum[-(1:n_sig_features)]!=0)
     
      # Intersection features
      aggr_f_prod <- apply(result$f_matrix, 2, prod)
      Result_Matrix[n_records, 17] <- sum(aggr_f_prod[1:n_sig_features]!=0)
      Result_Matrix[n_records, 18] <- sum(aggr_f_prod[-(1:n_sig_features)]!=0)
      
      # sumemrize features in f_matrix in result list for each row
      Result_Matrix[n_records, 19] <- round(mean(apply(result$f_matrix[,1:n_sig_features], 1, sum)),2)
      Result_Matrix[n_records, 20] <- round(mean(apply(result$f_matrix[,-(1:n_sig_features)], 1, sum)),2)
      
      # Select feature by threshold
      cnt <- 1
      for (f in 1:length(threshold_freq)){
        Result_Matrix[n_records, 20+cnt] <- sum(aggr_f_sum[1:n_sig_features]>threshold_freq[f])
        Result_Matrix[n_records, 20+cnt+1] <- sum(aggr_f_sum[-(1:n_sig_features)]>threshold_freq[f])
        cnt <- cnt+2
      }


      F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] <- F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] + result$f_matrix
      
      n_records <- n_records + 1
      algo_cnt <- algo_cnt + 1

      
      #####################
      ## ElasticNet wSVM ##
      #####################
      tictoc::tic()
      result <- ElasticNetwSVM(x.train, y.train, x.tune, y.tune, x.test, y.test, l2.option = FALSE, beta_precision = 1e-2)

      T.diff <- tictoc::toc()
      elapsed_time <- round(as.numeric(T.diff$toc - T.diff$tic)/60, 3)
      log_elapsed_time <- round(log(elapsed_time),3)
      
      Result_Matrix[n_records, 1] <- sim
      Result_Matrix[n_records, 2] <- algs[2]
      Result_Matrix[n_records, 3] <- "Primal"
      Result_Matrix[n_records, 4] <- "OSQP"
      Result_Matrix[n_records, 5] <- n_train 
      Result_Matrix[n_records, 6] <- p
      Result_Matrix[n_records, 7] <- elapsed_time
      Result_Matrix[n_records, 8] <- log_elapsed_time
      Result_Matrix[n_records, 9] <- "x1-x5"
      Result_Matrix[n_records, 10] <- result$best.lambda1
      Result_Matrix[n_records, 11] <- result$best.lambda2
      Result_Matrix[n_records, 12] <- result$beta_precision
      Result_Matrix[n_records, 13] <- result$egkl
      Result_Matrix[n_records, 14] <- result$TestErr
       # Union features
      aggr_f_sum <- colSums(result$f_matrix)
      Result_Matrix[n_records, 15] <- sum(aggr_f_sum[1:n_sig_features]!=0)
      Result_Matrix[n_records, 16] <- sum(aggr_f_sum[-(1:n_sig_features)]!=0)
     
      # Intersection features
      aggr_f_prod <- apply(result$f_matrix, 2, prod)
      Result_Matrix[n_records, 17] <- sum(aggr_f_prod[1:n_sig_features]!=0)
      Result_Matrix[n_records, 18] <- sum(aggr_f_prod[-(1:n_sig_features)]!=0)
      
      # sumemrize features in f_matrix in result list for each row
      Result_Matrix[n_records, 19] <- round(mean(apply(result$f_matrix[,1:n_sig_features], 1, sum)),2)
      Result_Matrix[n_records, 20] <- round(mean(apply(result$f_matrix[,-(1:n_sig_features)], 1, sum)),2)
      
      # Select feature by threshold
      cnt <- 1
      for (f in 1:length(threshold_freq)){
        Result_Matrix[n_records, 20+cnt] <- sum(aggr_f_sum[1:n_sig_features]>threshold_freq[f])
        Result_Matrix[n_records, 20+cnt+1] <- sum(aggr_f_sum[-(1:n_sig_features)]>threshold_freq[f])
        cnt <- cnt+2
      }


      F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] <- F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] + result$f_matrix
      
      n_records <- n_records + 1
      algo_cnt <- algo_cnt + 1


      ##########################
      ## ElasticNet + L2 wSVM ##
      ##########################
      tictoc::tic()
      result <- ElasticNetwSVM(x.train, y.train, x.tune, y.tune, x.test, y.test, l2.option = TRUE, beta_precision = 1e-3)

      T.diff <- tictoc::toc()
      elapsed_time <- round(as.numeric(T.diff$toc - T.diff$tic)/60, 3)
      log_elapsed_time <- round(log(elapsed_time),3)
      
      Result_Matrix[n_records, 1] <- sim
      Result_Matrix[n_records, 2] <- algs[3]
      Result_Matrix[n_records, 3] <- "Primal"
      Result_Matrix[n_records, 4] <- "OSQP"
      Result_Matrix[n_records, 5] <- n_train 
      Result_Matrix[n_records, 6] <- p
      Result_Matrix[n_records, 7] <- elapsed_time
      Result_Matrix[n_records, 8] <- log_elapsed_time
      Result_Matrix[n_records, 9] <- "x1-x5"
      Result_Matrix[n_records, 10] <- result$best.lambda1
      Result_Matrix[n_records, 11] <- result$best.lambda2
      Result_Matrix[n_records, 12] <- result$beta_precision
      Result_Matrix[n_records, 13] <- result$egkl
      Result_Matrix[n_records, 14] <- result$TestErr
       # Union features
      aggr_f_sum <- colSums(result$f_matrix)
      Result_Matrix[n_records, 15] <- sum(aggr_f_sum[1:n_sig_features]!=0)
      Result_Matrix[n_records, 16] <- sum(aggr_f_sum[-(1:n_sig_features)]!=0)
     
      # Intersection features
      aggr_f_prod <- apply(result$f_matrix, 2, prod)
      Result_Matrix[n_records, 17] <- sum(aggr_f_prod[1:n_sig_features]!=0)
      Result_Matrix[n_records, 18] <- sum(aggr_f_prod[-(1:n_sig_features)]!=0)
      
      # sumemrize features in f_matrix in result list for each row
      Result_Matrix[n_records, 19] <- round(mean(apply(result$f_matrix[,1:n_sig_features], 1, sum)),2)
      Result_Matrix[n_records, 20] <- round(mean(apply(result$f_matrix[,-(1:n_sig_features)], 1, sum)),2)
      
      # Select feature by threshold
      cnt <- 1
      for (f in 1:length(threshold_freq)){
        Result_Matrix[n_records, 20+cnt] <- sum(aggr_f_sum[1:n_sig_features]>threshold_freq[f])
        Result_Matrix[n_records, 20+cnt+1] <- sum(aggr_f_sum[-(1:n_sig_features)]>threshold_freq[f])
        cnt <- cnt+2
      }


      F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] <- F_matrix[[paste("F_Matrix_p", p, sep = "")]][((algo_index[algo_cnt]-1)*m+1):(algo_index[algo_cnt]*m),2:(p+1)] + result$f_matrix
      
      n_records <- n_records + 1

  }


}


T.diff <- tictoc::toc()
time_elasped <- round(as.numeric(T.diff$toc - T.diff$tic)/60, 3)

cat("Total Time for simulations:\n", time_elasped, "\n")
# Total Time for simulations:
# 63.276 

# write the result into CSV file
write.csv(Result_Matrix, file = 'example3.csv')
```

### Simulation Results (Example 3)

| **Algorithm** | **N** | **Dims** | **Time** | **λ1** | **λ2** | **EGKL** | **TE** | **$p_{signal}$** | **$p_{noisy}$** |
| :------------ | :---- | :------- | :------- | :----- | :----- | :------- | :----- | :--------------- | :-------------- |
| LTWSVM        | 100   | 100      | 0.795    | 0.280  | NA     | 0.360    | 0.156  | NA               | NA              |
| LOTWSVM       | 100   | 100      | 1.240    | 0.046  | NA     | 0.353    | 0.145  | 5.0              | 27.0            |
| ENPWSVM       | 100   | 100      | 1.269    | 0.055  | 0.006  | 0.340    | 0.135  | 5.0              | 13.0            |
| ENTPWSVM      | 100   | 100      | 1.337    | 0.055  | 0.006  | 0.372    | 0.157  | 5.0              | 95.0            |
| LTWSVM        | 100   | 200      | 0.728    | 0.550  | NA     | 0.402    | 0.163  | NA               | NA              |
| LOTWSVM       | 100   | 200      | 1.569    | 0.073  | NA     | 0.393    | 0.129  | 3.0              | 15.0            |
| ENPWSVM       | 100   | 200      | 2.354    | 0.055  | 0.006  | 0.387    | 0.135  | 4.0              | 25.0            |
| ENTPWSVM      | 100   | 200      | 2.442    | 0.006  | 55.0   | 0.365    | 0.153  | 5.0              | 41.0            |
| LTWSVM        | 100   | 400      | 0.791    | 0.910  | NA     | 0.428    | 0.192  | NA               | NA              |
| LOTWSVM       | 100   | 400      | 2.958    | 0.082  | NA     | 0.384    | 0.133  | 4.0              | 13.0            |
| ENPWSVM       | 100   | 400      | 5.168    | 0.055  | 0.001  | 0.361    | 0.136  | 4.0              | 38.0            |
| ENTPWSVM      | 100   | 400      | 5.254    | 0.055  | 55.0   | 0.373    | 0.126  | 5.0              | 16.0            |
| LTWSVM        | 100   | 1000     | 0.985    | 1.900  | NA     | 0.460    | 0.200  | NA               | NA              |
| LOTWSVM       | 100   | 1000     | 8.202    | 0.100  | NA     | 0.412    | 0.130  | 5.0              | 11.0            |
| ENPWSVM       | 100   | 1000     | 14.335   | 0.055  | 0.550  | 0.383    | 0.133  | 5.0              | 112.0           |
| ENTPWSVM      | 100   | 1000     | 14.353   | 0.055  | 0.055  | 0.369    | 0.152  | 5.0              | 257.0           |

### Conclusion

The **BSPwSVMs** package provides a comprehensive toolkit for high-dimensional binary classification where both predictive accuracy and model interpretability are paramount. By integrating weighted Support Vector Machines (wSVMs) with sparsity-inducing penalties, we address a critical gap in standard machine learning workflows: the need for reliable class probability estimation in the presence of noise and high-dimensional features.

**Our key contributions include:**

- **Accurate Probability Estimation:** We leverage the weighted SVM framework to ensure posterior probabilities are well-calibrated. This is evidenced by the low **EGKL** (Expected Generalized Kullback-Leibler) scores across our high-dimensional simulations, indicating superior estimation of class membership likelihoods compared to standard SVM implementations.
- **Automatic Feature Selection:** By utilizing $L_1$ and Elastic Net regularizers, the package spontaneously identifies informative predictors. This significantly reduces the impact of noise features—a critical capability for maintaining model performance in settings where the number of dimensions far exceeds the number of observations ($p \gg n$).
- **Robustness to Correlation:** As demonstrated in our correlated feature simulations ($\rho=0.8$), `BSPwSVMs` maintains high signal capture ($p_{signal}$) where traditional $L_2$ methods fail to provide sparsity. Furthermore, the Elastic Net component effectively handles the grouping effect of highly correlated features, ensuring that relevant variable clusters are selected together rather than arbitrarily excluded.

For a deeper dive into the mathematical proofs, complexity analysis, and extensive Monte Carlo performance metrics, please consult:

> **Zeng, L. and Zhang, H. H. (2023).** *Sparse Learning and Class Probability Estimation with Weighted Support Vector Machines.* [arXiv:2312.10618](https://arxiv.org/abs/2312.10618)
