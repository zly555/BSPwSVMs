#' Probability estimation of given data X
#'
#' @param x.train training data
#' @param y.train training label
#' @param x.test testing data
#' @param y.test testing label
#' @param svm.type types of weighted SVMs learning algorithms.
#'     'wl2svm': L2-norm weighted SVMs, without feature selection
#'     'wl1svm': Feature selection first by wL1SVM then wL2SVM (Dual Problem, same lambda for both)
#'     'wl1dsvm': Feature selection first by wL1SVM then wL2SVM (Dual Problem, different lambda for L1 and L2)
#'     'wl1pl2.svm': Feature selection first by wL1SVM then wL2SVM (Primal Problem)
#'     'wl1p.svm': Directly solve beta by wL1SVM (Primal Problem)
#'     'wENsvm': Feature selection and probability estimation directly by elastic net (Dual)
#'     'wENp.svm': Feature selection and probability estimation directly by elastic net (Primal)
#'     'wENl2svm': Feature selection by elastic net follow by L2SVM for probability estimation (Dual)
#'     'wENpl2.svm': Feature selection by elastic net follow by L2SVM for probability estimation (Primal)
#' @param kernel Kernel params list. default: 'linear' in sparse learning
#' @param plot.option plot feature importance map based on threshold: 'f_threshold' or top N features 'f_importance'
#' @param num.plot.imp.features number of important features in plot
#' @param f_threshold feature importance threshold value
#' @param lambda lambda for L1-norm penalty
#' @param lambda2 lambda for L2 penalty
#' @param osqp.option option to use osqp as solver
#' @param beta_precision minimum value of beta before set to 0
#' @param eps Small threshold
#'
#' @return Result matrix including estimated probability matrix and feature importance matrix
#' @export

wsvm.prob <- function(x.train, y.train, x.test, y.test, svm.type = 'wl2svm',
                      kernel = list(type = 'linear', param1 = NULL, param2 = NULL),
                      plot.option = '', num.plot.imp.features = 10, f_threshold = 3, lambda, lambda2, osqp.option, beta_precision=1e-5, eps = 1e-10){

  if(!is.matrix(x.train)) x.train <- as.matrix(x.train)
  if(!is.vector(y.train)) y.train <- as.vector(y.train)
  if(!is.matrix(x.test)) x.test <- as.matrix(x.test)
  if(!is.vector(y.test)) y.test <- as.vector(y.test)

  n.train <- nrow(x.train)
  n.test <- nrow(x.test)

  m <- floor(sqrt(n.train))
  PI <- rep(0, (m+1))
  PI_mid <- 1:(m-1)/m
  PI[2:m] <- PI_mid
  PI[1] <- 0
  PI[m+1]<- 1


  predicted_labels_train <- matrix(0, nrow = n.train, ncol =length(PI_mid))
  predicted_labels_test <- matrix(0, nrow = n.test, ncol = length(PI_mid))

  estimate_labels_train <- estimate_prob_train <- rep(0, n.train)
  estimate_labels_test <- estimate_prob_test <- rep(0, n.test)

  # store the series of wSVM classifier for future prediction
  c_matrix <- matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[1])
  d_vector <- rep(0, length(PI_mid))

  # for elasticNet SVM
  beta_matrix <- matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])
  beta0_vector <- rep(0, length(PI_mid))


  # Only do the wL2SVM without feature selection by L1SVM
  if(svm.type == 'wl2svm'){

    for (i in 1:length(PI_mid)){
      tryCatch({
        wsvm.model <- wsvm(x.train, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(x.train, x.train, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(x.test, x.train, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    # print("finish loop!!!!!")
    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))


  }


  # Feature selection first by wL1SVM then wL2SVM (Dual Problem, same lambda for both)
  if(svm.type == 'wl1svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        # first fit wL1SVM
        wl1svm.model <- wl1svm(x.train, y.train, PI = PI_mid[i], lambda = lambda, osqp.option = osqp.option, beta_precision = beta_precision)
        new.train.x <- wl1svm.model$Xp.train
        f_ind <- wl1svm.model$f_ind
        new.text.x <- as.matrix(x.test[,f_ind])
        f_matrix[i, f_ind] = 1

        # then fit wL2SVM
        wsvm.model <- wsvm(new.train.x, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(new.train.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(new.text.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,"\n","Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }


  # Feature selection first by wL1SVM then wL2SVM (Dual Problem, different lambda for L1 and L2)
  if(svm.type == 'wl1dsvm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        # first fit wL1SVM
        wl1svm.model <- wl1svm(x.train, y.train, PI = PI_mid[i], lambda = lambda, osqp.option = osqp.option, beta_precision = beta_precision)
        new.train.x <- wl1svm.model$Xp.train
        f_ind <- wl1svm.model$f_ind
        new.text.x <- as.matrix(x.test[,f_ind])
        f_matrix[i, f_ind] = 1

        # then fit wL2SVM
        wsvm.model <- wsvm(new.train.x, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda2, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(new.train.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(new.text.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,"\n","Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }



  # Feature selection first by wL1SVM then wL2SVM (Primal Problem)
  if(svm.type == 'wl1pl2.svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        # first fit wL1SVM
        wl1p.svm.model <- wl1p.svm(x.train, y.train, PI = PI_mid[i], lambda = lambda, beta_precision = beta_precision)
        new.train.x <- wl1p.svm.model$Xp.train
        f_ind <- wl1p.svm.model$f_ind
        new.text.x <- as.matrix(x.test[,f_ind])
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wl1p.svm.model$beta
        beta0_vector[i] <- wl1p.svm.model$beta0

        # then fit wL2SVM
        wsvm.model <- wsvm(new.train.x, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(new.train.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(new.text.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,"\n","Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }



  # Directly solve beta by wL1SVM (Primal Problem)
  if(svm.type == 'wl1p.svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        wl1p.svm.model <- wl1p.svm(x.train, y.train, PI = PI_mid[i], lambda = lambda, beta_precision = beta_precision)
        f_ind <- wl1p.svm.model$f_ind
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wl1p.svm.model$beta
        beta0_vector[i] <- wl1p.svm.model$beta0

        predicted_labels_train[,i] <- wENsvm.predict(x.train, wl1p.svm.model$beta, wl1p.svm.model$beta0)
        predicted_labels_test[,i] <- wENsvm.predict(x.test, wl1p.svm.model$beta, wl1p.svm.model$beta0)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }




  # Feature selection and probability estimation by elastic net
  if(svm.type == 'wENsvm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        wENsvm.model <- wENsvm(x.train, y.train, PI = PI_mid[i], lambda1 = lambda, lambda2 = lambda2, beta_precision = beta_precision)
        f_ind <- wENsvm.model$f_ind
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wENsvm.model$beta
        beta0_vector[i] <- wENsvm.model$beta0

        predicted_labels_train[,i] <- wENsvm.predict(x.train, wENsvm.model$beta, wENsvm.model$beta0)
        predicted_labels_test[,i] <- wENsvm.predict(x.test, wENsvm.model$beta, wENsvm.model$beta0)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }


  # Feature selection and probability estimation by elastic net
  if(svm.type == 'wENp.svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        wENp.svm.model <- wENp.svm(x.train, y.train, PI = PI_mid[i], lambda1 = lambda, lambda2 = lambda2, beta_precision = beta_precision)
        f_ind <- wENp.svm.model$f_ind
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wENp.svm.model$beta
        beta0_vector[i] <- wENp.svm.model$beta0

        predicted_labels_train[,i] <- wENsvm.predict(x.train, wENp.svm.model$beta, wENp.svm.model$beta0)
        predicted_labels_test[,i] <- wENsvm.predict(x.test, wENp.svm.model$beta, wENp.svm.model$beta0)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected", "\n", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }



  # Feature selection by elastic net follow by L2SVM for robability estimation
  if(svm.type == 'wENl2svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        wENsvm.model <- wENsvm(x.train, y.train, PI = PI_mid[i], lambda1 = lambda, lambda2 = lambda2, beta_precision = beta_precision)
        new.train.x <- wENsvm.model$Xp.train
        f_ind <- wENsvm.model$f_ind
        new.text.x <- as.matrix(x.test[,f_ind])
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wENsvm.model$beta
        beta0_vector[i] <- wENsvm.model$beta0

        # then fit wL2SVM
        wsvm.model <- wsvm(new.train.x, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(new.train.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(new.text.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }



  # Feature selection by elastic net follow by L2SVM for robability estimation
  if(svm.type == 'wENpl2.svm'){
    # matrix to hold all selected features
    f_matrix <-  matrix(0, nrow = length(PI_mid), ncol = dim(x.train)[2])

    for (i in 1:length(PI_mid)){
      tryCatch({
        wENp.svm.model <- wENp.svm(x.train, y.train, PI = PI_mid[i], lambda1 = lambda, lambda2 = lambda2, beta_precision = beta_precision)
        new.train.x <- wENp.svm.model$Xp.train
        f_ind <- wENp.svm.model$f_ind
        new.text.x <- as.matrix(x.test[,f_ind])
        f_matrix[i, f_ind] = 1

        beta_matrix[i,] <- wENp.svm.model$beta
        beta0_vector[i] <- wENp.svm.model$beta0

        # then fit wL2SVM
        wsvm.model <- wsvm(new.train.x, y.train, PI = PI_mid[i], kernel = kernel, lambda = lambda, eps = eps)
        c_matrix[i,] <- wsvm.model$c
        d_vector[i] <- wsvm.model$d
        predicted_labels_train[,i] <- wsvm.predict(new.train.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
        predicted_labels_test[,i] <- wsvm.predict(new.text.x, new.train.x, wsvm.model$c, wsvm.model$d, wsvm.model$kernel)
      }, error=function(e){})
    }

    predicted_labels_train <- as.matrix(cbind(rep(1, n.train), predicted_labels_train, rep(-1, n.train)))
    predicted_labels_test <- as.matrix(cbind(rep(1, n.test), predicted_labels_test, rep(-1, n.test)))

    F_sum <- colSums(f_matrix)
    F_index <- which(F_sum != 0)
    F_num <- length(F_index)
    F_importance_index <- F_index[sort(F_sum[F_index], decreasing = TRUE, index.return = TRUE)$ix]
    F_importance_value <- F_sum[F_importance_index]
    F_importance_matrix <- as.data.frame(cbind(F_importance_index,F_importance_value))
    F_importance_matrix$F_importance_index<- factor(F_importance_matrix$F_importance_index)

    # plot feature importance map based on threshold
    if(plot.option == 'f_threshold'){
      # plot the feature based on threshold

      F_plot_importance_matrix <- F_importance_matrix[F_importance_matrix[,2] > f_threshold, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Features Importance Values greater than ", f_threshold,'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }

    if(plot.option == 'f_importance'){
      # plot the top features

      if (F_num <= num.plot.imp.features){
        num.plot.imp.features <- F_num
      }

      F_plot_importance_matrix <- F_importance_matrix[1:num.plot.imp.features, ]
      colnames(F_plot_importance_matrix)[1] <- "F_index"

      # plot feature importance map
      f_plot <- ggplot(F_plot_importance_matrix, aes(x=F_index, y=F_importance_value, fill=F_index)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0("Top ", num.plot.imp.features, " Features Importance",'\n',"(Total ", F_num, " Feature Selected, ", "Max Importance Value = ", length(PI_mid), ")"),
             x = "Feature Index", y = "Importance") +
        geom_bar(stat = "identity") +
        geom_col() +
        coord_flip()
    }


  }



  # print(predicted_labels_train)
  # print('check probability')
  estimate_prob_train <- as.vector(apply(predicted_labels_train,1,prob.estimate,PI))
  # print(estimate_prob_train)
  estimate_prob_test <- as.vector(apply(predicted_labels_test,1,prob.estimate,PI))
  # print(estimate_prob_test)

  estimate_labels_train <- estimate_prob_train
  estimate_labels_train[estimate_labels_train>=0.5] <- 1
  estimate_labels_train[estimate_labels_train<0.5] <- -1

  estimate_labels_test <- estimate_prob_test
  estimate_labels_test[estimate_labels_test>=0.5] <- 1
  estimate_labels_test[estimate_labels_test<0.5] <- -1

  train_error <- mean(estimate_labels_train!=y.train)
  test_error <- mean(estimate_labels_test!=y.test)


  if(svm.type == 'wl2svm'){
    wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                            train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda, cmat = c_matrix, dvec =d_vector, PIseries = PI)
  }

  else if(svm.type == 'wl1svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }
  }


  else if(svm.type == 'wl1dsvm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }
  }


  else if(svm.type == 'wl1pl2.svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }
  }


  else if(svm.type == 'wl1p.svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda = lambda,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }
  }


  else if(svm.type == 'wENsvm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }
  }

  else if(svm.type == 'wENl2svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }
  }

  else if(svm.type == 'wENp.svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, PIseries = PI)
    }
  }

  else if(svm.type == 'wENpl2.svm'){
    if(plot.option == 'f_threshold' || plot.option == 'f_importance'){
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              f_imp_plot = f_plot, beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }

    else {
      wsvm.prob.model <- list(estimate_prob = estimate_prob_test, estimate_label = estimate_labels_test,
                              train.error = train_error, test.error = test_error, kernel = kernel, lambda1 = lambda, lambda2 = lambda2,
                              f_matrix = f_matrix, F_importance_matrix = F_importance_matrix, features_number = F_num,
                              beta_mat = beta_matrix, beta0_vec=beta0_vector, cmat = c_matrix, dvec =d_vector, PIseries = PI)
    }
  }

  return(wsvm.prob.model)

}
