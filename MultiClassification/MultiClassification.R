#
# Multi-classification version 0.1
#
# Author: Niklas Berliner (niklas.berliner@gmail.com)
#
# Collection of functions to perform multi-classification. By default
# one-vs-all classification is used, however, any form of classification
# can be performed by specifiying the model matrix M. Please refer to
# the wikipedia article for a brief description of one-vs-all:
# https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

require(randomForest)
require(parallel)
require(dplyr)

split.data = function(df, class.column, split.ratio=0.66, use.seed=123456) {
  #
  # Split a data.frame into test and training set.
  #
  # Args:
  #   df:           The data.frame containing the data
  #
  #   class.column: Name of the column containing the class labels
  #
  #   split.ratio:  The ratio between training and testing dataset
  #                 Default is set to have 66% of the data in the training set
  #
  #   use.seed:     The seed to use for the random splitting of the data
  #                 into test and training set.
  #
  # Returns:
  #   DataClass:  Classification Data Container. An object with the four elements,
  #                 "train.data",  i.e. the training data without class label
  #                 "test.data",   i.e. the test data without class label
  #                 "train.class", i.e the class labels for the training dataset
  #                 "test.class",  i.e the class labels for the test dataset
  
  # Initialise the Data Container Object
  DataClass = setRefClass("Classification Data Container", 
                          fields = list( train.data="data.frame", 
                                         test.data="data.frame", 
                                         train.class="matrix", 
                                         test.class="matrix"
                         ) 
  )
  data.object = DataClass$new()
  
  # Get the index of the class.column
  class.column.idx = grep(class.column, colnames(df))
  
  if ( floor(split.ratio*nrow(df)) == nrow(df) ) {
    # Get the test and training set
    data.object$train.data = df[, -c(class.column.idx)]
    data.object$test.data  = data.frame(c())
    
    # Get the corresponding classes
    data.object$train.class = matrix(df[, c(class.column.idx)])
    data.object$test.class  = matrix(c(0))
  }
  else {
    # Get random indexes to split the data into test and training set
    set.seed(use.seed)
    train_idx = sample(c(1:nrow(df)), floor(split.ratio*nrow(df)))
    test_idx  = c(1:nrow(df))[-train_idx]
    
    # Get the test and training set
    data.object$train.data = df[train_idx, -c(class.column.idx)]
    data.object$test.data  = df[test_idx,  -c(class.column.idx)]
    
    # Get the corresponding classes
    data.object$train.class = matrix(df[train_idx, c(class.column.idx)])
    data.object$test.class  = matrix(df[test_idx,  c(class.column.idx)])
  }
  
  return( data.object )
}


rf.multiclass = function(df, class.column, split.ratio=0.66, use.seed=123456) {
  #
  # Data container for multi-classification.
  #
  # The input data will be split into test and training set, and further prepared
  # for use in the multi-classification approach one-vs-all. This data container 
  # can then be used to train and test models using multi-classification.
  #
  # Note: The classification matrix M will be automaticall generated as one.vs.all
  #       but can be overwriten if required. For each column one binary model will be
  #       build. The number of rows must match the number of the original class labels.
  #       Class labels from the original data can be excluded for one specific
  #       model by setting the corresponging value to -1 in the classification matrix.
  #
  # Args:
  #   df:           The data.frame containing the data
  #
  #   class.column: Name of the column containing the class labels
  #
  #   split.ratio:  The ratio between training and testing dataset
  #                 Default is set to have 66% of the data in the training set
  #
  #   use.seed:     The seed to use for the random splitting of the data
  #                 into test and training set.
  # 
  # Returns:
  #   DataClass:  Classification Data Container. An object with the elements,
  #
  #                 "train.data":    the training data without class label
  #
  #                 "test.data":     test data without class label
  #
  #                 "train.class":   the (multi)class labels for the train dataset
  #
  #                 "test.class":    the (multi)class labels for the test dataset
  #
  #                 "M":             the one.vs.all matrix
  #
  #                 "class.labels":  a vector containing the unique class labels of 
  #                                  the input data
  #
  #                 "models":        a list holding the trained models for each 
  #                                  "sub-classification"
  #
  
  # Initialise the Data Container Object
  MultiClassData = setRefClass("Multi-classification Data Container", 
                               fields = list( train.data="data.frame", 
                                              test.data="data.frame", 
                                              train.class="matrix", 
                                              test.class="matrix",
                                              M="matrix",
                                              class.labels="vector",
                                              models="list"
                                            ) 
                               )
  data.object = MultiClassData$new()
  
  ## Split the data
  df.object = split.data(df, class.column, split.ratio=split.ratio)
  
  # Get the original class labels
  class.labels = c(as.character(unique(df[,class.column])))
  
  # Set the entries of the data.object
  data.object$M                 = one.vs.all( length(class.labels) )
  data.object$class.labels      = class.labels
  data.object$train.data        = df.object$train.data
  data.object$test.data         = df.object$test.data
  data.object$train.class       = df.object$train.class
  data.object$test.class        = df.object$test.class
  
  return(data.object)
}


one.vs.all = function(nr.classes) {
  #
  # Generate the one-vs-all matrix for nr.classes
  #
  # Args:
  #   nr.classes:  Number of classes in the original dataset
  #
  # Returns:
  #   M:  Matrix holding the binary classifications as columns
  #
  #
  # The idea behind the one-vs-all approach is to try for each
  # class to separate it from the others, i.e. the background.
  # Assuming three classes, the matrix would like
  #
  #  M = | 1   0   0|
  #      | 0   1   0|
  #      | 0   0   1|
  #
  # where each row represents one class and each column
  # represents one run of the model to separate one class
  # from all other classes.
  
  M = matrix(data = 0, nrow = nr.classes, ncol = nr.classes)
  M = M + diag(nrow = nr.classes, ncol = nr.classes)
  return(M)
}


rf.set.classes = function(train.class, class.labels) {
  #
  # Create the one.vs.all class definitions using 1 and 0.
  #
  # Based on the original classes this function will
  # generate a list containing binary class definitions.
  # Following the one.vs.all approach a new class vector
  # is created for every class against all the remaining
  # classes.
  #
  # Args:
  #   train.class: data.frame containing the (multi-)class definition
  #
  # Returns:
  #   List containing the new class definitions in data.frames

  # Generate the dataset for each column
  result = list()
  for (i in 1:length(class.labels)) {
    # Assign the data.frame to the list (i.e. make a copy of it)
    result[[i]] = train.class
    
    # Change the class assignments to make it one vs all
    multi.idx = result[[i]] == class.labels[i]
    result[[i]][  multi.idx ] = 1
    result[[i]][ !multi.idx ] = 0
  }
  
  return(result)
}




rf.build.model = function(data.object, ntree, mtry, nodesize, sampsize=NULL) {
  #
  # Build the randomForest model for each one.vs.all binary class
  # definition in the multi-classification data.object.
  #
  # Args:
  #   data.object:  A "Multi-classification Data Container" (see rf.multiclass)
  # 
  #   ntree:        Number of trees used in randomForest
  #
  #   mtry:         Number of variables that are sampled when building the trees
  #
  #   nodesize:     Final nodesize of each tree in randomForest
  #
  #   sampsize:     Vector of boolean indicating if balanced sampling should be
  #                 be used. The position in the vetor corresponds to the models
  #                 stored in the columns of the M matrix, i.e. sampsize[1]
  #                 corresponds to M[,1]. Must have the same length as M has columns.
  #                 Can be used to model highly unbalanced models as fully balanced. 
  #                 Note however that many data points of the more abundant class 
  #                 will be discarded for training.
  #
  # Returns:
  #   The input data.object containing the randomForests in "models"
  #
  
  run.rf = function(i, data.object, ntree, mtry, nodesize, sampsize) {
    
    # Get the row index of the data to consider in the model
    M            = data.object$M[,i]                                        # the current model that should be build
    keep.classes = data.object$class.labels[ ifelse( M >= 0, TRUE, FALSE) ] # check if some classes should be ignored
    idx          = data.object$train.class %in% keep.classes                # the indices of the rows to keep
    
    # Select the training data and class definitions
    df.data  = data.object$train.data[ idx, ]
    df.class = data.object$train.class[ idx, ]
    
    # Select the class that should be predicted as 1 and convert to binary
    this.class = data.object$class.labels[ which(M == 1) ]
    df.class[ df.class %in% this.class ] = 1 # consider the case that multiple classes are put together in one pot
    df.class[ df.class != 1 ] = 0
    
    
    ## Build the model
    # Check if the classes should be sampled to avoid class imbalances
    if ( !is.null(sampsize) && sampsize[i] ) {
      # Set the sample size of each class to the occurence of the smaller class
      if ( mean(as.numeric(df.class)) >= 0.5 ) {
        nr = nrow(df.data) * (1-mean(as.numeric(df.class))) - 1
      }
      else {
        nr = nrow(df.data) * mean(as.numeric(df.class)) -1
      }
      # Call randomForest with sampsize parameter
      rf = randomForest(x=df.data, y=factor(df.class, c("1", "0"), labels=c("One", "Zero")), ntree=ntree, mtry=mtry, nodesize=nodesize, sampsize=c(nr,nr))
    }
    else {
      # Call randomForest without sampsize parameter
      rf = randomForest(x=df.data, y=factor(df.class, c("1", "0"), labels=c("One", "Zero")), ntree=ntree, mtry=mtry, nodesize=nodesize)
    }
    rf
    
    return( list(rf, i) )
  }
  
  # Build each model
  result = mclapply(1:ncol(data.object$M), run.rf, data.object, ntree, mtry, nodesize, sampsize)
  
  # Add the models to the data object
  data.object$models = result
  
  return(data.object)
}


rf.predict.multi = function(data.object, df.data=NULL, wt=NULL) {
  #
  # Predict multi-classification models.
  #
  # Given a M matrix that specifies the models used for prediction
  # and the models itself in the data.object container, the class
  # of new data can be predicted. The class minimising the quadratic
  # error loss of all models simultaneously is selected as winning class.
  #
  # Args:
  #   data.object: A "Multi-classification Data Container" (see rf.multiclass)
  #
  #   df.data:     data.frame to predict. If NULL then the test data in
  #                data.object is used.
  #
  #   wt:          Weights given to the individual models specified in the
  #                model matrix M. If NULL no weights are given. Weights must
  #                be specified in a vector of length equal to the number of
  #                columns in the model matrix M. The order must match the
  #                model matrix M.
  #
  # Returns:
  #   A data.frame containing the multi-class labels
  #
  
  predict.mc = function(i, models, df.data) {
    # Get the predictions of each model
    rf  = models[[i]][[1]]
    idx = models[[i]][[2]]
    result = predict(rf, df.data, type="prob")
    result = data.frame(result)$One # the factor label was set to "One"
    return( result )
  }
  
  indices = function(i) {
    paste("M", i, sep='.')
  }
  
  predict.multi = function(df, wt) {
    # Separate the two data.frames again
    split    = (ncol(df)-1) / 2 # we have one "Index" columns
    df.pred  = df[,1:split+1]   # ignore the "Index" column
    df.class = df[,(split+2):ncol(df)]
    
    # Get the loss
    df.loss = quadratic.loss(df.pred, df.class, wt)
    #df.loss = one.zero.loss(df.pred, df.class, wt)
    
    return( which.min(df.loss) )
  }
  
  # If no data was specified, predict the test data
  if ( is.null(df.data) ) {
    df.data = data.object$test.data
  }
  
  # Some variable extraction (make the reference shorter)
  models = data.object$models
  
  # Get the predictions of all models
  predictions = mclapply(1:length(models), predict.mc, models, df.data)
  predictions = data.frame(predictions)
  predictions = data.frame(1:nrow(predictions), predictions) # add an index column for cross joining
  colnames(predictions) = c("Index", lapply(1:length(models), indices), recursive=TRUE)
  
  # Now we need to find for each data point which class it should be assigned to
  tmp = merge(x=predictions, y=data.object$M, by=NULL)
  col.names = colnames(tmp)[2:ncol(tmp)]
  result = tmp %>% group_by(Index) %>% do(class_id_list=predict.multi(., wt)) %>% summarise(class_id_idx=mean(class_id_list))
  
  # Lastly we can map the integer class label back to the original class labels
  result$class_id     = data.object$class.labels[result$class_id_idx]
  result$class_id_idx = NULL
  
  return(result)
}

quadratic.loss = function(prediction, class, wt) {
  #
  # Calculate the quadratic loss for multi-classification.
  #
  # Args:
  #   prediction: data.frame containing the predictions for each class. The
  #               number of rows must be equal to the number of classes in
  #               the multi-classification problem. The number of columns
  #               must be equal to the number of models chosen for classification.
  #
  #   class:      data.frame containing the model matrix M. Models (i.e. columns)
  #               with -1 will not consider the respective class and will not
  #               contribute to the loss of that class.
  #
  #   wt:         (optional) weights given to the individual models. If NULL
  #               then no weights are given. Must be a vector with length equal
  #               to the number of columns of M.
  # Returns:
  #   The loss associated to each class of the multi-classification problem.
  #
  M    = ncol(class)
  diff = class-prediction
  if ( !is.null(wt) ) { # apply the weights
    stopifnot( ncol(diff) == length(wt) )
    for ( ii in 1:ncol(diff) ) {
      diff[,ii] = diff[,ii] * wt[ii]
    }
  }
  if ( length( which(class == -1) > 0 ) ) {
    diff[ class == -1 ] = 0 # set the value to zero
  }
  
  loss = rowSums( diff*diff/M )
  return(loss)
}

one.zero.loss = function(prediction, class, wt) {
  #
  # Calculate the zero-one loss for multi-classification.
  #
  # Please refer to the quadratic.loss function description for further details.
  #
  diff = class-prediction
  if ( !is.null(wt) ) {
    stopifnot( ncol(diff) == length(wt) )
    for ( ii in 1:ncol(diff) ) {
      diff[,ii] = diff[,ii] * wt[ii]
    }
  }
  if ( length( which(class == -1) > 0 ) ) {
    diff[ class == -1 ] = 0 # set the value to zero
  }
  loss = rowSums( abs(diff) )
  return(loss)
}



rf.accuracy = function(prediction, true.class) {
  #
  # Calculate the accuracy of the prediction.
  #
  # Args:
  #   prediction: data.frame containing the predicted classes.
  #
  #   true.class: data.frame containing the true class labels.
  # 
  # Returns:
  #   Accuracy of the prediction in percent.
  #
  acc = prediction == true.class
  return( mean(acc, na.rm=TRUE) )
}

rf.confusion.matrix = function(prediction, true.class) {
  #
  # Calculate the confusion matrix of the multi-classification problem.
  #
  # Args:
  #   prediction: data.frame containing the predicted classes.
  #
  #   true.class: data.frame containing the true class labels.
  # 
  # Returns:
  #   A list containing (1) data.frame of the confusion matrix
  #                     (2) overall accuracy of the model
  
  # Get the original class labels
  class.labels = c(as.character(unique(true.class)))
  
  # Calculate the accuracy
  acc = rf.accuracy(prediction, true.class)
  
  # Calculate the confusion matrix
  confusion.matrix = matrix(nrow=length(class.labels), ncol=(length(class.labels)+1))
  for ( ii in 1:length(class.labels) ) {  # let ii denote the row
    for (jj in 1:length(class.labels) ) { # then jj denotes the column
      
      idx.class = true.class == class.labels[ii] # only look at relevant entries
      idx.pred  = prediction == class.labels[jj]
      
      confusion.matrix[ii,jj] = sum(idx.pred & idx.class)
    }
  }
  
  # Calculate the accuracy of each class
  row.sums = rowSums(confusion.matrix, na.rm=TRUE)
  acc.idx  = ncol(confusion.matrix)
  for ( ii in 1:length(class.labels) ) {
    confusion.matrix[ii,acc.idx] = round(confusion.matrix[ii,ii] / row.sums[ii], digits=2)
  }
  
  # Convert the matrix to a data.frame and add column and row labels
  confusion.matrix = data.frame(confusion.matrix)
  colnames(confusion.matrix) = c(class.labels, "Accuracy in %")
  rownames(confusion.matrix) = class.labels
  
  return( list(confusion.matrix, acc) )
}


rf.cross.validate = function(data.container, Ms, ntrees, mtrys, nodesizes, sampsizes, wts) {
  #
  # Cross validate different paramters to obtain optimal accuracy.
  #
  # The model can either be optimised for accuracy alone or it can be optimised
  # for a balanced confusion matrix. Input data must be lists (of equal length)
  # containing the parameters whose performance should be reported. For each
  # paramter set the accuracy and the confusion matrix are returned in the same
  # order than the input paramters.
  #
  # Args:
  #   data.container: Classification Data Container as returned from rf.multiclass()
  #
  #   Ms:             List of model matrices M
  #
  #   ntrees:         List of ntree (see randomForest)
  #
  #   mtrys:          List of mtry (see randomForest)
  #
  #   nodesizes:      List of nodesize (see randomForest)
  #
  #   sampsizes:      List of boolean indicating if the classes should be sampled equally
  #
  #   wts:            List of weigths given to each column of the corresponding
  #                   model matrix M. If NULL no weighting is performed. Must be a
  #                   vector of length ncol(M) containing the weights otherwise.
  #
  # Returns:
  #   List containing tuples of accuracy and confusion matrix for each paramter combination.
  #
  
  accuracy  = c()
  confusion = c()
  for ( i in 1:length(Ms) ) {
    # Set the current model matrix M and select the parameters
    data.container$M = Ms[[i]]
    ntree            = ntrees[[i]]
    mtry             = mtrys[[i]]
    nodesize         = nodesizes[[i]]
    sampsize         = sampsizes[[i]]
    wt               = wts[[i]]
    
    # Calculate the actual model
    rf.model = rf.build.model(data.container, ntree, mtry, nodesize, sampsize=sampsize)
    
    # Get the predictions
    rf.test = rf.predict.multi(rf.model, data.container$test.data, wt=wt)
    
    # Check the accuracy
    acc = rf.accuracy(rf.test, data.container$test.class)
    con = rf.confusion.matrix(rf.test, data.container$test.class)
    accuracy  = c(accuracy, acc)
    confusion = c(confusion, con)
  }
  return( list(accuracy, confusion) )
}





