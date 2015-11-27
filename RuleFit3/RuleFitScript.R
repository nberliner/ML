#
# RuleFitScript version 0.1
#
# Author: Niklas Berliner (niklas.berliner@gmail.com)
#
# Collection of function to aid in using the RuleFit3 package (see
# http://statweb.stanford.edu/~jhf/r-rulefit/rulefit3/RuleFit_help.html )
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

platform = "linux"
rfhome   = "/home/niklas/R/RuleFit3/"
source(paste(rfhome, "rulefit.r", sep="/"))
library(akima, lib.loc=rfhome)
library(ROCR)

prepare.data = function(df, class.column, cond, split.ratio=0.66, use.seed=123456) {
  #
  # Split the data in training and test set.
  #
  # This will automatically split the data set into two parts corresponding to
  # the size ratio specified. It furthermore creates the class necessary for
  # RuleFit, i.e. based on a condition creates the classes -1 and +1.
  #
  # Args:
  #   df:           Input data.frame which will be split into test and training set.
  #
  #   class.column: Column name that should be used to define the classes (must be string)
  #
  #   cond:         Condition to use for class definition. Must be a function that returns
  #                 TRUE or FALSE based on the desired classification.
  #                 Example: if the classification column is already binary one can use
  #                          cone = function(x) { x==1 }
  #
  #   split.ratio:  The ratio for splitting the dataset into test and training (in percent)
  #                 The input data.frame will be randomly divided.
  #
  #   use.seed:     The the seed for the random splitting of the dataset.
  #
  # Returns:
  #   result:  RuleFit3 Data Container Object with four fields.
  #              train.data:  The training data.frame (without the class definition)
  #              train.class: The class definition of the taining dataset
  #              test.data:   The test data.frame (without the class definition)
  #              test.class:  The class definition of the test dataset
  
  # Initialise the Data Container Object
  RuleFitData = setRefClass("RuleFit3 Data Container", fields = list( train.data="data.frame", 
                                                                      test.data="data.frame", 
                                                                      train.class="matrix", 
                                                                      test.class="matrix") )
  data.object = RuleFitData$new()
  
  # Get the index of the class.column
  class.column.idx = grep(class.column, colnames(df))
  
  # Add the class based on the cutoff provided
  df[,c(class.column.idx)] = ifelse( cond(df[c(class.column.idx)]), 1, -1 )
  
  # Do not split the data if the split.ratio does not allow it
  if ( floor(split.ratio*nrow(df)) == nrow(df) ) { # don't split
    # Get the test and training set
    data.object$train.data           = data.frame( df[, -c(class.column.idx)] )
    colnames(data.object$train.data) = colnames(df)[-c(class.column.idx)]
    data.object$test.data            = data.frame(c())
    
    # Add empty train data
    data.object$train.class = matrix(df[, c(class.column.idx)])
    data.object$test.class  = matrix(c(0))
  }
  else { # do split
    # Get random indexes to split the data into test and training set
    set.seed(use.seed)
    train_idx = sample(c(1:nrow(df)), floor(split.ratio*nrow(df)))
    test_idx  = c(1:nrow(df))[-train_idx]
    
    # Get the test and training set
    data.object$train.data           = data.frame(df[train_idx, -c(class.column.idx)])
    colnames(data.object$train.data) = colnames(df)[-c(class.column.idx)]
    data.object$test.data            = data.frame(df[test_idx,  -c(class.column.idx)])
    colnames(data.object$test.data)  = colnames(df)[-c(class.column.idx)]
    
    # Get the corresponding classes
    data.object$train.class = matrix(df[train_idx, c(class.column.idx)])
    data.object$test.class  = matrix(df[test_idx,  c(class.column.idx)])
  }
  
  return(data.object)
}


analyse.model = function(test.data, test.class) {
  #
  # Analyse the RuleFit3 model and calculate the True Positve and False Negative rates
  #
  # Take the RuleFit3 model in the current workspace and predict the class of the
  # test.data. Calculate the false postives, true positives, true negatives, false 
  # negatives, true positive rate, and false positive rate for all possible
  # classification cutoffs.
  #
  # Args:
  #   test.data:   data.frame containing the test data
  #   test.class:  Classification of the test data.frame
  #
  # Result:
  #   result:   RuleFit3 Model Analysis Container Object with three fields.
  #
  #               eval:  data.frame with 7 columns:
  #                        (1) cutoffs:  The classification cutoff (above class 1,
  #                                      below class -1)
  #                        (2) fp:       False positives
  #                        (3) tp:       True positives
  #                        (4) tn:       True negatives
  #                        (5) fn:       False negatives
  #                        (6) tpr:      True positive rate
  #                        (7) fpr:      False positve rate
  #
  #               pred:  Prediction object returned by prediction() from ROCR
  #
  #               perf:  Performance object returned by performance() from ROCR
  
  # Initialise the Data Container Object
  RuleFitData = setRefClass("RuleFit3 Model Analysis Container", fields = list( pred="prediction", eval="data.frame", perf="performance", prediction="numeric") )
  result = RuleFitData$new()
  
  # Get the predictions of the test set
  yp = rfpred( test.data )
  yp = 1.0/(1.0+exp(-yp))
  
  # Calculate the data for the ROC curve
  pred   = prediction(yp, test.class)
  eval = data.frame(pred@cutoffs, pred@fp, pred@tp, pred@tn, pred@fn)
  colnames(eval) = c('cutoffs','fp','tp','tn','fn')
  eval$accuracy = (eval$tp + eval$tn) / (eval$tp + eval$fp + eval$tn + eval$fn)
  eval$tpr = eval$tp/(eval$tp+eval$fn)
  eval$fpr = eval$fp/(eval$fp+eval$tn)
  
  # Set the result fields
  result$prediction = yp
  result$eval = eval
  result$pred = pred
  result$perf = performance(pred, measure="tpr", x.measure="fpr")
  
  return(result)
}



rulefit.rules = function(rulesout.hlp=paste(rfhome, "rulesout.hlp", sep="/")) {
  #
  # Read the rulesout.hlp file generated by RuleFit3
  #
  # The rules specified in the rulesout.hlp are read and stored into
  # small class containers. Each rule will have the following fields
  #
  #    name        The number of the rule
  #    support     The support of the rule
  #    coeff       The coefficient
  #    importance  The importance
  #    rules       A data.frame containing the actual rule definitions.
  #                It has three columns, "feature", "range_min", "range_max"
  #                corresponding to the fields in the rulesout.hlp file.
  #
  # Args:
  #   rulesout.hlp (string):  Path to the rulesout.hlp file generated by RuleFit3
  #
  # Returns:
  #   A list containing the rule objects as described above.
  #
  
  
  # Initialise the RuleFit Rule Container Object
  RuleFitRule = setRefClass("RuleFit3 Rule", fields = list( name="numeric", 
                                                            support="numeric", 
                                                            coeff="numeric", 
                                                            importance="numeric",
                                                            rules="data.frame") )
  
  # Define a function to remove leadning and trailing ':' from strings
  strip = function (x) gsub("^\\:+|\\:+$", "", x) # credit to: http://stackoverflow.com/a/2261149
  
  # Read the rules file
  rules.string = scan(rulesout.hlp, character(0), skip=3, sep="\n")
  
  rules.list = c() # stores the output
  for ( i in 1:length(rules.string) ) {
    # Read the current line
    current.line = unlist( strsplit(rules.string[i], " ") )
    current.line = current.line[ which( current.line != "" ) ] # remove empty fields
    
    # Check what to do with this line
    if ( current.line[1] == "Rule") { # a new rule
      # Sanity check
      stopifnot( current.line[ length(current.line) ] == 'variables' )
      
      # If not the first rule, add it to the result list
      if ( i != 1 ) {
        newRule$rules = newRule$rules[ -c(1), ]
        rules.list    = c(rules.list, newRule)
      }
      
      # Initialise a new rule container object
      newRule      = RuleFitRule()
      newRule$name = as.numeric( strip(current.line[2]) )
      START        = TRUE # we start with a new rule
      
    }
    else if (START) { # first line after the new rule containing the support
      
      # Do some sanity checks
      stopifnot( current.line[1]      == 'support' )
      stopifnot( current.line[2]      == '=' )
      stopifnot( current.line[4]      == 'coeff' )
      stopifnot( current.line[5]      == '=' )
      stopifnot( current.line[7]      == 'importance' )
      stopifnot( current.line[8]      == '=' )
      stopifnot( length(current.line) == 9 )
      
      # Extract the support, coefficient, and support
      newRule$support    = as.numeric(current.line[3])
      newRule$coeff      = as.numeric(current.line[6])
      newRule$importance = as.numeric(current.line[9])
      
      # Create an empty rules data.frame
      newRule$rules           = data.frame( c(0), c(0), c(0) )
      colnames(newRule$rules) = c("feature", "range_min", "range_max")
      
      START = FALSE # now we deal with the actual rules
    }
    else if ( current.line[1] == "print_rules:" ) { # This can appear if no rule was found
      next
    }
    else { # the actual rule definitions
      # Some sanity checks
      stopifnot( current.line[2]      == 'range' )
      stopifnot( current.line[3]      == '=' )
      stopifnot( length(current.line) == 5 )
      
      # Put the rule into the data.frame
      new.data           = data.frame( c(strip(current.line[1])), c(as.numeric(current.line[4])), c(as.numeric(current.line[5])) )
      colnames(new.data) = c("feature", "range_min", "range_max")
      newRule$rules      = rbind( newRule$rules, new.data )
    }
  }
  
  # Add the last rule to the result list
  newRule$rules = newRule$rules[ -c(1), ]
  rules.list    = c(rules.list, newRule)
  
  # Done, return the collected results
  return(rules.list)
}



generate.plots.single = function(folder, df) {
  #
  # Generate single partial dependence plots.
  #
  # Args:
  #   folder:  Output folder. The partial dependence plots will be saved here
  #
  #   df:      data.frame for which to calculate the partial depence plots,
  #            i.e. the column names are extracted and taken as input.
  # 
  for ( col.name in colnames(df)) {
    # Assmble the filename
    fname        = paste(folder, "partial_dependence", sep='/')
    fname.base   = paste(fname, col.name, sep="_")
    fname.single = paste(fname.base, ".png", sep="")
    
    # Plot the partial dependence plot
    png(filename=fname.single)
    idx = grep(col.name, colnames(df))
    singleplot(idx)
    dev.off()
  }
}

generate.plots.double = function(folder, df, col.names, nval=FALSE) {
  #
  # Generate pair partial dependence plots of all combinations in col.names
  #
  # Args:
  #   folder:    Output folder. The partial dependence plots will be saved here
  #
  #   df:        data.frame for which to calculate the partial depence plots,
  #              i.e. the column names are extracted and used to determine their index.
  # 
  #   col.names: List of column names for which the pairplots should be calculated.
  #              All possible combinations will be generated and plotted.
  #
  #   nval:      Maximum number of evaluation points (see RuleFit3 manual). If set to
  #              FALSE and abolute (optimal) maximum is calculated for each plot.
  
  # Quick sanity check if all specified col.names are found
  for ( col.name in col.names) {
    if ( !(col.name %in% colnames(df)) ) {
      stop( sprintf("Variable %s not found!", col.name) )
    }
  }
  
  # Generate the pairplots
  for ( i in 1:(length(col.names)-1) ) {
    for ( j in (i+1):length(col.names) ) {
      # Create the filenames
      fname        = paste(folder,     "partial_dependence", sep='/')
      fname.base   = paste(fname,      col.names[i], sep="_")
      fname.base   = paste(fname.base, col.names[j], sep="_AND_")
      fname.double = paste(fname.base, ".png", sep="")
      
      # Get the column indexes
      idx1 = grep(col.names[i], colnames(df))
      idx2 = grep(col.names[j], colnames(df))
      
      # Estimate a good, i.e. pretty much maximum value for nval
      if ( !nval ) {
        points.x = range(df[idx1])
        points.y = range(df[idx2])
        nval.auto = (points.x[2] - points.x[1]) * (points.y[2] - points.y[1])
      }
      else {
        nval.auto = nval
      }

      # Plot the partial dependence plot
      png(filename=fname.double)
      pairplot(idx1, idx2, nval=nval.auto)
      dev.off()
    }
    
  }
}

dump.range = function(df, fname="/tmp/ruleFitRange.dat") {
  #
  # Save the range of the data.frame variables. Use in combination with plot.rules()
  #
  # Args:
  #   df:    data.frame to use for getting the variable range
  #
  #   fname: Filename to write the output to
  #
  
  if ( !(class(df) == "data.frame") ) {
    stop("Input data type not understood. Not dumping the range!")
  }
  
  f = file(fname)
  sink(f)
  for ( col.name in colnames(df) ) {
    str = sprintf( "%s,%f,%f", col.name, max(df[col.name]), min(df[col.name]) )
    print(str)
  }
  sink()
  close(f)
}

plot.rules = function(outputDir, fname.rules=FALSE, fname.values="/tmp/ruleFitRange.dat", fname.program=FALSE) {
  #
  # Plot RuleFit3 rules using an external python script.
  #
  # Note: This will delete the entire outputDir!!
  #
  # Plot the each rule in a barplot using an external python script. Requires rules() and
  # dump.range() to be run. The blue region in the barplots indicate the selected region,
  # red indicates the ignored range.
  #
  # Args:
  #   outputDir: Output directory to save the plots
  #
  #   fname.rules: If the rulesout.hlp file is not kept in the rfhome folder it can
  #                be specified here (must include the filename).
  #
  #   fname.values: The filename of the dump.range() output.
  #
  #   fname.program: If not placed in the rfhome directory the external python script
  #                  can be specified here.
  # 
  
  # Check the fname.rules and fname.program input
  if ( fname.rules == FALSE ) {
    fname.rules = paste(rfhome, "rulesout.hlp", sep="/")
  }
  if ( fname.program == FALSE ) {
    fname.program = paste(rfhome, "RuleFitRules.py", sep="/")
  }
  
  # Run the python script
  cmd = sprintf("python %s -b %s -r %s -v %s", fname.program, outputDir, fname.rules, fname.values)
  system(cmd)
}



