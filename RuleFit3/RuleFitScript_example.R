#
# How-To use RuleFitScript
#

# Load the functions from RuleFitScript
source("/home/niklas/R/RuleFit3/RuleFitScript.R")

# Load someDataSet
path.to.datasets = "/tmp" #  Rstudio does not remember where the script is executed..
data.someDataSet = read.csv( paste(path.to.datasets, "someDataSet.csv", sep="/") )

# Prepare the data for RuleFit. We need to split the data into training and test sets
# and create the class definition. In this case we want to predict based on a binary
# variable in someDataSet. We can create an object holding all relevant information 
# using the prepare.data() function (each function in RuleFitScript.R is documented).
func = function(x) { x==1 }  # the classification criterium
df = prepare.data(data.someDataSet, "someVariable", func)

# Run the RuleFit algorithm (optimise the parameters if needed)
ruleData = rulefit(df$train, df$train.class, rfmode="class", tree.size=5, mod.sel=3)
rfxval(nfold=10, quiet=T)

# Get the predictions on the test set using rulefit.prediction()
# We can look at the ROC curve and select a cutoff as well.
rulefit.prediction = analyse.model(df$test, df$test.class)

plot(rulefit.prediction$perf)+abline(a=0, b=1, col="red")  # plot the ROC
head(rulefit.prediction$eval)  # select the cutoff from here

# Generate all single partial depence plots (make sure to create the folder)
generate.plots.single("/tmp/rulefit/single/", df$test)

# Generate the pairwise partial depence plots for a selected subset of columns.
# For the selected columns all possible combinations will be chosen. If you need
# the full list you can simply select all columns.
col.names.pair = c("someVariable", "someVariable1", "someVariable2", "etc")
generate.plots.double("/tmp/rulefit/pairwise/", df$test, col.names.pair)

# Finally have a look at the rules. If you have python installed you can generate
# barplots for each rule automatically. In order to have correct scaling of the
# plot you need to export the range of all variables in the original data.frame
# via dump.range()
dump.range(data.someDataSet)
rules()
# rules(x=subset(df$test[ df$test$is_mobile_user==1, ]))  # if you want to plot the
                                                          # rules for a subset
plot.rules("/tmp/rulefit/ruleplots")  # WARNING: this will delete all content in the folder!!

