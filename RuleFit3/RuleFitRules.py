# -*- coding: utf-8 -*-
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
"""
RuleFitRules version 0.1

Author: Niklas Berliner (niklas.berliner@gmail.com)

Visualise rules from the RuleFit3 package (see http://statweb.stanford.edu/~jhf/r-rulefit/rulefit3/RuleFit_help.html )

"""
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Rule(object):
    
    def __init__(self, ruleNr):
        
        self.ruleNr  = int(ruleNr)
        self.rules   = None
        self.names   = list()
    
    def __repr__(self):
        return "Rule %d: %s" %(self.ruleNr, ', '.join(self.names))
    
    def add(self, line):
        if line.strip() == "":
            pass
        elif line[5:12] == "support":
            parameters       = [ item for item in line.split(' ') if item != '' and item != '=' ]
            self.support     = float(parameters[1])
            self.coefficient = float(parameters[3])
            self.importance  = float(parameters[5])
        else:
            rule  = [ item.strip(':') for item in line.split(' ') if item !='' and item != '=' ]
            variable  = rule[0]
            lower     = float(rule[2])
            upper     = float(rule[3])
            
            self.names.append(variable)
            d = {'Rule'       : self.ruleNr      ,\
                 'Variable'   : variable         ,\
                 'Lower'      : lower            ,\
                 'Upper'      : upper            ,\
                 'Coefficient': self.coefficient ,\
                 'Support'    : self.support     ,\
                 }
            df = pd.DataFrame(d, index=[1])
                     
            if self.rules is None:
                self.rules = df
            else:
                self.rules = pd.concat([self.rules, df])
        
        return


def readRuleFit3Rules(fname):
    with open(fname) as f:
        # Skip the first lines
        for i in range(3):
            f.readline()
            
        rules = list()
        for line in f:
            if line[:4] == "Rule":
                try:
                    rules.append(rule)
                except NameError:
                    pass
                ruleNr = line[5:8].strip()
                rule = Rule(ruleNr)
            else:
                rule.add(line)
        
        # Append the last rule
        rules.append(rule)
    
    return rules
    
def ZeroToOne(x, stretch=1.0):
    return 1. / (1. + np.exp(-(1./float(stretch))*x))

def readRange(fname):
    result = dict()
    with open(fname) as f:
        for line in f:
            name, maxValue, minValue = line[4:-1].strip('"').split(',')
            result[name] = (float(minValue), float(maxValue))
    return result

if __name__ == "__main__":
    
    ##################################
    #                                #
    #  Read the command line option  #
    #                                #
    ##################################
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--baseDir", help="Output directory")
    parser.add_argument("-r", "--rules", help="rulesout.hlp file with RuleFit3 rules")
    parser.add_argument("-v", "--values", help="File containing the variable ranges of the original data.frame")
    args = parser.parse_args()

    if args.baseDir is None:
        print('baseDir not specified.\nExiting.\n')
        sys.exit()
    else:
        baseDir = args.baseDir
        
    if args.rules is None:
        print("rulesout.hlp not found.\Exiting.\n")
        sys.exit()
    else:
        fnameRules = args.rules
    
    if args.values is None:
        print("rulesout.hlp not found.\Exiting.\n")
        sys.exit()
    else:
        fnameValues = args.values
    

    #####################
    #                   #
    #  Run the program  #
    #                   #
    #####################

    # Check the output folder (delete content if existend, create if not)
    if os.path.isdir(baseDir):
        shutil.rmtree(baseDir)
    os.mkdir(baseDir)
    
    # Read the rules from RuleFit
    rules = readRuleFit3Rules(fnameRules)
    
    # Read the min and max readings for each variable
    valueRange = readRange(fnameValues)
    
    # Convert the rules into a DataFrame
    d = [ item.rules for item in rules ]
    df = pd.concat(d)
    
    # Create the figure for each rule
    for ruleNr in sorted(list(set(df.Rule))):
        # Create the filename
        fname = baseDir + "/rulePlot_rule%2.d.png" %(ruleNr)
    
        # Select the rule
        df1 = df[ df.Rule == ruleNr ]
        
        # Set up the figure, dynamically allocate the size and layout
        nrows = np.ceil(len(df1.Variable) / 2.)
        fig = plt.figure(figsize=(6,3.5*nrows))
#        fig.suptitle('Rule %2.d' %(int(ruleNr)+1), fontsize=14, fontweight="bold")
        
        # Plot each split in the rule
        for i, item in enumerate(df1.Variable):
            l = df1[ df1.Variable == item ]["Lower"]
            u = df1[ df1.Variable == item ]["Upper"]
            
            ax = fig.add_subplot(nrows,2,i+1, axisbg='lightgrey')
            ax.grid(b=True, color='w', linewidth=0.8, linestyle='-', zorder=0)
            ax.set_xlim([-0.2,1.5])
            ax.set_ylim(valueRange[item])
            ax.set_xticks([])
            ax.set_title(item)
        
            ax.vlines(0.2,l,u, lw=30, color="darkblue", zorder=10)
            ax.vlines(0.2,valueRange[item][0],l, lw=30, color="darkred", zorder=10)
            ax.vlines(0.2,u,valueRange[item][1], lw=30, color="darkred", zorder=10)
            
            
            if float(l) < valueRange[item][0]:
                lp = valueRange[item][0]
            else:
                lp = l
            if float(u) > valueRange[item][1]:
                up = valueRange[item][1]
            else:
                up = u
            

            ax.text(.65, 0.9, 'Coefficient:\n%.3f' %np.asarray(df1.Coefficient)[0],
                    horizontalalignment='center',
                    fontweight="bold",
                    fontsize=12,
                    backgroundcolor="lightgrey",
                    verticalalignment='top',
                    transform=ax.transAxes,
                    zorder=10)
            
            ax.text(.65, 0.7, 'Support:\n%.3f' %np.asarray(df1.Support)[0],
                    horizontalalignment='center',
                    fontweight="bold",
                    fontsize=12,
                    backgroundcolor="lightgrey",
                    verticalalignment='top',
                    transform=ax.transAxes,
                    zorder=10)
            
            ax.text(.65, 0.4, 'Upper bound:\n%.2f' %up,
                    horizontalalignment='center',
                    fontweight="bold",
                    fontsize=12,
                    backgroundcolor="lightgrey",
                    verticalalignment='top',
                    transform=ax.transAxes,
                    zorder=10)
                    
            ax.text(.65, 0.2, 'Lower bound:\n%.2f' %lp,
                    horizontalalignment='center',
                    fontweight="bold",
                    fontsize=12,
                    backgroundcolor="lightgrey",
                    verticalalignment='top',
                    transform=ax.transAxes,
                    zorder=10)

        fig.tight_layout()
        fig.savefig(fname, dpi=120)
    