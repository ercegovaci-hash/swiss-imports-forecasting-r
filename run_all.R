# Reproduce every table, number and figure in the README.
# Run from the repository root:  Rscript run_all.R
source("R/01_box_jenkins.R")
source("R/02_backtest.R")
source("R/03_reality_check.R")
cat("\nDone. Results in output/, figures in figures/.\n")
