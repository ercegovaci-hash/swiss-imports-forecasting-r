# -----------------------------------------------------------------------------
# 02_backtest.R - does the chosen model actually forecast well?
#
# AIC only says which model fits the past best. To judge forecasting skill,
# every model is re-estimated on an expanding window and scored on years it
# has NOT seen (rolling-origin evaluation, the time-series version of
# cross-validation). No information from after the forecast origin is used.
# -----------------------------------------------------------------------------

source("R/00_setup.R")

y <- load_series("2025-07")
H <- 5                       # forecast horizons 1..5 years
first_origin <- 1994         # first training window: 1970-1994 (25 years)
origins <- first_origin:(end(y)[1] - 1)

# Each model returns point forecasts and (where available) 80/95% intervals
models <- list(
  "Naive (random walk)" = function(x, h) forecast(Arima(x, order = c(0, 1, 0)), h = h),
  "Random walk + drift" = function(x, h) rwf(x, h = h, drift = TRUE),
  "Historical mean"     = function(x, h) meanf(x, h = h),
  "auto.arima"          = function(x, h) forecast(auto.arima(x), h = h),
  "ETS"                 = function(x, h) forecast(ets(x), h = h),
  "Theta"               = function(x, h) thetaf(x, h = h),
  "NNAR (neural net)"   = function(x, h) {
    set.seed(42)
    forecast(nnetar(x, repeats = 20), h = h)
  }
)

rows <- list()
for (o in origins) {
  train <- window(y, end = o)
  for (m in names(models)) {
    fc <- models[[m]](train, H)
    for (h in seq_len(H)) {
      target_year <- o + h
      if (target_year > end(y)[1]) next
      actual <- as.numeric(window(y, start = target_year, end = target_year))
      has_pi <- !is.null(fc$lower) && ncol(as.matrix(fc$lower)) >= 2
      rows[[length(rows) + 1]] <- data.frame(
        model = m, origin = o, h = h, year = target_year,
        actual = actual, point = as.numeric(fc$mean)[h],
        in80 = if (has_pi) actual >= fc$lower[h, 1] & actual <= fc$upper[h, 1] else NA,
        in95 = if (has_pi) actual >= fc$lower[h, 2] & actual <= fc$upper[h, 2] else NA
      )
    }
  }
}
bt <- do.call(rbind, rows)
bt$error <- bt$actual - bt$point
write.csv(bt, "output/02_backtest_forecasts.csv", row.names = FALSE)

sink("output/02_backtest.txt", split = TRUE)
cat("Rolling-origin evaluation: origins", min(origins), "-", max(origins),
    "| horizons 1 -", H, "| forecasts scored:", nrow(bt), "\n\n")

# --- Accuracy by model and horizon ------------------------------------------
acc <- aggregate(cbind(abs_err = abs(error), sq_err = error^2) ~ model + h,
                 data = bt, FUN = mean)
acc$MAE  <- acc$abs_err
acc$RMSE <- sqrt(acc$sq_err)
acc <- acc[, c("model", "h", "MAE", "RMSE")]

naive_mae <- acc[acc$model == "Naive (random walk)", c("h", "MAE")]
names(naive_mae)[2] <- "naive_MAE"
acc <- merge(acc, naive_mae, by = "h")
acc$rel_MAE <- acc$MAE / acc$naive_MAE     # < 1 means better than naive
acc$naive_MAE <- NULL
acc <- acc[order(acc$h, acc$MAE), ]
write.csv(acc, "output/02_accuracy_by_horizon.csv", row.names = FALSE)

overall <- aggregate(cbind(MAE = abs(error), MSE = error^2) ~ model, data = bt, FUN = mean)
overall$RMSE <- sqrt(overall$MSE); overall$MSE <- NULL
cov <- aggregate(cbind(cover80 = in80, cover95 = in95) ~ model, data = bt,
                 FUN = mean, na.action = na.pass)
overall <- merge(overall, cov, by = "model", all.x = TRUE)
overall <- overall[order(overall$MAE), ]
cat("Overall (all horizons pooled):\n")
print(overall, digits = 3, row.names = FALSE)
write.csv(overall, "output/02_accuracy_overall.csv", row.names = FALSE)

cat("\nMAE by horizon (percentage points):\n")
wide <- reshape(acc[, c("model", "h", "MAE")], idvar = "model",
                timevar = "h", direction = "wide")
names(wide) <- sub("MAE.", "h=", names(wide), fixed = TRUE)
print(wide[order(wide[["h=1"]]), ], digits = 3, row.names = FALSE)

# --- Is the best model significantly better than naive? ---------------------
# Diebold-Mariano test on 1-step-ahead errors (same target years for both)
e_naive <- bt[bt$model == "Naive (random walk)" & bt$h == 1, ]
for (m in setdiff(names(models), "Naive (random walk)")) {
  e_m <- bt[bt$model == m & bt$h == 1, ]
  stopifnot(identical(e_m$year, e_naive$year))
  d <- dm.test(e_m$error, e_naive$error, h = 1, power = 2)
  cat(sprintf("DM test, %-20s vs naive (h=1): DM = %6.3f, p = %.3f\n",
              m, d$statistic, d$p.value))
}

# --- Why does auto.arima lose to the plain random walk? ---------------------
# Look at what it selected at each origin: with short samples the unit-root
# test does not reject stationarity, so it picks mean-reverting models.
picked <- data.frame(origin = origins,
                     model = sapply(origins, function(o)
                       as.character(auto.arima(window(y, end = o)))))
cat("\nauto.arima selection by forecast origin:\n")
print(aggregate(origin ~ model, data = picked,
                FUN = function(o) paste(range(o), collapse = "-")), row.names = FALSE)
print(table(picked$model))
write.csv(picked, "output/02_auto_arima_selection.csv", row.names = FALSE)
sink()

# --- Figure ------------------------------------------------------------------
# The historical mean (MAE 9-11 pp) is left out of the chart so the others stay readable.
plot_acc <- acc[acc$model != "Historical mean", ]
h5 <- plot_acc[plot_acc$h == H, ]
plot_acc$model <- factor(plot_acc$model, levels = h5$model[order(-h5$MAE)])
palette <- c("Naive (random walk)" = "#1a202c", "Random walk + drift" = "#2b6cb0",
             "auto.arima" = "#38a169", "ETS" = "#d69e2e", "Theta" = "#805ad5",
             "NNAR (neural net)" = "#e53e3e")
p_bt <- ggplot(plot_acc, aes(h, MAE, colour = model)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
  scale_colour_manual(values = palette) +
  expand_limits(y = 0) +
  labs(title = "Out-of-sample error by forecast horizon",
       subtitle = sprintf("Expanding-window backtest, forecast origins %d-%d (historical mean not shown: MAE 9-11)",
                          min(origins), max(origins)),
       x = "Horizon (years ahead)", y = "MAE (percentage points)", colour = NULL)
save_plot(p_bt, "05_backtest_mae")
