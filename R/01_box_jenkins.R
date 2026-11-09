# -----------------------------------------------------------------------------
# 01_box_jenkins.R - the original course analysis, reproducible end to end
#
# Identification -> estimation -> diagnostics -> forecast (Box-Jenkins)
# on Switzerland's imports of goods and services (% of GDP), 1970-2024.
# Every number quoted in the README is printed to output/01_box_jenkins.txt.
# -----------------------------------------------------------------------------

source("R/00_setup.R")

y <- load_series("2025-07")
sink("output/01_box_jenkins.txt", split = TRUE)

cat("Series:", start(y)[1], "-", end(y)[1], " n =", length(y), "\n")
print(summary(as.numeric(y)))

# --- 1. Look at the data -----------------------------------------------------
p_series <- autoplot(y) +
  geom_point(size = 0.9) +
  labs(title = "Switzerland: imports of goods and services",
       subtitle = "% of GDP, annual, World Bank WDI (July 2025 release)",
       x = NULL, y = "% of GDP")
save_plot(p_series, "01_series")

# --- 2. Is the level stationary? ---------------------------------------------
# Ljung-Box on the level: is there any autocorrelation at all?
print(Box.test(y, lag = 10, type = "Ljung-Box"))
# ADF: H0 = unit root (non-stationary)
print(adf.test(y))

# --- 3. First difference -----------------------------------------------------
dy <- diff(y)
print(adf.test(dy))                         # H0 rejected at 5% -> d = 1
for (k in c(5, 10, 15)) print(Box.test(dy, lag = k, type = "Ljung-Box"))
cat("ndiffs (KPSS-based):", ndiffs(y), "\n")

acf_frame <- function(x, label, max_lag = 20) {
  a <- acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]   # drop lag 0
  p <- pacf(x, lag.max = max_lag, plot = FALSE)$acf
  rbind(data.frame(series = label, fn = "ACF",  lag = seq_along(a), value = a),
        data.frame(series = label, fn = "PACF", lag = seq_along(p), value = p))
}
af <- rbind(acf_frame(y, "Level"), acf_frame(dy, "First difference"))
af$series <- factor(af$series, levels = c("Level", "First difference"))
ci <- qnorm(0.975) / sqrt(length(y))

p_acf <- ggplot(af, aes(lag, value)) +
  geom_hline(yintercept = 0, colour = "grey40") +
  geom_hline(yintercept = c(-ci, ci), linetype = "dashed", colour = "#2b6cb0") +
  geom_segment(aes(xend = lag, yend = 0), linewidth = 0.8) +
  facet_grid(series ~ fn) +
  labs(title = "Correlograms: the level decays slowly, the difference is white noise",
       x = "Lag (years)", y = NULL)
save_plot(p_acf, "02_acf_pacf", height = 5)

# --- 4. Candidate models, compared on information criteria -------------------
candidates <- list(
  "ARIMA(1,1,0)" = c(1, 1, 0),
  "ARIMA(0,1,1)" = c(0, 1, 1),
  "ARIMA(1,1,1)" = c(1, 1, 1),
  "ARIMA(0,1,0)" = c(0, 1, 0)
)
fits <- lapply(candidates, function(o) Arima(y, order = o))
ic <- data.frame(
  model = names(fits),
  k     = sapply(fits, function(f) length(coef(f)) + 1),
  AIC   = sapply(fits, AIC),
  AICc  = sapply(fits, function(f) f$aicc),
  BIC   = sapply(fits, BIC),
  row.names = NULL
)
ic <- ic[order(ic$AICc), ]
print(ic, digits = 6)
write.csv(ic, "output/01_model_comparison.csv", row.names = FALSE)

# Coefficients of the richer models are tiny and not significant
for (nm in c("ARIMA(1,1,0)", "ARIMA(0,1,1)")) {
  f <- fits[[nm]]
  cat(nm, ": coef =", round(coef(f), 4), " s.e. =",
      round(sqrt(diag(f$var.coef)), 4), "\n")
}

# Automatic search agrees
final <- auto.arima(y)
print(final)
print(accuracy(final))

# --- 5. Residual diagnostics -------------------------------------------------
res <- residuals(final)
print(Box.test(res, lag = 10, type = "Ljung-Box", fitdf = 0))
arch <- arch_lm_test(res, lags = 10)
cat(sprintf("ARCH LM test (10 lags): Chi-squared = %.4f, df = %d, p-value = %.4f\n",
            arch$statistic, arch$df, arch$p.value))
print(shapiro.test(res))

rf <- data.frame(year = as.numeric(time(res)), resid = as.numeric(res))
p_res <- ggplot(rf, aes(year, resid)) +
  geom_hline(yintercept = 0, colour = "grey40") +
  geom_col(width = 0.7, fill = "#4a5568") +
  labs(title = "ARIMA(0,1,0) residuals = year-on-year changes",
       subtitle = "No autocorrelation (Ljung-Box p = 0.68) and no ARCH effects (p = 0.74)",
       x = NULL, y = "percentage points")
save_plot(p_res, "03_residuals")

# --- 6. Forecast 2025-2035 ---------------------------------------------------
fc <- forecast(final, h = 11, level = c(80, 95))
print(fc)
write.csv(data.frame(year = as.numeric(time(fc$mean)),
                     point = as.numeric(fc$mean),
                     lo80 = fc$lower[, "80%"], hi80 = fc$upper[, "80%"],
                     lo95 = fc$lower[, "95%"], hi95 = fc$upper[, "95%"]),
          "output/01_forecast_2025_2035.csv", row.names = FALSE)

p_fc <- autoplot(fc) +
  labs(title = "Random-walk forecast, 2025-2035",
       subtitle = "Flat point forecast; 80% and 95% intervals widen with sqrt(h)",
       x = NULL, y = "% of GDP")
save_plot(p_fc, "04_forecast")

sink()
