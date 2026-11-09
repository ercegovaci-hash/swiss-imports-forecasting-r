# -----------------------------------------------------------------------------
# 03_reality_check.R - the forecast meets new data
#
# The model was built on the World Bank release of July 2025 (data to 2024).
# The July 2026 release adds 2025 and also revises the history. This script
#   1. measures how much the past was revised, and
#   2. checks the published 2025 value against the 2025 forecast.
# -----------------------------------------------------------------------------

source("R/00_setup.R")

old <- load_series("2025-07")    # 1970-2024, used for the model
new <- load_series("2026-07")    # 1970-2025, revised

sink("output/03_reality_check.txt", split = TRUE)

# --- 1. Data revisions -------------------------------------------------------
common <- window(new, end = end(old)[1])
rev <- common - old
cat("Revision (new - old), percentage points\n")
cat(sprintf("  mean absolute revision 1970-2024: %.2f\n", mean(abs(rev))))
cat(sprintf("  mean absolute revision 2015-2024: %.2f\n",
            mean(abs(window(rev, start = 2015)))))
print(round(window(cbind(old = old, new = common, revision = rev), start = 2018), 2))

# --- 2. The 2025 forecast vs the published 2025 value -------------------------
actual_2025 <- as.numeric(window(new, start = 2025))

fc_old <- forecast(Arima(old, order = c(0, 1, 0)), h = 1, level = c(80, 95))
# Same model, refit on the revised history up to 2024 (fair comparison)
fc_new <- forecast(Arima(window(new, end = 2024), order = c(0, 1, 0)),
                   h = 1, level = c(80, 95))

report <- function(fc, label) {
  sigma <- sqrt(fc$model$sigma2)
  z <- (actual_2025 - fc$mean[1]) / sigma
  cat(sprintf("\n%s\n  point %.2f | 80%% [%.2f, %.2f] | 95%% [%.2f, %.2f]\n",
              label, fc$mean[1], fc$lower[1, 1], fc$upper[1, 1],
              fc$lower[1, 2], fc$upper[1, 2]))
  cat(sprintf("  actual 2025 = %.2f -> error %+.2f pp = %.1f residual standard deviations\n",
              actual_2025, actual_2025 - fc$mean[1], z))
}
report(fc_old, "Forecast from the July 2025 release (original model)")
report(fc_new, "Same model refit on the July 2026 release up to 2024")

# How unusual is the 2024 -> 2025 move compared with all earlier years?
d <- diff(new)
cat(sprintf("\nChange 2024->2025: %+.2f pp. Largest earlier yearly change: %+.2f pp (%d).\n",
            tail(d, 1), d[which.max(abs(head(d, -1)))],
            as.integer(time(d)[which.max(abs(head(d, -1)))])))
sink()

# --- Figure ------------------------------------------------------------------
df <- rbind(
  data.frame(year = as.numeric(time(old)), value = as.numeric(old),
             release = "July 2025 release (model input)"),
  data.frame(year = as.numeric(time(new)), value = as.numeric(new),
             release = "July 2026 release (revised + 2025)")
)
df <- df[df$year >= 2000, ]
fcd <- data.frame(year = 2025, point = fc_old$mean[1],
                  lo80 = fc_old$lower[1, 1], hi80 = fc_old$upper[1, 1],
                  lo95 = fc_old$lower[1, 2], hi95 = fc_old$upper[1, 2])

p <- ggplot() +
  geom_linerange(data = fcd, aes(x = year, ymin = lo95, ymax = hi95),
                 linewidth = 6, colour = "#bee3f8") +
  geom_linerange(data = fcd, aes(x = year, ymin = lo80, ymax = hi80),
                 linewidth = 6, colour = "#63b3ed") +
  geom_point(data = fcd, aes(year, point), shape = 95, size = 9, colour = "#2b6cb0") +
  geom_line(data = df, aes(year, value, colour = release, linetype = release), linewidth = 0.8) +
  geom_point(data = df[df$year == 2025, ], aes(year, value), colour = "#c53030", size = 2.5) +
  annotate("text", x = 2024.6, y = actual_2025, label = "published 2025",
           hjust = 1, size = 3.3, colour = "#c53030") +
  scale_colour_manual(values = c("#1a202c", "#c53030")) +
  labs(title = "Reality check: the 2025 value landed outside the 95% interval",
       subtitle = "Blue bar = 2025 forecast (80% dark, 95% light) from the July 2025 data",
       x = NULL, y = "% of GDP", colour = NULL, linetype = NULL) +
  theme(legend.position = "bottom")
save_plot(p, "06_reality_check", height = 4.6)
