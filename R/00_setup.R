# -----------------------------------------------------------------------------
# 00_setup.R - packages, data loading and small helpers shared by all scripts
# -----------------------------------------------------------------------------

required <- c("forecast", "tseries", "ggplot2", "svglite", "scales")
missing  <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Install missing packages first: install.packages(c(",
       paste0('"', missing, '"', collapse = ", "), "))")
}

suppressPackageStartupMessages({
  library(forecast)
  library(tseries)
  library(ggplot2)
})

dir.create("figures", showWarnings = FALSE)
dir.create("output",  showWarnings = FALSE)

# World Bank WDI indicator NE.IMP.GNFS.ZS - Imports of goods and services (% of GDP)
# Two vintages are kept on purpose (see README, "Data revisions"):
#   2025-07: the release the original course analysis was run on (1970-2024)
#   2026-07: a later release with revised history and the first 2025 value
load_series <- function(vintage = c("2025-07", "2026-07")) {
  vintage <- match.arg(vintage)
  path <- file.path("data", sprintf("che_imports_pct_gdp_wdi_%s.csv", vintage))
  d <- read.csv(path)
  d <- d[!is.na(d$imports_pct_gdp), ]
  ts(d$imports_pct_gdp, start = min(d$year), frequency = 1)
}

# Engle's ARCH LM test: regress e_t^2 on its own `lags` lags, LM = n * R^2.
# Same statistic as FinTS::ArchTest(x, lags, demean = FALSE).
arch_lm_test <- function(x, lags = 10) {
  x  <- as.numeric(na.omit(x))
  e2 <- x^2
  m  <- embed(e2, lags + 1)
  fit <- lm(m[, 1] ~ m[, -1])
  stat <- nrow(m) * summary(fit)$r.squared
  list(statistic = stat, df = lags,
       p.value = pchisq(stat, df = lags, lower.tail = FALSE))
}

save_plot <- function(plot, name, width = 8, height = 4.2) {
  ggsave(file.path("figures", paste0(name, ".svg")), plot,
         width = width, height = height, device = svglite::svglite, bg = "white")
  invisible(plot)
}

theme_set(theme_minimal(base_size = 11) +
            theme(plot.title = element_text(face = "bold"),
                  panel.grid.minor = element_blank(),
                  plot.background = element_rect(fill = "white", colour = NA)))
