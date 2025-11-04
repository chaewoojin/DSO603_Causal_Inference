# Problem 2: Simulation Exercise
# Causal Forest Comparison: Adaptive (RF) vs Honest (CF)

library(grf)
library(ggplot2)
library(dplyr)

set.seed(123)  # For reproducibility

# Parameters
n <- 1000
n_train <- 600
n_test <- 400
n_trees <- 500
n_simulations <- 50

# True CATE function
true_cate <- function(x) {
  return(1 + x)
}

# Function to generate data
generate_data <- function(n) {
  X <- runif(n, 0, 1)
  W <- rbinom(n, 1, 0.5)
  epsilon <- rnorm(n, 0, 1)
  Y <- 1 + 2*X + W*(1 + X) + epsilon
  
  return(list(X = X, W = W, Y = Y))
}

# Initialize storage for results
results_rf <- matrix(NA, nrow = n_simulations, ncol = n_test)
results_cf <- matrix(NA, nrow = n_simulations, ncol = n_test)
X_test_all <- matrix(NA, nrow = n_simulations, ncol = n_test)

mse_rf <- numeric(n_simulations)
mse_cf <- numeric(n_simulations)
bias_rf <- numeric(n_simulations)
bias_cf <- numeric(n_simulations)

coverage_rates <- numeric(n_simulations)
mean_se <- numeric(n_simulations)

# Run Monte Carlo simulations
cat("Running", n_simulations, "Monte Carlo simulations...\n")

for (sim in 1:n_simulations) {
  if (sim %% 10 == 0) {
    cat("Simulation", sim, "/", n_simulations, "\n")
  }
  
  # Generate data
  data <- generate_data(n)
  
  # Split into training and testing
  train_idx <- 1:n_train
  test_idx <- (n_train + 1):n
  
  X_train <- matrix(data$X[train_idx], ncol = 1)
  W_train <- data$W[train_idx]
  Y_train <- data$Y[train_idx]
  
  X_test <- matrix(data$X[test_idx], ncol = 1)
  W_test <- data$W[test_idx]
  Y_test <- data$Y[test_idx]
  
  X_test_all[sim, ] <- X_test
  
  # Fit adaptive causal forest (honesty = FALSE)
  cf_rf <- causal_forest(
    X = X_train,
    Y = Y_train,
    W = W_train,
    num.trees = n_trees,
    honesty = FALSE
  )
  
  # Fit honest causal forest (honesty = TRUE)
  cf_honest <- causal_forest(
    X = X_train,
    Y = Y_train,
    W = W_train,
    num.trees = n_trees,
    honesty = TRUE
  )
  
  # Predict on test set
  pred_rf <- predict(cf_rf, X_test, estimate.variance = FALSE)$predictions
  pred_cf <- predict(cf_honest, X_test, estimate.variance = TRUE)
  
  results_rf[sim, ] <- pred_rf
  results_cf[sim, ] <- pred_cf$predictions
  
  # Calculate true CATE for test set
  true_effect <- true_cate(X_test)
  
  # Calculate MSE and Bias for this simulation
  mse_rf[sim] <- mean((pred_rf - true_effect)^2)
  mse_cf[sim] <- mean((pred_cf$predictions - true_effect)^2)
  
  bias_rf[sim] <- mean(pred_rf - true_effect)
  bias_cf[sim] <- mean(pred_cf$predictions - true_effect)
  
  # Calculate uncertainty metrics for honest CF
  se_estimates <- sqrt(pred_cf$variance.estimates)
  mean_se[sim] <- mean(se_estimates)
  
  # Calculate 95% coverage
  lower_bound <- pred_cf$predictions - 1.96 * se_estimates
  upper_bound <- pred_cf$predictions + 1.96 * se_estimates
  coverage <- mean((true_effect >= lower_bound) & (true_effect <= upper_bound))
  coverage_rates[sim] <- coverage
}

cat("\nSimulations complete!\n\n")

# ============================================================================
# Output 1: Predicted Effect Plot
# ============================================================================

cat("Creating predicted effect plot...\n")

# Calculate average predictions across simulations
avg_pred_rf <- colMeans(results_rf)
avg_pred_cf <- colMeans(results_cf)
avg_X_test <- colMeans(X_test_all)

# Create data frame for plotting
plot_data <- data.frame(
  X = avg_X_test,
  True_Effect = true_cate(avg_X_test),
  RF_Effect = avg_pred_rf,
  CF_Effect = avg_pred_cf
)

# Sort by X for better plotting
plot_data <- plot_data[order(plot_data$X), ]

# Create plot
p1 <- ggplot(plot_data, aes(x = X)) +
  geom_line(aes(y = True_Effect, color = "True CATE"), size = 1.2) +
  geom_line(aes(y = RF_Effect, color = "Adaptive RF"), size = 1, linetype = "dashed") +
  geom_line(aes(y = CF_Effect, color = "Honest CF"), size = 1, linetype = "dashed") +
  scale_color_manual(values = c("True CATE" = "black", 
                                  "Adaptive RF" = "red", 
                                  "Honest CF" = "blue")) +
  labs(
    title = "Treatment Effect Estimates: True vs Estimated",
    subtitle = "Averaged over 50 Monte Carlo simulations",
    x = "X",
    y = "Treatment Effect τ(x)",
    color = "Method"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave("/Users/wchae/Developer/Courseworks/DSO603/hw4/problem2_treatment_effects.png", 
       p1, width = 8, height = 6, dpi = 300)

print(p1)

# ============================================================================
# Output 2: Performance Summary Table
# ============================================================================

cat("\nComputing performance metrics...\n")

# Calculate average MSE and Bias across simulations
performance_table <- data.frame(
  Method = c("Adaptive RF (honesty=FALSE)", "Honest CF (honesty=TRUE)"),
  MSE = c(mean(mse_rf), mean(mse_cf)),
  Bias = c(mean(bias_rf), mean(bias_cf))
)

cat("\n=== Performance Summary Table ===\n")
print(performance_table, row.names = FALSE)

# Save to CSV
write.csv(performance_table, 
          "/Users/wchae/Developer/Courseworks/DSO603/hw4/problem2_performance_table.csv", 
          row.names = FALSE)

# ============================================================================
# Output 3: Uncertainty Output for Honest CF
# ============================================================================

cat("\n=== Uncertainty Analysis for Honest CF ===\n")

uncertainty_table <- data.frame(
  Metric = c("Mean Standard Error", "95% Coverage Rate"),
  Value = c(mean(mean_se), mean(coverage_rates))
)

cat("\n")
print(uncertainty_table, row.names = FALSE)

# Save to CSV
write.csv(uncertainty_table, 
          "/Users/wchae/Developer/Courseworks/DSO603/hw4/problem2_uncertainty_table.csv", 
          row.names = FALSE)

# ============================================================================
# Discussion Summary
# ============================================================================

cat("\n=== Simulation Results Discussion ===\n\n")

cat("1. Treatment Effect Estimation:\n")
cat("   - Both methods capture the linear trend τ(x) = 1 + x\n")
cat("   - The honest CF tends to be less biased but may have higher variance\n")
cat("   - The adaptive RF may overfit to training data\n\n")

cat("2. Performance Comparison (MSE and Bias):\n")
cat("   - MSE measures overall prediction accuracy\n")
cat("   - Bias measures systematic deviation from true effect\n")
cat("   - Lower values indicate better performance\n\n")

cat("3. Uncertainty Quantification:\n")
cat(sprintf("   - Mean Standard Error: %.4f\n", mean(mean_se)))
cat(sprintf("   - 95%% Coverage Rate: %.4f (%.1f%%)\n", 
            mean(coverage_rates), mean(coverage_rates)*100))
cat("   - Coverage rate near 95% indicates well-calibrated uncertainty estimates\n\n")

cat("Results saved to:\n")
cat("  - problem2_treatment_effects.png\n")
cat("  - problem2_performance_table.csv\n")
cat("  - problem2_uncertainty_table.csv\n")
