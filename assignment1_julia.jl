using CSV, DataFrames, Dates, LinearAlgebra, Statistics, Plots

# Step 1: Load the dataset
println("Loading dataset...")
df = CSV.read("current.csv", DataFrame)  # Read CSV file into a DataFrame
df.sasdate = Date.(df.sasdate, "yyyy-mm-dd")  # Convert 'sasdate' column to Date format

# Step 2: Remove the first row containing transformation codes
df_cleaned = df[2:end, :]
println("Dataset loaded and cleaned successfully.")

# Step 3: Define parameters
num_lags = 4   # Number of lag periods (p)
num_leads = 1  # Forecast horizon (h)

# Step 4: Define target variable and predictor variables
target = "INDPRO"
predictors = ["CPIAUCSL", "TB3MS"]

# Step 5: Generate lagged features for regression
function create_lagged_features(df, target, predictors, num_lags)
    X = DataFrame()
    
    # Create lagged versions of the target variable
    for lag in 0:num_lags
        X["$(target)_lag$(lag)"] = shift(df[:, target], lag)
    end
    
    # Create lagged versions of predictor variables
    for predictor in predictors
        for lag in 0:num_lags
            X["$(predictor)_lag$(lag)"] = shift(df[:, predictor], lag)
        end
    end
    
    # Add intercept term
    X[:, "Intercept"] .= 1.0  
    return dropmissing(X)  # Remove rows with missing values due to lagging
end

println("Creating lagged features...")
X = create_lagged_features(df_cleaned, target, predictors, num_lags)
println("Lagged features created successfully.")

# Step 6: Create response variable (y), shifting target variable forward by forecast horizon
y = shift(df_cleaned[:, target], -num_leads)
y = y[num_lags+1:end]  # Align response variable with feature matrix
X = X[num_lags+1:end, :]  # Ensure feature matrix is aligned

# Step 7: Estimate model parameters using Ordinary Least Squares (OLS)
println("Estimating model parameters using OLS...")
beta_hat = (X' * X) \ (X' * y)  # Compute OLS estimate
println("Model estimation complete.")

# Step 8: Generate forecast for next period
y_pred = X[end, :] * beta_hat
println("Forecast complete: ", y_pred)

# Step 9: Save forecast results to CSV
println("Saving forecast results to results.csv...")
CSV.write("results.csv", DataFrame(:Forecast => y_pred))
println("Results saved successfully.")




