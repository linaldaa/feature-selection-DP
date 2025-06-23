import numpy as np
import pandas as pd

# Load data from CSVs
# CSVs have to be formatted with headers for easier understanding

df_features_kpis = pd.read_csv('features_kpis_table.csv')
feature_impact_matrix = df_features_kpis.iloc[:, 1:].astype(int).to_numpy()


df_hours = pd.read_csv('feature_development_hours.csv')
feature_hours = df_hours.iloc[:, 1].astype(int).to_numpy()


df_relevance = pd.read_csv('category_kpi_weights_modified.csv')
relevance_matrix = df_relevance.iloc[0:, 1:].astype(float).to_numpy()




# Parameters example
max_hours = 100
category_index = 4


def knapsack_feature_selection(relevance_matrix, feature_impact_matrix, feature_hours, max_hours, category_index):
    """
    relevance_matrix: shape (19, 20) -> KPI relevance per category
    feature_impact_matrix: shape (32, 20) -> KPI impact per feature
    feature_hours: shape (32,) -> hours per feature
    max_hours: int -> total hours constraint
    category_index: int -> which category row to use from relevance_matrix
    """

    # Compute value of each feature for the selected category
    category_weights = relevance_matrix[category_index]  # shape (20,)
    feature_values = feature_impact_matrix @ category_weights  # shape (32,)

    # Convert to int for dynamic programming (optional)
    n = len(feature_values)
    max_hours = int(max_hours)
    feature_hours = [int(h) for h in feature_hours]
    feature_values = list(feature_values)

    # Initialize DP table
    dp = [[0] * (max_hours + 1) for _ in range(n + 1)]

    # DP: Build table
    for i in range(1, n + 1):
        for w in range(max_hours + 1):
            if feature_hours[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - feature_hours[i - 1]] + feature_values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find selected features ?
    selected = []
    w = max_hours
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= feature_hours[i - 1]

    selected.reverse()
    return selected, dp[n][max_hours]




# # Inputs as numpy arrays for randomized data if no tables are there yet
# relevance_matrix = np.random.dirichlet(np.ones(20), size=19)
# feature_impact_matrix = np.random.randint(1, 26, size=(32, 20))
# feature_hours = np.random.randint(10, 50, size=32)
# max_hours = 100
# category_index = 4  

selected_features, total_impact = knapsack_feature_selection(
    relevance_matrix,
    feature_impact_matrix,
    feature_hours,
    max_hours,
    category_index
)

print("Selected feature indices:", selected_features)
print("Total weighted impact:", total_impact)


# Check feature_impact_matrix
print("Feature Impact Matrix shape:", feature_impact_matrix.shape)
print("Feature Impact Matrix preview:\n", feature_impact_matrix[:5])  # show first 5 rows

# Check feature_hours
print("Feature Hours shape:", feature_hours.shape)
print("Feature Hours preview:\n", feature_hours[:10])  # first 10 hours

# Check relevance_matrix
print("Relevance Matrix shape:", relevance_matrix.shape)
print("Relevance Matrix preview:\n", relevance_matrix[:5])  # show first 5 categories
