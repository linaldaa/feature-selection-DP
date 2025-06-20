import numpy as np

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




# Inputs as numpy arrays
relevance_matrix = np.random.dirichlet(np.ones(20), size=19)
feature_impact_matrix = np.random.randint(1, 26, size=(32, 20))
feature_hours = np.random.randint(10, 50, size=32)
max_hours = 100
category_index = 4  

selected_features, total_impact = knapsack_feature_selection(
    relevance_matrix,
    feature_impact_matrix,
    feature_hours,
    max_hours,
    category_index
)

print("Selected feature indices:", selected_features)
print("Total weighted impact:", total_impact)