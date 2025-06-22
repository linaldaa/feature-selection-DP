# Feature Selection for Webshop Optimization (Thesis Project)

This repository contains the implementation of a feature prioritization model developed as part of a master's thesis. The project focuses on helping webshop stakeholders make **data-driven decisions** when selecting features to implement under **limited development hours**, based on their **estimated performance impact**.

---

##  Objective

The goal is to support strategic feature planning using a **0-1 knapsack optimization algorithm**, where each feature has:
- **Cost**: estimated development time in hours
- **Value**: weighted impact on KPIs (Key Performance Indicators) relevant to a webshop category

The model identifies the subset of features that **maximizes total KPI impact** without exceeding a given resource constraint (e.g. 100 development hours).

---

##  Methodology

The project uses a **dynamic programming** approach to solve the knapsack problem. It takes the following inputs:

- `relevance_matrix` (`19 x 20`): importance of each KPI across 19 webshop categories  
- `feature_impact_matrix` (`32 x 20`): how much each of 32 features affects each KPI  
- `feature_hours` (`32,`): estimated development time for each feature  
- `max_hours`: the total number of hours available for development  
- `category_index`: selects which webshop category the optimization should be run for

###  How It Works
1. Calculates **feature values** by computing the dot product between:
   - KPI relevance scores of the selected category  
   - Feature impact scores on KPIs  
2. Builds a **dynamic programming table** to find the maximum total value without exceeding the time budget.
3. **Backtracks** to retrieve the optimal set of features.

---


