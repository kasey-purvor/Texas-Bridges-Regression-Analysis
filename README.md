### Project Title: US Department of Transport(USDT) - Bridge Condition Analysis

**[See The Full Report](report/report.md)**


### Project Description

This project investigates the condition of bridges in Texas using a dataset provided by the US Department of Transport (USDT). The goal is to develop a model to predict bridge condition based on five predictor variables: **Age, AverageDailyTraffic, Trucks_percent, Material, and Design**.  

The project explores the following questions:

* How well do the proposed variables predict bridge condition?
* Which variables are most influential in determining bridge condition?

### Findings

**Data Preprocessing:**

* **Target variable engineering:**  The three original target variables (Deck\_rating, Superstr\_rating, and Substr\_rating) were combined to create a single 'Condition' variable. This simplification aimed to enhance signal, reduce redundancy, and mitigate multicollinearity issues.
* **Category consolidation:**  Material and Design categories with low frequencies were grouped into an 'Other' category to reduce noise and simplify analysis.
* **Outlier Handling:**  Outliers in Age, AverageDaily, and Trucks\_percent were addressed to improve model robustness.

**Exploratory Data Analysis:**

* **Age** was identified as a strong predictor of bridge condition, showing a negative correlation. This suggests that older bridges tend to be in poorer condition.
* **Material** and **Design** showed some influence on condition. Concrete bridges generally exhibited better conditions than those made of steel or other materials. The 'Slab' design was associated with slightly worse conditions.
* **AverageDaily** and **Trucks\_percent** showed negligible influence on bridge condition, suggesting traffic load may not be a primary driver of deterioration within the dataset.

**Regression Analysis:**

* A linear regression model was developed to predict bridge condition.
* The model achieved an R<sup>2</sup> of 0.452, meaning it explains 45.2% of the variance in bridge condition.
* **Age** was confirmed as the most influential predictor, followed by **Material** and **Design**.
* Residual analysis indicated a near-normal distribution, suggesting the model's errors are generally random. However, there was a tendency to underestimate the condition of bridges at the lower end of the scale.

### Conclusion

The linear regression model provides a reasonable starting point for predicting bridge condition. However, its predictive accuracy could be enhanced by incorporating additional variables, exploring non-linear relationships, and further addressing outliers. 

### Potential Improvements

* Incorporate interaction terms to capture the combined effects of predictors.
* Consider non-linear relationships to improve model fit.
* Include additional predictor variables to account for unexplained variance.
* Further investigate and address outliers.

### Data Source

The data used in this project is sourced from the US Department of Transport (USDT) National Bridge Inventory (NBI) database.

### Repository Contents

* **README.md:**  This file.
* **CW2_Kasey_purvor.ipynb:**  Jupyter Notebook containing the complete analysis code and findings.
* **tx19_bridges_sample.csv:**  Dataset used in the analysis (if applicable).
* **report.md:** a markdown version of the pre-run-notebook to allow viewing in github. 



