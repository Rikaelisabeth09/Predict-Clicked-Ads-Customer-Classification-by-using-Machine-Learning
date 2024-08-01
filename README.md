# Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning
In this project, we aim to develop a machine learning model to classify customers based on their likelihood of clicking on advertisements. The primary goal is to predict which customers are more likely to engage with ads, thereby optimizing marketing strategies and improving ad targeting.


**Table of Contents**
- [Problem Statement](#problem-statement)
    - [Project Overview](#project-overview)
    - [Goal](#goal)
    - [Objective](#objective)
- [Data Preparation](#data-preparation)
    - [Insight](#insight)
    - [Statistical Summary](#statistical-summary)
- [Data Exploration](#data-exploration)
    - [Univariate Analysis](#univariate-analysis)
    - [Bivariate Analysis](#bivariate-analysis)
    - [Multivariate Analysis](#multivariate-analysis)
- [Preprocessing](#data-preprocessing)
- [Modeling And Evaluation](#modeling-and-evaluation)
    - [Modelling Before Standardization](#modelling-before-standardization)
    - [Modelling afer Standardization](#modelling-afer-standardization)
    - [Feature Importance](#feature-importance)
    - [Confusion Matrix](#confusion-matrix)
- [Business Recomendation And Simulation](#business-recomendations-and-simulation)
    - [Business Simulation](#business-simulation)
    - [Business Recommendation](#business-recommendation)


## Problem Statement

## Project Overview
“A company in Indonesia wants to determine the effectiveness of an advertisement they have
aired. This is important for the company to understand the reach of the marketed advertisement
so that it can attract customers to view the ad. By processing historical advertisement data and
discovering insights and patterns, this can help the company in determining their marketing
targets. The focus of this case is to create a machine learning classification model that functions
to identify the right target customers. ”

## Problem
The business team needs to refine their digital advertising strategies to maximize user engagement with their product while minimizing advertising costs.

## Goal
Make target marketing effective by using machine learning so that it can increase the click-through rate (CTR) and reduce costs incurred.

## Objective
- Predict which users are likely to click on ads. 
- Gain insight into probable trends of consumers who click on ads.
- Based on the study and model results, make business recommendations.

## Data
The data to be used is ``Clicked Ads Dataset.csv.`` The data has nine features with one target, the following is the variable information used:  

Variable Information:

| Column   | Descriptioan |
|-----------|--------------|
|Daily Time Spent on Site| : Length of stay at a site (daily) in minutes|
|Age | : User's age in years|
|Area Income |: User income in rupiah units|
|Daily Internet Usage | : Daily internet usage in minutes|
|Male | : Gender user|
|Timestamp | : When a user visits a site|
|Clicked on Ad | : Clicking on ads or not|
|city | : City of origin of the user|
|province | : Province of origin of the user| 

# Data Preparation
- Handling missing values
- Handling Duplicated Data
- Check the type and consistency of values
- Handling outliers or unusual data (anomalies)

### Insight 
Based on dataframe information, here is a summary of the key observations:
1. The dataset had 1000 rows and 11 columns
2. The dataset has 6 categoric and 5 numeric features.
3. There are missing values in `Daily Time Spent on Site`, `Area Income`, `Daily Internet Usage`, and `Gender`.


### Statistical Summary
Based on statistical summary of the Numerical columns and Categorical Columns , here is a summary of the key observations:
Numerical Columns:

1. Most features are normally distributed.
2.` Daily Time Spent on Site`: The minimum time spent on the site is 32.60 minutes, while the maximum is 91.43 minutes.
3. `Age`: The minimum customer age is 19 years old, with an average age of 36 years old and a maximum age of 61 years old.

Categorical Columns:

1. `Gender`: The majority of customers are women, with a frequency of 518.
2. `City`: Surabaya is the top city of residence for customers.
3. `Province`: The province with the most customers has a frequency of 253.
4. `Category`: The most clicked ad category is Otomotif, with a frequency of 112.
5. `Clicked on Ad`: The distribution of customers who clicked on an ad is normal.


# Data Exploration 

### Univariate analysis
![Numericals](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/nums%20ua.png)

### Insight : 
The numerical data exhibits skewed distributions, indicating a concentration of users in the lower range of the variables, with the exception of `Daily Internet Usage`, which shows a bi-modal distribution.

![Categorical](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/cats.png)

### Insight : 
This plot shows the distribution of numerical and categorical features:
Categorical Features:

1. `City`: The most common city is Surabaya, with 64 occurrences.
Province: The most common province is Daerah Khusus Ibukota Jakarta, with 253 occurrences.
2. `Category`: The most common category is Otomotif, with 112 occurrences.
3. `Gender`: There are two categories: Perempuan (female) and Laki-Laki (male). The dataset seems to be fairly balanced, with 518 females and 479 males.
4. `Clicked on Ad`: There are two categories: No and Yes. The dataset is balanced with 500 individuals not clicking on the ad, and 500 clicking on the ad.
5. The city and category distributions seem to be skewed, with a few dominant categories (Surabaya and Otomotif, respectively) and a long tail of less frequent categories.
6. The province distribution is also skewed, with Daerah Khusus Ibukota Jakarta being the dominant category.
7. The gender and clicked on ad distributions are balanced, with no category dominating the others.


### Bivariate Analysis
![Bivariatae Analysis](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/cats%20ua.png)


### Insight : 
1. `Gender` : Females are more likely to click on ads compared to males, indicating a higher engagement with advertisements among women.

2. `Province` :
    *   Jawa Barat takes the lead, with residents more likely to click on ads than those in other provinces.
    *   Bandung stands out as a city with a higher ad click-through rate compared to other cities.
    *   Daerah Khusus Ibukota Jakarta he click-through rate for ads in Jakarta is comparable to other provinces, indicating average performance in ad engagement.

![Bivariate Analysis](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/raw/main/bi.png)


### Insight  : 
Key Inferences from Bivariate Analysis
1. `Area Income`:
    *   The highest density of customers who clicked on ads is at an area income of around le8.
    *   The highest density of customers who did not click on ads is at an area income of around le9
    *   This suggests that customers who click on ads are more likely to have a lower area income, while those who do not click on ads are more likely to have a higher area income.

2. `Daily Time Spent on Site`:

    *   The highest density of customers who clicked on ads is at a daily time spent on site of around 40 minutes.
    *   he highest density of customers who did not click on ads is at a daily time spent on site of around 80 minutes.
    *   This suggests that customers who click on ads are more likely to spend around 40 minutes on the site, while those who do not click on ads are more likely to spend around 80 minutes on the site.

3. `Daily Internet Usage`:

    *   The highest density of customers who clicked on ads is at a daily internet usage of around 115 minutes.
    *  The highest density of customers who did not click on ads is at a daily internet usage of around 210 minutes.
    *   This suggests that customers who click on ads are more likely to spend around 115 minutes or less on the internet daily, while those who do not click on ads are more likely to spend around 210 minutes or more.

4. `Age`:
    *  Younger customers tend to click on ads less frequently than older customers.
    *  This indicates that older customers are more likely to engage with ads compared to younger customers.

5. `Income`:

    *  Customers with higher incomes tend to click on ads less frequently than customers with lower incomes.
    *  This suggests that lower-income customers are more likely to engage with ads compared to higher-income customers.


## Multivariate analysis 

![Multivariate Analysis](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/heatmap%20correlation.png)


### Insight : 
1. `Daily Time Spent on Site` and `Daily Internet Usage` have a moderate positive correlation (0.52), indicating that people who spend more time on the site tend to use the internet more often.
2. `Daily Time Spent on Site` and `Age` have a moderate negative correlation (-0.33), suggesting that older people tend to spend less time on the site.
4. `Daily Time Spent on Site` and `Area Income` have a strong negative correlation (-0.8), indicating that people from areas with higher income tend to spend less time on the site.
5. `Age` and `Area Income` have a moderate negative correlation (-0.4), suggesting that older people tend to come from areas with lower income.
6. `Area Income` and `Daily Internet Usage` have a moderate positive correlation (0.34), indicating that people from areas with higher income tend to use the internet more often.
6. `Age` and `Daily Internet Usage` have a moderate negative correlation (-0.37), suggesting that older people tend to use the internet less often.

# Data PreProcessing 
### Handle missing value
After handling the missing values, all columns now have 0 missing values, indicating that the dataset is complete and ready for further analysis or modeling.
 
## Feature Selection 
![Ch2](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/Chi2.png)
The p-values in the table are actually very small (e.g., 5.4e-46), which indicates that the null hypothesis of independence can be rejected for most of the variable pairs. This suggests that the variables are statistically dependent.

In particular, the p-values suggest that:
  *   `Week` and `Clicked on Ad` are highly dependent
  *   `Gender` and `Age Segment` are highly dependent
  *   `Province` and `Category` are highly dependent
  *   `City` and `Age Segment` are highly dependent

And the small p-values indicate that the variables are not independent, and there is a significant association between them.


![Anova](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/anova.png)
Based on the ANOVA test results, we can infer the following:
1. `Area Income` is a significant feature: This means that the income of the area where the individual lives has a significant effect on the outcome variable. In other words, the income of the area is likely to influence the outcome, and this relationship is not due to chance (p-value < 0.05).

2. `Age` is a significant feature: This means that the age of the individual has a significant effect on the outcome variable. In other words, the age of the individual is likely to influence the outcome, and this relationship is not due to chance (p-value < 0.05).

3. `Hour` is a significant feature: This means that the hour of the day when the individual uses the site has a significant effect on the outcome variable. In other words, the hour of the day is likely to influence the outcome, and this relationship is not due to chance (p-value < 0.05).

4. `Daily Time Spent on` Site is a significant feature: This means that the amount of time the individual spends on the site daily has a significant effect on the outcome variable. In other words, the daily time spent on the site is likely to influence the outcome, and this relationship is not due to chance (p-value < 0.05).

5. `Day`, `Month`, `Minute`, ` Unnamed: 0`, and `Daily Internet Usage` are not significant features: This means that these features do not have a significant effect on the outcome variable. In other words, these features are not likely to influence the outcome, and any observed relationships are likely due to chance (p-value ≥ 0.05).

# Modeling And Evaluation 
## Modelling Before Standardization
![Train Test](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/Training%20Test.png)
![ROC Curve](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/ROC%20Curve.png)


### Insight :
1. **Logistic Regression**
    *   AUC score: 0.77
    *   Ranking: Worst performing model
    *   Performance: Fairly good, but not as good as the other models
    *   Summary: Logistic Regression has a relatively lower AUC score, indicating it may not be the best choice for this classification task.
2. **Decision Tree Classifier**
    *   AUC Score: 0.93
    *   Ranking: Third best performing model
    *   Performance: Good, able to distinguish between positive and negative classes with some accuracy
    *   Summary: Decision Tree Classifier has a good AUC score, indicating it can distinguish between classes fairly well.
3. **Random Forest Classifier**
    *   AUC Score: 0.98
    *   Ranking: Second best performing model
    *   Performance: Excellent, able to distinguish between positive and negative classes with high accuracy
    *   Summary: Random Forest Classifier has a very high AUC score, indicating it can distinguish between classes with high accuracy.
4. **XGB Classifier**
    *   AUC Score: 0.98
    *   Ranking: Best performing model
    *   Performance: Excellent, able to distinguish between positive and negative classes with high accuracy
    *   Summary: XGB Classifier has the highest AUC score, indicating it is the best performing model for this classification task.

**Overall Summary** :
All four algorithms have good performance, with AUC scores above 0.77, indicating their ability to distinguish between positive and negative classes with some accuracy. However, XGB Classifier and Random Forest Classifier stand out as the top performers.

## Modelling afer Standardization
![Train Test](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/Training%20Test2.png)
![ROC Curve](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/ROC%20Curve2.png)
 ### Insight :
 Here is a summary of the chart in point form:

1. **Logistic Regression**
    *   AUC score: 0.98
    *   Ranking: Top performing model (tied)
    *   Performance:  Excellent, able to distinguish between positive and negative classes with high accuracy
    *   Summary: Logistic Regression has a very high AUC score, indicating it can distinguish between classes with high accuracy.
2. **Decision Tree Classifier**
    *   AUC Score: 0.93
    *   Ranking: Third best performing model
    *   Performance: Good, able to distinguish between positive and negative classes with some accuracy
    *   Summary: Decision Tree Classifier has a good AUC score, indicating it can distinguish between classes fairly well.
3. **Random Forest Classifier**
    *   AUC Score: 0.98
    *   Ranking: Top performing model (tied)
    *   Performance: Excellent, able to distinguish between positive and negative classes with high accuracy
    *   Summary: Random Forest Classifier has a very high AUC score, indicating it can distinguish between classes with high accuracy.
4. **XGB Classifier**
    *   AUC Score: 0.98
    *   Ranking: Top performing model (tied)
    *   Performance: Excellent, able to distinguish between positive and negative classes with high accuracy
    *   Summary:  XGB Classifier has a very high AUC score, indicating it can distinguish between classes with high accuracy.

**Overall Summary** :
All four algorithms have excellent performance, with AUC scores above 0.93, indicating ability to distinguish between positive and negative classes with high accuracy. Logistic Regression, Random Forest Classifier, and XGB Classifier are tied as the top performers.

## Feature Importance
![Feature Importance](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/feature%20importance.png)

## Insight
Based on the feature Importance of Logistic Regression Model, we can see :

The most important feature for predicting the target variable is `Daily Internet Usage` with a coefficient of 1.0, followed by `Daily Time Spent on Site` and `Area Income` both with a coefficient of 0.5. The rest of the features, mostly provinces, have a smaller impact. `AgeSegment` is the most important feature among the provinces with a coefficient of -0.5. Note that the coefficients are all negative, except for `Daily Internet Usage`. This means that an increase in these features is associated with a decrease in the likelihood of the target variable, except for ` Daily Internet Usage` which has a positive relationship.

Here is the ranking of the features by importance:
1.   `Daily Internet Usage` (1.0)
2.   `Daily Time Spent on Site` (0.5)
3.   `Area Incom`e (0.5)
4.   `AgeSegment `(-0.5)
5.   Other provinces (coefficients ranging from -2.5 to -0.5)


## Confusion Matrix
![cm](https://github.com/Rikaelisabeth09/Predict-Clicked-Ads-Customer-Classification-by-using-Machine-Learning/blob/main/LR.png)
The confusion matrix from the logistic regression model provides insights into the model's performance in classifying instances as either "Clicked" or "Not Clicked".

Key observations:


1.   **True Negative (Not Clicked)**: 143 instances were correctly identified as not being clicked, which is a strong true negative rate.
2.  **True Positive (Clicked)**: 144 instances were correctly identified as being clicked, indicating a high true positive rate.
3.   **False Positive (False Positive)**: 4 instances were incorrectly identified as being clicked when they were not, representing a low false positiv
4. **False Negative (False Negative)** : 9 instances were incorrectly identified as not being clicked when they were, indicating a relatively low false negative rate.

Overall, the confusion matrix suggests that the logistic regression model has performed well, with high accuracy in correctly identifying both clicked and not-clicked instances. The low false positive and false negative rates indicate effective balance in correctly classifying positive and negative instances.


# Business Recomendations and Simulation
## Business Simulation
The insights derived from this business impact simulation of implementing a machine learning model for targeted advertising are quite promising. Here are the key takeaways:

1. Reduced Advertising Cost:
    *   Before ML: $600.00
    *   After ML: $294.00
    *   Insight: The machine learning model helps in identifying potential customers more accurately, leading to a significant reduction in advertising costs. By targeting ads only to likely converters, the overall expenditure on ads decreases substantially.
2. Increased Revenue:
    *   Before ML: $750.00
    *   After ML: $670.00
    *   Insight: While the total revenue after implementing ML is slightly lower compared to the revenue before, this minor decrease is outweighed by the substantial cost savings and profit increase. This indicates that the ads are being shown to a more relevant audience, resulting in more effective spending.
3. Higher Profit:
    *   Before ML: $150.00
    *   After ML: $376.00
    *   Insight: The implementation of ML has led to a significant increase in profit (150.67%). This is a direct result of the reduction in costs and the more efficient allocation of ad spend towards users who are more likely to click on the ads.
4. Improved Click Through Rate (CTR):
    *   Before ML: 50.00%
    *   After ML: 97.28%
    *   Insight: The CTR has almost doubled after implementing the ML model. This improvement suggests that the machine learning model is highly effective in targeting users who are genuinely interested in the ads, leading to higher engagement rates.
Summary of Insights:
1. **Cost Efficiency**: The ML model significantly reduces advertising costs by focusing on users with a higher likelihood of clicking the ads.
2. **Revenue Optimization**: Although there's a slight decrease in total revenue, the significant cost savings and profit increase demonstrate a more efficient use of the advertising budget.
3. **Profit Maximization**: The increase in profit highlights the overall financial benefit of implementing machine learning for targeted advertising.
4. **Enhanced Engagemen**t: The improved CTR indicates that the ads are being served to a more relevant audience, resulting in better user engagement and higher chances of conversion.

## Business Recommendation
Based on the insights from the business impact simulation of implementing machine learning for targeted advertising, here are some business recommendations:


1. Optimize Advertising Spend:
    *   Recommendation: Focus the advertising budget on users identified as likely to click on ads.
    *   Action:
        1. Targeted Campaigns: Implement ad campaigns targeting only the predicted potential customers (those identified by the model as likely to click).
        2. Budget Reallocation: Allocate more of the advertising budget towards these targeted campaigns to maximize ROI.
        3. Performance Monitoring: Continuously track the performance of these targeted ads and adjust the budget allocation based on their effectiveness.
2. Enhance Data Collection:
    *   Recommendation: Improve the data quality and diversity to enhance the model's predictive accuracy.
    *   Action:
        1. Data Sources: Integrate additional data sources, such as customer interactions, purchase history, and demographic information.
        2. Data Quality Management: Regularly clean and validate the data to ensure accuracy and reliability.
        3. Compliance: Ensure all data collection practices comply with privacy regulations to maintain customer trust.
3. Personalize Marketing Efforts:
    *   Recommendation: Leverage the insights from the ML model to create personalized marketing messages tailored to the preferences and behaviors of high-potential customers.
    *   Action: Segmentation:
        1. Segment customers based on their likelihood to click on ads and tailor marketing messages accordingly.
        2. Personalized Content: Develop personalized email campaigns, social media ads, and special offers based on customer segments.
        3. Consistent Experience: Ensure a consistent and personalized experience across all marketing channels.
        4. Age Range Targeting: Create content specifically tailored to different age groups identified as high potential, ensuring that messaging and offers resonate with each demographic.
        5. Re-engagement Campaigns: Develop strategies to re-engage inactive or less active customers using personalized offers and targeted communication based on past behavior and preferences.
4. Continuously Measure and Optimize:
    *   Recommendation: Establish metrics and continuously improve marketing strategies based on performance data.
    *   Action:
        1. KPIs: Define and track key performance indicators such as click-through rates (CTR), conversion rates, and return on ad spend (ROAS).
        2. A/B Testing: Conduct A/B tests to determine the most effective marketing strategies and optimize them based on the results.
        3. Feedback Loop: Use the insights gained from each campaign to refine the model and improve future marketing efforts.
5. Invest in Team Training and Development:
    *   Recommendation: Ensure the marketing and data teams are proficient in using machine learning tools and insights.
    *   Action:
        1. Training Programs: Provide training on the latest machine learning and data analysis techniques.
        2. Collaboration: Foster collaboration between marketing and data science teams to effectively leverage model insights.
        3. Advanced Tools: Equip teams with advanced analytics tools to enable effective data analysis and decision-making.
