# Machine Learning Project Report: Online News Popularity (Regression)

*UCI Heart Disease (Online News Popularity (Regression)* - **Note:** The title seems to combine two distinct projects. This README focuses on the "Online News Popularity (Regression)" part as detailed in the document.

**Date:** May 17, 2025

---

## Table of Contents

1.  [Online News Popularity (Regression)](#online-news-popularity-regression)
    * [Abstract](#abstract)
    * [Introduction](#introduction)
    * [Data Description](#data-description)
    * [Exploratory Data Analysis](#exploratory-data-analysis)
    * [Methodology](#methodology)
        * [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
        * [Data Splitting](#data-splitting)
        * [Model Selection and Training](#model-selection-and-training)
        * [Model Evaluation](#model-evaluation)
    * [Results](#results)
        * [Performance on Transformed Scales](#performance-on-transformed-scales)
        * [Performance on Original Scale (after Transformation)](#performance-on-original-scale-after-transformation)
        * [Hyperparameter Tuning for SVR](#hyperparameter-tuning-for-svr)
    * [Conclusion](#conclusion)
2.  [References](#references)

---

## Online News Popularity (Regression)

### Abstract

This report details a machine learning project focused on predicting the popularity of online news articles, as measured by the number of social network shares. The project explores various regression algorithms to model the complex relationship between article features and share counts. Key challenges identified include the highly skewed distribution of the features and the presence of outliers. Despite comprehensive preprocessing techniques and the application of multiple regression models, achieving accurate prediction of the exact number of shares proved challenging. The findings highlight the inherent difficulty of this regression task with the given data characteristics, suggesting potential limitations for precise popularity forecasting using standard regression approaches.

### Introduction

In the digital age, understanding and predicting the popularity of online content is crucial for publishers, marketers, and content creators. The ability to forecast which articles are likely to gain traction can inform content strategy, distribution efforts, and resource allocation. This project investigates the application of machine learning techniques to predict the popularity of online news articles based on their features.

### Data Description

This project focused on predicting the popularity of online news articles, measured by the number of social network shares (`shares`), using the Online News Popularity dataset. Sourced from Mashable articles published over two years, the dataset contains 39,797 instances and 58 predictive features. These features capture diverse aspects of each article, including content characteristics, metadata, multimedia elements, keyword performance, publication timing, latent topics derived from LDA, and sentiment/subjectivity scores. The `url` and `timedelta` features were identified as non-predictive and excluded from the analysis. Features are a mix of integer and real data types.

### Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the dataset's structure and key characteristics. Features were conceptually grouped into categories such as textual, multimedia, keyword, content type, self-reference, weekday, LDA topics, and subjectivity/polarity features to facilitate analysis.

Beyond the statistical analysis, initial exploration yielded several qualitative insights into factors potentially influencing popularity. It was observed that both the number and quality of keywords appear relevant, with more and better keywords correlating with increased shares. Titles of a moderate word count seemed more likely to be shared than very short ones. Visual content in the form of images appeared more associated with shares than videos, possibly reflecting user preference. Observations on reader behavior suggested Tuesday and Wednesday as popular days for consuming articles, potentially linked to weekly routines.

Analysis of the target variable, `shares`, revealed a highly skewed distribution. A large majority of articles have a low number of shares, while a small subset receives significantly higher share counts, forming a long tail. This severe positive skewness is clearly depicted in the histogram of the target variable (Figure 1). Similar skewness was also evident in the distributions of many other features (Figure 2). Descriptive statistics for `shares` highlight this challenge: a mean of approximately $3349$, a median of $1400$, a $75^{th}$ percentile of $2700$, and a maximum exceeding $663,000$, indicating a wide range and the presence of numerous outliers.

**Figure 1: Distribution of Shares**
![Distribution of Shares](shares_distribution.png)

**Figure 2: Distribution of Various Features**
![Distribution of Various Features](distribution.png)

**Figure 3: Distribution of Shares After Log Transformation**
![Distribution of Shares After Log Transformation](shares_with_log.png)

Temporal analysis showed distinct patterns in publication frequency and popularity. Article publication is highest on weekdays, particularly Wednesday and Tuesday, with significantly fewer articles published on weekends (Figure 4). Correspondingly, the total number of shares broadly align with this, with weekdays accumulating more shares than individual weekend days (Figure 5).

**Figure 4: Number of Articles Published Per Day**
![Number of Articles Published Per Day](numberOfArticles_data.png)

**Figure 5: Total Shares Per Day of Week**
![Total Shares Per Day of Week](shares_date.png)

An examination of content categories revealed that the 'World', 'Tech', 'Entertainment', and 'Business' channels published the most articles (Figure 6). However, in terms of total shares, the 'Tech' and 'Entertainment' lead in total shares, suggesting higher average popularity in these categories compared to others with similar or higher publication rates (Figure 7). LDA topic analysis also indicated that the dominant topic is associated with varying average share counts (Figure 8).

**Figure 6: Number of Articles Published Per Category**
![Number of Articles Published Per Category](articles_per_category.png)

**Figure 7: Total Shares Per Content Type**
![Total Shares Per Content Type](shares_per_category.png)

**Figure 8: Average Shares by Dominant LDA Topic**
![Average Shares by Dominant LDA Topic](LDA.png)

A correlation heatmap (Figure 9) provided an overview of feature relationships. While some features were correlated with each other, linear correlations between most individual features and the target variable (`shares`) were generally low. Keyword statistics (e.g., `kw_max_avg`, `kw_avg_avg`) and self-reference shares (`self_reference_avg_sharess`) showed some of the relatively stronger positive linear relationships, though these were still moderate. Scatter plots of textual features and topic purity against shares further illustrated the complex, often non-linear, relationship with popularity, particularly for high-share articles (Figure 10, Figure 11).

**Figure 9: Correlation Matrix Heatmap of All Features**
![Correlation Matrix Heatmap of All Features](heatmap_of_all.png)

**Figure 10: Textual Features vs Shares**
![Textual Features vs Shares](scatter-plotwithtarget.png)

**Figure 11: Topic Purity vs Shares**
![Topic Purity vs Shares](topic-purity.png)

Overall, the EDA revealed the inherent challenges of predicting the exact number of shares, primarily due to the target variable's severe skewness and the complex, often non-linear, relationships between features and popularity.

### Methodology

The project followed a structured methodology for training and evaluating regression models to predict online news popularity.

#### Data Preprocessing and Feature Engineering

The initial data loading and inspection involved handling any duplicate entries and ensuring correct data types. The non-predictive `url` and `timedelta` features were removed. Addressing the observed data skewness in many features and the target variable was a critical step. To explore different strategies for handling skewness and compare their impact on model performance, both Log transformation (specifically $\log(x+1)$ for positive data) and Yeo-Johnson transformation were applied to the relevant skewed features and the target variable. The Yeo-Johnson transformation was particularly useful as it can normalize data regardless of positive or negative values. To handle the influence of outliers, which were prevalent, **Robust Scaling** was applied to the features. Robust Scaling, based on quartiles, is less sensitive to extreme values than standard scaling methods.

**Principal Component Analysis (PCA) was also applied to the features.** PCA is a dimensionality reduction technique used to transform the features into a smaller set of uncorrelated components while retaining most of the variance in the data. This step was likely performed to potentially reduce the dimensionality of the feature space and address multicollinearity observed in the correlation heatmap, preparing the data for the regression models.

Furthermore, several new features were engineered to potentially provide more informative inputs to the models by combining existing sentiment and polarity-related features:

* `sentiment_balance` = `global_rate_positive_words` - `global_rate_negative_words` (Net sentiment).
* `sentiment_intensity` = `global_rate_positive_words` + `global_rate_negative_words` (Overall emotional language prominence).
* `pos_polarity_strength` = (`max_positive_polarity` - `min_positive_polarity`) * `avg_positive_polarity` (Range and intensity of positive sentiment polarity).
* `neg_polarity_strength` = (`max_negative_polarity` - `min_negative_polarity`) * `avg_negative_polarity` (Range and intensity of negative sentiment polarity).
* `title_impact` = `abs_title_sentiment_polarity` * `title_subjectivity` (Strength of the title's emotional tone and opinionatedness).
* `title_bias` = `title_sentiment_polarity` / (`abs_title_sentiment_polarity` + $10^{-6}$) (with a small constant to prevent division by zero).

These engineered features were added to the dataset prior to modeling.

#### Data Splitting

The prepared dataset, consisting of features and the continuous target variable (`shares`), was split into three sets: training, validation, and test, to ensure robust model evaluation. This resulted in a final distribution of approximately $60\%$ for training, $20\%$ for validation, and $20\%$ for testing. The training set was used for model training, the validation set for hyperparameter tuning and model selection, and the test set for final evaluation. The split was performed with `random_state=42` for reproducibility.

#### Model Selection and Training

For this project, seven different regression algorithms were implemented and compared to evaluate their performance in predicting online news shares: Linear Regression, Bayesian Ridge, K-Nearest Neighbors (KNN), Support Vector Regressor (SVM), Decision Tree Regressor, Random Forest Regressor, and Neural Network. These models were trained on the preprocessed training data. Hyperparameter tuning was performed for some models, such as SVR, using GridSearchCV with cross-validation on the training set and evaluation on the validation set (example GridSearchCV output from Figure 12 shows tuning for SVR).

#### Model Evaluation

The performance of each trained model was evaluated using four standard regression metrics:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R-squared ($R^2$)

Models were evaluated on both the validation and test sets, with the test set evaluation providing the final assessment of the model's generalization ability.

### Results

The regression models were evaluated on the test set using MAE, MSE, RMSE, and $R^2$ metrics. To understand the impact of different skewness handling approaches, performance was assessed for models trained using data prepared with Log transformation versus Yeo-Johnson transformation, and evaluated on both the transformed and original target scales.

#### Performance on Transformed Scales

When evaluating the models using MSE and $R^2$ directly on the log-transformed shares target (using the results from Figure 12), the error magnitudes were significantly smaller compared to the original scale. For example, MSE values ranged from approximately $0.74$ to $1.48$, and some models achieved positive $R^2$ values, albeit low ones (e.g., Random Forest at $0.14$, Bayesian Ridge and Linear Regression at $0.13$, SVM at $0.11$). This suggests that the models were able to capture some variance in the transformed target space, which has a more normalized distribution.

**Figure 12: Model Performance Metrics on Log-Transformed Scale**
![Model Performance Metrics on Log-Transformed Scale](results_with_log_transformation.png)

#### Performance on Original Scale (after Transformation)

Evaluating the models on the original target scale after training with Yeo-Johnson transformed data provides a more direct measure of performance in predicting the actual number of shares. Table 1 presents these results.

**Table 1: Regression Model Performance on the Test Set (Original Shares Scale) after Yeo-Johnson Transformation and Robust Scaling**

| Model           | Test MAE | Test MSE             | Test RMSE | Test R²   |
| :-------------- | :------- | :------------------- | :-------- | :-------- |
| SVM             | 2321.16  | $1.274784 \times 10^8$ | 11290.63  | -0.00751  |
| Random Forest   | 2341.60  | $1.274289 \times 10^8$ | 11288.44  | -0.00712  |
| Bayesian Ridge  | 2358.19  | $1.280922 \times 10^8$ | 11317.78  | -0.01236  |
| Linear Regression | 2359.34| $1.281673 \times 10^8$ | 11321.09  | -0.01296  |
| KNN             | 2396.89  | $1.655150 \times 10^8$ | 12865.26  | -0.00777  |
| Neural Network  | 2421.32  | $1.278625 \times 10^8$ | 11307.63  | -0.01055  |
| Decision Tree   | 4051.12  | $3.375524 \times 10^8$ | 18372.60  | -1.66781  |

As observed from the results in Table 1 and visually represented in the bar charts (Figure 13), the **Decision Tree Regressor performed significantly worse** than all other models across all metrics (MAE, MSE, RMSE, $R^2$). Its errors were roughly twice as high as the next worst models, and its $R^2$ value was highly negative.

**Figure 13: Test Set Performance Metrics (Original Scale)**
![Test Set Performance Metrics (Original Scale)](final_result_with_yoe_transformation.png)

Among the remaining six models, performance was relatively similar, although some minor distinctions can be made:

* **SVM** achieved the lowest Mean Absolute Error (MAE) at $2321.16$, suggesting it had the smallest average prediction error magnitude.
* **Random Forest** and **SVM** had the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), which are more sensitive to larger errors. Random Forest had a slightly lower MSE ($1.274289 \times 10^8$) and RMSE ($11288.44$) than SVM (MSE $1.274784 \times 10^8$, RMSE $11290.63$).
* **Neural Network** and **KNN** generally showed slightly higher MAE compared to SVM, Random Forest, Bayesian Ridge, and Linear Regression within this group. KNN also had noticeably higher MSE and RMSE than the other five non-Decision Tree models.
* All six models (excluding Decision Tree) exhibited very low or negative $R^2$ values, ranging from approximately $-0.013$ to $-0.007$.

#### Hyperparameter Tuning for SVR

As part of the model development, hyperparameter tuning was performed for the Support Vector Regressor using GridSearchCV with cross-validation. Tuning was explored using data prepared with both Log and Yeo-Johnson transformations to identify optimal parameters under different data distributions.

When trained and evaluated on data processed with **Yeo-Johnson transformation**, GridSearchCV identified the best SVR parameters as `{'C': 0.1, 'epsilon': 0.2, 'kernel': 'rbf'}`. The performance metrics for this tuned model on the validation and test sets (e.g., Test MAE $\approx 2317.4$, Test RMSE $\approx 11324.4$) align closely with the SVR results presented in Table 1, confirming its performance when trained with Yeo-Johnson transformed data and evaluated on the original scale.

Tuning was **also performed using data prepared with Log transformation** (Figure 14). For example, tuning for an SVR with a 'linear' kernel resulted in a Validation MAE of approximately $2317.5$ (when evaluated on the original scale).

**Figure 14: Example GridSearchCV Output for SVR Tuning (Log Transformation)**
![Example GridSearchCV Output for SVR Tuning (Log Transformation)](Screenshot 2025-05-17 132023.png)

These tuning efforts demonstrate that while hyperparameter optimization was conducted for SVR under different transformation strategies, the resulting models consistently showed similar performance levels when evaluated on the original shares scale. This further reinforces the observation that accurately predicting the precise, highly variable share count remains a significant challenge regardless of the specific transformation or SVR parameters used.

Despite the minor differences among the top six models (SVM and Random Forest showing a slight edge in minimizing error metrics), their overall performance was poor. The consistently low and negative $R^2$ values across virtually all models (including those with the "best" MAE/RMSE) indicate that none of the implemented algorithms were able to explain a significant portion of the variance in the actual number of shares.

Comparing the error metrics (MAE, RMSE) on the original scale to the descriptive statistics of the target variable (mean $\approx 3349$, median $\approx 1400$, $75^{th}$ percentile $\approx 2700$), an MAE of around $2300-2500$ shares is substantial. It suggests that, on average, the models' predictions were off by an amount comparable to or exceeding the typical share counts for most articles. The large RMSE values further emphasize that larger errors were frequent, highlighting the difficulty in predicting high share counts.

The scatter plot of Actual vs. Predicted shares for the SVM model (Figure 15) visually confirms the models' limitations on the original scale. It shows that even the model with the lowest MAE primarily predicts within a relatively narrow range of low share counts, failing to accurately predict the articles with high or very high numbers of shares. This inability to capture and predict the viral content, which contributes significantly to the overall variance and the long tail of the distribution, is the main reason for the high error metrics and the low/negative $R^2$ values when evaluating on the original scale.

**Figure 15: Actual vs. Predicted Shares for SVM (Original Scale)**
![Actual vs. Predicted Shares for SVM (Original Scale)](results.png)

### Conclusion

In conclusion, while different data transformation techniques (Log and Yeo-Johnson) were explored to handle skewness and showed some ability to model the transformed target, the evaluation on the original shares scale revealed significant challenges. The models struggled to effectively predict the precise number of shares, particularly for high-popularity articles. While SVM and Random Forest showed marginally better performance in minimizing error metrics like MAE and RMSE among the evaluated models (excluding the poor-performing Decision Tree), the consistently high error magnitudes relative to the typical share counts and the very low/negative $R^2$ values indicate that, based on the evaluated models and methodology, the data is not well-suited for accurately predicting the precise popularity score as a regression problem using these standard techniques. Given the significant difficulty in predicting the exact share count, this dataset might be more amenable to a classification task, such as predicting whether an article will be popular (e.g., above median shares) or not popular.

---

## References

* Szymanowski, J. (2023). Predicting the Popularity of an Online News Article. *Medium*. Available at: [https://joshua-szymanowski.medium.com/predicting-the-popularity-of-an-online-news-article-5b07591146bf](https://joshua-szymanowski.medium.com/predicting-the-popularity-of-an-online-news-article-5b07591146bf)
* Balaji, S. (n.d.). ML Capstone Project. *GitHub Pages*. Available at: [https://swebalaji.github.io/ML_Capstone_Project/](https://swebalaji.github.io/ML_Capstone_Project/)
* Rangarajan, S. (n.d.). Online-News-Popularity-Regression. *GitHub*. Available at: [https://github.com/Sushama-Rangarajan/Online-News-Popularity-Regression?tab=readme-ov-file](https://github.com/Sushama-Rangarajan/Online-News-Popularity-Regression?tab=readme-ov-file)
