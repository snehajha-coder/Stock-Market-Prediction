# Stock Market Prediction using Numerical and Textual Analysis

## Overview
This repository presents a hybrid model for predicting stock prices by leveraging numerical analysis of historical stock prices and textual analysis of news headlines. The primary objective of this project is to develop an effective approach that combines both quantitative and qualitative factors to enhance stock price/performance prediction.

## Features
- Data preprocessing
- Data Visualization
- Time Series Analysis
- Implementation of various machine learning models
- Evaluation of model performance
- Prediction visualization
- Understanding Important Aspects

## Requirements
- Python (version 3.9.16)
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - nltk
  - matplotlib
  - seaborn
  - tensorflow
  - keras

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/snehajha-coder/Stock-Market-Prediction.git
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Download the Jupyter Notebook file (ipynb) from the repository.

2. Open the Jupyter Notebook in your preferred Python environment.

3. Modify the code according to your requirements:
   - Update the data sources and file paths as per your dataset.
   - Customize the data preprocessing steps based on your specific needs.
   - Adjust the machine learning models or algorithms as desired.
   - Modify the visualization code to suit your preferences.

4. Ensure that you have all the necessary libraries installed:
   - Install the required libraries by running the following command in a code cell:
     ```
     !pip install library_name
     ```
   - Replace "library_name" with the actual name of the library if it is not already installed.

5. Execute the code cells in the Jupyter Notebook to run the entire code or specific sections as needed:
   - Use the "Run" button or press "Shift + Enter" to execute a code cell.
   - Follow the instructions and prompts provided within the code for any user inputs or interactions.

6. View the output, predictions, and evaluation results directly in the Jupyter Notebook:
   - The output will be displayed in the output cells below the corresponding code cells.
   - Evaluate the performance of the models and analyze the predicted stock prices or performance.

7. Customize the code and experiment with different settings or approaches to further refine the predictions:
   - Modify the parameters of the machine learning models.
   - Try different feature engineering techniques or preprocessing methods.
   - Explore alternative algorithms or models to enhance the predictions.

8. Save your modifications and the output generated from the Jupyter Notebook for future reference and analysis.

Note: Ensure that you have the required dataset available and properly formatted before running the code. Modify the code and adapt it to your specific dataset and analysis objectives.

By following these steps and modifying the provided Jupyter Notebook code, you can leverage the stock market prediction functionality according to your specific needs and dataset.

## Steps Involved

### Data Collection
To gather the necessary data for Stock Market Prediction using Numerical and Textual Analysis, the following steps were followed:

1. Historical Stock Price Data:
   - The historical stock price data was collected from Yahoo Finance.
   - Yahoo Finance provides a comprehensive and reliable source of historical stock price information for various stocks.
   - The data includes the date, open price, close price, high price, low price, and volume traded for each trading day.

2. News Headlines and Articles:
   - News headlines and articles related to the target stock were collected using two APIs: newsAPI and BingNewsAPI.
   - These APIs allow access to a wide range of news sources and provide relevant news content for analysis.
   - In addition, the dataset available at [this website](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQMQH) was used.

Note: It is important to ensure the proper usage and compliance with the terms and conditions of the APIs used for data collection.

### 2. Data Preprocessing
1. Stock Price Data:
   - The dataset has 5303 entries with 8 columns: Date, Open, High, Low, Close, Volume, Dividends, and Stock Splits.
   - There are no missing values in the dataset.
   - Descriptive statistics of the data show summary statistics for each column.
   - Two stock split events were identified on February 12, 2004, and July 15, 2015.

2. Adjusting for Stock Splits and Dividends:
   - The code adjusts the stock prices for stock splits and dividends to accurately reflect the true value of the stock.
   - The 'Stock Splits' column is replaced with 1 where the value is 0 to avoid division errors.
   - The adjusted close prices are calculated by dividing the 'Close' prices by the cumulative product of the 'Stock Splits' column.
   - Dividend amounts are subtracted from the adjusted close prices to account for the impact of dividends.
   - The unnecessary columns are dropped, and the resulting DataFrame ('data_preprocessed') contains the adjusted close prices.

3. Preprocessing News Headlines for Sentiment Analysis:
   - The code demonstrates text preprocessing techniques applied to news headlines for sentiment analysis.
   - The 'preprocess_text' function is defined to tokenize, remove stopwords, perform stemming, and lemmatize the text.
   - The function is applied to the 'Headline' column in the 'news_api_netfix_data' and 'bing_netfilx_data' DataFrames.
   - Preprocessed headlines are stored in the 'Headline_preprocessed' column.

4. Making Indian News Headlines Data Short:
   - Duplicates are dropped from the 'india_news_headlines' DataFrame.
   - The 'publish_date' column is converted to the 'datetime' data type.
   - The DataFrame is filtered to include only the 'publish_date' and 'headline_text' columns.
   - News headlines are grouped by date, joining them into a single string per date.
   - The 'publish_date' column is set as the index.
   - The data is sorted by date in ascending order.
   - Finally, the 'publish_date' column is reset as a regular column.

### 3. Numerical Analysis
Perform exploratory data analysis (EDA) on the historical stock price data to understand trends, patterns, and relationships.
Extract important features such as moving averages, relative strength index (RSI), and other technical indicators.
Split the data into training and testing sets.

### 4. Textual Analysis
Perform sentiment analysis on the collected news headlines using techniques like Natural Language Processing (NLP).
Assign sentiment scores or labels (positive, negative, neutral) to each headline based on the sentiment analysis results.

### 5. Feature Integration
Combine the numerical features derived from the historical stock price data with the sentiment scores from the textual analysis.
Create a merged dataset that includes both numerical and textual features.

### 6. Model Training
Select a suitable machine learning algorithm such as regression, support vector machines, or neural networks.
Split the merged dataset into training and testing sets.
Train the machine learning model using the training data.

### 7. Model Evaluation
Evaluate the performance of the trained model using appropriate evaluation metrics such as mean squared error (MSE), root mean squared error (RMSE), or accuracy.
Fine-tune the model parameters if necessary to improve the performance.

### 8. Prediction and Analysis
Use the trained model to make predictions on the unseen data (testing set).
Analyze the predicted stock prices/performance and compare them with the actual values.
Evaluate the accuracy and effectiveness of the hybrid model in predicting stock prices/performance.

### 9. Iterative Improvement
Iterate and refine the model by incorporating additional data, adjusting feature selection, or trying different machine learning algorithms.
Continuously monitor and update the model as new data becomes available.

## Data
The dataset used for this project should be

 stored in the `data/` directory. It should contain historical stock price data in CSV format.

## Model Selection
You can experiment with different machine learning models by modifying the `main.py` script. The current implementation includes [list the models used].

## Evaluation
Model performance is evaluated using [mention the evaluation metrics used, such as mean squared error (MSE) or mean absolute percentage error (MAPE)].

## Results
The results of the stock price prediction, along with the evaluation metrics, can be found in the `results/` directory.

## Contributing
Contributions to this project are welcome. If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License
[Specify the license under which your code is released, such as MIT or Apache 2.0.]

## Contact
[Provide your contact information or any other relevant links, such as your email or a link to your website.]

Feel free to customize this template according to your specific project requirements. Good luck with your stock price prediction code!
