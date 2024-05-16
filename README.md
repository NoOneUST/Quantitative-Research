# 📈 Quantitative Research

This repository :file_folder: contains two sub-projects dedicated to quantitative research in finance and economics. Each embodies a comprehensive research study backed by corresponding paper 📝 and code.

---

## 📋 Table of contents

1. 🏡 [Predicting the S&P/Case-Shiller U.S. National Home Price Index](#home_price_index)
2. 💹 [Idea Generation for Price Prediction Signal for a Macro Asset or ETF](#macro_asset_prediction)
3. ⚙️ [Installation](#installation)
4. 🎬 [Running the Scripts](#running_the_scripts)

---

<a name="home_price_index"></a>

## 🏠 Project 1: Predicting the S&P/Case-Shiller U.S. National Home Price Index

This project is accompanied by a [**research paper**](./Home_Price_Index_Prediction.pdf) :page_with_curl:. The code for this project can be found in the following Python scripts:

1. 🐍 [Home_Price_Index_Prediction_based_on_Linear_Regression.py](./Code/Home_Price_Index_Prediction_based_on_Linear_Regression.py)
2. 🐍 [Home_Price_Index_Prediction_based_on_LSTM.py](./Code/Home_Price_Index_Prediction_based_on_LSTM.py)

Both files contain standalone scripts that generate the results mentioned in the paper 📝 upon execution.

---

<a name="macro_asset_prediction"></a>

## 💰 Project 2: Idea Generation for Price Prediction Signal for a Macro Asset or ETF

Corresponding to this project, you can find the [**research paper**](./Macro_Asset_Price_Prediction.pdf) 📜 and the Python script [Macro_Asset_Price_Prediction_based_on_LSTM.py](./Code/Macro_Asset_Price_Prediction_based_on_LSTM.py):snake:. By running the script, you will be able to reproduce the results discussed in the paper.

---

<a name="installation"></a>

## 🔧 Installation

In order to run the Python scripts, you'll need to install the required Python packages. You can do this by running the following command:

```shell
pip install torch sklearn pandas numpy matplotlib yfinance
```

---

<a name="running_the_scripts"></a>
## ▶️ Running the Scripts 

Each Python script can be run independently to reproduce the corresponding project's results. Here's how you can run them from the command line:

```shell
python Home_Price_Index_Prediction_based_on_Linear_Regression.py
```

```shell
python Home_Price_Index_Prediction_based_on_LSTM.py
```

```shell
python Macro_Asset_Price_Prediction_based_on_LSTM.py
```

The scripts are set up to automatically take advantage of GPU acceleration if available. However, they can also be executed using just a CPU. This ensures that regardless of your system configuration, you should be able to run the scripts and replicate our findings.

---
