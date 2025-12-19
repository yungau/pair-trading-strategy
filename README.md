# Pair Trading Strategy with Kalman Filter & Regime Shift Detection

## **Overview**
This project implements and backtests a **pair trading strategy** using the Hong Kong indices **HSCEI** and **XIN9I**. The strategy uses a **Kalman filter** to estimate the dynamic hedge ratio and spread, and generates trading signals based on the **20-day rolling z-score** of the spread.  
Additionally, the strategy is enhanced with a **regime shift detection filter** using **Maximum Mean Discrepancy (MMD)** to avoid trading during periods of structural market change.

---

## **Key Features**
- **Kalman Filter** for dynamic hedge ratio estimation
- **Pair Trading Signals**:
  - Short when z-score > 2
  - Long when z-score < -2
  - Exit when |z-score| < 1
- **Regime Shift Detection**:
  - Uses MMD to compare recent vs historical distribution
  - Suspends trading during detected regime shifts
- **Performance Metrics**:
  - Cumulative PnL
  - Sharpe Ratio
  - Max Drawdown
- **Visualizations**:
  - Spread & z-score plots
  - Entry/exit points
  - Equity curve

---


## **Project Structure**
```
pair-trading-strategy/
│
├── data/                             # Sample data
│   ├── fx_data.csv                   # historical prices for fx 
│   └── indices_data.csv              # historical prices for indices
├── pair_trading_signal.py            # Pair trading signal
├── mmd_regime_shift_detector.py      # Regime shift detection filter
├── pair_trading_backtest.ipynb       # Main Jupyter Notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## **Installation**
Clone the repository and install dependencies:

```bash
git clone https://github.com/yungau/pair-trading-strategy.git
cd pair-trading-strategy
pip install -r requirements.txt
```

---

## **Usage**
Open the Jupyter notebook:

```bash
jupyter lab pair_trading_backtest.ipynb
```

Follow the steps in the notebook:
1. Load data
2. Apply Kalman filter
3. Generate trading signals
4. Backtest strategy
5. Apply regime shift detection
6. Analyze performance

---

## **Dependencies**
Minimal dependencies based on imports:
```
numpy
pandas
matplotlib
vectorbt
pykalman
scikit-learn
jupyter
```

---

## **Results**
- Strategy performance metrics and visualizations are included in the notebook.
- Regime shift detection improves total PnL and Sharpe ratio.

---

## **Disclaimer**
This project is for **educational purposes only** and does not constitute financial advice.
