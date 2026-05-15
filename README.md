# Global Football Transfer Market Analysis & Prediction

A full data science project analyzing 25 years of high-value football transfers — from exploratory analysis to a machine learning model that attempts to predict player transfer fees.

---

## Problem Statement

The football transfer market has undergone extraordinary financial transformation since 2000. What was once driven by sporting merit alone is now shaped by commercial hype, market inflation, league prestige, and the "superstar economy." This project aims to quantify those forces, identify the structural patterns behind transfer valuations, and test whether a machine learning model can predict a player's fee using only demographic and market data.

### Specific Questions Answered

1. **Transfer Inflation** — How have fees evolved over 25 years, and is that growth evenly distributed across all players, or concentrated at the top?
2. **League Dynamics** — Which leagues are the dominant buyers? Which consistently export talent to the elite tier ("feeder leagues")?
3. **Age & Player Value** — Does age follow a simple linear relationship with fee, or is the relationship more complex? What is the "wonderkid premium"?
4. **Position Premium** — Are attacking players systematically more expensive than defenders?
5. **Predictive Power** — Can a model predict transfer fees from age, position, league, and year alone? What does its failure or success tell us about how football actually works?

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | [Kaggle — Global Football Transfer Dataset](https://www.kaggle.com/datasets/rajbirahmed/global-football-transfer-dataset) |
| Coverage | 2000-01 season to 2025-26 season |
| Scope | Top 100 transfers by fee per season |
| Records | ~2,600 transfers |
| Key Columns | `Player`, `Season`, `Age`, `Position`, `Fee_Euros`, `From_Club`, `To_Club`, `From_League`, `To_League` |

---

## Project Structure

```
football-transfers/
├── data/
│   ├── global_football_transfer_dataset.csv   # Raw source data
│   ├── processed_transfer_data.csv            # After cleaning
│   └── ml_ready_transfer_data.csv             # After feature engineering
├── notebooks/
│   └── football_transfers.ipynb               # Main analysis notebook
├── images/                                    # All generated charts
├── README.md
├── STUDY.md                                   # Non-technical walkthrough guide
└── requirements.txt
```

---

## Process

```
Raw CSV Data
    │
    ▼
Phase 1: Data Audit & Cleaning
    │   → Rename ambiguous column (Unnamed: 10 → Loan_Fee)
    │   → Parse Season text into numeric Start_Year / End_Year
    │   → Save processed_transfer_data.csv
    │
    ▼
Phase 2: Exploratory Data Analysis (EDA)
    │   → Fee distribution analysis (raw vs log-transformed)
    │   → Transfer inflation trends over time (mean vs median)
    │   → Age vs fee scatter with regression line
    │   → League-by-league spending comparison
    │   → Position premium analysis
    │   → Domestic vs international transfer premium
    │   → Correlation matrix (Age, Start_Year, Fee_Euros)
    │
    ▼
Phase 3: Feature Engineering
    │   → Positional encoding via traditional shirt numbers
    │   → Inflation adjustment (base-year normalization)
    │   → League tier ranking by empirical spending power
    │   → Transfer type flag (international vs domestic)
    │   → Log transformation of target variable
    │   → Save ml_ready_transfer_data.csv
    │
    ▼
Phase 4: Machine Learning
    │   → 80/20 train-test split
    │   → Random Forest Regressor (100 estimators)
    │   → R² evaluation
    │   → Feature importance analysis
    │
    ▼
Insights & Conclusions
```

---

## Data Cleaning

### Column Renaming
The raw dataset contained a column called `Unnamed: 10`, which is a default label pandas assigns when a CSV has a trailing comma on each row — indicating a column with no header. Inspection revealed it contained sparse loan fee data. Rather than drop it and lose information, it was renamed to `Loan_Fee` for clarity, but excluded from aggregate mathematical operations since most values were null.

### Season Parsing
The `Season` column stored values in the format `"2000-2001"` — human-readable, but mathematically useless. It was split on the hyphen (`-`) using pandas string methods and the start year extracted as an integer column (`Start_Year`). This single step unlocked the entire time-series analysis: without a numeric year, there is no way to model inflation, chart trends over time, or use the year as an ML feature.

---

## Key Findings & Chart Explanations

### 1. Fee Distribution: The Case for Log Transformation
**Two histograms side by side** — raw fee on the left, log-transformed fee on the right.

The raw distribution is severely right-skewed: the vast majority of transfers cluster between €5M–€50M, while a handful of extreme outliers (Neymar at €222M, Mbappé at €180M) stretch the x-axis into the hundreds of millions. This shape makes statistical analysis unreliable and confuses ML models. Applying `np.log1p()` (log of 1 + fee) compresses those outliers proportionally and produces a near-normal bell curve — a prerequisite for robust regression.

### 2. Transfer Fee Inflation Over Time: Mean vs. Median
**Two lines on one chart** — mean fee and median fee plotted across every season from 2000 to 2025.

The most important finding in the entire project lives here. Until approximately 2014, the mean and median track each other closely — suggesting broadly uniform fee growth. After 2014, the mean detaches sharply upward while the median continues a more moderate rise. This is the "Superstar Effect" made quantitatively visible: a small number of record-breaking transfers (PSG's Neymar deal in 2017 being the clearest single inflection point) inflate the average without proportionally affecting what most players cost. The transfer market is bifurcating into a luxury tier and a standard tier.

### 3. Age vs. Transfer Fee: The Age Paradox
**Scatter plot with regression line** — each dot is one transfer, age on the x-axis, fee on the y-axis. The red regression line is nearly flat.

The Pearson correlation between age and fee is **-0.03** — effectively zero. A naive assumption would predict that prime-age (24–27) players are most expensive and value drops off on either side. The scatter disproves this. Teenage "wonderkids" (18–21) carry enormous price tags because clubs are buying future potential and resale upside, not current ability. A 19-year-old and a 26-year-old can command identical fees for entirely different reasons. This finding is what rules out linear regression as a modelling approach for this dataset.

### 4. Top 10 Leagues by Average Transfer Fee
**Horizontal bar chart** — leagues ranked by mean outgoing fee for transfers into that league.

This chart makes the wealth hierarchy of global football immediately legible. The Premier League and La Liga sit at the top by a significant margin. There is a steep drop after the traditional "Big 5" European leagues (PL, La Liga, Bundesliga, Serie A, Ligue 1), and then a long tail of lower-spending competitions. This chart also justifies using `To_League_Rank` as a machine learning feature: league destination is a strong structural signal of price.

### 5. Average Transfer Fee by Position
**Horizontal bar chart** — positions ranked by mean fee.

Forwards and attacking midfielders command higher average fees than defensive players. This reflects market demand: goal contributions are the most directly monetisable skill in football (through broadcast deals, shirt sales, and commercial partnerships), which drives clubs to pay a premium for attackers. Goalkeepers and centre-backs, despite being equally critical to results, are systematically undervalued by the market.

### 6. Correlation Matrix
**3×3 heatmap** — showing correlations between `Age`, `Start_Year`, and `Fee_Euros`.

Two numbers matter here. The correlation between `Start_Year` and `Fee_Euros` is **0.54** — a strong positive signal meaning the year of transfer is a more powerful price predictor than any player attribute. The correlation between `Age` and `Fee_Euros` is **-0.03** — confirming what the scatter plot showed. Market era dominates individual demographics.

---

## Feature Engineering

Raw data cannot be fed into a machine learning model directly. Text categories are meaningless to an algorithm; fees must be normalized; time must be adjusted for inflation.

**Positional Encoding:** Player positions (e.g., "Centre-Forward", "GoalKeeper") were mapped to traditional football shirt numbers — a domain-knowledge encoding that creates an ordinal "attacking threat" spectrum rather than treating positions as arbitrary categories. Goalkeepers map to 1, strikers to 9, creative midfielders to 10.

**Inflation Adjustment:** A base-year multiplier was calculated using yearly mean fees. Each transfer fee was scaled up by the factor representing how much more expensive the market is today versus the year of transfer. This allows a €50M deal in 2005 to be compared fairly against a €50M deal in 2024 — without adjustment, the model would systematically undervalue older transfers.

**League Tier Ranking:** Rather than using league names as raw text (which the model cannot interpret), leagues were ranked empirically by their average transfer spend. The highest-spending league receives rank 1, and so on. This preserves the ordinal signal (moving to a richer league = premium price) without requiring manual tier assignments.

**International Transfer Flag:** A binary column (`is_international`) was created — 1 if origin and destination leagues differ, 0 if not. Analysis showed international transfers average roughly €1.9M more than domestic moves.

**Log Transformation of Target:** The prediction target (`Adjusted_Fee`) was log-transformed before model training. Predicting log-fees rather than raw fees forces the model to think in proportional terms rather than absolute euros, preventing massive outlier transfers from dominating the error signal during training.

---

## Technical Decisions

### Why Random Forest, not Linear Regression?

Linear regression assumes a straight-line relationship between inputs and the target variable. The EDA, specifically the age vs. fee scatter plot and the correlation of -0.03, proved this assumption is violated. Age does not predict fee linearly — a 19-year-old wonderkid and a 26-year-old peak player can cost identical amounts for structurally different reasons.

A Random Forest Regressor resolves this by building hundreds of independent decision trees, each asking a sequence of "if/then" questions (e.g., *if age < 21 AND destination = Premier League THEN predict high*). The ensemble of trees can capture curved, conditional relationships that no straight line could model.

The choice of 100 estimators (`n_estimators=100`) with `random_state=42` (for reproducibility) provides a reasonable balance between model stability and computational cost on a 2,600-row dataset.

### Why Log-Transform the Target?

Transfer fees are log-normally distributed — the log of the fee follows a normal distribution. Predicting on this scale means the model treats a €5M error on a €10M transfer (50% off) the same as a €5M error on a €200M transfer (2.5% off). Log-transformation makes the error metric proportionally sensible across the full price range.

### The R² Outcome as a Finding

The model produced a low R² score, meaning it explains only a small proportion of variance in transfer fees. This is not a failure — it is the most important result. It is mathematical proof that age, position, league, and year collectively cannot account for most of what drives a player's price. The unmodelled factors — contract length remaining, selling club's financial desperation, media hype, agent relationships, deadline-day dynamics — are the real drivers. Feature importance from the model confirmed that `Start_Year` (market era) was the strongest predictor by a significant margin, far outweighing any player attribute.

---

## How to Run Locally

**Clone the repository:**
```bash
git clone https://github.com/yourusername/football-transfers.git
cd football-transfers
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Place the raw dataset:**
Download `global_football_transfer_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/rajbirahmed/global-football-transfer-dataset) and move it into the `data/` directory.

**Launch the notebook:**
```bash
jupyter notebook notebooks/football_transfers.ipynb
```
Run cells sequentially from top to bottom. Each phase saves an intermediate CSV that the next phase depends on.

---

## Tech Stack

| Tool | Role |
|---|---|
| Python 3.11+ | Core language |
| Pandas | Data loading, cleaning, transformation, groupby analysis |
| NumPy | Log transformations, array-level math |
| Matplotlib | Base charting engine |
| Seaborn | Statistical visualizations (histograms, scatter, bar, heatmap) |
| Scikit-Learn | Train-test split, Random Forest model, R² scoring |
| Jupyter Notebook | Interactive cell-by-cell execution environment |

---

## Conclusion & Real-World Implications

The predictive model's low R² score is the finding, not the obstacle. This project demonstrates mathematically what football insiders have always known intuitively: **transfer fees are fundamentally irrational by quantitative standards.** Demographic variables — age, position, league — account for a fraction of what a player costs. Market era (year) is the single strongest predictor, confirming that you are buying into an inflationary market as much as you are buying a specific player.

For **scouting departments and technical directors**, this suggests that fee prediction models built purely on player demographics will structurally underperform. The missing variables are qualitative: remaining contract length (urgency premium), a club's financial position on deadline day, individual statistical output, and the player's commercial marketability.

For **financial analysts and club ownership groups**, the mean-vs-median inflation finding highlights a bifurcating market. The elite tier of transfers (€80M+) is growing at an exponential rate disconnected from the median transfer. Clubs without access to sovereign wealth or broadcast-rich league income are structurally unable to compete for the top tier and should build strategy around the feeder league pipeline instead.

For **data science practitioners**, this project illustrates a core professional lesson: a model that fails to predict accurately can produce more actionable insight than one that succeeds — if you understand why it fails.

---

*Built by Samarth Goradia. Data sourced from Kaggle.*