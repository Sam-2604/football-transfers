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
| Source | [Kaggle — Global Football Transfer Dataset](#) *(placeholder)* |
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

### 7 & 8. The Global Talent Pipeline — Feeder and Consumer Leagues
**Two side-by-side horizontal bar charts** — left panel shows the top feeder leagues; right panel shows the biggest consuming leagues by total spend.
 
**Chart 7 — Feeder leagues (left):** Filters every transfer where the destination is a Big 5 league (Premier League, LaLiga, Serie A, Bundesliga, Ligue 1) but the origin is not. The solid bars show total fee revenue that league received from selling players upward — the dashed pink outline shows how many individual players were exported. These two signals tell different stories: a league with long solid bars but a modest outline exports fewer but higher-value players (elite pipeline). A league with a large outline but shorter bars exports many players cheaply (volume pipeline).
 
Liga Portugal, the Eredivisie, and the Jupiler Pro League consistently rank as the top three feeder leagues — not just by headcount but by fee value. This is structurally explained: these leagues have developed youth systems and scouting networks specifically oriented toward producing players for Big 5 clubs, and their domestic clubs have become expert at buying low, developing, and selling high.
 
**Chart 8 — Consumer leagues (right):** Total transfer spend across all records in the dataset, ranked by league. Pink bars are Big 5 leagues; blue are everyone else. The Premier League dominates by a significant margin — not just in average fee (Chart 4) but in raw total outlay, reflecting both the volume of transfers and the price per transfer. The gap between the Premier League and the rest of the Big 5 is visible; the gap between the Big 5 collectively and all other leagues is stark. The Saudi Pro League and MLS appear as the only non-European competitions with meaningful spend, reflecting the recent Gulf investment wave and designated player rules respectively.
 
The printed summary below the chart shows what percentage of Big 5 spending was on players imported from outside the Big 5 — the proportion of the elite transfer market that runs through the feeder pipeline.

---

## Feature Engineering

Raw data cannot be fed into a machine learning model directly. Text categories are meaningless to an algorithm; fees must be normalized; time must be adjusted for inflation.

**Positional Encoding:** Player positions (e.g., "Centre-Forward", "GoalKeeper") were mapped to traditional football shirt numbers — a domain-knowledge encoding that creates an ordinal "attacking threat" spectrum rather than treating positions as arbitrary categories. Goalkeepers map to 1, strikers to 9, creative midfielders to 10.

**Inflation Adjustment:** A base-year multiplier was calculated using yearly mean fees. Each transfer fee was scaled up by the factor representing how much more expensive the market is today versus the year of transfer. This allows a €50M deal in 2005 to be compared fairly against a €50M deal in 2024 — without adjustment, the model would systematically undervalue older transfers.

**League Tier Ranking:** Rather than using league names as raw text (which the model cannot interpret), leagues were ranked empirically by their average transfer spend. The highest-spending league receives rank 1, and so on. This preserves the ordinal signal (moving to a richer league = premium price) without requiring manual tier assignments.

**International Transfer Flag:** A binary column (`is_international`) was created — 1 if the player moved between countries — a country-level flag rather than a league-level one (for e.g. England alone has both the Premier League and EFL Championship), 0 if not. Analysis showed international transfers average roughly €2.78M more than domestic moves.

**Log Transformation of Target:** The prediction target (`Adjusted_Fee`) was log-transformed before model training. Predicting log-fees rather than raw fees forces the model to think in proportional terms rather than absolute euros, preventing massive outlier transfers from dominating the error signal during training.

---

## Technical Decisions

### Why Random Forest, not Linear Regression?

Linear regression assumes a straight-line relationship between inputs and the target variable. The EDA, specifically the age vs. fee scatter plot and the correlation of -0.03, proved this assumption is violated. Age does not predict fee linearly — a 19-year-old wonderkid and a 26-year-old peak player can cost identical amounts for structurally different reasons.

A Random Forest Regressor resolves this by building hundreds of independent decision trees, each asking a sequence of "if/then" questions (e.g., *if age < 21 AND destination = Premier League THEN predict high*). The ensemble of trees can capture curved, conditional relationships that no straight line could model.

The choice of 100 estimators (`n_estimators=100`) with `random_state=42` (for reproducibility) provides a reasonable balance between model stability and computational cost on a 2,600-row dataset.

### Why Log-Transform the Target?

Transfer fees are log-normally distributed — the log of the fee follows a normal distribution. Predicting on this scale means the model treats a €5M error on a €10M transfer (50% off) the same as a €5M error on a €200M transfer (2.5% off). Log-transformation makes the error metric proportionally sensible across the full price range.

### Why the Model Produced a Negative R² — and Why That's the Finding

R² measures what fraction of the variance in actual fees the model explains. A score of 1.0 is perfect. A score of 0.0 means the model is no better than always predicting the mean fee for every transfer. A **negative** R² means the model's predictions are actively worse than that baseline — it would have been more accurate to just guess the average every time.

Two compounding technical reasons explain this result:

**Reason 1 — Severe overfitting on noise.** Random Forest is a powerful algorithm — so powerful it will find patterns even where none genuinely exist. During training on the 80% split, the model memorised specific historical quirks: rules like *"a 24-year-old moving to a Rank 3 league in 2018 commands a premium."* These quirks are market noise, not universal laws. When the model applied those memorised rules to the unseen 20% test set, they backfired — producing predictions further from the truth than a simple average would have been.

**Reason 2 — The missing feature problem.** The correlation analysis showed that `Age` has a -0.03 correlation with fee — effectively zero. The features available (age, position, league tier, year, transfer type) are the 10% of the puzzle the public dataset provides. The true drivers of a transfer fee — individual performance statistics (goals, assists, progressive carries), remaining contract length, commercial appeal, agent leverage, and selling club desperation — are absent entirely. Forced to predict without them, the model makes wild errors on the test set.

Together these mean the negative R² is not an implementation error — it is a correct result from a correctly built model on an insufficient feature set. **That is precisely the insight.** It proves quantitatively that publicly available demographic and market data alone cannot predict transfer fees, which has direct implications for how scouting and valuation departments should think about data infrastructure.

---

## How to Run Locally

**Clone the repository:**
```bash
git clone https://github.com/Sam-2604/football-transfers.git
cd football-transfers
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Place the raw dataset:**
Download `global_football_transfer_dataset.csv` from [Kaggle](#) *(placeholder)* and move it into the `data/` directory.

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

The model's negative R² is the central finding of this project. It does not mean the model was built incorrectly — it means the features available in any public transfer dataset are structurally insufficient to predict fees. A negative R² is worse than predicting the mean for every transfer, which proves that the patterns the model memorised during training (market noise from the 80% split) actively misled it on unseen data. **Demographic and market variables account for less than zero net predictive value once the model attempts to generalise.**

This demonstrates mathematically what football insiders have always known intuitively: transfer fees are driven by qualitative, private information — contract length remaining, a club's financial desperation on deadline day, a player's commercial marketability, and agent leverage. None of these appear in any public dataset.

For **scouting departments and technical directors**, this result reframes the data infrastructure question. Fee prediction models built purely on demographics will not just underperform — they will produce predictions worse than a baseline guess. Useful valuation tooling requires performance metrics (StatsBomb, Opta), contract data, and commercial indicators at minimum.

For **financial analysts and club ownership groups**, the mean-vs-median inflation finding is the more actionable output. The market is bifurcating: the elite tier (€80M+) inflates exponentially while the median transfer rises modestly. Clubs without sovereign wealth or top-tier broadcast revenue cannot compete in the superstar market and should build strategy around feeder league pipelines instead — Liga Portugal, Eredivisie, and EFL Championship being the primary talent exporters into the Big 5.

For **data science practitioners**, this project illustrates a core professional principle: a model that fails informatively is more valuable than one that succeeds without explanation. The negative R² is a number; understanding the two reasons it happened — overfitting on noise and the missing feature problem — is the knowledge.

---

*Built by Samarth Goradia. Data sourced from Kaggle.*