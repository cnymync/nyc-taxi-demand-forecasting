# Chasing the Surge
Predicting NYC Taxi Demand for Fleet Positioning

University of Melbourne, MAST30034 Applied Data Science, 2026

## Overview
Predicts taxi demand at zone and hour level across New York City so fleet operators can decide where to position drivers before a shift, rather than reacting after demand moves.

## Approach
- Cleaned and processed over 110 million Yellow and Green taxi trips (January 2024 to May 2026)
- Joined hourly weather, permitted events and public holidays at zone and hour level
- Built a rolling 8 week historical baseline, computed so it never uses future data
- Trained four regression models (Linear, Ridge, LASSO, Gradient Boosting) on 2024 to 2025 and tested on 2026 only
- Judged models on how well they rank the busiest zone hours, not just raw accuracy

## Key findings
- The most accurate model was not the most useful. Gradient Boosting had the lowest error, but Linear Regression and LASSO ranked the busiest zone hours better
- Linear Regression recovered 38% of the true top 50 and 60% of the top 1,000 busiest zone hours, over 1,000 times better than random
- On the busiest test day, a top 20 list correctly identified 13 of the 20 actual busiest zone hours
- In severe weather, demand falls rather than spikes, and every model's error rises about 1.6 times, so operators should not surge position drivers on those days

## Report
See the full report in this repository.

## Tools
Python, PySpark, pandas, scikit learn

## Data sources
- NYC Taxi and Limousine Commission trip records
- Iowa Environmental Mesonet hourly weather (LaGuardia station)
- NYC Open Data permitted events
- US Office of Personnel Management federal holidays

Raw data is not included due to size.

## Acknowledgement
Project template provided by the MAST30034 teaching team.
