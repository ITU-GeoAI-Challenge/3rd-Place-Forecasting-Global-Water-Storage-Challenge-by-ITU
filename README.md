# 3rd-Place-Forecasting-Global-Water-Storage-Challenge-by-ITU
1. Data & Model Bias
I did not use AI Fairness 360. I checked holdout bias and RMSE by latitude band and by horizon.
Training TWS gaps last 1–2 months; test gaps run 3–18 months (about 28% of test rows). Tropics are too dry (bias −0.095, RMSE 0.696); 15–30° is the worst band. SPEI and soil moisture vanish across long voids; GDO SPI and fAPAR do not.A per-band offset helped CAL and failed HOLD. v8 calibration cost +0.004 on the public board, so both stay off. No lat/lon features. Recency weights (τ = 120 months) favour the later GRACE era.
<div style="display: flex; justify-content: center; align-items: center; min-height: 100vh;">
  <img style="width: 600px; max-width: 100%; height: auto;" alt="image" src="https://github.com/user-attachments/assets/965dfeac-9ce3-4b86-872d-000a08b2226d" />
</div>
2. Model Transparency
I used TreeSHAP on 20,000 holdout rows (Figure 1), not LIME. Top feature is e_S_s40 (0.124): smoothed ERA5-Land column storage. That beat the Kalman base (0.094). Next: GDO SPI-3 (0.077), clim_delta (0.074), e_S_s25 (0.037). Provided SPEI and soil moisture are not in the top five.
Unexpected: ERA5 LAI is almost unused (it is a climatology), so I added MODIS fAPAR. Snow and deep soil appear because SPEI cannot see those stores. ERA5 is 23% of total |SHAP|, GDO 14.5%.
<img style="width: 100vw; height: 600px; object-fit: cover; display: block;" alt="image" src="https://github.com/user-attachments/assets/94c39d16-fd9d-4057-8c4e-cad1a513f82e" />
3. Approach Reusability
A new cell needs TWS history, ERA5-Land, and GDO SPI/fAPAR. No latitude dummy. Extra data is public Copernicus. Kalman and EOF look backward only, so a new month appends. Lead times 1–7 are already mixed, as in the test file.
FAST/SMOKE flags cut cost. Limit: monthly TWS on a 1° grid, not a 1 km basin. The reusable parts are the causal filter, residual trees, and physical features instead of coordinates.
4. Sustainability and Efficiency
CodeCarbon on the full v11 run: 56.5 g CO2e, 0.33 kWh. One Windows laptop (i7-1185G7, 6 threads, ~23 GB RAM), no GPU. Wall clock 11.2 hours, almost all training. The 280,961 test rows take a few minutes.
No deep net. fAPAR is cut from 2.8 GB to a 34 MB cache. Extra trees cost ~1.3 hours for 0.001–0.002 RMSE; FAST mode skips them. Fine for a monthly GRACE fill. A daily job would keep LightGBM only

<img width="311" height="104" alt="image" src="https://github.com/user-attachments/assets/7b6ef6ad-95d9-4087-b2ec-ca50072bccc1" />

Innovation and practicality (20% score, not in the 4-item form)
v8 copied the test gap pattern into validation and got worse on the board (0.657 vs 0.653). A one-line EOF-delta bug fix in v9 did help (public LB 0.651802). v11 adds GDO after a side test of every extra source: SEAS5 was worse (+0.0017), even perfect target-month ERA5 added nothing, GDO SPI/fAPAR was the only gain (−0.0032). Holdout RMSE stays 0.60–0.65 from h=1 to h=7. Known failure: tropical dry bias ~0.09 — I would publish that map, not hide it. Predictions are raw floats, no rounding.

<img width="263" height="123" alt="image" src="https://github.com/user-attachments/assets/31b91070-67f6-41d1-9d6c-326b9d0a8b73" />
How to rerun
Local Windows laptop, not Colab. pip install -r requirements.txt. Place Train.csv and Test.csv next to the notebook, plus era5/, era5_extra/, and gdo/ caches. Run TWS_Solution_v11.ipynb. Output: outputs_v11/submission_v11.csv. Seeds: LightGBM (1,2,3). LightGBM: lr 0.03, 255 leaves, early stopping on CAL.
