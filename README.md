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
<img width="455" height="466" alt="image" src="https://github.com/user-attachments/assets/94c39d16-fd9d-4057-8c4e-cad1a513f82e" />

