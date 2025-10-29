# Bankruptcy Risk Predictor
End-to-end ML pipeline to predict U.S. company bankruptcy risk using 78k+ Nasdaq firms' financial data. The project explores and compares different techniques of data normalization, regularization, scaling, standardization, and sampling. It also demonstrates data cleaning, feature encoding, outlier handling, feature selection, class-imbalance strategies, and model benchmarking (Logistic Regression, Random Forest, XGBoost), with plots and saved artifacts for deployment.


## Dataset
- **Source:** 78,682 NYSE/Nasdaq company-year rows (1999 – 2018) with 18 financial features (X1 – X18) and ```status_label``` (alive/failed).  
- **Target encoding:** ```status_label``` → ```status_encoding``` (alive = 0, failed = 1).  
- **Dropped:** ```company_name```, ```year```.


## Pipeline
1. **Load & Encode**
   - Read ```american_bankruptcy.csv```.
   - Drop ```company_name```, ```year```.
   - Map ```status_label``` → ```status_encoding``` (alive = 0, failed = 1).
2. **Normalization Trials**
   - Square, Inverse, and **Log** (final choice applied via ```np.log1p``` for ```x > 0```, else ```0```).
   - Plot before and after each normalization technique.
3. **Outliers**
   - IQR clipping per feature: ```[Q1 − 1.5 · IQR, Q3 + 1.5 · IQR]```.
   - Update boxplots.  
4. **Correlation-Based Feature Selection**
   - Correlation heatmap.
   - Drop columns with ```|r| > 0.85```.
   - Confirm with updated heatmap.  
5. **Train/Test Split & Scaling**
   - 80/20 split.
   - ```StandardScaler``` fit on train only.
   - Transform train and test.
6. **Class Imbalance**
   - **Undersampling** is used for Logistic Regression and Random Forest training.  
   - **SMOTE** and **SMOTEENN** are used for alternative training of the XGBoost grid on SMOTEENN data.
7. **Models & Evaluation**
   - **RandomForestClassifier** (n_estimators=100); feature importances plot; tested on ```X_test_scaled```.  
   - **LogisticRegression** (+ GridSearchCV over ```penalty ∈ {l1,l2}```, ```C ∈ {0.01…100}```, ```max_iter ∈ {100…400}```) trained on undersampled data; confusion matrix heatmap + ROC/AUC.  
   - **XGBClassifier** (+ GridSearchCV over standard params) trained on SMOTEENN data; confusion matrix + ROC/AUC.
8. **Artifacts**
   - ```col_averages.pkl``` train column means for simple imputation.
   - ```scaler.pkl``` z-score scaler.
   - ```best_model.pkl``` the Random Forest instance saved at the end of the run.


## Requirements
Install the required packages:
```sh
!pip install pandas numpy seaborn scikit-learn imbalanced-learn xgboost matplotlib joblib
```


## How to Run
1. Open the Jupyter notebook or script and execute the cells sequentially from top to bottom.
2. Place the ```american_bankruptcy.csv``` file in the working directory.
3. The code will:
   - Try Square/Inverse/Log normalization (plots shown), then apply Log normalization.
   - Clip outliers (IQR), drop highly correlated features (```|r| > 0.85```).
   - Split ```80/20```, scale with StandardScaler (train stats only).
   - Train/evaluate:
     - Random Forest (undersampled)
     - Logistic Regression (undersampled + GridSearchCV)
     - XGBoost (SMOTEENN + GridSearchCV).
   - Produce plots (box/violin, heatmaps, ROC) and save ```col_averages.pkl```, ```scaler.pkl``, ```best_model.pkl```.
   > If you want to switch sampling strategy per model, uncomment the corresponding blocks (SMOTE/SMOTEENN or undersample) and pass the matching arrays to the chosen estimator.


## Notes
- **No leakage:** scaler fit on train only and applied to test after split.
- **Imbalance caution:** Logistic Regression on imbalanced data can look "good" in terms of accuracy, but it can fail on the minority class—use a confusion matrix and a Receiver Operating Characteristic (ROC) curve, which show TPR (recall) vs. FPR across all classification thresholds; this demonstrates the trade-off between catching positives and triggering false alarms.
- **Plots:** The script generates histograms, box plots, violin plots, correlation heatmaps, confusion matrix heatmaps, and ROC curves for each model.


## Conclusion
Log normalization, IQR (Interquartile Range) outlier clipping, and z-score scaling produced the most stable feature distributions. Undersampling was the most reliable class-imbalance fix among those tried (vs. SMOTE/SMOTEENN), improving minority detection but lowering overall accuracy. Regarding the models trained, Logistic Regression underperformed on the balanced data (≈57% test accuracy; ROC-AUC ≈0.64), indicating that linear models were insufficient. Ensemble tree methods (Random Forest, XGBoost) are recommended primary models given the dataset's non-linear structure and prior evidence.

| Aspect                    | What Worked / Finding                                                                             |
|---------------------------|---------------------------------------------------------------------------------------------------|
| Normalization             | Log normalization chosen as best distributional transform                                         |
| Outliers                  | IQR clipping ```(Q1 − 1.5 · IQR, Q3 + 1.5 · IQR)``` stabilized features                           |
| Scaling & Leakage Control | z-score scaling after ```80/20``` split; fit stats on train only to avoid leakage                 |
| Sampling                  | Undersampling most effective vs. SMOTE/SMOTEENN; improved minority class at cost of ```0```       |
| Logistic Regression       | ```≈57%``` test accuracy (balanced set), ROC-AUC ```≈0.64```; minority improved but still limited |
| Best Models               | Tree ensembles (Random Forest/XGBoost) indicated as better suited for this task                   |
