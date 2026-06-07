import pandas as pd
import numpy as np

oly = pd.read_csv(r"C:\\Users\\sarac\\OneDrive\\Desktop\\ss4\\olympics.csv")
noc = pd.read_csv(r"C:\\Users\\sarac\\OneDrive\\Desktop\\ss4\\noc_regions.csv")
wb = pd.read_csv(r"C:\\Users\\sarac\\OneDrive\\Desktop\\ss4\\wb_development.csv")

# Uklanjamo NA vrednosti i ostavljamo samo letnje olimpijske igre
oly = oly.dropna()
oly = oly[oly['Season'] == 'Summer']

# mapiramo drzave sa njihovom "NOC" skracenicom
noc_subset = noc[['NOC', 'region']].copy()
noc_subset.rename(columns={'region': 'Nation'}, inplace=True)
oly = oly.merge(noc_subset, on='NOC', how='left')

# Uklanjamo duplikate zbog timskih sportova
oly = oly.drop_duplicates(subset=['Year', 'Event', 'Medal', 'Nation'])

# Zadrzavamo samo relevantne kolone
oly = oly[['Year', 'Nation', 'Medal']]

# Pripremamo data set WB
wb = wb[['country', 'date', 'population', 'GDP_current_US', 'life_expectancy_at_birth']]
wb = wb.dropna()

# Potrebna nam je samo godina iz baze WB da bi podatke mogli da spojimo sa bazom OLY 
wb['Year'] = wb['date'].astype(str).str[:4].astype(int)
wb = wb.drop(columns=['date'])

# Brojimo ukupan broj medalja po zemlji i godini
medals_number = oly.groupby(['Nation', 'Year']).size().reset_index(name='Medals')

# Spajamo podatke
final_data = medals_number.merge(wb, how='left', left_on=['Nation', 'Year'], right_on=['country', 'Year'])
final_data = final_data.drop(columns=['country'])
final_data = final_data.dropna()

# skaliramo odredjene prediktore i sve kombinujemo
features = ['population', 'GDP_current_US', 'life_expectancy_at_birth']
X = final_data[features].astype(float)
X_scaled = (X - X.mean()) / X.std()

final_scaled = pd.concat([final_data[['Nation', 'Year', 'Medals']].reset_index(drop=True),
                          X_scaled.reset_index(drop=True)], axis=1)

# Pripremamo odmah i bazu za klasifikaciju (u njoj ce biti i one drzave koje nisu osvojile nijednu medalju)
all_years = sorted(oly['Year'].unique())
all_countries = sorted(wb['country'].unique())
grid = pd.DataFrame([(country, year) for country in all_countries for year in all_years],
                    columns=['Nation', 'Year'])

clf_base = grid.merge(medals_number, on=['Nation', 'Year'], how='left')
clf_base['Medals'] = clf_base['Medals'].fillna(0).astype(int)
clf_base['HasMedal'] = (clf_base['Medals'] > 0).astype(int)

wb_small = wb[['country', 'Year', 'population', 'GDP_current_US', 'life_expectancy_at_birth']]
clf_data = clf_base.merge(wb_small, how='left', left_on=['Nation', 'Year'], right_on=['country', 'Year'])
clf_data = clf_data.drop(columns=['country'])
clf_data = clf_data.dropna()
# Skaliramo i ovde prediktore
Xc = clf_data[features].astype(float)
Xc_scaled = (Xc - Xc.mean()) / Xc.std()
clf_scaled = pd.concat([clf_data[['Nation', 'Year', 'Medals', 'HasMedal']].reset_index(drop=True),
                        Xc_scaled.reset_index(drop=True)], axis=1)

# Podela na test i trening za regresiju
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(final_scaled, test_size=0.3, random_state=123)
# Razdvajamo prediktore i zavisnu promenljivu radi lakseg rada
X_train_reg = train_set[features].values
Y_train_reg = train_set['Medals'].values
X_test_reg = test_set[features].values
Y_test_reg = test_set['Medals'].values

# Linearna regresija
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train_reg, Y_train_reg)

import statsmodels.api as sm
X_train_const = sm.add_constant(X_train_reg)
ols_model = sm.OLS(Y_train_reg, X_train_const).fit()
print(ols_model.summary())

# Evaluacija na test skupu
pred_lin = lin_model.predict(X_test_reg)
pred_lin_int = np.round(pred_lin)
pred_lin_int[pred_lin_int < 0] = 0  # menjamo negativno predvidjene vrednosti za 0, jer to jedino ima smisla
# RMSE i MAE za linearnu regresiju
rmse_lin = np.sqrt(np.mean((Y_test_reg - pred_lin_int) ** 2))
mae_lin = np.mean(np.abs(Y_test_reg - pred_lin_int))
print(f"Linearni model - RMSE: {rmse_lin:.3f}, MAE: {mae_lin:.3f}")

# Ridge regresija + unakrsna validacija za izbor lambde
from sklearn.linear_model import Ridge, RidgeCV
alphas = np.logspace(-4, 4, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=10)
ridge_cv.fit(X_train_reg, Y_train_reg)
best_alpha = ridge_cv.alpha_
print(f"Najbolje lambda: {best_alpha}")
# Treniranje sa najboljim lambda
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_reg, Y_train_reg)
# Predvidjanje i evaluacija kao i ranije
pred_ridge = ridge_model.predict(X_test_reg)
pred_ridge_int = np.round(pred_ridge)
pred_ridge_int[pred_ridge_int < 0] = 0
rmse_ridge = np.sqrt(np.mean((Y_test_reg - pred_ridge_int) ** 2))
mae_ridge = np.mean(np.abs(Y_test_reg - pred_ridge_int))
print(f"Ridge model - RMSE: {rmse_ridge:.3f}, MAE: {mae_ridge:.3f}")

# Random Forest regresija
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=500, max_features=2, random_state=123)
rf_model.fit(X_train_reg, Y_train_reg)
pred_rf = rf_model.predict(X_test_reg)
pred_rf_int = np.round(pred_rf)
pred_rf_int[pred_rf_int < 0] = 0
rf_rmse = np.sqrt(np.mean((Y_test_reg - pred_rf_int) ** 2))
print(f"Random Forest (regresija) - RMSE: {rf_rmse:.3f}")

# Puasonova regresija
X_train_const = sm.add_constant(X_train_reg)
X_test_const = sm.add_constant(X_test_reg)
poisson_model = sm.GLM(Y_train_reg, X_train_const, family=sm.families.Poisson()).fit()
print(poisson_model.summary())
poisson_pred = poisson_model.predict(X_test_const)
poisson_pred_int = np.round(poisson_pred)
poisson_pred_int[poisson_pred_int < 0] = 0  # should not happen for Poisson
poisson_rmse = np.sqrt(np.mean((Y_test_reg - poisson_pred_int) ** 2))
print(f"Puasonova regresija - RMSE: {poisson_rmse:.3f}")

# Uporedjivanje i vizuelizacija modela
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
# LM
plt.subplot(2, 2, 1)
plt.scatter(Y_test_reg, pred_lin_int, color='blue')
plt.plot([0, max(Y_test_reg.max(), pred_lin_int.max())],
         [0, max(Y_test_reg.max(), pred_lin_int.max())], color='red', linewidth=2)
plt.xlabel("Stvarne medalje")
plt.ylabel("Predikcije (int, >=0)")
plt.title("Linearni model")
# RM
plt.subplot(2, 2, 2)
plt.scatter(Y_test_reg, pred_ridge_int, color='darkgreen')
plt.plot([0, max(Y_test_reg.max(), pred_ridge_int.max())],
         [0, max(Y_test_reg.max(), pred_ridge_int.max())], color='red', linewidth=2)
plt.xlabel("Stvarne medalje")
plt.ylabel("Predikcije (int, >=0)")
plt.title("Ridge model")
# RF
plt.subplot(2, 2, 3)
plt.scatter(Y_test_reg, pred_rf_int, color='darkorange')
plt.plot([0, max(Y_test_reg.max(), pred_rf_int.max())],
         [0, max(Y_test_reg.max(), pred_rf_int.max())], color='red', linewidth=2)
plt.xlabel("Stvarne medalje")
plt.ylabel("Predikcije (RF)")
plt.title("Random Forest")
# PR
plt.subplot(2, 2, 4)
plt.scatter(Y_test_reg, poisson_pred_int, color='purple')
plt.plot([0, max(Y_test_reg.max(), poisson_pred_int.max())],
         [0, max(Y_test_reg.max(), poisson_pred_int.max())], color='red', linewidth=2)
plt.xlabel("Stvarne medalje")
plt.ylabel("Predikcije (Poisson)")
plt.title("Poisson regresija")
plt.tight_layout()
plt.show()

# Podela na trening i test skupu za klasifikaciju
train_clf, test_clf = train_test_split(clf_scaled, test_size=0.3, random_state=123, stratify=clf_scaled['HasMedal'])
X_train_clf = train_clf[features].values
Y_train_clf = train_clf['HasMedal'].values
X_test_clf = test_clf[features].values
Y_test_clf = test_clf['HasMedal'].values

# Logisticka regresija
X_train_clf_const = sm.add_constant(X_train_clf)
X_test_clf_const = sm.add_constant(X_test_clf)
logit_model = sm.Logit(Y_train_clf, X_train_clf_const).fit(disp=0)
print(logit_model.summary())
prob_logit = logit_model.predict(X_test_clf_const)
pred_logit = (prob_logit >= 0.5).astype(int)
acc_logit = np.mean(pred_logit == Y_test_clf)
print(f"Logisticka regresija - Accuracy: {acc_logit:.3f}")

# Random Forest klasifikator
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, max_features=2, random_state=123)
rf_clf.fit(X_train_clf, Y_train_clf)
pred_rf = rf_clf.predict(X_test_clf)
prob_rf = rf_clf.predict_proba(X_test_clf)[:, 1]
acc_rf = np.mean(pred_rf == Y_test_clf)
print(f"Random Forest (klasifikacija) - Accuracy: {acc_rf:.3f}")

# Naivni Bajes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train_clf, Y_train_clf)
pred_nb = nb_clf.predict(X_test_clf)
prob_nb = nb_clf.predict_proba(X_test_clf)[:, 1]
acc_nb = np.mean(pred_nb == Y_test_clf)
print(f"Naivni Bajes (klasifikacija) - Accuracy: {acc_nb:.3f}")

# MLP
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, alpha=1e-3, random_state=123)
mlp_clf.fit(X_train_clf, Y_train_clf)
prob_mlp = mlp_clf.predict_proba(X_test_clf)[:, 1]
pred_mlp = (prob_mlp >= 0.5).astype(int)
acc_mlp = np.mean(pred_mlp == Y_test_clf)
print(f"MLP - Accuracy: {acc_mlp:.3f}")

# Uporedjivanje modela i ROC i AUC
from sklearn.metrics import roc_curve, roc_auc_score
fpr_logit, tpr_logit, _ = roc_curve(Y_test_clf, prob_logit)
fpr_rf, tpr_rf, _ = roc_curve(Y_test_clf, prob_rf)
fpr_nb, tpr_nb, _ = roc_curve(Y_test_clf, prob_nb)
fpr_mlp, tpr_mlp, _ = roc_curve(Y_test_clf, prob_mlp)
auc_logit = roc_auc_score(Y_test_clf, prob_logit)
auc_rf = roc_auc_score(Y_test_clf, prob_rf)
auc_nb = roc_auc_score(Y_test_clf, prob_nb)
auc_mlp = roc_auc_score(Y_test_clf, prob_mlp)

# Plot ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr_logit, tpr_logit, color='blue', linewidth=2, label=f"Logit AUC = {auc_logit:.3f}")
plt.plot(fpr_rf, tpr_rf, color='red', linewidth=2, label=f"RF AUC = {auc_rf:.3f}")
plt.plot(fpr_nb, tpr_nb, color='darkgreen', linewidth=2, label=f"NB AUC = {auc_nb:.3f}")
plt.plot(fpr_mlp, tpr_mlp, color='purple', linewidth=2, label=f"MLP AUC = {auc_mlp:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.gca().set_aspect('equal', adjustable='box')
plt.title("ROC krive: Logit vs RF vs Naive Bayes")
plt.legend(loc='lower left')
plt.show()
