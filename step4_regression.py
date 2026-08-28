"""
PASSO 4 - Modelagem de Regressao / Explicabilidade do ADR
Hackathon Jovens Talentos Seazone 2026
Analista: Thays Cambi
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from math import radians, sin, cos, asin, sqrt

np.random.seed(42)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"


# =============================================================================
# 1. CARREGAR BASE CONSOLIDADA (com superhost corrigido)
# =============================================================================
print("=" * 80)
print("1. CARREGANDO BASE CONSOLIDADA")
print("=" * 80)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "base_airbnb_consolidada.csv"))
print(f"Total listings: {len(df):,}")
print(f"Listings com ADR: {int(df['adr'].notna().sum()):,}")


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 80)
print("2. FEATURE ENGINEERING")
print("=" * 80)


def haversine(lat1, lon1, lat2, lon2):
    """Distancia em km entre dois pontos (haversine)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


# --- Linha de costa de Itapema (pontos densos ao longo da orla) ---
# Direcao: de sul (Canto da Praia) subindo para norte (Meia Praia)
coast_lat = np.interp(np.linspace(0, 1, 40),
                      [0, 0.3, 0.5, 0.8, 1.0],
                      [-27.083, -27.090, -27.097, -27.115, -27.125])
coast_lon = np.interp(np.linspace(0, 1, 40),
                      [0, 0.3, 0.5, 0.8, 1.0],
                      [-48.618, -48.615, -48.612, -48.607, -48.605])


def dist_beach(lat, lon):
    if np.isnan(lat) or np.isnan(lon):
        return np.nan
    d = np.min([
        haversine(lat, lon, clat, clon) for clat, clon in zip(coast_lat, coast_lon)
    ])
    return d


df["dist_praia_km"] = [
    dist_beach(a, b) for a, b in zip(df["latitude_num"], df["longitude_num"])
]

# --- Vagas: derivar de amenities (garagem/estacionamento) ---
# Detectar presenca de estacionamento/garagem nos amenities (string JSON-like)
amen_str = df["amenities"].fillna("").astype(str)
df["tem_estacionamento"] = amen_str.str.contains("Estacionamento|Garagem|Vaga", case=False).astype(int)

# --- Avaliacoes: valores 0.0 = sem reviews (nao sao notas reais) ---
REVIEW_COLS = ["star_rating", "guest_satisfaction_overall", "accuracy_rating",
               "checkin_rating", "cleanliness_rating", "communication_rating",
               "location_rating", "value_rating"]
for c in REVIEW_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].where(df[c] > 0)  # 0.0 -> NaN (sem reviews)

# --- Booleans padronizados ---
df["can_instant_book"] = df["can_instant_book"].map({True: 1, False: 0})
df["is_professional"] = df["is_professional"].map({True: 1, False: 0})
df["is_new_listing"] = df["is_new_listing"].map({True: 1, False: 0})
df["is_superhost"] = df["is_superhost_bool"].astype(float)
df["is_guest_favorite"] = df["is_guest_favorite"].astype(float)
df["listing_type_apart"] = (df["listing_type_std"] == "apartamento").astype(float)

# --- Target transformado -> log(ADR) para normalizar ---
df["target_log_adr"] = np.log(df["adr"])

print("Features criadas: dist_praia_km, tem_estacionamento, avaliacoes limpas.")


# =============================================================================
# 3. DATASET DE MODELAGEM
# =============================================================================
print("\n" + "=" * 80)
print("3. DATASET DE MODELAGEM (alvo: log ADR)")
print("=" * 80)

FEATURES = [
    "dist_praia_km",
    "number_of_bedrooms",
    "number_of_bathrooms",
    "number_of_beds",
    "number_of_guests",
    "cleaning_fee",
    "picture_count",
    "star_rating",
    "guest_satisfaction_overall",
    "cleanliness_rating",
    "location_rating",
    "value_rating",
    "number_of_reviews",
    "can_instant_book",
    "tem_estacionamento",
    "is_superhost",
    "is_professional",
    "is_guest_favorite",
    "is_new_listing",
    "listing_type_apart",
    "years_host",
    "months_host",
]

X = df[FEATURES].copy()
y = df["target_log_adr"].copy()

# Remover linhas sem ADR
mask = y.notna()
X = X[mask]
y = y[mask]

# Pre-processar: preencher NaN das avaliacoes com a mediana (variaveis continuas)
X_full = X.copy()
for c in X_full.columns:
    if X_full[c].isna().any():
        X_full[c] = X_full[c].fillna(X_full[c].median())

print(f"Amostras para modelagem: {len(X_full):,}")
print(f"Features: {len(FEATURES)}")
print("\nCorrelacao de cada feature com log(ADR) (bruto, Pearson):")
corrs = X_full.apply(lambda col: col.corr(y)).sort_values(ascending=False)
print(corrs.round(3).to_string())
X_full.to_csv(os.path.join(OUTPUT_DIR, "dataset_modelagem.csv"), index=False)


# =============================================================================
# 4. REGRESSAO LINEAR (coeficientes interpretaveis)
# =============================================================================
print("\n" + "=" * 80)
print("4. REGRESSAO LINEAR (coeficientes padronizados)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_tr_s, y_train)

y_pred_lr = lr.predict(X_te_s)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))

print(f"\nR2 (teste): {r2_lr:.3f} | RMSE (em log): {rmse_lr:.3f}")

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coef_padronizado": lr.coef_,
}).sort_values("coef_padronizado", key=abs, ascending=False)
coef_df["exp_coef"] = np.exp(coef_df["coef_padronizado"])
err = np.abs(coef_df["coef_padronizado"]) / np.abs(coef_df["coef_padronizado"]).sum() * 100
coef_df["import_rel_pct"] = err.round(1)

print("\nCoeficientes padronizados (top):")
print(coef_df.to_string(index=False))
coef_df.to_csv(os.path.join(OUTPUT_DIR, "coeficientes_linear.csv"), index=False)


# =============================================================================
# 5. RANDOM FOREST (importancia nao-linear + interacoes)
# =============================================================================
print("\n" + "=" * 80)
print("5. RANDOM FOREST (feature importance)")
print("=" * 80)

rf = RandomForestRegressor(
    n_estimators=400, max_depth=8, min_samples_leaf=8,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print(f"R2 (teste): {r2_rf:.3f} | RMSE (em log): {rmse_rf:.3f}")

# CV
cv = cross_val_score(rf, X_full, y, cv=5, scoring="r2")
print(f"CV R2 (5-fold): media={cv.mean():.3f} +- {cv.std():.3f}")

imp = pd.DataFrame({
    "feature": FEATURES,
    "importance_rf": rf.feature_importances_,
}).sort_values("importance_rf", ascending=False)
imp["import_pct"] = (imp["importance_rf"] / imp["importance_rf"].sum() * 100).round(1)
print("\nImportancia Random Forest (top 10):")
print(imp.head(10).to_string(index=False))
imp.to_csv(os.path.join(OUTPUT_DIR, "importancia_rf.csv"), index=False)

# Permutation importance (mais honesto p/ colinearidade)
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": FEATURES,
    "perm_importance": perm.importances_mean,
}).sort_values("perm_importance", ascending=False)
print("\nPermutation Importance (teste):")
print(perm_df.head(10).to_string(index=False))
perm_df.to_csv(os.path.join(OUTPUT_DIR, "importance_permutation.csv"), index=False)


# =============================================================================
# 6. GRAFICOS
# =============================================================================
print("\n" + "=" * 80)
print("6. GERANDO GRAFICOS")
print("=" * 80)

# Grafico 1: importancia RF vs Permutation (top 12)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_imp = imp.head(12)
axes[0].barh(top_imp["feature"][::-1], top_imp["importance_rf"][::-1], color="#2e86ab")
axes[0].set_title("Feature Importance - Random Forest")
axes[0].set_xlabel("Importancia (Gini)")

top_perm = perm_df.sort_values("perm_importance", ascending=False).head(12)
axes[1].barh(top_perm["feature"][::-1], top_perm["perm_importance"][::-1], color="#d98324")
axes[1].set_title("Permutation Importance (teste)")
axes[1].set_xlabel("Reducao R2")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico_importancia_features.png"), dpi=150)
plt.close()
print("  salvo: grafico_importancia_features.png")

# Grafico 2: correlacao com ADR
corr_series = corrs.sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(corr_series.index, corr_series.values, color="#5b9aa5")
ax.axvline(0, color="grey", lw=1)
ax.set_title("Correlacao Pearson com log(ADR)")
ax.set_xlabel("Correlacao")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico_correlacao_adr.png"), dpi=150)
plt.close()
print("  salvo: grafico_correlacao_adr.png")

# Grafico 3: ADR real vs previsto (RF)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_rf, alpha=0.4, s=20)
lim = [min(y_test.min(), y_pred_rf.min()) - 0.2, max(y_test.max(), y_pred_rf.max()) + 0.2]
ax.plot(lim, lim, "r--", lw=1)
ax.set_xlabel("log(ADR) real")
ax.set_ylabel("log(ADR) previsto")
ax.set_title(f"Random Forest - real vs previsto (R2={r2_rf:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grafico_rf_real_vs_previsto.png"), dpi=150)
plt.close()
print("  salvo: grafico_rf_real_vs_previsto.png")


# =============================================================================
# 7. RESUMO
# =============================================================================
print("\n" + "=" * 80)
print("7. SINTESE - TOP DRIVERS DO ADR")
print("=" * 80)

fusion = imp.rename(columns={"importance_rf": "rf", "import_pct": "rf_pct"}) \
           .merge(perm_df.rename(columns={"perm_importance": "perm"}),
                  on="feature", how="outer")
fusion = fusion.merge(coef_df[["feature", "coef_padronizado"]], on="feature", how="outer")
fusion["rf_rank"] = fusion["rf"].rank(ascending=False)
fusion["perm_rank"] = fusion["perm"].rank(ascending=False)
fusion["media_rank"] = (fusion["rf_rank"] + fusion["perm_rank"]) / 2
fusion = fusion.sort_values("media_rank")
print(fusion.to_string(index=False))
fusion.to_csv(os.path.join(OUTPUT_DIR, "sintese_drivers_adr.csv"), index=False)

print("\n[OK] PASSO 4 CONCLUIDO - arquivos em outputs/")
print("  1. dataset_modelagem.csv")
print("  2. coeficientes_linear.csv")
print("  3. importancia_rf.csv")
print("  4. importance_permutation.csv")
print("  5. sintese_drivers_adr.csv")
print("  6. graficos PNG (importancia, correlacao, real vs previsto)")
