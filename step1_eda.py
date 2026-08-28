"""
PASSO 1 - EDA & Sanitização
Hackathon Jovens Talentos Seazone 2026
Analista: Thays Cambi
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. CARREGAMENTO DOS DATASETS
# =============================================================================
print("=" * 80)
print("1. CARREGAMENTO DOS DATASETS")
print("=" * 80)

files = {
    "details": "Details_Itapema.csv",
    "hosts": "Hosts_ids_Itapema.csv",
    "mesh": "Mesh_Ids_Data_Itapema.csv",
    "prices": "Price_AV_Itapema.csv",
    "vivareal": "VivaReal_Itapema.csv",
}

dfs = {}
for name, fname in files.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    dfs[name] = df
    print(f"\n{'-'*60}")
    print(f"  {fname}")
    print(f"{'-'*60}")
    print(f"Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"Colunas: {list(df.columns)}")


# =============================================================================
# 2. SCHEMA & TIPOS DETALHADOS
# =============================================================================
print("\n" + "=" * 80)
print("2. SCHEMA & TIPOS DE CADA DATAFRAME")
print("=" * 80)

for name, df in dfs.items():
    print(f"\n{'-'*60}")
    print(f"  SCHEMA: {files[name]}")
    print(f"{'-'*60}")
    schema = pd.DataFrame({
        "dtype": df.dtypes,
        "n_unique": df.nunique(),
        "n_null": df.isnull().sum(),
        "pct_null": (df.isnull().sum() / len(df) * 100).round(1),
        "sample": [str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "ALL NULL" for col in df.columns]
    })
    print(schema.to_string())


# =============================================================================
# 3. AUDITORIA DE NULOS & DUPLICADOS
# =============================================================================
print("\n" + "=" * 80)
print("3. AUDITORIA DE NULOS & DUPLICADOS")
print("=" * 80)

for name, df in dfs.items():
    print(f"\n{'-'*60}")
    print(f"  {files[name]}")
    print(f"{'-'*60}")
    print(f"  Duplicatas (todas as colunas): {df.duplicated().sum():,}")
    
    key_col = None
    if "airbnb_listing_id" in df.columns:
        key_col = "airbnb_listing_id"
    elif "owner_id" in df.columns and name == "hosts":
        key_col = "owner_id"
    elif "listing_id" in df.columns:
        key_col = "listing_id"
    
    if key_col:
        dup_key = df[key_col].duplicated().sum()
        print(f"  Duplicatas por chave ({key_col}): {dup_key:,}")
    
    # Top colunas com nulos
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
    if len(null_cols) > 0:
        print(f"  Colunas com nulos ({len(null_cols)}):")
        for col, n in null_cols.items():
            pct = n / len(df) * 100
            print(f"    {col}: {n:,} ({pct:.1f}%)")
    else:
        print("  [OK] Sem nulos")


# =============================================================================
# 4. CHECAGEM DE STRING "<NA>" COMO MISSING REAL
# =============================================================================
print("\n" + "=" * 80)
print("4. DETECCAO DE '<NA>' COMO STRING (falso non-null)")
print("=" * 80)

for name, df in dfs.items():
    na_string_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            count = (df[col] == "<NA>").sum()
            if count > 0:
                na_string_cols.append((col, count))
    if na_string_cols:
        print(f"\n  {files[name]}:")
        for col, count in na_string_cols:
            pct = count / len(df) * 100
            print(f"    '{col}': {count:,} ocorrências ({pct:.1f}%)")


# =============================================================================
# 5. ESTATISTICAS DESCRITIVAS — DETAILS
# =============================================================================
print("\n" + "=" * 80)
print("5. ESTATISTICAS DESCRITIVAS — Details_Itapema")
print("=" * 80)

det = dfs["details"]
numeric_cols = ["number_of_bathrooms", "number_of_bedrooms", "number_of_beds",
                "number_of_guests", "number_of_reviews", "cleaning_fee",
                "star_rating", "picture_count", "min_nights",
                "guest_satisfaction_overall"]
print("\nVariáveis numéricas:")
print(det[numeric_cols].describe().round(2).to_string())

print("\nDistribuição de listing_type:")
print(det["listing_type"].value_counts().to_string())

print("\nDistribuição de number_of_bedrooms:")
print(det["number_of_bedrooms"].value_counts().sort_index().to_string())

print("\nDistribuição de number_of_bathrooms:")
print(det["number_of_bathrooms"].value_counts().sort_index().to_string())


# =============================================================================
# 6. ESTATISTICAS DESCRITIVAS — PRICES
# =============================================================================
print("\n" + "=" * 80)
print("6. ESTATISTICAS DESCRITIVAS — Price_AV_Itapema")
print("=" * 80)

prc = dfs["prices"]
print(f"\nShape: {prc.shape}")
print(f"Listings únicos: {prc['airbnb_listing_id'].nunique():,}")
print(f"Período de datas: {prc['date'].min()} a {prc['date'].max()}")
print(f"\nEstatísticas de price:")
print(prc["price"].describe().round(2).to_string())

# Preço por listing (média, mediana, std)
price_stats = prc.groupby("airbnb_listing_id")["price"].agg(["mean", "median", "std", "count"])
price_stats.columns = ["price_mean", "price_median", "price_std", "n_obs"]
print(f"\nEstatísticas por listing (n={len(price_stats):,}):")
print(price_stats.describe().round(2).to_string())

# Número de observações por listing
print(f"\nDistribuição de observações por listing:")
print(price_stats["n_obs"].describe().round(1).to_string())


# =============================================================================
# 7. ESTATISTICAS DESCRITIVAS — VIVAREAL
# =============================================================================
print("\n" + "=" * 80)
print("7. ESTATISTICAS DESCRITIVAS — VivaReal_Itapema")
print("=" * 80)

vr = dfs["vivareal"]
print(f"\nShape: {vr.shape}")

# Converter sale_price para numérico
vr["sale_price_num"] = pd.to_numeric(vr["sale_price"], errors="coerce")
vr["usable_area_num"] = pd.to_numeric(vr["usable_area"], errors="coerce")
vr["bedrooms_num"] = pd.to_numeric(vr["bedrooms"], errors="coerce")
vr["monthly_condo_fee_num"] = pd.to_numeric(vr["monthly_condo_fee"], errors="coerce")

print(f"\nSale_price:")
print(vr["sale_price_num"].describe().round(0).to_string())
print(f"\nUsable_area (m²):")
print(vr["usable_area_num"].describe().round(1).to_string())
print(f"\nBedrooms:")
print(vr["bedrooms_num"].value_counts().sort_index().to_string())
print(f"\nSuburb:")
print(vr["suburb"].value_counts().to_string())
print(f"\nListing type (property_type):")
print(vr["property_type"].value_counts().to_string())
print(f"\nMonthly condo fee:")
print(vr["monthly_condo_fee_num"].describe().round(0).to_string())


# =============================================================================
# 8. ESTATISTICAS DESCRITIVAS — MESH (LOCALIZAÇÃO)
# =============================================================================
print("\n" + "=" * 80)
print("8. DISTRIBUIÇÃO POR BAIRRO — Mesh_Ids_Data")
print("=" * 80)

mesh = dfs["mesh"]
print(f"\nSuburb:")
print(mesh["suburb"].value_counts().to_string())
print(f"\nCoordenadas:")
print(f"  Latitude:  {mesh['latitude'].min()} a {mesh['latitude'].max()}")
print(f"  Longitude: {mesh['longitude'].min()} a {mesh['longitude'].max()}")


# =============================================================================
# 9. ESTATISTICAS DESCRITIVAS — HOSTS
# =============================================================================
print("\n" + "=" * 80)
print("9. ESTATISTICAS DESCRITIVAS — Hosts_ids_Itapema")
print("=" * 80)

hosts = dfs["hosts"]
print(f"\nSuperhost:")
print(hosts["is_superhost"].value_counts().to_string())
print(f"\nVerified:")
print(hosts["is_verified"].value_counts().to_string())
print(f"\nYears hosting:")
print(hosts["years_host"].value_counts().sort_index().head(10).to_string())
print(f"\nStar rating (host):")
hosts["star_rating_host_num"] = pd.to_numeric(hosts["star_rating_host"], errors="coerce")
print(hosts["star_rating_host_num"].describe().round(2).to_string())


# =============================================================================
# 10. SANITIZAÇÃO BÁSICA & PREPARAÇÃO
# =============================================================================
print("\n" + "=" * 80)
print("10. SANITIZAÇÃO BÁSICA")
print("=" * 80)

# --- DETAILS ---
det_clean = det.copy()
det_clean["number_of_bathrooms"] = pd.to_numeric(det_clean["number_of_bathrooms"], errors="coerce")
det_clean["number_of_bedrooms"] = pd.to_numeric(det_clean["number_of_bedrooms"], errors="coerce")
det_clean["number_of_beds"] = pd.to_numeric(det_clean["number_of_beds"], errors="coerce")
det_clean["number_of_guests"] = pd.to_numeric(det_clean["number_of_guests"], errors="coerce")
det_clean["cleaning_fee"] = pd.to_numeric(det_clean["cleaning_fee"], errors="coerce")
det_clean["star_rating"] = pd.to_numeric(det_clean["star_rating"], errors="coerce")
det_clean["guest_satisfaction_overall"] = pd.to_numeric(det_clean["guest_satisfaction_overall"], errors="coerce")
det_clean["aquisition_date_dt"] = pd.to_datetime(det_clean["aquisition_date"], errors="coerce")

# Listing type padronizado
det_clean["listing_type_std"] = det_clean["listing_type"].str.strip().str.lower()

# Quartos categorizados
det_clean["bedrooms_cat"] = det_clean["number_of_bedrooms"].apply(
    lambda x: "Studio/1Q" if x <= 1 else ("2Q" if x == 2 else ("3Q" if x == 3 else "4Q+"))
)

print("[OK] Details sanitizado")
print(f"  listing_type: {det_clean['listing_type_std'].value_counts().to_dict()}")
print(f"  bedrooms_cat: {det_clean['bedrooms_cat'].value_counts().to_dict()}")

# --- PRICES ---
prc_clean = prc.copy()
prc_clean["price_num"] = pd.to_numeric(prc_clean["price"], errors="coerce")
prc_clean["date_dt"] = pd.to_datetime(prc_clean["date"], errors="coerce")
prc_clean["year"] = prc_clean["date_dt"].dt.year
prc_clean["month"] = prc_clean["date_dt"].dt.month
prc_clean["dow"] = prc_clean["date_dt"].dt.dayofweek  # 0=Seg, 6=Dom

print("\n[OK] Prices sanitizado")

# --- MESH ---
mesh_clean = mesh.copy()
mesh_clean["latitude_num"] = pd.to_numeric(mesh_clean["latitude"], errors="coerce")
mesh_clean["longitude_num"] = pd.to_numeric(mesh_clean["longitude"], errors="coerce")
mesh_clean["suburb_std"] = mesh_clean["suburb"].str.strip()

print(f"[OK] Mesh sanitizado - bairros: {mesh_clean['suburb_std'].value_counts().to_dict()}")

# --- HOSTS ---
hosts_clean = hosts.copy()
hosts_clean["star_rating_host_num"] = pd.to_numeric(hosts_clean["star_rating_host"], errors="coerce")
hosts_clean["number_of_reviews_host_num"] = pd.to_numeric(hosts_clean["number_of_reviews_host"], errors="coerce")
# is_superhost pode chegar como bool nativo (True/False) ou string ("true"/"false")
hosts_clean["is_superhost_bool"] = hosts_clean["is_superhost"].map(
    {True: True, False: False, "true": True, "false": False}
)

print("[OK] Hosts sanitizado")

# --- VIVAREAL ---
vr_clean = vr.copy()
vr_clean["sale_price_num"] = pd.to_numeric(vr_clean["sale_price"], errors="coerce")
vr_clean["usable_area_num"] = pd.to_numeric(vr_clean["usable_area"], errors="coerce")
vr_clean["bedrooms_num"] = pd.to_numeric(vr_clean["bedrooms"], errors="coerce")
vr_clean["bathrooms_num"] = pd.to_numeric(vr_clean["bathrooms"], errors="coerce")
vr_clean["parking_spaces_num"] = pd.to_numeric(vr_clean["parking_spaces"], errors="coerce")
vr_clean["monthly_condo_fee_num"] = pd.to_numeric(vr_clean["monthly_condo_fee"], errors="coerce")
vr_clean["yearly_iptu_num"] = pd.to_numeric(vr_clean["yearly_iptu"], errors="coerce")
vr_clean["suburb_std"] = vr_clean["suburb"].str.strip().str.lower()

# Preço/m²
vr_clean["price_per_m2"] = vr_clean["sale_price_num"] / vr_clean["usable_area_num"]

# Quartos categorizados
vr_clean["bedrooms_cat"] = vr_clean["bedrooms_num"].apply(
    lambda x: "Studio/1Q" if x <= 1 else ("2Q" if x == 2 else ("3Q" if x == 3 else "4Q+"))
)

print(f"[OK] VivaReal sanitizado")
print(f"  Bairros: {vr_clean['suburb_std'].value_counts().to_dict()}")
print(f"  bedrooms_cat: {vr_clean['bedrooms_cat'].value_counts().to_dict()}")

# Vetores de merge deduplicados
# HOSTS: linhas repetidas por owner_id (mesmos dados, snapshot diferente) -> manter 1
hosts_merge = hosts_clean.drop_duplicates(subset="owner_id", keep="first").copy()
hosts_merge = hosts_merge[["owner_id", "is_superhost_bool", "star_rating_host_num",
                           "number_of_reviews_host_num", "years_host", "months_host"]]
print(f"[OK] Hosts deduplicado: {len(hosts_merge):,} owners unicos p/ merge")

# VIVAREAL: duplicatas reais por listing_id
vr_merge = vr_clean.drop_duplicates(subset="listing_id", keep="first").copy()
print(f"[OK] VivaReal deduplicado: {len(vr_merge):,} listings unicos p/ merge")


# =============================================================================
# 11. CÁLCULO DA RECEITA ESTIMADA POR LISTING (ADR x Dias Observados)
# =============================================================================
print("\n" + "=" * 80)
print("11. RECEITA ESTIMADA POR LISTING")
print("=" * 80)

# ADR = média de preço diário observado no Price_AV
# Número de dias com preço disponível = proxy de disponibilidade/ocupação
# Receita Bruta Anual Estimada = ADR x 365 x taxa_de_ocupação_estimada
# Usaremos: dias_com_preco / dias_totais_do_periodo como proxy de ocupação

price_per_listing = prc_clean.groupby("airbnb_listing_id").agg(
    adr=("price_num", "mean"),
    price_median=("price_num", "median"),
    price_std=("price_num", "std"),
    n_dates=("date_dt", "count"),
    first_date=("date_dt", "min"),
    last_date=("date_dt", "max"),
).reset_index()

# Período total em dias
price_per_listing["period_days"] = (price_per_listing["last_date"] - price_per_listing["first_date"]).dt.days + 1
price_per_listing["occupancy_proxy"] = (price_per_listing["n_dates"] / price_per_listing["period_days"]).clip(0, 1)

# Receita bruta anual estimada
# Nota: occupancy_proxy é a fração de dias com preço disponível
# Usamos isso como proxy de ocupação real
DAYS_IN_YEAR = 365
price_per_listing["est_annual_revenue"] = (
    price_per_listing["adr"] * DAYS_IN_YEAR * price_per_listing["occupancy_proxy"]
)

# revenue p/ dia do período observado
price_per_listing["est_revenue_observed_period"] = (
    price_per_listing["adr"] * price_per_listing["n_dates"]
)

print(f"Listings com dados de preço: {len(price_per_listing):,}")
print(f"\nADR (diária média) por listing:")
print(price_per_listing["adr"].describe().round(2).to_string())
print(f"\nProxy de ocupação (dias_com_preco / periodo_total):")
print(price_per_listing["occupancy_proxy"].describe().round(3).to_string())
print(f"\nReceita Bruta Anual Estimada:")
print(price_per_listing["est_annual_revenue"].describe().round(0).to_string())


# =============================================================================
# 12. CRUZAMENTO: DETAILS + MESH + RECEITA
# =============================================================================
print("\n" + "=" * 80)
print("12. BASE CONSOLIDADA AIRBNB (Details + Mesh + Revenue)")
print("=" * 80)

# Merge Details + Mesh
base_airbnb = det_clean.merge(
    mesh_clean[["airbnb_listing_id", "latitude_num", "longitude_num", "suburb_std"]],
    on="airbnb_listing_id",
    how="left",
    suffixes=("", "_mesh"),
)

# Merge + Revenue
base_airbnb = base_airbnb.merge(
    price_per_listing,
    on="airbnb_listing_id",
    how="left",
)

# Merge + Hosts (deduplicado)
base_airbnb = base_airbnb.merge(
    hosts_merge,
    on="owner_id",
    how="left",
)

print(f"Base consolidada: {base_airbnb.shape[0]:,} linhas × {base_airbnb.shape[1]} colunas")
print(f"Listings com receita estimada: {base_airbnb['est_annual_revenue'].notna().sum():,}")
print(f"Listings sem receita: {base_airbnb['est_annual_revenue'].isna().sum():,}")

# Estatísticas por bairro
print("\n--- ADR por Bairro ---")
adr_bairro = base_airbnb.groupby("suburb_std").agg(
    n_listings=("airbnb_listing_id", "count"),
    adr_medio=("adr", "mean"),
    adr_mediano=("price_median", "mean"),
    receita_anual_media=("est_annual_revenue", "mean"),
    receita_anual_mediana=("est_annual_revenue", "median"),
    pct_superhost=("is_superhost_bool", "mean"),
).round(0)
print(adr_bairro.to_string())

print("\n--- ADR por Tipologia (bedrooms_cat) ---")
adr_tipo = base_airbnb.groupby("bedrooms_cat").agg(
    n_listings=("airbnb_listing_id", "count"),
    adr_medio=("adr", "mean"),
    adr_mediano=("price_median", "mean"),
    receita_anual_media=("est_annual_revenue", "mean"),
    receita_anual_mediana=("est_annual_revenue", "median"),
).round(0)
print(adr_tipo.to_string())

print("\n--- ADR por Bairro x Tipologia ---")
cross = base_airbnb.groupby(["suburb_std", "bedrooms_cat"]).agg(
    n=("airbnb_listing_id", "count"),
    adr=("adr", "mean"),
    receita_anual=("est_annual_revenue", "median"),
).round(0)
print(cross.to_string())


# =============================================================================
# 13. EXPORTAR BASE CONSOLIDADA
# =============================================================================
base_airbnb.to_csv(os.path.join(OUTPUT_DIR, "base_airbnb_consolidada.csv"), index=False)
vr_merge.to_csv(os.path.join(OUTPUT_DIR, "vivareal_clean.csv"), index=False)
price_per_listing.to_csv(os.path.join(OUTPUT_DIR, "price_per_listing.csv"), index=False)

print("\n" + "=" * 80)
print("[OK] PASSO 1 CONCLUIDO - Arquivos exportados para pasta outputs/")
print("=" * 80)
print("  1. base_airbnb_consolidada.csv")
print("  2. vivareal_clean.csv")
print("  3. price_per_listing.csv")
