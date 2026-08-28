"""
PASSO 3 - Teste de Hipotese & Validacao Estatistica dos Yields
Hackathon Jovens Talentos Seazone 2026
Analista: Thays Cambi
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

np.random.seed(42)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. CARREGAR BASES
# =============================================================================
print("=" * 80)
print("1. CARREGANDO BASES")
print("=" * 80)

base_airbnb = pd.read_csv(os.path.join(OUTPUT_DIR, "base_airbnb_consolidada.csv"))
vr = pd.read_csv(os.path.join(OUTPUT_DIR, "vivareal_quality_filtered.csv"))

print(f"Airbnb: {len(base_airbnb):,} | VivaReal (qualidade): {len(vr):,}")


# =============================================================================
# 2. NORMALIZACAO DE BAIRROS + TIPOLOGIA (reusar logica do Passo 2)
# =============================================================================
SUBURB_MAP = {
    "alto s\u00e3o bento": "alto sao bento",
    "sert\u00e3o do trombudo": "sertao do trombudo",
    "sert\u00e3ozinho": "sertaozinho",
    "jardim praia mar": "jardim praiamar",
    "meia praia - frente mar": "meia praia",
    "taboleiro": "tabuleiro",
    "ocean tower": "canto da praia",
}


def norm_suburb(s):
    s = str(s).strip().lower()
    if s.endswith("(itapema)"):
        s = s[: -len("(itapema)")].strip()
    return SUBURB_MAP.get(s, s)


base_airbnb["suburb_key"] = base_airbnb["suburb_std"].apply(norm_suburb)
vr["suburb_key"] = vr["suburb"].str.strip().str.lower().apply(norm_suburb)

# ADR medio por bairro x tipologia (Airbnb)
adr_map = (
    base_airbnb[base_airbnb["adr"].notna()]
    .groupby(["suburb_key", "bedrooms_cat"])["adr"]
    .agg(["mean", "median", "count"])
    .rename(columns={"mean": "adr_bairro", "median": "adr_med_bairro", "count": "n_preco"})
    .reset_index()
)

# Atribuir ADR do bairro/tipologia a cada anúncio VivaReal
vr = vr.merge(adr_map, on=["suburb_key", "bedrooms_cat"], how="left")

# Proxy de ocupacao por bairro x tipologia (Airbnb)
# dias_com_preco / periodo_total - proxy de intensidade de precificacao
occ_map = (
    base_airbnb[base_airbnb["adr"].notna()]
    .groupby(["suburb_key", "bedrooms_cat"])["occupancy_proxy"]
    .mean()
    .rename("occ_proxy_bairro")
    .reset_index()
)
vr = vr.merge(occ_map, on=["suburb_key", "bedrooms_cat"], how="left")

# Ocupacao padrao assumida (cenarios)
OCC = {"baixo": 0.35, "base": 0.45, "alto": 0.55}
OPEX_RATE = 0.35
ITBI = 0.03
REGISTRO = 0.01

# Custo de aquisicao
vr["acquisition_cost"] = vr["sale_price_num"] * (1 + ITBI + REGISTRO)
# Condominio + IPTU anual
vr["carry_annual"] = vr["monthly_condo_fee_num"].fillna(0) * 12 + vr["yearly_iptu_num"].fillna(0)

# FILTRO DE SANIDADE DE AREA POR TIPOLOGIA
# Remove aberracoes (ex.: "1 quarto" com 1430 m2 = casas/terrenos mal classificados)
AREA_LIM = {"Studio/1Q": 150, "2Q": 120, "3Q": 180, "4Q+": 300}
area_lim = vr["bedrooms_cat"].map(AREA_LIM)
vr = vr[(vr["usable_area_num"] <= area_lim)].copy()
print(f"\nApos filtro de area por tipologia: {len(vr):,} anuncios")

# Yield liquido por anúncio para cada cenario de ocupacao
for lab, occ in OCC.items():
    gross = vr["adr_bairro"] * occ * 365
    net = gross * (1 - OPEX_RATE) - vr["carry_annual"]
    vr[f"yield_liq_{lab}"] = net / vr["acquisition_cost"] * 100

print("\nVivaReal com ADR atribuido:", len(vr), "anuncios")
print("  com ADR (par no Airbnb):", int(vr["adr_bairro"].notna().sum()))
print("  sem ADR (sem par):", int(vr["adr_bairro"].isna().sum()))


# =============================================================================
# 3. TESTES DE HIPOTESE: MORRETES vs CENTRO vs MEIA PRAIA (por tipologia)
# =============================================================================
print("\n" + "=" * 80)
print("3. TESTES DE HIPOTESE DE YIELD (cenario base) - Mann-Whitney U + Bootstrap CI")
print("=" * 80)


def bootstrap_ci(series, n_boot=2000, ci=0.95):
    """Bootstrap CI para a media do yield."""
    s = series.dropna().values
    if len(s) < 5:
        return np.nan, np.nan
    boots = np.array([
        np.mean(np.random.choice(s, size=len(s), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi


RESULTADOS = []

for cat in ["Studio/1Q", "2Q", "3Q"]:
    print(f"\n{'-'*70}")
    print(f"TIPOLOGIA: {cat}")
    print(f"{'-'*70}")
    bairros = ["morretes", "centro", "meia praia"]
    for b in bairros:
        sub = vr[(vr["suburb_key"] == b) & (vr["bedrooms_cat"] == cat)
                 & vr["adr_bairro"].notna()]["yield_liq_base"]
        lo, hi = bootstrap_ci(sub)
        print(f"  {b:<12} n={len(sub):<5} yield_mediana={sub.median():.2f}%  "
              f"yield_media={sub.mean():.2f}%  CI95=[{lo:.2f}, {hi:.2f}]")
    # Pares
    for a, b in [("morretes", "centro"), ("morretes", "meia praia"), ("centro", "meia praia")]:
        sa = vr[(vr["suburb_key"] == a) & (vr["bedrooms_cat"] == cat) & vr["adr_bairro"].notna()]["yield_liq_base"]
        sb = vr[(vr["suburb_key"] == b) & (vr["bedrooms_cat"] == cat) & vr["adr_bairro"].notna()]["yield_liq_base"]
        if len(sa.dropna()) < 5 or len(sb.dropna()) < 5:
            continue
        stat, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        direction = ">" if sa.median() > sb.median() else "<"
        sig = "SIGNIFICATIVO" if p < 0.05 else "n.s."
        diff = sa.median() - sb.median()
        RESULTADOS.append({
            "tipologia": cat, "bairro_A": a, "bairro_B": b,
            "med_A": round(sa.median(), 2), "med_B": round(sb.median(), 2),
            "diff_mediana_pp": round(diff, 2), "p_value": round(p, 4), "significancia": sig,
        })
        print(f"  {a} {direction} {b} | diff mediana={diff:+.2f}pp | MannWhitney p={p:.4f} | {sig}")

df_res = pd.DataFrame(RESULTADOS)
if len(df_res):
    df_res.to_csv(os.path.join(OUTPUT_DIR, "testes_hipotese_yield.csv"), index=False)


# =============================================================================
# 4. OCUPACAO: comparacao do proxy e analise de sensibilidade
# =============================================================================
print("\n" + "=" * 80)
print("4. OCUPACAO & ANALISE DE SENSIBILIDADE")
print("=" * 80)

# comparacao do proxy de ocupacao entre bairros (Airbnb)
print("\nProxy de ocupacao (dias com preco / periodo) por bairro:")
occ_bairro = (
    base_airbnb[base_airbnb["adr"].notna()]
    .groupby("suburb_key")["occupancy_proxy"]
    .agg(["count", "mean", "median"])
    .round(3)
)
occ_bairro = occ_bairro[occ_bairro["count"] >= 10].sort_values("mean", ascending=False)
print(occ_bairro.to_string())

# Sensibilidade: rank dos bairros (tipologia 2Q e 3Q) pelos 3 cenarios
print("\nResiliencia de Morretes - yield liquido por cenario de ocupacao (2Q):")
for b in ["morretes", "centro", "meia praia", "tabuleiro dos oliveiras", "casa branca"]:
    sub = vr[(vr["suburb_key"] == b) & (vr["bedrooms_cat"] == "2Q") & vr["adr_bairro"].notna()]
    if len(sub) < 5:
        continue
    linha = [f"{lab}={sub[f'yield_liq_{lab}'].median():.2f}%" for lab in OCC]
    print(f"  {b:<24} n={len(sub):<4}  " + "  ".join(linha))

print("\nResiliencia de Morretes - yield liquido por cenario de ocupacao (3Q):")
for b in ["morretes", "centro", "meia praia"]:
    sub = vr[(vr["suburb_key"] == b) & (vr["bedrooms_cat"] == "3Q") & vr["adr_bairro"].notna()]
    if len(sub) < 5:
        continue
    linha = [f"{lab}={sub[f'yield_liq_{lab}'].median():.2f}%" for lab in OCC]
    print(f"  {b:<24} n={len(sub):<4}  " + "  ".join(linha))


# =============================================================================
# 5. SUBANALISE: ROI/Payback simples para o combo campeao (Morretes 3Q)
# =============================================================================
print("\n" + "=" * 80)
print("5. INDICADORES DE RETORNO - Morretes 3Q (cenarios)")
print("=" * 80)

sub = vr[(vr["suburb_key"] == "morretes") & (vr["bedrooms_cat"] == "3Q") & vr["adr_bairro"].notna()]
preco_med = sub["sale_price_num"].median()
area_med = sub["usable_area_num"].median()
adr_val = sub["adr_bairro"].dropna().iloc[0]
carry = sub["carry_annual"].median()

print(f"Preco mediano compra: R$ {preco_med:,.0f} | area: {area_med:.0f} m2 | ADR: R$ {adr_val:.0f} | carry anual: R$ {carry:,.0f}")

for lab, occ in OCC.items():
    gross = adr_val * occ * 365
    net = gross * (1 - OPEX_RATE) - carry
    acq = preco_med * (1 + ITBI + REGISTRO)
    yl = net / acq * 100
    payback = acq / net
    capex_turno = acq * 0.25  # 25% de IM (mobilia/obras) -> ITBI/reg já incluidos; usamos acq
    print(f"  Ocup. {int(occ*100)}%: receita_bruta=R$ {gross:,.0f} | net=R$ {net:,.0f} "
          f"| yield_liq={yl:.2f}% | payback(sem alavancagem)={payback:.1f} anos") 

# =============================================================================
# 6. EXPORT
# =============================================================================
print("\n" + "=" * 80)
print("6. EXPORTANDO")
print("=" * 80)

vr.to_csv(os.path.join(OUTPUT_DIR, "vivareal_yield_listing_level.csv"), index=False)
occ_bairro.to_csv(os.path.join(OUTPUT_DIR, "ocupacao_proxy_por_bairro.csv"))
print("  1. vivareal_yield_listing_level.csv")
print("  2. ocupacao_proxy_por_bairro.csv")
print("  3. testes_hipotese_yield.csv")
print("\n[OK] PASSO 3 CONCLUIDO")
