"""
PASSO 2 - Cruzamento Airbnb x VivaReal e calculo de YIELD de investimento
Hackathon Jovens Talentos Seazone 2026
Analista: Thays Cambi
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. CARREGAMENTO DAS BASES SANITIZADAS DO PASSO 1
# =============================================================================
print("=" * 80)
print("1. CARREGANDO BASES SANITIZADAS (PASSO 1)")
print("=" * 80)

base_airbnb = pd.read_csv(os.path.join(OUTPUT_DIR, "base_airbnb_consolidada.csv"))
vr = pd.read_csv(os.path.join(OUTPUT_DIR, "vivareal_clean.csv"))

print(f"Airbnb consolidado: {base_airbnb.shape[0]:,} listings")
print(f"VivaReal limpo: {vr.shape[0]:,} anúncios de venda")


# =============================================================================
# 2. MAPEADOR DE BAIRROS (normalizacao de sinonimos entre as duas bases)
# =============================================================================
print("\n" + "=" * 80)
print("2. NORMALIZACAO DE BAIRROS (sinonimos Airbnb x VivaReal)")
print("=" * 80)

# Nomes divergentes entre as bases. Mapear variantes para um nome canonico.
SUBURB_MAP = {
    # variante -> canonico
    "alto s\u00e3o bento": "alto sao bento",
    "alto s\u00e3o bento (itapema)": "alto sao bento",
    "sert\u00e3o do trombudo": "sertao do trombudo",
    "sert\u00e3o do trombudo": "sertao do trombudo",
    "sert\u00e3ozinho": "sertaozinho",
    "sert\u00e3ozinho (itapema)": "sertaozinho",
    "jardim praia mar": "jardim praiamar",
    "jardim praia mar (itapema)": "jardim praiamar",
    "meia praia - frente mar": "meia praia",
    "meia praia - frente mar (itapema)": "meia praia",
    "taboleiro": "tabuleiro",
    "tabuleiro (itapema)": "tabuleiro",
    "tabuleiro dos oliveiras": "tabuleiro dos oliveiras",
    "canto da praia": "canto da praia",
    # bairros que NAO existem no Airbnb (sem par Airbnb) -> mantidos com prefixo
    "andorinha": "andorinha",
    "castelo branco": "castelo branco",
    "estreito": "estreito",
    "itapema": "itapema",
    "ocean tower": "canto da praia",  # Ocean Tower fica no Canto da Praia
}


def norm_suburb(s):
    s = str(s).strip().lower()
    # remover sufixo "(itapema)"
    if s.endswith("(itapema)"):
        s = s[: -len("(itapema)")].strip()
    return SUBURB_MAP.get(s, s)


base_airbnb["suburb_key"] = base_airbnb["suburb_std"].apply(norm_suburb)
vr["suburb_key"] = vr["suburb_std"].apply(norm_suburb)

print("Bairros normalizados (Airbnb):")
print(sorted(base_airbnb["suburb_key"].dropna().unique()))
print("\nBairros normalizados (VivaReal):")
print(sorted(vr["suburb_key"].dropna().unique()))
print("\nBairros VivaReal SEM par no Airbnb:", sorted(set(vr["suburb_key"]) - set(base_airbnb["suburb_key"])))


# =============================================================================
# 3. TIPOLOGIA: padronizar categorias entre as bases
# =============================================================================
# Airbnb ja tem bedrooms_cat (Studio/1Q, 2Q, 3Q, 4Q+)
# VivaReal ja tem bedrooms_cat. Garantir que casam.
assert set(base_airbnb["bedrooms_cat"].dropna()) == set(vr["bedrooms_cat"].dropna()), "cats divergem"


# =============================================================================
# 4. AGREGACOES VIVAREAL - PREÇO DE AQUISICAO POR BAIRRO x TIPOLOGIA
# =============================================================================
print("\n" + "=" * 80)
print("4. PRECO DE AQUISICAO VIVAREAL POR BAIRRO x TIPOLOGIA")
print("=" * 80)

# Filtros de qualidade no VivaReal
vr_q = vr[
    vr["sale_price_num"].notna()
    & vr["usable_area_num"].notna()
    & (vr["usable_area_num"] > 10)          # área útil realista (>10 m2)
    & (vr["sale_price_num"] >= 50000)       # excluir lances absurdos
].copy()

# Custo total anual de carregamento (condominio + IPTU)
vr_q["condo_annual"] = vr_q["monthly_condo_fee_num"].fillna(0) * 12
vr_q["iptu_annual"] = vr_q["yearly_iptu_num"].fillna(0)
vr_q["carry_annual"] = vr_q["condo_annual"] + vr_q["iptu_annual"]

# Preco de compra total + custos de entrada (ITBI ~3%, registro ~1%)
ITBI = 0.03
REGISTRO = 0.01
vr_q["acquisition_cost"] = vr_q["sale_price_num"] * (1 + ITBI + REGISTRO)

print(f"VivaReal apos filtros de qualidade: {len(vr_q):,} anúncios")

# Grupos com dados suficientes
grp_vr = vr_q.groupby(["suburb_key", "bedrooms_cat"]).agg(
    n_venda=("listing_id", "count"),
    preco_mediana=("sale_price_num", "median"),
    preco_media=("sale_price_num", "mean"),
    area_mediana=("usable_area_num", "median"),
    preco_m2_mediano=("price_per_m2", "median"),
    preco_m2_medio=("price_per_m2", "mean"),
    condo_mediano=("carry_annual", "median"),
).reset_index()

print("\nPreco de venda e preco/m2 por bairro x tipologia (grupos com n>=10):")
print(grp_vr[grp_vr["n_venda"] >= 10].sort_values("preco_m2_mediano", ascending=False).to_string(index=False))


# =============================================================================
# 5. AGREGACOES AIRBNB - RECEITA POR BAIRRO x TIPOLOGIA
# =============================================================================
print("\n" + "=" * 80)
print("5. RECEITA AIRBNB POR BAIRRO x TIPOLOGIA")
print("=" * 80)

# Filtrar listings com preço (tenham ADR)
base_r = base_airbnb[base_airbnb["adr"].notna()].copy()

# Tratar ocupacao. O proxy do PASSO 1 (dias com preco / periodo) ~1.0 superestima.
# Para o Passo 2 usaremos RECEITA OBSERVADA REAL por listing:
#   receita_periodo = adr * n_dates  (soma de diárias precificadas no período)
# e o ADR como diária média.
# Para estimar receita anual bruta usamos ADR ajustado por sazonalidade real.
# Vamos reportar ambos: Receita Bruta POTENCIAL (ADR*365) e Receita ESTIMADA com
# ocupacao sazonal assumida. A ocupação real de short stay litorâneo gira em
# torno de 40-55% anual (alta temporada dez-fev + feriados). Modelamos 3 cenários.
OCC = {"baixo": 0.35, "base": 0.45, "alto": 0.55}

print(f"Airbnb listings com ADR (preco disponivel): {len(base_r):,}")

grp_airbnb = base_r.groupby(["suburb_key", "bedrooms_cat"]).agg(
    n_airbnb=("airbnb_listing_id", "count"),
    n_com_preco=("adr", "count"),
    adr_media=("adr", "mean"),
    adr_mediana=("adr", "median"),
    n_dates_med=("n_dates", "median"),
).reset_index()

print("\nADR por bairro x tipologia (grupos com n_com_preco>=10):")
print(grp_airbnb[grp_airbnb["n_com_preco"] >= 10].sort_values("adr_media", ascending=False).to_string(index=False))


# =============================================================================
# 6. CRUZAMENTO AIRBNB x VIVAREAL -> YIELD
# =============================================================================
print("\n" + "=" * 80)
print("6. YIELD DE INVESTIMENTO (Receita / Preco de Compra)")
print("=" * 80)

# Merge das duas agregações
m = grp_vr.merge(grp_airbnb, on=["suburb_key", "bedrooms_cat"], how="inner")

# Ocupacao de referencia
m["occupancy"] = OCC["base"]

# Receita Bruta Anual Estimada = ADR media x Ocupacao x 365
m["gross_annual_rev"] = m["adr_media"] * m["occupancy"] * 365

# Custo total de aquisicao (preco + ITBI + registro)
m["total_acquisition"] = m["preco_mediana"] * (1 + ITBI + REGISTRO)

# Yield Bruto anual = Receita Bruta / Preco de compra (mediana)
m["yield_bruto"] = m["gross_annual_rev"] / m["preco_mediana"]

# Yield Liquido (apos despesas operacionais ~35%: limpeza sazonal, energia,
# internet, manutencao, ABnB fees, impostos) e custo de carregamento (condominio+IPTU)
OPEX_RATE = 0.35
m["net_rev"] = m["gross_annual_rev"] * (1 - OPEX_RATE) - m["condo_mediano"]
m["yield_liquido"] = m["net_rev"] / m["total_acquisition"]

# Preco por m2
m["preco_m2"] = m["preco_mediana"] / m["area_mediana"]

# Exigir amostras mínimas nos dois lados
m_q = m[(m["n_venda"] >= 10) & (m["n_com_preco"] >= 10)].copy()
m_q = m_q.sort_values("yield_liquido", ascending=False)

print("\n--- YIELD (preco mediano, ocupacao", OCC["base"], "anual) ---")
cols = [
    "suburb_key", "bedrooms_cat",
    "n_venda", "n_com_preco",
    "preco_mediana", "preco_m2", "area_mediana",
    "adr_media", "gross_annual_rev", "carry_annual_q" if False else "condo_mediano",
    "yield_bruto", "yield_liquido",
]
show = m_q[["suburb_key", "bedrooms_cat", "n_venda", "n_com_preco",
            "preco_mediana", "preco_m2", "area_mediana", "adr_media",
            "gross_annual_rev", "condo_mediano", "yield_bruto", "yield_liquido"]].copy()
show["yield_bruto"] = (show["yield_bruto"] * 100).round(1)
show["yield_liquido"] = (show["yield_liquido"] * 100).round(1)
show["gross_annual_rev"] = show["gross_annual_rev"].round(0)
show["preco_mediana"] = show["preco_mediana"].round(0)
show["preco_m2"] = show["preco_m2"].round(0)
print(show.to_string(index=False))


# =============================================================================
# 7. ANALISE DIRETA DA TESE DO CENTRO (Studio/1Q Centro vs 3Q Meia Praia)
# =============================================================================
print("\n" + "=" * 80)
print("7. TESTE DIRETO DA TESE: Studio/1Q CENTRO vs 3Q MEIA PRAIA")
print("=" * 80)


def linha(comb):
    """Retorna linha do cruzamento para um combo bairro/tipologia."""
    sub, cat = comb
    row = m_q[(m_q["suburb_key"] == sub) & (m_q["bedrooms_cat"] == cat)]
    return row.iloc[0] if len(row) else None


ALVOS = [
    ("centro", "Studio/1Q"),
    ("meia praia", "Studio/1Q"),
    ("centro", "2Q"),
    ("meia praia", "2Q"),
    ("centro", "3Q"),
    ("meia praia", "3Q"),
]

rows = []
for comb in ALVOS:
    r = linha(comb)
    if r is None:
        print(f"  {comb}: SEM DADOS SUFICIENTES")
        continue
    rows.append({
        "segmento": f"{comb[1]} - {comb[0]}",
        "n_venda": r["n_venda"],
        "n_preco": r["n_com_preco"],
        "preco_mediana": r["preco_mediana"],
        "preco_m2": r["preco_m2"],
        "adr": r["adr_media"],
        "receita_bruta_anual": r["gross_annual_rev"],
        "condo_anual": r["condo_mediano"],
        "yield_bruto_pct": r["yield_bruto"] * 100,
        "yield_liquido_pct": r["yield_liquido"] * 100,
    })

df_tese = pd.DataFrame(rows)
print(df_tese.to_string(index=False))


# =============================================================================
# 8. ANALISE SENSIBILIDADE DA OCUPACAO PARA O COMBO VENCEDOR
# =============================================================================
print("\n" + "=" * 80)
print("8. SENSIBILIDADE DA OCUPACAO")
print("=" * 80)

# Para cada trio comb x producao (baixo/base/alto)
print("\nPassthrough de yield liquido por cenario de ocupacao (top 8 combos):")
for i, row in m_q.head(8).iterrows():
    sub = row["suburb_key"]
    cat = row["bedrooms_cat"]
    prec = row["preco_mediana"]
    adr = row["adr_media"]
    condo = row["condo_mediano"]
    acq = prec * (1 + ITBI + REGISTRO)
    out = []
    for lab, occ in OCC.items():
        gross = adr * occ * 365
        net = gross * (1 - OPEX_RATE) - condo
        yl = net / acq * 100
        out.append(f"{lab}={yl:.1f}%")
    print(f"  {cat}-{sub}: preco={prec:,.0f} adr={adr:.0f} | " + "  ".join(out))


# =============================================================================
# 9. EXPORTAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("9. EXPORTANDO RESULTADOS")
print("=" * 80)

m_q.to_csv(os.path.join(OUTPUT_DIR, "yield_cross_by_neighborhood_tipology.csv"), index=False)
df_tese.to_csv(os.path.join(OUTPUT_DIR, "tese_centro_comparativo.csv"), index=False)
vr_q.to_csv(os.path.join(OUTPUT_DIR, "vivareal_quality_filtered.csv"), index=False)

# Matriz pivot resumida de yield liquido
pivot = m_q.pivot_table(
    index="suburb_key", columns="bedrooms_cat", values="yield_liquido"
) * 100
pivot = pivot.round(1)
pivot.to_csv(os.path.join(OUTPUT_DIR, "matriz_yield_liquido.csv"))
print("Matriz de Yield Liquido (%) por bairro x tipologia:")
print(pivot.to_string())

print("\n[OK] PASSO 2 CONCLUIDO - arquivos em outputs/")
print("  1. yield_cross_by_neighborhood_tipology.csv")
print("  2. tese_centro_comparativo.csv")
print("  3. vivareal_quality_filtered.csv")
print("  4. matriz_yield_liquido.csv")
