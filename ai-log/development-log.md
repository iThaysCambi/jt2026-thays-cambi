# Log de Desenvolvimento — Hackathon Seazone 2026 · Itapema (SC)

**Analista:** Thays Cambi (com copiloto técnico IA)
**Período da sessão:** sessão única de desenvolvimento (análise ponta a ponta)
**Objetivo:** recomendar o melhor investimento em short stay em Itapema/SC e refutar/validar a tese interna
de studios/1Q no Centro.

---

## Cronologia e decisões tomadas

### Fase 0 — Leitura do desafio e mapeamento dos dados
- Identificados os **5 datasets** em `data/` e as **chaves de ligação**:
  - `Details_Itapema.csv` (airbnb_listing_id, owner_id)
  - `Hosts_ids_Itapema.csv` (owner_id)
  - `Mesh_Ids_Data_Itapema.csv` (airbnb_listing_id, lat/long, bairro)
  - `Price_AV_Itapema.csv` (airbnb_listing_id, date, price)
  - `VivaReal_Itapema.csv` (listing_id — chave **diferente** do Airbnb)
- **Decisão:** nomes de bairros divergem entre bases; criei um **mapeador de sinônimos** (ex.: Sertãozinho↔Sertaozinho, Jardim Praia Mar↔Jardim Praiamar, Ocean Tower→Canto da Praia).

### Passo 1 — EDA & Sanitização (`step1_eda.py`)
- **Bug crítico encontrado e corrigido:** `Hosts_ids_Itapema.csv` tinha **1.383 duplicatas por `owner_id`**
  (mesmas linhas, snapshots em horários diferentes). Sem deduplicar, o merge multiplicava a base de 4.441
  para 30.822 linhas (7×).
- **Segundo bug:** `is_superhost` chegava como **bool nativo** do pandas, quebrando o `.map` de strings —
  a feature ficava 100% NaN. Corrigido com mapeamento híbrido (bool + string).
- Colunas inúteis removidas da análise: `response_rate_shown`/`response_time_shown` (100% nulos),
  `rental_price` do VivaReal (100% nulo).
- **Decisão metodológica:** Receita Anual = ADR × dias × ocupação. Como o `Price_AV` só captura dias
  precificados, o proxy de ocupação tende a ~1,0. **Decisão:** nunca usar yield único — trabalhar com
  **cenários de ocupação (35/45/55%)**.
- Output: `outputs/base_airbnb_consolidada.csv` (4.441 listings, 1 linha cada).

### Passo 2 — Cruzamento Airbnb × VivaReal → Yield (`step2_cross.py`)
- Cruzamento por **bairro + tipologia** (não há chave comum de imóvel entre as bases).
- Yield Bruto = ADR×365×ocupa÷preço; **Yield Líquido** desconta 35% OPEX + condomínio/IPTU, sobre custo de
  aquisição (preço + ITBI 3% + registro 1%).
- **Achado dominante:** **Morretes lidera** em 2Q (8,7% líquido) e 3Q (9,2%), graças ao preço/m² baixo
  (R$ 7,9–10,9 mil) apesar do ADR menor.
- **Refutação parcial da tese do Centro:** Studio/1Q Centro tem yield 5,3% — melhor *dentro* do Centro, mas
  longe de Morretes, porque o preço/m² do Centro é alto (R$ 17,4 mil/m²).
- Outputs: `yield_cross_by_neighborhood_tipology.csv`, `tese_centro_comparativo.csv`, `matriz_yield_liquido.csv`.

### Passo 3 — Teste de Hipótese & Validação (`step3_stats.py`)
- **Método:** atribuir o ADR médio do bairro/tipologia (Airbnb) a cada anúncio de venda (VivaReal) → distribuição
  de yield por anúncio. Testes **Mann-Whitney U** + **Bootstrap 95% CI**.
- **Filtro de sanidade:** removidos "1 quarto" com ~1.400 m² (casas/terrenos mal classificados), pois distorciam
  médias. Limites de área por tipologia aplicados.
- **Resultado:** Morretes **estatisticamente superior** a Centro e Meia Praia em 2Q e 3Q (`p < 0,0001`,
  Δ +3,5 a +5,3 p.p.). A tese do Centro **refutada**: Studio/1Q Centro não difere de Morretes (p=0,76) e é
  marginal vs Meia Praia (p=0,18).
- **Sensibilidade de ocupação:** Morretes robusto em todos os cenários (35/45/55%) — mesmo a 35% supera
  Centro/Meia Praia a 55%.
- **Retorno (Morretes 3Q):** yield líquido 7,1/9,2/11,2% · payback 14/10,9/8,9 anos.
- Outputs: `testes_hipotese_yield.csv`, `vivareal_yield_listing_level.csv`, `ocupacao_proxy_por_bairro.csv`.

### Passo 4 — Modelagem / Explicabilidade (`step4_regression.py`)
- Alvo = log(ADR); **Regressão Linear** (coefs. padronizados) + **Random Forest** (importância + permutation).
- Desempenho: Linear R² = 0,46 · RF R² = 0,48 (CV 5-fold 0,463 ± 0,071).
- **Top drivers:** nº quartos (47,5% importância / perm 0,54), listing apartamento, banheiros; **host
  profissional +12% ADR**; superhost sem prêmio controlando tamanho; **distância da praia não-linear**.
- `dist_praia_km` calculada por **haversine até uma linha de costa aproximada** (proxy; sem coords exatas da orla).
- Outputs: `dataset_modelagem.csv`, `coeficientes_linear.csv`, `importancia_rf.csv`,
  `importance_permutation.csv`, `sintese_drivers_adr.csv` + 3 gráficos PNG.

### Passo 5 — Recomendação Executiva & README
- Escrito o `README.md` completo: resumo executivo, decisão (Morretes 2Q/3Q), refutação da tese, matriz de
  risco, plano de ação e instruções de reprodução.

---

## Decisões-chave de entrega

| Decisão | Justificativa |
|---|---|
| **Recomendar Morretes 2Q/3Q** (não Studio/Centro) | Yield líquido 8,7–9,2% vs 5,3%; significância p<0,0001; robusto à sazonalidade |
| **Trabalhar com cenários de ocupação** (35/45/55%) | Proxy do Price_AV ≈1,0 superestima ocupação; honestidade do report |
| **Filtrar outliers de área** | "1 quarto" com 1.400 m² distorciam estatística; medianas robustas adotadas |
| **Não commitar `.env` e `opencode.json`** | `.env` contém segredos; `opencode.json` é config local do ambiente |
| **Usar Mann-Whitney U (medianas)** | Robusto à não-normalidade e outliers do mercado de venda |

---

## Limitações & ressalvas (transparência)
1. **Ocupação não medida** — o `Price_AV` captura dias precificados, não reservas reais; por isso cenários.
2. **Amostras pequenas** em alguns segmentos (3Q Morretes = 11 preços, Studio Centro = 28 vendas).
3. **Distância da praia aproximada** (proxy de linha de costa, sem coordenadas exatas da orla).
4. **Modelo explica ~46–48%** da variância do ADR (R²) — fatores não capturados (estado do imóvel, vista) importam.

---

## Verificação final
- Todos os 4 scripts (`step1_eda.py` → `step4_regression.py`) rodam ponta a ponta **sem erros**, em sequência.
- 18 artefatos de resultado em `outputs/` (CSV + PNG).
- `.gitignore` e `requirements.txt` criados.
