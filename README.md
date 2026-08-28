# Hackathon Jovens Talentos Seazone 2026 — Recomendação de Investimento em Short Stay · Itapema (SC)

> **Analista:** Thays Cambi & Copiloto Técnico (Data Science Imobiliário / M&A Short Stay)
>
> **Artefatos:** scripts modulares (`step1_eda.py` à `step4_regression.py`), resultados em `outputs/`,
> conversas com a IA em `ai-log/`.

---

## 🎯 Resumo Executivo & Decisão Final de Investimento

### Decisão
**Comprar apartamentos de 2 e 3 quartos em Morretes (Itapema/SC) para operação short stay** — boa
localização a um **preço de aquisição por m² ~45–55% menor** que Centro/Meia Praia, com **yield líquido
estimado entre 7% e 11% a.a.** e retorno (payback) de 9–14 anos conforme o cenário de ocupação.

### Refutação da Tese Interna (Apartamentos compactos Studio/1Q no Centro)
A tese de que "studios/1Q no **Centro** são a aposta mais eficiente" **não se sustenta** nos dados:

| Métrica | Studio/1Q Centro | 2Q Morretes | 3Q Morretes |
|---|---|---|---|
| Preço mediano de compra | R$ 930 mil | R$ 750 mil | R$ 790 mil |
| Preço por m² | **R$ 17.383/m²** | R$ 10.870/m² | R$ 7.900/m² |
| ADR médio | R$ 482 | R$ 655 | R$ 711 |
| **Yield líquido (ocupa 45%)** | **5,3%** | **8,7%** | **9,2%** |
| Amostra (anúncios vend/ADR) | 28 / 82 | 1.242 / 60 | 306 / 11 |

**Rationale:** o Studio/Centro **gera pouco ADR** (é a tipologia de menor diária — o tamanho é o driver nº 1 do
preço, ~48% na modelagem) **e é caro por m²** (R$ 17,4 mil/m²). O custo de aquisição corrói o retorno:
mesmo o Studio/1Q no **Meia Praia** rende menos (4,6%) por ter preço/m² ainda maior (R$ 21,6 mil/m²).
A tese só "pode" valer *dentro* do Centro, mas **não é a melhor oportunidade** do mercado.

### Oportunidade vencedora: **Morretes 2Q/3Q**
- **Yield líquido 8,7% (2Q) e 9,2% (3Q)** — o mais alto do dataset, **estatisticamente superior** a Centro e
  Meia Praia (Mann-Whitney `p < 0,0001`, diferença de +3,5 a +5,3 p.p.).
- **Robusto à sazonalidade:** mesmo a 35% de ocupação, Morretes 3Q (7,1%) supera Centro/Meia Praia no
  cenário otimista de 55% (4,8–4,9%). O proxy de ocupação é uniforme entre bairros (~0,93), sem viés.
- **Baixa despesa de carregamento:** condomínio+IPTU mediano de apenas ~R$ 500/ano no 3Q de Morretes.
- Morretes é adjacente ao Centro, aproveitando a infraestrutura urbana sem pagar o tenant da orla.

### Estimativa de retorno — 3º quarto em Morretes (cenário de compra típico)

| Cenário de ocupação | Receita bruta | Receita líquida* | **Yield líquido** | **Payback (sem alavancagem)** |
|---|---|---|---|---|
| Baixa temporada (35%) | R$ 90,8 mil | R$ 58,5 mil | **7,1%** | 14,0 anos |
| Cenário base (45%) | R$ 116,7 mil | R$ 75,4 mil | **9,2%** | **10,9 anos** |
| Alta temporada (55%) | R$ 142,7 mil | R$ 92,2 mil | **11,2%** | 8,9 anos |

*50% de ocupação acima do ponto em que o yield cobre os custos; *liquida após 35% de OPEX (limpeza,
energia, internet, manutenção, taxas Airbnb, impostos) e despesa de condomínio/IPTU, sobre custo de
aquisição (preço + ITBI 3% + registro 1%).*

---

## 🧭 Metodologia & Pipeline de Dados

### Passo 1 — EDA & Sanitização (`step1_eda.py`)

Carrega os 5 datasets, audita nulos/duplicados e consolida Airbnb + Mesh + Hosts + Receita em 1 linha por
listing (`outputs/base_airbnb_consolidada.csv`, 4.441 listings).

**Achados e correções-chave:**
- **Bug crítico corrigido:** a tabela `Hosts` tinha **1.383 duplicatas de `owner_id`** (snapshots repetidos) e
  `is_superhost` chegava como *bool nativo* (quebrando o `.map` de string). Sem correção, o merge explodia a
  base para 30.822 linhas (7×) e o superhost ficava 100% nulo.
- Colunas inúteis: `response_rate_shown`/`response_time_shown` (100% nulos) e `rental_price` do VivaReal (100% nulo).
- **Receita:** ADR médio por listing a partir do `Price_AV`; **proxy de ocupação** = nº de dias com preço /
  período. **Cuidado:** o proxy tende a ~1,0 (superestima), por isso todo o yield trabalha com **cenários** de
  ocupação ajustados (35/45/55%).

### Passo 2 — Cruzamento Airbnb × VivaReal & Yield (`step2_cross.py`)

Cruza as bases por **bairro + tipologia** (com mapeador de sinônimos: "Sertãozinho"↔"Sertaozinho",
"Jardim Praia Mar"↔"Jardim Praiamar", "Ocean Tower"→Canto da Praia etc.). Calcula preço mediano de compra,
preço/m² e **Yield Bruto e Líquido** por segmento.

**Achado dominante:** Morretes lidera o yield em 2Q (14,4% bruto / 8,7% líquido) e 3Q (14,8% / 9,2%), impulsionado
pelo **preço/m² baixo (R$ 7,9–10,9 mil)** apesar de ADR médio menor que Centro/Meia Praia.

> ⚠️ Ressalva: alguns combos têm amostra pequena (ex.: 3Q Morretes = 11 listings com preço; Studio Centro =
> 28 anúncios). A recomendação final se apoia nas células com **n grande e significância estatística** (2Q/3Q).

### Passo 3 — Teste de Hipótese & Validação Estatística (`step3_stats.py`)

Atribui o ADR médio do bairro/tipologia (Airbnb) a cada anúncio de venda (VivaReal), gerando distribuições de
yield por anúncio. Testes: **Mann-Whitney U** (medianas) + **Bootstrap 95% CI**.

**Resultados (teste Mann-Whitney, cenário base 45%):**

| Comparação | Δ mediana (p.p.) | p-value | Veredicto |
|---|---|---|---|
| 2Q Morretes vs Centro | +3,51 | < 0,0001 | **Significativo** |
| 2Q Morretes vs Meia Praia | +3,63 | < 0,0001 | **Significativo** |
| 3Q Morretes vs Centro | +5,23 | < 0,0001 | **Significativo** |
| 3Q Morretes vs Meia Praia | +5,32 | < 0,0001 | **Significativo** |
| Studio/1Q Morretes vs Meia Praia | +0,44 | 0,003 | Significativo (fraco) |
| Studio/1Q Morretes vs Centro | −0,09 | 0,76 | **não significativo** |

**Resiliência de ocupação** (yield líquido por cenário):
- Morretes 3Q: 7,1% → 9,2% → 11,2%
- Morretes 2Q: 6,7% → 8,7% → 10,7%
- Centro 3Q: 3,0% → 3,9% → 4,9% · Meia Praia 3Q: 2,9% → 3,8% → 4,8%

### Passo 4 — Regressão / Explicabilidade (`step4_regression.py`)

Alvo = log(ADR); modelos Linear (coef. padronizados) e Random Forest (importância + permutation), CV 5-fold.

**Desempenho:** Linear R² = 0,46 · RF R² = 0,48 (CV 5-fold 0,463 ± 0,071).

**Top drivers do ADR:**

| Variável | Importância RF | Permutation | Coef. Linear | Leitura |
|---|---|---|---|---|
| **Nº de quartos** | **47,5%** | 0,54 | +0,14 | Tamanho = driver nº 1 |
| Listing = apartamento | 11,1% | 0,02 | +0,08 | Apto > casa |
| Nº de banheiros | 7,0% | 0,05 | +0,14 | Correlato ao tamanho |
| Nº de reviews | 5,1% | 0,028 | ~0 | Prova social |
| **Distância da praia** | 3,5% | 0,037 | ~0 | Não-linear; penaliza só longe |
| **Host profissional** | 1,6% | 0,013 | **+0,12** | **Premia +12% no ADR** |
| Superhost | 1,7% | 0,013 | −0,04 | Sem prêmio (controlando tamanho) |

**Insights:**
1. **Tamanho explica ~48%** do ADR → por isso 3Q/2Q superam studios em diária.
2. **Host profissional adiciona ~+12%** de ADR — mais relevante que o selo superhost/guest-favorite.
3. **Distância da praia importa de forma não-linear** (alto na permutation, ~0 no linear): forte apenas para
   quem está muito afastado. Morretes compensa a diária menor com preço de compra bem mais baixo.

---

## ⚠️ Matriz de Risco

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| **Ocupação superestimada** (proxy ~1,0 no Price_AV) | Média | Alto | Trabalhar com 3 cenários (35/45/55%); nunca apresentar yield único otimista |
| **Amostras pequenas** em alguns segmentos (3Q Morretes, Studio) | Média | Médio | Decisão apoiada em células com n grande e `p<0,0001` (2Q/3Q) |
| **Distância da praia aproximada** (proxy de linha de costa, sem coords exatas) | Média | Baixo | Validar 5–10 imóveis-alvo com visita/Georreferenciamento |
| **Variação de preço de aquisição** (oferecidas/negociadas) | Alta | Médio | Negociar alvo ≤ 5–10% abaixo do preço mediano do bairro |
| **Regulação de aluguel por temporada** (fortalecimento de short stay) | Média | Alto | Acompanhar legislação municipal de Itapema; diversificar em 2–3 operadores |
| **Dependência do Airbnb** (concentração de canal) | Média | Médio | Multi-canal: Vrbo/Booking; fluxo de gestão da Seazone |

---

## 🗺 Plano de Ação p/ a Seazone em Itapema

1. **Curto prazo (0–3 meses)** — *Due diligence de alvos:*
   - Selecionar 5–8 unidades 2Q/3Q em **Morretes** (faixa R$ 750–850 mil), validar área real, matrícula,
     condomínio e cobranças. Priorizar prédios com **vagas de garagem e baixo condomínio**.
   - Rodar o `step4_regression.py` por imóvel candidato para estimar o ADR esperado de cada unidade.
2. **Médio prazo (3–12 meses)** — *Estrutura de operação:*
   - **Gestão profissional** (host profissional = +12% ADR) — padrão Seazone, não terceirizar para
     superhost individual.
   - **Otimização de preço dinâmico** por sazonalidade Itapema (dez–fev alta, feriados nacionais), mapeada
     no `Price_AV`.
3. **Acompanhamento contínuo:**
   - Recalcular yield a cada semestre; **portfolio KPI de ocupação real** (substituir o proxy por telemetria
     de reservas próprias).
   - Monitorar regulação de short stay e competição (novos lançamentos em Morretes).

---

## 🔁 Reprodução dos Scripts

Requisitos: **Python 3.11+** (testado em 3.14) e bibliotecas `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

```bash
# instalar dependencias
pip install pandas numpy scipy scikit-learn matplotlib

# Passo 1 - EDA, sanitizacao e consolidacao Airbnb
python step1_eda.py

# Passo 2 - Cruzamento Airbnb x VivaReal e calculo de yield
python step2_cross.py

# Passo 3 - Testes de hipotese (Mann-Whitney, Bootstrap) e sensibilidade de ocupacao
python step3_stats.py

# Passo 4 - Modelagem de regressao / sobre o que explica o ADR
python step4_regression.py
```

> Os scripts **dependem entre si** e devem rodar em ordem (1 → 2 → 3 → 4). Todos leem os artefatos sanitizados
> de `outputs/` e exportam novos arquivos na mesma pasta.

### Estrutura de entrega

```
├── README.md                 # Este relatorio executivo
├── step1_eda.py              # EDA & sanitizacao
├── step2_cross.py            # Yield de investimento
├── step3_stats.py            # Testes de hipotese / validacao
├── step4_regression.py       # Explicabilidade do ADR
├── outputs/                  # Resultados intermediarios e finais (CSV/PNG)
│   ├── base_airbnb_consolidada.csv
│   ├── yield_cross_by_neighborhood_tipology.csv
│   ├── tese_centro_comparativo.csv
│   ├── testes_hipotese_yield.csv
│   ├── vivareal_yield_listing_level.csv
│   ├── matriz_yield_liquido.csv
│   ├── sintese_drivers_adr.csv
│   └── grafico_*.png
├── data/                     # Dados brutos fornecidos
└── ai-log/                   # Conversas com a IA (texto)
```

---

*Seazone — Hackathon Jovens Talentos 2026 · Itapema (SC) · short stay data-driven investment recommendation*
