# Relatório — Previsão do Preço de Alojamentos Airbnb em Lisboa

**Machine Learning — Practical Project 1**
Notebook principal: `project_analisys_c.ipynb` · Código: `src/`

---

## Resumo

O objetivo é prever o preço por noite (`price`) de anúncios Airbnb em Lisboa. O modelo base é uma regressão linear treinada com **gradient descent implementado de raiz**, com expansão polinomial. Foram comparadas três estratégias de tratamento de outliers, validação cruzada, regularização L2, a generalização para outra cidade (Nova Iorque), uma versão de **classificação** (4 faixas de preço, com regressão logística feita de raiz) e uma abordagem que usa apenas o **texto** dos anúncios, codificado por um LLM de embeddings.

| Abordagem | Melhor configuração | R² teste | MAE teste |
|---|---|---|---|
| Regressão tabular, sem tratar outliers | grau 3 | 0.140 | 80.4 € |
| Regressão tabular, outliers **cap** | grau 3 | **0.590** | 44.0 € |
| Regressão tabular, outliers **filter** | grau 3 | 0.455 | **40.7 €** |
| Regressão só com texto (LLM, 1024 dims, filter no preço) | linear | 0.460 | 42.0 € |
| Classificação em 4 faixas de preço | softmax grau 2 | accuracy 0.566 · AUC 0.81 | 91% na classe certa ou vizinha |

Principais conclusões:
1. **Tratar outliers é o fator com mais impacto.** Sem tratamento, o R² fica em ~0.15, porque meia dúzia de preços extremos dominam o erro quadrático.
2. **Graus 2–3 são o ponto ideal.** O grau 4 diverge sempre (15–17 mil features, sem tratamento das restantes variáveis contínuas).
3. **A L2 testada (λ ≤ 0.75) não tem efeito.** Na implementação, o termo é dividido por *n* ≈ 14 mil.
4. **O texto sozinho prevê o preço quase tão bem como as variáveis estruturadas.**
5. **O modelo não se transfere para Nova Iorque.** Mesmo com distância ao centro em vez de coordenadas, o R² fica negativo.

---

## 1. Dados e preparação

**Dataset:** `data/listings.csv` — 24 876 anúncios × 90 colunas (Inside Airbnb, Lisboa, recolha de junho/julho 2026).

### 1.1 Limpeza inicial de colunas
Foram removidas as colunas que não contribuem para prever o preço ou que o prejudicam:
- **Identificadores e URLs:** `id`, `listing_url`, `scrape_id`, `picture_url`, `host_id`, `host_url`, `host_profile_*`, `host_*_url`, `license`, `host_name`, `source`.
- **Metadados do scraping:** `last_scraped`, `calendar_updated`, `calendar_last_scraped`.
- **Texto livre** (`name`, `description`, `neighborhood_overview`, `host_about`): removido da análise tabular e reaproveitado mais tarde na secção 15 (embeddings).
- **Leakage:** `price_quote_*` e `estimated_revenue_l365d` contêm informação direta do preço.

### 1.2 Valores em falta
| Passo | Linhas |
|---|---|
| Dataset original | 24 876 |
| 9 colunas 100% vazias removidas (`host_since`, `host_response_*`, `host_neighbourhood`, `neighbourhood`, `instant_bookable`, …) | 24 876 |
| Remoção das linhas sem `price` | 22 636 |
| Drops pontuais: linhas com NaN em `minimum_nights`, nos dados do anfitrião, em `bathrooms_text` ou em `has_availability` | **22 575** |

Os valores em falta vêm em grupos coerentes (as mesmas linhas em todas as colunas do grupo). Por exemplo, as 2 683 linhas sem reviews têm em falta todos os `review_scores_*` e as datas das reviews. O tratamento foi o seguinte:
- `host_location` → `"Não especificado"`;
- `bathrooms` (1 917 NaN), `bedrooms` (3 890) e `beds` (745) → mediana do grupo (`room_type`, `accommodates`);
- reviews → `has_reviews` (flag), `reviews_per_month = 0`, `review_scores_* = −1` (sentinela fora do intervalo 0–5);
- `first_review` e `last_review` foram removidas.

O dataset final tem 22 575 linhas × 60 colunas, sem NaN.

---

## 2. Outliers

### 2.1 Deteção
Foram comparados dois critérios robustos:
- **IQR (Tukey, k = 1.5)**
- **MAD (z-score modificado > 3.5, Iglewicz & Hoaglin)**: mais resistente quando a distribuição tem cauda longa, como acontece com o preço. **Foi o critério usado.**

| Variável | Outliers IQR | Outliers MAD | Limites MAD | Concordância IQR/MAD |
|---|---|---|---|---|
| `price` | 1 678 (7.43%) | 1 390 (6.16%) | [−126 ; 398] € | 82.8% |
| `minimum_nights` | 1 401 (6.21%) | 1 124 (4.98%) | [−3.2 ; 7.2] | 80.2% |
| `accommodates` | 1 640 (7.26%) | 179 (0.79%) | [−6.4 ; 14.4] | 10.9% |

### 2.2 Estratégias de tratamento (aplicadas às 3 variáveis acima)
| Estratégia | O que faz | Linhas |
|---|---|---|
| `none` | nada | 22 575 |
| `cap` | winsorization: corta os valores nos limites MAD | 22 575 |
| `filter` | remove as linhas com outliers (coluna a coluna, sequencialmente) | 19 717 (−2 858) |

Os ficheiros resultantes estão em `outlier_versions/listings_{none,cap,filter}.csv`.

> **Nota:** as restantes variáveis contínuas **não** foram tratadas e continuam com valores extremos, mesmo no `filter`. Exemplos: `bedrooms` até 43 (z = 47.7), `maximum_nights` até 10 000 (z = 20.3), `host_listings_count` até 1 150 (z = 12.2). A expansão polinomial não trata estes valores, **amplifica-os** (43⁴ ≈ 3.4 milhões), e o `StandardScaler` também não os resolve. É uma das causas prováveis da divergência no grau 4.

---

## 3. Features e modelo

### 3.1 Preparação de features (`src/prepare_features.py`)
Cada coluna é classificada automaticamente:
- **contínua**: expandida polinomialmente e standardizada;
- **binária** (t/f → 0/1);
- **categórica**: one-hot;
- **excluída**: redundante, leakage ou alta cardinalidade.

No `filter` resultam **51 features**: 22 contínuas (coordenadas, quartos, camas, casas de banho, noites mín./máx., disponibilidade, reviews, scores, tempo de anfitrião, …) e 29 passthrough (flags e one-hot de `room_type`, `neighbourhood_group_cleansed` e `accommodates`). Não há seleção de features.

### 3.2 Regressão linear com gradient descent (`src/linear_regression.py`)
- Full-batch gradient descent, loss MSE + λ·Σw²/n (L2 opcional, sem penalizar o bias).
- Expansão `PolynomialFeatures(grau)` + `StandardScaler`, só nas features contínuas. O fit é feito apenas no treino.
- Checkpoint do melhor modelo pela loss de **treino**, para que a validação não influencie os pesos. Se a loss ficar NaN/Inf, o treino para e repõe o último checkpoint válido.
- Hiperparâmetros base: `learning_rate = 0.001`, `n_epochs = 1000`, split 70/15/15 com `random_state = 42`.

| Grau | Nº de features (none/cap · filter) |
|---|---|
| 1 | 45 · 51 |
| 2 | 321 · 304 |
| 3 | 2 621 · 2 328 |
| 4 | 17 571 · 14 978 |

---

## 4. Resultados — split único 70/15/15 (secção 8)

Métricas no **conjunto de teste**:

| Estratégia | Grau | MAE | RMSE | R² |
|---|---|---|---|---|
| none | 1 | 86.88 | 374.85 | 0.116 |
| none | 2 | 83.42 | 377.66 | 0.102 |
| none | 3 | 80.40 | 369.72 | 0.140 |
| none | 4 | 309.50 | 625.23 | −1.460 |
| cap | 1 | 49.06 | 67.70 | 0.491 |
| cap | 2 | 46.11 | 63.70 | 0.549 |
| **cap** | **3** | 44.05 | 60.71 | **0.590** |
| cap | 4 | 239.78 | 343.91 | −12.15 |
| filter | 1 | 44.88 | 60.71 | 0.323 |
| filter | 2 | 42.49 | 56.50 | 0.414 |
| **filter** | **3** | **40.65** | **54.46** | 0.455 |
| filter | 4 | 161.92 | 190.24 | −5.65 |

**Baseline ingénuo** (prever sempre a mediana do treino, no mesmo conjunto de teste):

| Estratégia | MAE baseline | RMSE baseline | Melhor MAE do modelo | Redução do MAE |
|---|---|---|---|---|
| none | 103.24 | 402.60 | 80.40 | −22% |
| cap | 70.56 | 97.77 | 44.05 | −38% |
| filter | 56.45 | 75.10 | 40.65 | −28% |

**Leitura dos resultados:**
- O `cap` tem o melhor R² e o `filter` o menor erro absoluto. As duas estratégias não são diretamente comparáveis, porque cada uma altera o próprio target: o `cap` corta os preços altos e o `filter` remove-os.
- Os graus 2 e 3 melhoram face ao grau 1 em todas as estratégias. O grau 4 falha em todas.

---

## 5. Validação cruzada 5-fold (secção 12)

Mesmos hiperparâmetros da secção 8, `KFold(shuffle=True, random_state=42)`. Média dos 5 folds de **validação**:

| Estratégia | Grau 1 R² | Grau 2 R² | Grau 3 R² | Grau 4 R² | Melhor MAE |
|---|---|---|---|---|---|
| none | 0.141 ± 0.054 | 0.169 ± 0.063 | **0.188** ± 0.076 | −0.874 | 79.56 (g3) |
| cap | 0.488 ± 0.019 | 0.542 ± 0.015 | **0.559** ± 0.065 | −8.450 | 45.33 (g3) |
| filter | 0.346 ± 0.020 | **0.413** ± 0.014 | 0.389 ± 0.154 | −5.575 | 40.94 (g3) |

- **Confirma o split único:** com tratamento de outliers, o R² passa de ~0.15 para 0.4–0.56.
- **Nos graus 1–2, o desvio-padrão entre folds é baixo** (≈0.015–0.02), por isso as diferenças entre estratégias são reais.
- **No grau 3 a variância sobe** (dp 0.065 no cap, 0.154 no filter). No `filter`, o grau 3 fica até **abaixo do grau 2** em média e o gap treino/validação aumenta (0.47 vs 0.39). **O grau 2 é a escolha mais segura.**
- **O grau 4 diverge em 15 de 15 folds.** O gradient descent explode logo nas primeiras épocas e o checkpoint restaurado é o da época 0. Com ~15–17 mil features mal condicionadas, um `learning_rate` de 0.001 já é grande demais.

---

## 6. Regularização L2 (secção 13)

λ ∈ {0.25, 0.5, 0.75}, estratégia `filter`, o mesmo split da secção 8 e graus 1–4.

| Grau | R² teste (λ = 0) | R² teste (λ = 0.25 / 0.5 / 0.75) |
|---|---|---|
| 1 | 0.3231 | 0.3231 / 0.3231 / 0.3231 |
| 2 | 0.4138 | 0.4138 / 0.4138 / 0.4138 |
| 3 | 0.4554 | 0.4554 / 0.4554 / 0.4554 |
| 4 | −5.6471 | −5.6471 / −5.6471 / −5.6471 |

- **Não há efeito prático:** as métricas são iguais até à 4.ª casa decimal e a norma dos pesos só muda na 4.ª casa (69.661 → 69.659 no grau 1).
- **A causa está na implementação:** o gradiente do termo L2 é `2λ·w/n`, com n ≈ 13 800. A força efetiva fica em ~10⁻⁴, e em 1000 épocas encolhe os pesos menos de 0.01%. Para se notar, λ teria de estar na ordem de 10²–10⁴.
- **A L2 também não salva o grau 4.** Aí o problema é a divergência do otimizador, não overfitting.

---

## 7. Generalização para outra cidade — Nova Iorque (secções 10–11)

Os modelos treinados em Lisboa (estratégia `filter`) foram aplicados ao dataset de Nova Iorque (375 anúncios após o mesmo tratamento), sem novo treino.

| Grau | R² com lat/lon absolutos | R² com distância ao centro | MAE com distância ao centro |
|---|---|---|---|
| 1 | −4 478 | −0.062 | 57.1 € |
| 2 | −34 430 | −0.073 | 57.9 € |
| 3 | −340 368 | −0.594 | 72.6 € |
| 4 | −1 043 | −6.885 | 196.9 € |

- **Com latitude/longitude absolutas, os resultados não fazem sentido.** As coordenadas de NY ficam a centenas de desvios-padrão da distribuição de Lisboa usada pelo `scaler`, e a expansão polinomial amplifica-as ainda mais.
- **Substituir as coordenadas por `dist_centro_km`** (distância Haversine ao centro geográfico de cada cidade) resolve o colapso numérico. Mesmo assim, o R² continua **negativo**: o modelo é pior do que prever a média de NY. O nível de preços e a estrutura de cada mercado são diferentes, e um modelo de Lisboa não se transfere diretamente.

---

## 8. Classificação em 4 faixas de preço (secção 14)

### 8.1 Problema e modelo
- **Novo target `price_cat`:** o preço é dividido pelos quartis **do treino** (Q1 = 90.5 €, Q2 = 131 €, Q3 = 183 €):
  - 0 **barato** (≤ Q1)
  - 1 **medio-barato**
  - 2 **medio-caro**
  - 3 **caro** (> Q3)

  As classes ficam equilibradas (~25% cada), por isso o acaso tem ~25% de accuracy.
- **Modelo (`src/logistic_regression.py`):** regressão logística **multinomial (softmax)** implementada de raiz só com numpy, com entropia cruzada e full-batch gradient descent (`learning_rate = 0.1`, 1000 épocas, sem L2). Usa a estratégia `filter`, o mesmo split da secção 8 e graus 1–3.
- **Métricas implementadas de raiz:** matriz de confusão, precision/recall/F1, balanced accuracy, ROC e AUC (one-vs-rest, macro e micro), curva precision-recall e AP, e log loss. Foram validadas contra o `sklearn` e dão valores iguais.

### 8.2 Resultados (teste)
| Grau | Features | Accuracy | Macro F1 | AUC macro | AUC micro | Log loss | Accuracy ±1 classe |
|---|---|---|---|---|---|---|---|
| 1 | 51 | 0.551 | 0.549 | 0.796 | 0.813 | 1.028 | 0.897 |
| **2** | 304 | **0.566** | **0.563** | **0.811** | **0.827** | **0.982** | **0.913** |
| 3 | 2 328 | 0.398 | 0.351 | 0.662 | 0.663 | 2.794 | 0.735 |

Métricas por classe do grau 2:

| Classe | Precision | Recall | F1 | AUC | AP |
|---|---|---|---|---|---|
| barato | 0.732 | 0.770 | 0.751 | 0.924 | 0.829 |
| medio-barato | 0.475 | 0.527 | 0.499 | 0.753 | 0.453 |
| medio-caro | 0.409 | 0.332 | 0.366 | 0.707 | 0.398 |
| caro | 0.626 | 0.643 | 0.634 | 0.859 | 0.685 |

- **O modelo acerta 57% das vezes** (contra 25% do acaso), e **91% das previsões ficam na classe certa ou numa vizinha**. Erros grosseiros, como trocar barato por caro, são raros.
- **As classes extremas são fáceis e as do meio difíceis.** O `medio-caro` fica "espremido" entre as vizinhas (recall 0.33). As fronteiras estão próximas (91 → 131 → 183 €), e o MAE da regressão (~41 €) é da mesma ordem de grandeza que a largura das classes do meio.
- **As probabilidades são informativas.** A accuracy sobe de forma consistente com a confiança do modelo: 37% quando a probabilidade máxima é ≤ 0.4 e 90% quando passa de 0.8.
- **O grau 3 falha por otimização.** A log loss (2.79) é **pior que o acaso** (ln 4 ≈ 1.39), porque o gradient descent oscila com 2 328 features e não converge.

---

## 9. Representações latentes do texto com um LLM (secção 15)

### 9.1 Método
- **Texto (`src/llm_embeddings.py`):** parte do **dataset cru** e usa todas as colunas de texto, exceto links, identificação, metadados do scraping e `price`/`price_quote_*` (leakage). Ficam 16 colunas: nome, descrição, tipo de propriedade e de quarto, bairro, casas de banho, amenities, flags do anfitrião, datas das reviews, localização e "about" do anfitrião. Cada anúncio vira um texto `Campo: valor` (mediana de 1 414 caracteres).
- **LLM:** `Qwen/Qwen3-Embedding-0.6B` (0.6 mil milhões de parâmetros), na GPU (RTX 3060, float16, máx. 512 tokens). Gerar os embeddings dos 24 876 anúncios levou cerca de 15 minutos. O modelo foi treinado com *Matryoshka Representation Learning*: as primeiras 256 e 512 componentes também são embeddings válidos. Por isso os 3 CSVs (`Data_clean/embeddings/listings_emb_{256,512,1024}.csv`) vêm do **mesmo** modelo, truncado e renormalizado, e diferem apenas na dimensão.
- **Regressão:** `LinearRegressionGD`, split 70/15/15, `learning_rate = 0.005`. Com 0.001 o modelo fica longe de convergir e com 0.01 a versão quadrática de 1024 dims diverge.
  - **linear:** as `d` componentes;
  - **quadrática:** `x_i` e `x_i²` (2d features). Os termos cruzados são impraticáveis: seriam 525 mil features para d = 1024, ~60 GB.
- **Target:** `none` (preço cru) e `filter` (MAD aplicado só ao `price`, 21 236 linhas).

### 9.2 Resultados (teste)
| Dim | Tratamento | Linear R² | Quadrática R² | Linear MAE | Quadrática MAE |
|---|---|---|---|---|---|
| 256 | filter | 0.402 | 0.412 | 44.8 | 44.5 |
| 512 | filter | 0.433 | 0.444 | 43.3 | 42.9 |
| 1024 | filter | **0.460** | 0.457 | **42.0** | 42.2 |
| 256 | none | 0.136 | 0.191 | 119.6 | 126.4 |
| 512 | none | 0.219 | 0.296 | 124.3 | 135.0 |
| 1024 | none | 0.299 | 0.403 | 130.2 | 144.3 |

- **Só com o texto, o R² chega a 0.46 e o MAE a 42 €.** É melhor que o modelo tabular `filter` de grau 1 (0.32) e de grau 2 (0.41), e só fica abaixo do tabular `cap` (0.49–0.59). O texto contém quase toda a informação estrutural (tipo, bairro, casas de banho, amenities) e acrescenta qualidade e estilo.
- **Mais dimensões ajudam, com retornos decrescentes.** De 256 para 1024 o R² sobe +0.06, mas o gap treino−teste também cresce (0.04 → 0.07 no linear, 0.05 → 0.13 no quadrático).
- **Os termos quadráticos quase não acrescentam nada** (+0.01 ou empate). Os embeddings já são uma representação não-linear do texto.
- **Sem tratar outliers, o R² engana.** O R² sobe com a dimensão, mas o **MAE piora** (120 → 144 €): o modelo ajusta-se aos preços extremos à custa do anúncio típico.
- **A comparação com o modelo tabular é aproximada.** Os conjuntos de linhas são diferentes (aqui o `filter` só filtra o preço).

---

## 10. Limitações e trabalho futuro

1. **Outliers das restantes variáveis contínuas.** Seria útil aplicar `log1p` às contagens e cap a `bedrooms`, `beds`, `bathrooms` e `maximum_nights`, ou usar `RobustScaler`/`QuantileTransformer` antes da expansão polinomial.
2. **Otimização em graus altos.** Um learning rate menor, adaptado ao condicionamento da matriz (por exemplo 1/λ_max), ou normalização após a expansão, permitiria avaliar o grau 4 de forma justa.
3. **L2 com λ na ordem de 10²–10⁴**, ou a fórmula do gradiente sem dividir por *n*, para testar regularização a sério, sobretudo no grau 3 e nos embeddings de 1024 dims.
4. **Combinar embeddings de texto com as variáveis tabulares** num único modelo, ou reduzir os embeddings com PCA para permitir uma expansão quadrática completa.
5. **Transferência entre cidades.** Seriam precisas features relativas ao mercado local (preço normalizado pela mediana da cidade, distâncias, tipologia) ou algum re-treino com dados da cidade alvo.
6. **Problema conhecido noutro ficheiro:** `gd_results_all_strategies_degrees.csv` (gerado pelo `project_analisys.ipynb` antigo) tem resultados **idênticos** para none/cap/filter. Isto indica que foi carregada a mesma versão dos dados nas três estratégias, por isso esse ficheiro não deve ser usado para comparar tratamentos.

---

## 11. Reprodutibilidade

### Ambiente
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt              # base: numpy, pandas, scikit-learn, matplotlib, joblib, ipykernel
.venv/bin/pip install -r requirements-embeddings.txt   # só para a secção 15: torch, sentence-transformers (~5 GB)
```
Versões usadas: Python 3.12.3, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, torch 2.14.1 (CUDA), sentence-transformers 6.1.0.

> Com o torch 2.14 sem o pacote `python3-dev`, os kernels Triton "nativos" não compilam. O módulo `llm_embeddings.py` define `TORCH_DISABLE_NATIVE_JIT=1` para contornar isto.

### Tempo de execução aproximado (16 CPU, 13 GB RAM, RTX 3060)
| Secção | Tempo |
|---|---|
| 12 (5-fold, 3 estratégias × 4 graus) | ~35 min (o grau 4 diverge cedo; precisa de até ~7 GB de RAM) |
| 13 (L2) | ~5 min |
| 14 (classificação) | ~3 min |
| 15 (embeddings) | ~15 min a gerar (só na 1.ª vez; `REGENERAR_EMBEDDINGS=False` reutiliza os CSVs) + ~5 min de regressão |

### Ficheiros
| Ficheiro | Conteúdo |
|---|---|
| `project_analisys_c.ipynb` | notebook principal (secções 1–15) |
| `src/prepare_features.py` | classificação automática e tratamento das features |
| `src/linear_regression.py` | regressão linear com GD, k-fold, gravação/carregamento de modelos |
| `src/logistic_regression.py` | regressão logística softmax e métricas de classificação (ROC, AUC, …) |
| `src/llm_embeddings.py` | texto → embeddings LLM → CSVs 256/512/1024 |
| `src/data_preparation.py` | pipeline de preparação reutilizável (usado para Nova Iorque) |
| `outlier_versions/listings_{none,cap,filter}.csv` | dataset após cada estratégia de outliers |
| `Modelos_treinados/*.joblib` | pesos e histórico dos modelos da secção 8 e 11 |
| `gd_results_single_split.csv` | resultados da secção 8 |
| `gd_results_kfold_all_strategies.csv` | resultados da secção 12 (por fold) |
| `gd_results_l2.csv` | resultados da secção 13 |
| `gd_resultados_nova_york*.csv` | resultados das secções 10–11 |
| `logistic_results.csv`, `logistic_results_detalhado.csv` | resultados da secção 14 |
| `gd_results_embeddings.csv` | resultados da secção 15 |
| `Data_clean/embeddings/` | CSVs de embeddings e texto enviado ao LLM (no `.gitignore`, ~460 MB) |
