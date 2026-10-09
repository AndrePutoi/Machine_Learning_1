# Previsão do Preço de Alojamentos Airbnb em Lisboa

**Machine Learning — Practical Project 1**

| | |
|---|---|
| **Autor** | André Pinheiro Putoi |
| **Curso** | Mestrado em Engenharia Informática — Sistemas Inteligentes |
| **Nº de aluno** | M16626 |

---

## Sobre o projeto

O objetivo é prever o preço por noite (`price`) de anúncios Airbnb. Os algoritmos de aprendizagem foram **implementados de raiz com numpy**, sem usar modelos do scikit-learn:

- **Regressão linear** com gradient descent (full-batch), expansão polinomial (graus 1 a 4) e regularização L2 opcional.
- **Regressão logística multinomial (softmax)** com entropia cruzada, para classificar o preço em 4 faixas, com as métricas (matriz de confusão, ROC/AUC, precision-recall, log loss) também feitas de raiz.
- **Regressão sobre embeddings de texto** gerados por um LLM (`Qwen3-Embedding-0.6B`), usando só a descrição dos anúncios.

### Dados: treino só em Lisboa

> **Todos os modelos foram treinados exclusivamente com os dados de Lisboa** (`data/listings.csv`, 24 876 anúncios, Inside Airbnb).
> Qualquer outro dataset (**Nova Iorque, Amesterdão, Barcelona e Barossa Valley**) foi usado **apenas para avaliação**, sem qualquer re-treino, para testar se um modelo treinado em Lisboa generaliza para outras cidades.

---

## Resultados principais

Todas as métricas são no **conjunto de teste** (split 70/15/15, `random_state=42`), com treino em Lisboa.

### Regressão do preço (R² e MAE em €)

| Abordagem | Melhor configuração | R² | MAE |
|---|---|---|---|
| Variáveis tabulares, sem tratar outliers | grau 3 | 0.140 | 80.4 € |
| Variáveis tabulares, outliers **cap** (winsorization MAD) | grau 3 | **0.590** | 44.0 € |
| Variáveis tabulares, outliers **filter** (remoção MAD) | grau 3 | 0.455 | **40.7 €** |
| Só texto, via LLM (1024 dims, `filter` no preço) | linear | 0.460 | 42.0 € |

Com **validação cruzada de 5 folds**, o grau 2 é a escolha mais estável (R² de validação `cap` 0.54, `filter` 0.41). O grau 4 diverge em todos os folds.

### Classificação em 4 faixas de preço (softmax, grau 2)

| Accuracy | Macro F1 | AUC macro | Classe certa ou vizinha |
|---|---|---|---|
| 0.566 (acaso ≈ 0.25) | 0.563 | 0.811 | 91.3% |

### Avaliação noutras cidades (modelos de Lisboa, sem re-treino, R² com `filter`)

| Cidade | Grau 1 | Grau 2 | Grau 3 | Grau 4 |
|---|---|---|---|---|
| Nova Iorque | −0.06 | −0.07 | −0.59 | −6.88 |
| Barcelona | −0.77 | −0.66 | −0.80 | −3.67 |
| Barossa Valley | −1.60 | −1.49 | −1.34 | −3.71 |
| Amesterdão | −1.7×10⁶ | −7×10¹⁵ | −2×10²⁶ | −5×10³³ |

### Principais conclusões

1. **Tratar outliers é o fator com mais impacto.** Sem tratamento, o R² fica em ~0.15; com `cap` ou `filter` passa a 0.4–0.6.
2. **Os graus 2–3 são o ponto ideal.** O grau 4 (15–17 mil features) diverge sempre.
3. **A L2 testada (λ ≤ 0.75) não tem efeito**, porque o termo de regularização é dividido por *n* ≈ 14 mil.
4. **O texto sozinho prevê o preço quase tão bem como as variáveis estruturadas** (R² 0.46).
5. **Os modelos de Lisboa não generalizam para outras cidades.** O R² é negativo em todas. Em Amesterdão o erro explode porque `maximum_nights` tem valores de 2×10⁹ (até ~4,6 milhões de desvios-padrão de Lisboa), amplificados pela expansão polinomial.

---

## Onde ver mais detalhes

O **notebook [`Airbnb_Lisboa_Predicao_Preco.ipynb`](Airbnb_Lisboa_Predicao_Preco.ipynb) tem muito mais detalhe** do que este resumo: todo o processo, com código, gráficos, tabelas e conclusões de cada passo.

| Secção | Conteúdo |
|---|---|
| 1–3 | Setup, extração e limpeza inicial de colunas |
| 4–5 | Valores em falta e análise exploratória |
| 6 | Deteção e tratamento de outliers (IQR vs. MAD; estratégias `none`, `cap`, `filter`) |
| 7–9 | Preparação de features, treino (gradient descent) e comparação com baseline |
| 10–11 | Avaliação em Nova Iorque, lat/lon vs. distância ao centro, e novas cidades |
| 12 | Validação cruzada 5-fold, graus 1–4, por estratégia de outliers |
| 13 | Regularização L2 |
| 14 | Classificação em 4 faixas de preço (regressão logística) |
| 15 | Embeddings de texto com um LLM |

O [`RELATORIO.md`](RELATORIO.md) contém o relatório escrito completo, com limitações e trabalho futuro.

---

## Estrutura do repositório

```
├── Airbnb_Lisboa_Predicao_Preco.ipynb   # notebook principal (secções 1–15, com todos os outputs)
├── RELATORIO.md                         # relatório escrito
├── README.md
├── requirements.txt                     # dependências base
├── requirements-embeddings.txt          # extra, só para a secção 15 (torch, sentence-transformers)
├── inspect_columns.py                   # utilitário para inspecionar as colunas dos CSVs
├── src/
│   ├── data_preparation.py              # pipeline de preparação reutilizável (outras cidades)
│   ├── prepare_features.py              # classificação automática e tratamento das features
│   ├── linear_regression.py             # regressão linear com GD, k-fold, gravar/carregar modelos
│   ├── logistic_regression.py           # regressão logística softmax e métricas de classificação
│   ├── llm_embeddings.py                # texto → embeddings LLM → CSVs 256/512/1024
│   └── generate_text_embeddings.py      # script autónomo (alternativa TF-IDF), não usado no notebook
├── data/listings.csv                    # dataset de Lisboa (treino)
├── outlier_versions/                    # Lisboa após cada estratégia de outliers (none, cap, filter)
├── Data_clean/                          # datasets tratados (Nova Iorque e restantes cidades)
├── Modelos_treinados/                   # modelos treinados (.joblib): pesos, histórico e scaler
├── nova_york.csv.gz, amesterdam.csv.gz, barcelona.csv.gz, Barossa_Valley_australia.csv.gz
│                                        # datasets das outras cidades (SÓ avaliação)
└── gd_*.csv, logistic_results*.csv      # tabelas de resultados de cada secção
```

Os CSVs de embeddings (`Data_clean/embeddings/`, ~460 MB) não estão no repositório. São regeneráveis com `src/llm_embeddings.py`, e o notebook faz isso automaticamente se não existirem.

---

## Como executar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt               # numpy, pandas, scikit-learn, matplotlib, joblib, ipykernel
.venv/bin/pip install -r requirements-embeddings.txt    # opcional: só para a secção 15 (~5 GB)
.venv/bin/jupyter notebook Airbnb_Lisboa_Predicao_Preco.ipynb
```

- Versões usadas: Python 3.12, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1 (usado apenas em `PolynomialFeatures`, `StandardScaler` e `KFold`, e para validar as métricas; os modelos e o gradient descent são numpy).
- Tempo de execução do notebook completo: cerca de 1 hora num CPU de 16 núcleos (a secção 12, validação cruzada, demora ~35 min).
- A secção 15 gera os embeddings com um LLM na GPU (~15 min numa RTX 3060). Com `REGENERAR_EMBEDDINGS = False` e os CSVs já presentes, esse passo é ignorado.
