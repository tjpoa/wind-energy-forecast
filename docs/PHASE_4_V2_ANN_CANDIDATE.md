# ANN v2 — candidato local governado

Este caminho é separado do notebook, do modelo v1 e do controlled retraining
existente. Não altera `/predict`, o batch ativo, `retraining_policy_v1.json`, os
artefactos v1 ou o estado Azure.

## Contrato

- Entrada: a tabela `feature_ready_ren_era5_land_v2/feature_ready_daily.csv`,
  com 56 atributos na ordem do dataset e sem scaling no chamador.
- Scalers: apenas o manifesto aceite em
  `models/v2/scalers/feature_ready_ren_era5_land_v2/`; o dataset, a ordem dos
  atributos, os quatro scalers e os seus hashes são verificados antes do treino.
- Variantes: `original` e `log1p`, com scalers temporários ajustados só em
  `2010-01-15`–`2022-12-31`. A escolha usa MAE em unidades originais em
  `2023-01-01`–`2024-12-31`, com desempate por `original`; o teste selado nunca
  participa na escolha.
- Receita fixa: `Dense(32, relu)`, `Dropout(0.2)`, `Dense(1)`, Adam `1e-3`, MSE,
  batch `32`, no máximo 200 épocas, patience 50, seed 42 e operações
  determinísticas em CPU. A variante vencedora é recriada sobre treino+validação
  com os scalers versionados e o número de épocas selecionado.
- O bundle é `keras_scaled_v2`: o pyfunc recebe os 56 atributos crus, aplica os
  scalers internamente e devolve `Wind_Production` na escala original. Saídas
  não finitas ou negativas fazem o fluxo falhar; não há clipping implícito.

## Execução sequencial

```powershell
.\venv\Scripts\python.exe .\scripts\train_v2_ann_candidate.py `
  --output-dir outputs\training\v2_ann_candidate_<sha>

.\venv\Scripts\python.exe .\scripts\backtest_v2_ann_challenger.py `
  --candidate-bundle <bundle-aceite> `
  --incumbent-bundle outputs\training\v2_reference_mlflow `
  --incumbent-calibration <calibracao-incumbent> `
  --output-root outputs\backtests\v2_ann_challenger_<sha>

.\venv\Scripts\python.exe .\scripts\calibrate_monitoring.py `
  --challenger-candidate <backtest-aceite> `
  --output-root data\processed\v2\monitoring\reporting\ann_challenger_<sha>

.\venv\Scripts\python.exe .\scripts\log_v2_ann_candidate.py `
  --candidate-bundle <backtest-aceite> `
  --backtest-bundle <backtest-aceite> `
  --calibration-dir <calibracao-candidato>
```

O backtest usa 15 folds consecutivos de 30 observações em
`2025-01-01`–`2026-06-27`, sempre com o champion v2 congelado e
`Wind_Production_Lag1`. Um resultado rejeitado deixa apenas evidência de
auditoria e não é registável. A calibração do candidato é um novo diretório
content-addressed e não move o ponteiro ativo.

## Registry e promoção

O registo requer Git limpo no mesmo SHA selado, run MLflow `FINISHED`, assinatura
numérica ordenada, hash do `model.keras`, reload pyfunc equivalente (`rtol=1e-7`,
`atol=1e-5`) e expectativas explícitas para `candidate`, `champion` e `stable`.
Só o alias `candidate` pode ser alterado; o recibo content-addressed inclui
hashes, lineage e estado anterior/posterior. O dry-run não escreve.

Promoção local e qualquer mutação Azure (`production`, imagens, Terraform/Bicep,
rollback ou deployment pointer) estão fora deste caminho e exigem nova
autorização explícita. Enquanto o runtime batch/monitoring e
`manage_v2_deployment.py` não validarem `keras_scaled_v2`, a prontidão para
promoção é `NO-GO`.

Os modelos, backtests e calibrações gerados permanecem locais e ignorados pelo
Git. A execução de aceitação deve ocorrer num `master` limpo e no mesmo SHA,
seguida de uma revisão documental dos IDs, métricas e hashes.
