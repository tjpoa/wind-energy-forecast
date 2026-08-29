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
- Receita autorizada: `Dense(32, relu)`, `Dropout(0.2)`, `Dense(1, softplus)` (saída
  não-negativa), Adam `1e-3`, MSE,
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

## Estado da execução desta implementação

Após autorização explícita para a receita não-negativa, a execução local foi
repetida no SHA `29e7955ba82ecb0bf96d187a832c8ee7b16f97b6`, com `softplus` apenas
na saída final; os scalers, seed, particionamento e restantes hiperparâmetros
foram preservados. O treino terminou com a variante `original` e 123 épocas.
Os hashes de entrada, scalers e split são, respetivamente,
`d0d073748c5d963cba30212e6b0ab666ec2000197b8f61a5c439b4aaf786b2a6`,
`8d97424a0ee63d603922e8d39f3a2e6532f8c5f4be79711e644c1ed04db520f8` e
`196456643397d5ba6b3f64b55b4d3783fd5df51381014ffe7e97dff09a252e3b`.

O backtest selado de 15 folds × 30 observações (`2025-01-01`–`2026-06-27`)
produziu o ID
`2b2043fe16c1996c12fbf248734db8eb96702d2a216fb1e4a4d479a0dcbee165` e foi
`rejected`: a MAE agregada do candidato foi `25133.27`, inferior à do incumbent
(`25541.04`) e à persistência (`69577.15`), e passou a calibração do incumbent,
mas `every_fold_passed=false` (por exemplo, fold 1: `39738.13` contra
`34279.32` do incumbent; fold 15: `22144.40` contra `20146.07`).

O resultado rejeitado conserva métricas, previsões, hashes e lineage apenas para
auditoria e não é um bundle registável. Consequentemente, a calibração ANN do
candidato, o run/versão MLflow e o alias `candidate` não foram produzidos nem
alterados. Não houve promoção local, alteração de `champion`/`stable`, mudança
de deployment pointer ou qualquer operação Azure. O treino e o backtest não
fizeram pedidos de rede nem escrita no Registry.

As suites ANN direcionadas passaram após a alteração (`4 passed`) e Ruff nos
ficheiros tocados passou; a suite completa anterior passou (`772 passed,
17 skipped`) antes desta alteração limitada à ativação de saída. A decisão
governada é manter o candidato rejeitado e abrir novo plano apenas se for
proposta outra receita ou alteração dos gates.

### Experiência v2.1 — cabeça `sigmoid`

No branch experimental, a única alteração foi tornar a cabeça final
`Dense(1, sigmoid)` configurável, mantendo a seleção `original`/`log1p`, os
scalers, splits, seed e gates. A execução foi selada no SHA
`b92bb7b7f96b92a397a673e24d86722f6838eb11`; selecionou `original` em 195 épocas,
com os mesmos hashes de dataset, scalers e split acima. A MAE de validação foi
`18905.28` e a de teste `23393.29`.

O backtest correspondente, com os mesmos 15 folds × 30 observações e os mesmos
comparadores, recebeu o ID
`66d67030a3cf75e6e24ec4b951664b7d3d5bbc61d91976d672fbea10290c9aae` e foi
`rejected`. A MAE agregada do candidato (`23393.28`) foi inferior à do incumbent
(`25541.04`) e à persistência (`69577.15`), mas os folds 4, 5, 12, 13 e 15
continuaram piores que o incumbent. O número de folds aprovados subiu de 6/15
com `softplus` para 10/15 com `sigmoid`, sem satisfazer o gate de todos os
folds.

O bundle rejeitado conserva evidência, hashes e lineage apenas localmente. Não
foi criada calibração de candidato, não houve run/versão MLflow nem alteração de
`candidate`, `champion` ou `stable`; promoção e Azure continuam sem execução.
Esta experiência termina aqui. Qualquer nova receita, retreino temporal ou
alteração de gates exige um plano e autorização próprios.
