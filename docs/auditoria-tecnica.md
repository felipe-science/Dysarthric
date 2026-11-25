# Auditoria técnica do Dysarthric

**O projeto tem potencial para ser um destaque de portfólio em software científico e ML aplicado a áudio. Ainda não recomendo tornar público o repositório atual.** Os dois bloqueios são a autorização para distribuir os artefatos derivados do TORGO e a interpretação dos resultados obtidos com divisão por gravação. Há também um defeito confirmado na variante multifractal. A coluna de Δα contém valores não finitos em **55 gravações de treino e 12 de teste**.

Auditoria realizada em 25 de novembro de 2026. Foram examinados os sete arquivos Python, os tensores, os checkpoints, os gráficos e o histórico Git disponível localmente. Os sete arquivos passaram pela análise sintática. As três matrizes de confusão foram reproduzidas em NumPy a partir dos pesos e tensores armazenados, sem executar o conteúdo pickle. O treinamento completo não foi executado. O checkout não contém os áudios nem o CSV de origem, e o interpretador local não tem PyTorch, librosa ou Parselmouth instalados. Nenhum arquivo do projeto foi modificado durante a auditoria, nenhuma dependência foi instalada e nenhum código ou dado do projeto foi enviado a serviços externos.

## 1. Objetivo e arquitetura atual

O Dysarthric é um protótipo de pesquisa para **classificação binária de gravações de fala no contexto da disartria**. A pergunta experimental que o código permite investigar é se descritores de picos e de comportamento multifractal acrescentam informação aos atributos acústicos convencionais. Os scripts esperam dados organizados a partir do TORGO, em um CSV com colunas `path` e `label`. O mapeamento dos rótulos `0` e `1` para os grupos do corpus não está documentado no checkout.

| Variante | Atributos por gravação | Entrada e saída atuais |
|---|---|---|
| `normal` | 33. Médias de 13 MFCCs, 12 coeficientes chroma, 7 contrastes espectrais e 1 taxa de cruzamento por zero | Extrai a partir do CSV e dos áudios. Grava tensores, gráficos e modelo. |
| `max_density` | 34. Acrescenta a razão entre máximos locais com amplitude ≥ 0 e o número de amostras | Treina a partir de tensores já salvos. O bloco de extração está desativado. |
| `multifrac` | 35. Acrescenta Δα à variante de 34 atributos | Tem extração e treino no mesmo script. O checkout contém tensores, sem checkpoint ou gráficos dessa variante. |

As três variantes centrais usam um **MLP PyTorch** com duas camadas ocultas de 64 unidades, ReLU, saída sigmoide, `BCELoss` e Adam com taxa de aprendizado de `0.001`. O treino usa o conjunto inteiro por atualização durante 5.000 épocas. O código detecta CUDA, mas treina em CPU porque não move o modelo nem os tensores para o dispositivo selecionado. Os resultados saem em arquivos com nomes fixos no diretório de execução. [Baseline](../train/normal/train.py#L127) · [Densidade de picos](../train/max_density/train_max.py#L26) · [Variante multifractal](../train/multifrac/train_max.py#L129)

A implementação multifractal aplica média móvel causal à forma de onda, calcula flutuações RMS em escalas de 32 a 4.096 amostras e estima h(q) por regressão log–log para q de −5 a 5. O atributo final é α(−5) − α(5). Como o código usa a forma de onda diretamente, sem a soma cumulativa do método publicado, a documentação deve identificá-lo como **uma variante inspirada em MF-DMA** e descrever as escolhas de janela e borda. [Implementação](../train/multifrac/core.py#L11) · [Gu e Zhou, 2010](https://arxiv.org/html/1005.0877)

**Tecnologias identificadas.** Python é a única linguagem de implementação. NumPy, SciPy e librosa processam os sinais, pandas lê os metadados, scikit-learn divide e avalia os dados, imbalanced-learn aplica SMOTE, PyTorch treina o MLP, e Matplotlib e seaborn geram os gráficos. Parselmouth/Praat é importado, mas não participa dos cálculos. `librosa.display`, `sklearn.utils.resample` e o alias `scipy.signal as signal` também estão sem uso. SciPy continua necessário para `find_peaks` e `linregress`.

## 2. Resultados disponíveis e pontos fortes

As métricas foram recalculadas a partir das matrizes registradas, com **3.527 gravações de teste** em cada caso. A inferência independente com NumPy reproduziu exatamente as três matrizes.

| Artefato | Atributos | Acurácia | Acurácia balanceada | F1 da classe `1` |
|---|---:|---:|---:|---:|
| `train/normal/` | 33 | 95,04% | 94,43% | 92,80% |
| `train/max_density/` | 34 | 95,44% | 95,14% | 93,45% |
| `train/` | 34 | 95,04% | 94,86% | 92,93% |

A classe majoritária representa **65,41%** desse teste. O checkpoint da raiz usa os mesmos tensores de `max_density`, mas tem pesos e resultados diferentes, sem registro da configuração que gerou a execução. O acréscimo de picos corresponde a **+0,40 ponto percentual de acurácia** na comparação entre `normal` e `max_density`. Esse valor descreve os artefatos históricos. A variação entre inicializações e a generalização para falantes inéditos continuam sem estimativa. [Matriz normal](../train/normal/confusion.png) · [Matriz com picos](../train/max_density/confusion.png) · [Matriz da raiz](../train/confusion.png)

O ponto forte científico é a **ablação incremental**. Os tensores têm 14.105 linhas de treino e 3.527 de teste, com os mesmos rótulos armazenados e igualdade linha a linha nas colunas compartilhadas entre as variantes. Isso fornece uma base concreta para medir a contribuição de cada descritor após revisar o protocolo. O MLP pequeno, os atributos explícitos e as matrizes por classe também facilitam o diagnóstico de erros. A variante multifractal ainda carece de resultado verificável.

## 3. Problemas encontrados

| Achado | Consequência |
|---|---|
| **Divisão 80/20 por gravação.** Os trechos de extração usam `random_state=42`, sem estratificação nem agrupamento. | O mesmo falante, a mesma sessão ou gravações simultâneas em microfones distintos podem aparecer dos dois lados. Os tensores não preservam IDs para confirmar ou quantificar essa sobreposição. |
| **SMOTE no teste legado.** `extract_amplitude_balanceamento.py` reamostra treino e teste separadamente. Os três fluxos centrais aplicam SMOTE somente no treino. | A métrica do script legado passa a representar um teste sintético com outra distribuição. Nos fluxos centrais, a escala desigual dos atributos também pode distorcer as distâncias usadas pelo SMOTE. |
| **Δα não finito.** A maior escala precisa de pelo menos 8.192 amostras para produzir um bloco com a fórmula atual. RMS zero afeta os logaritmos e os momentos negativos. | A falha foi confirmada com ruído curto e silêncio sintéticos, além das ocorrências nos tensores. O treino entrega a coluna diretamente ao SMOTE, sem validação. Os avisos são globalmente suprimidos. |
| **Avaliação e carregamento frágeis.** `carregar.py` define um modelo `10 → 50 → 1`, enquanto os checkpoints usam entradas de 33 ou 34 atributos e camadas com outros nomes. Os scripts legados esperam 33 colunas, mas os tensores da raiz têm 34. | Os pontos de entrada legados são incompatíveis com os artefatos que encontram na execução padrão. Salvar a classe inteira em pickle também acopla o checkpoint ao código e exige confiança na origem do arquivo. |
| **Experimentos difíceis de reproduzir.** Faltam versões fixadas, seed de inicialização do PyTorch, divisão de validação, busca de hiperparâmetros e registro das execuções. | A perda disponível é só de treino. Ela não permite diagnosticar sobreajuste ou subajuste. Também faltam baselines treinados sob o mesmo protocolo e medidas de incerteza. |
| **Portabilidade e erros de entrada.** Os caminhos dependem do diretório de trabalho e de `path[37:]`. Os scripts executam trabalho no nível do módulo, descartam falhas individuais sem contabilização estruturada e abrem janelas de gráficos. | Importar, testar ou executar em CI é difícil. Um lote unitário pode perder a dimensão com `.squeeze()`, e uma classe ausente pode causar divisão por zero na matriz normalizada. |
| **Custo computacional evitável.** A variante multifractal decodifica cada áudio até três vezes e recalcula o mesmo RMS 12 vezes por escala. A média móvel percorre as amostras em Python. | São candidatos claros a otimização. Ainda faltam medições de tempo e memória para quantificar o efeito no pipeline completo. |

[Divisão do baseline](../train/normal/train.py#L80) · [SMOTE legado](../train/extract_amplitude_balanceamento.py#L105) · [Flutuações multifractais](../train/multifrac/core.py#L32) · [Carregamento do modelo](../train/carregar.py#L10)

## 4. Avaliação profissional

As notas avaliam o repositório atual como projeto de pesquisa que outra pessoa precisaria executar, entender e manter. Para desempenho, a nota reflete a implementação observada, pois não há benchmark de execução.

| Critério | Nota | Justificativa e melhoria necessária |
|---|---:|---|
| Qualidade do código | 4/10 | As etapas são reconhecíveis, mas há duplicação e incompatibilidades de dimensão e checkpoint. Extrair funções compartilhadas e validar entradas. |
| Organização | 4/10 | As pastas distinguem variantes, mas misturam scripts, pesos, dados derivados e figuras. Separar pacote, testes e saídas de execução. |
| Legibilidade | 5/10 | Os comentários ajudam a seguir o fluxo, mas nomes como `aux`, `pht` e `N` dificultam manutenção. Adotar nomes descritivos, PEP 8, tipos nas interfaces e docstrings em inglês. |
| Documentação | 1/10 | Faltam instruções de instalação, preparação do CSV e interpretação dos rótulos e resultados. Escrever um README executável em inglês. |
| Reprodutibilidade | 3/10 | Os artefatos permitem conferir a inferência, mas faltam ambiente, sementes, IDs e configuração da execução para repetir o treino. |
| Desempenho | 4/10 | O modelo é compacto, porém há trabalho repetido na extração e treino em lote inteiro. Medir o fluxo e otimizar os trechos dominantes. |
| Testabilidade | 2/10 | Não há suíte automatizada, e a execução no nível do módulo dificulta testes isolados. Tornar extração, divisão e avaliação importáveis. |
| Boas práticas de engenharia | 3/10 | Faltam empacotamento, controles de qualidade e contratos de entrada e saída. Adotar configuração explícita, logging e CI. |
| Relevância para o mercado | 8/10 | O projeto demonstra processamento de sinais, engenharia de atributos e modelagem. Um experimento reproduzível por falante tornaria essas competências verificáveis. |
| Maturidade | 3/10 | Há um protótipo e evidência histórica de inferência. Ainda faltam um fluxo sustentado por testes e critérios claros de publicação. |

## 5. Riscos antes da publicação

- **Corpus, derivados e autoria.** Há 16 arquivos de tensores derivados, três checkpoints e nove gráficos. A página da Universidade de Toronto informa uso acadêmico sem fins lucrativos e exige referência em publicações. O catálogo LDC indica termos próprios para sua edição. O checkout não documenta a edição usada, autorização de redistribuição dos derivados, titularidade do código ou liberação das figuras e resultados. Para um candidato público, a recomendação é um snapshot de código revisado, com exemplo sintético, após confirmar autoria, permissões institucionais e eventual embargo. **MIT é uma candidata para o código original autorizado.** Dados, pesos e figuras exigem análise própria. [Termos oficiais do TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html#papers) · [Catálogo LDC](https://catalog.ldc.upenn.edu/products/LDC2012S02)
- **Segredos e dados pessoais.** A busca por assinaturas comuns de credenciais, chaves privadas e URLs autenticadas não encontrou correspondências no material pesquisado. Foram revisados os seis commits de `main` e as demais referências locais, que somavam 11 commits acessíveis. Não foram encontrados `.env`, áudios brutos, CSV bruto ou caminhos pessoais absolutos nos scripts. Há **um e-mail de autoria nos metadados Git de `main`** e **três `.DS_Store` versionados**, também presentes no histórico. O resultado cobre as referências locais e os padrões pesquisados. Dados industriais e material científico inédito não puderam ser descartados pela inspeção técnica dos derivados.
- **Binários e tamanho.** São 39 arquivos rastreados, cerca de 9,73 MiB, dos quais 9,52 MiB são `.pth`. O maior arquivo tem 1,88 MiB, e cerca de 2,50 MiB correspondem a cópias idênticas além da primeira. O volume atual dispensa Git LFS. A proveniência, a necessidade e a permissão de cada binário são os critérios relevantes para incluí-lo. O `.gitignore` cobre apenas `dataset/`. [.gitignore](../.gitignore)
- **Licenças das dependências.** A checagem preliminar encontrou licenças BSD nas bibliotecas científicas principais, MIT em imbalanced-learn, licença baseada na PSF em Matplotlib, ISC em librosa e GPLv3 no texto de licença do Parselmouth. Como as versões do projeto não estão fixadas, a revisão final precisa usar as versões efetivamente empacotadas e seus avisos. O import sem uso de Parselmouth é um candidato a remoção depois de cobrir o fluxo com testes. [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) · [librosa](https://github.com/librosa/librosa/blob/main/LICENSE.md) · [Parselmouth](https://github.com/YannickJadoul/Parselmouth/blob/master/LICENSE)

## 6. Estrutura recomendada

```text
README.md
LICENSE                         # After confirming rights
.gitignore
pyproject.toml
src/
  dysarthric_speech/
    data.py                     # Manifest, paths, and splits
    features.py                 # Acoustic features
    multifractal.py             # Delta alpha calculation
    model.py
    train.py
    evaluate.py
tests/                          # Unit tests and synthetic audio integration
.github/
  workflows/
    ci.yml                      # Once the tests can run
```

O `pyproject.toml` pode ser a fonte das dependências diretas, com um ambiente de referência fixado para reproduzir os experimentos. O README em inglês deve explicar o problema, a instalação, o esquema do manifesto, como executar o exemplo sintético, o protocolo de avaliação, a citação do corpus e as limitações. Um `docs/` passa a fazer sentido se a metodologia detalhada crescer. Um `results/` pode guardar tabelas agregadas aprovadas. Dados locais e checkpoints podem ser apontados por configuração.

**Docker e FastAPI ficam em P3.** Um container CPU faria sentido se surgir necessidade de distribuir um ambiente de execução reproduzível. Nesse caso, usaria uma imagem base pequena com versão fixada, `.dockerignore` cobrindo dados e artefatos, e montagem dos dados em modo somente leitura. Uma API faria sentido com demanda concreta de inferência, checkpoint validado e direitos de distribuição esclarecidos.

## 7. Plano por prioridade

| Prioridade | Melhoria | Complexidade |
|---|---|---|
| **P0** | Confirmar edição e termos do TORGO, autoria, permissões sobre derivados e status de divulgação dos resultados. Definir a lista de arquivos autorizados para um eventual snapshot público. | **Média a alta.** A vistoria técnica é direta. A autorização depende da documentação e dos titulares. |
| **P0** | Revisar o candidato à publicação quanto a e-mail, metadados, segredos e binários. Registrar a origem verificável dos resultados históricos e os limites do split atual. | **Média.** Envolve revisar o conteúdo selecionado e conferir as alegações públicas. |
| **P1** | Criar testes de caracterização e modularizar os fluxos. Configurar caminhos com `pathlib` e CLI, adicionar logging, documentar formatos e registrar parâmetros, versões e sementes. | **Média.** A duplicação ajuda a localizar o código comum. Os testes precisam demonstrar equivalência nos casos válidos. |
| **P1** | Corrigir o contrato entre atributos e modelo. Migrar para `state_dict` com arquitetura e ordem de atributos explícitas. Adicionar testes de dimensão, sinais curtos, silêncio, matrizes de confusão e ida e volta do checkpoint. Integrar Pytest e Ruff ao CI. | **Média.** Requer compatibilidade planejada com os artefatos já existentes e um teste de integração com WAV e CSV sintéticos. |
| **P1** | Propor política para Δα não finito e reavaliar com grupos por falante. Fixar a mesma amostra elegível entre variantes, separar validação e teste e comparar maioria, regressão logística e MLP. Ajustar escalonamento e SMOTE apenas no treino de cada iteração. Reportar métricas por classe e incerteza por falante. | **Alta.** Depende do manifesto original e de decisões sobre sinais inválidos, particionamento e seleção de hiperparâmetros. Exige aprovação científica antes da implementação. |
| **P2** | Medir tempo e memória. Eliminar releituras e recálculo de RMS. Avaliar vetorização da média móvel e cache com proveniência. Publicar agregados autorizados com a configuração correspondente. | **Média.** A vetorização precisa de testes de equivalência numérica, inclusive nas bordas. |
| **P3** | Empacotar um ambiente CPU em Docker ou avaliar uma API FastAPI se houver caso de uso definido. | **Baixa a média** para Docker. **Média a alta** para uma API com validação de entradas e do modelo. |

A sequência proposta começa por caracterizar o comportamento atual. Mudanças nas janelas, no tratamento de silêncio, na seleção de gravações, nas sementes de treino, no split e no balanceamento podem alterar resultados científicos e ficam sujeitas à aprovação do responsável pelo projeto.

## 8. Potencial para o portfólio

O melhor recorte é **um estudo reproduzível sobre a contribuição de atributos acústicos e multifractais para classificação de fala**. A história do projeto fica mais forte quando o leitor consegue executar um exemplo, entender a ablação e verificar como a avaliação trata falantes inéditos. Esse recorte demonstra competências relevantes para vagas de **Research Engineer, Scientific Software Developer, ML Engineer, AI Engineer aplicada a áudio e Data Scientist experimental**.

**Nome recomendado** — `dysarthric-speech-classification`. Outras opções são `dysarthric-speech-analysis`, `dysarthria-acoustic-features`, `multifractal-speech-analysis` e `speech-dysarthria-ml`.

**Descrição curta em inglês**

> Research code for binary speech classification using acoustic and multifractal features with PyTorch.

**Topics sugeridos** — `speech-processing` · `audio-classification` · `dysarthria` · `pytorch` · `signal-processing` · `feature-engineering` · `multifractal-analysis`
