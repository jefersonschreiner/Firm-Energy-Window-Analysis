# Orquestrador do pipeline completo do proejto

# 1. Carrega a série histórica de vazão (usa o banco processado, se existir;
#    senão processa o CSV bruto uma vez e salva o processado para acelerar
#    as próximas execuções);
# 2. Roda a análise estatística (Mann-Kendall, Spearman, Pettitt) e salva
#    o resultado em resultado/estatisticas_resultado.csv;
# 3. Roda as três metodologias de geração de séries sintéticas (MFB, MOF, MJD),
#    cada uma salvando seus próprios CSVs dentro de resultado/<metodologia>/.

# Bibliotecas e Classes
import os
import time
from concurrent.futures import ProcessPoolExecutor
 
from modulo.banco import banco, CAMINHO_BANCO_PROCESSADO
from modulo.estatistica import estatistica
from modulo.metodologia import mfb, mof, mjd

RODAR_ESTATISTICA = True
RODAR_MFB = True
RODAR_MOF = True
RODAR_MJD = True
 
ANO_INICIO_BASE = 1931 # início da série / do período estacionário pré-quebra
ANO_QUEBRA = 1968 # fim do período estacionário (quebra de Pettitt)
N_ANOS_SIMULACAO = 50
N_SERIES = 500
SEMENTE_BASE = 13

N_WORKERS = 4 # esse aqui é o número de nucleos que vamos usar do processador

def carregarSerie():
    b = banco(carregar_ao_iniciar=False)
    try:
        serie = b.carregarSerieProcessada()
        print(f"[MAIN] Série carregada do banco processado ({len(serie)} dias).")
    except FileNotFoundError:
        print("[MAIN] Banco processado não encontrado. Processando o CSV bruto original...")
        serie = b.carregarDados()
        b.salvarSerieProcessada()
        print(f"[MAIN] Série processada e salva em: {CAMINHO_BANCO_PROCESSADO} ({len(serie)} dias).")
    return serie
 
 
def rodarEstatistica(serie) -> None:
    print("\n" + "=" * 70)
    print("ETAPA 1/4 - ANÁLISE ESTATÍSTICA (Mann-Kendall, Spearman, Pettitt)")
    print("=" * 70)
 
    modelo = estatistica(serie)
    modelo.executar_analise()
    modelo.imprimir_resumo()
    caminho = modelo.salvarResultados()
    print(f"[MAIN] Resultado da análise estatística salvo em: {caminho}")
 
 
def rodarMfb(serie, executor: ProcessPoolExecutor) -> dict:
    print("\n" + "=" * 70)
    print("ETAPA 2/4 - MFB (Modelo Fixo Base)")
    print("=" * 70)
 
    modelo = mfb(serie, ano_inicio=ANO_INICIO_BASE, ano_quebra=ANO_QUEBRA)
    modelo.executar(n_anos_simulacao=N_ANOS_SIMULACAO, n_series=N_SERIES, semente_base=SEMENTE_BASE,
                     executor=executor)
    caminho = modelo.salvar()
    resumo = modelo.resumo()
 
    print(f"[MAIN] MFB concluído. Arquivo salvo em: {caminho}")
    return resumo
 
 
def rodarMof(serie, executor: ProcessPoolExecutor) -> list:
    print("\n" + "=" * 70)
    print("ETAPA 3/4 - MOF (Modelo Origem Fixa)")
    print("=" * 70)
 
    modelo = mof(serie, ano_inicio_fixo=ANO_INICIO_BASE, ano_inicio_simulacao=ANO_QUEBRA + 1)
    log = modelo.executarTodos(n_anos_simulacao=N_ANOS_SIMULACAO, n_series=N_SERIES, semente_base=SEMENTE_BASE,
                                executor=executor)
    caminho_log = modelo.salvarLog()
 
    print(f"[MAIN] MOF concluído: {len(log)} ciclos. Log salvo em: {caminho_log}")
    return log
 
 
def rodarMjd(serie, executor: ProcessPoolExecutor) -> list:
    print("\n" + "=" * 70)
    print("ETAPA 4/4 - MJD (Modelo de Janela Deslizante)")
    print("=" * 70)
 
    modelo = mjd(serie, ano_inicio_base=ANO_INICIO_BASE, ano_quebra=ANO_QUEBRA)
    log = modelo.executarTodos(n_anos_simulacao=N_ANOS_SIMULACAO, n_series=N_SERIES, semente_base=SEMENTE_BASE,
                                executor=executor)
    caminho_log = modelo.salvarLog()
 
    print(f"[MAIN] MJD concluído: {len(log)} ciclos. Log salvo em: {caminho_log}")
    return log
 
 
def main():
    tempo_inicio = time.time()
 
    serie = carregarSerie()
 
    n_workers = N_WORKERS if N_WORKERS is not None else (os.cpu_count() or 1)
    print(f"[MAIN] Pool de {n_workers} processos criado (reaproveitado em todas as etapas).")

    # Um único ProcessPoolExecutor para o pipeline inteiro: a busca em grade
    # do SAMS (ARMA + SARIMA, ~50 ajustes independentes por janela) roda em
    # paralelo nele, e o mesmo pool é reaproveitado nas 111 janelas de
    # MFB/MOF/MJD em vez de recriar processos a cada janela.
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        if RODAR_ESTATISTICA:
            rodarEstatistica(serie)
 
        if RODAR_MFB:
            rodarMfb(serie, executor)
 
        if RODAR_MOF:
            rodarMof(serie, executor)
 
        if RODAR_MJD:
            rodarMjd(serie, executor)
 
    tempo_total = time.time() - tempo_inicio
    horas = int(tempo_total // 3600)
    minutos = int((tempo_total % 3600) // 60)
 
    print("\n" + "=" * 70)
    print(f"[MAIN] PIPELINE CONCLUÍDO em {horas}h {minutos}min")
    print("=" * 70)
 
 
if __name__ == '__main__':
    main()