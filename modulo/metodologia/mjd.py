# MJD - Modelo de Janela Deslizante

# Bibliotecas e Classes
import os
import time
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from ..banco import banco
from ..sams import sams
 
PASTA_METODOLOGIA = os.path.dirname(os.path.abspath(__file__))
PASTA_MODULO = os.path.dirname(PASTA_METODOLOGIA)
PASTA_RAIZ = os.path.dirname(PASTA_MODULO)
PASTA_RESULTADO = os.path.join(PASTA_RAIZ, "resultado", "mjd")

class mjd:
 
    ANO_INICIO_BASE_PADRAO = 1931 # início do período estacionário pré-quebra
    ANO_QUEBRA_PADRAO = 1968 # fim do período estacionário (quebra de Pettitt)
    N_ANOS_SIMULACAO_PADRAO = 50
    N_SERIES_PADRAO = 500
    SEMENTE_BASE_PADRAO = 13

    def __init__(self, serie: pd.Series, ano_inicio_base: int = ANO_INICIO_BASE_PADRAO,
                 ano_quebra: int = ANO_QUEBRA_PADRAO, ano_fim_final: int = None):
        self.serie = serie
        self.ano_inicio_base = ano_inicio_base
        self.ano_quebra = ano_quebra

        self.tamanho_janela = self.ano_quebra - self.ano_inicio_base + 1
        self.ano_inicio_primeira_janela = self.ano_inicio_base + 1
        self.ano_fim_primeira_janela = self.ano_inicio_primeira_janela + self.tamanho_janela - 1
 
        self.ano_fim_final = ano_fim_final if ano_fim_final is not None else int(serie.index.year.max())

        self._banco = banco(carregar_ao_iniciar=False)
        self._banco.serie = serie
 
        self.log = []

    def executarCiclo(self, ano_inicio_win: int, ano_fim_win: int, janela: pd.Series, n_anos_simulacao: int,
                       n_series: int, semente_base: int, verbose: bool, executor: ProcessPoolExecutor = None) -> dict:

        tempo_inicio = time.time()
 
        modelo_sams = sams(janela)
        modelo_vencedor = modelo_sams.selecionar(executor=executor, verbose=verbose)
        ensemble = modelo_sams.gerar(n_anos=n_anos_simulacao, n_series=n_series, semente_base=semente_base)
 
        tempo_ciclo = time.time() - tempo_inicio
 
        caminho = self._salvarEnsemble(ano_inicio_win, ano_fim_win, ensemble)
 
        resumo = {
            'metodologia': 'MJD',
            'janela': f"{ano_inicio_win}-{ano_fim_win}",
            'ano_inicio_win': ano_inicio_win,
            'ano_fim_win': ano_fim_win,
            'dias_janela': len(janela),
            'modelo': modelo_vencedor,
            'aic': modelo_sams.aic_vencedor,
            'bic': modelo_sams.bic_vencedor,
            'tempo': tempo_ciclo,
            'arquivo': caminho,
        }
 
        if verbose:
            print(f"[MJD] Janela {resumo['janela']} | Modelo: {modelo_vencedor} | "
                  f"Tempo: {tempo_ciclo:.1f}s | Salvo em: {caminho}")
 
        return resumo

    def _salvarEnsemble(self, ano_inicio_win: int, ano_fim_win: int, ensemble: np.ndarray,
                         pasta: str = PASTA_RESULTADO) -> str:
        os.makedirs(pasta, exist_ok=True)
 
        n_series, n_dias = ensemble.shape
        n_anos = n_dias // sams.DIAS_ANO
 
        ano_simulado = np.repeat(np.arange(1, n_anos + 1), sams.DIAS_ANO)
        dia_ano = np.tile(np.arange(1, sams.DIAS_ANO + 1), n_anos)
 
        dados = {'ano_simulado': ano_simulado, 'dia_ano': dia_ano}
        for i in range(n_series):
            # Arredonda para 4 casas decimais (precisão hidrológica padrão do projeto)
            dados[f'serie_{i + 1:04d}'] = np.round(ensemble[i], 4)
 
        df = pd.DataFrame(dados)
        nome_arquivo = f"MJD_{ano_inicio_win}-{ano_fim_win}.csv"
        caminho_completo = os.path.join(pasta, nome_arquivo)
        df.to_csv(caminho_completo, index=False, sep=';', decimal=',')

    def executarTodos(self, n_anos_simulacao: int = N_ANOS_SIMULACAO_PADRAO, n_series: int = N_SERIES_PADRAO,
                       semente_base: int = SEMENTE_BASE_PADRAO, verbose: bool = True, executor: ProcessPoolExecutor = None) -> list:

        self.log = []
        total_ciclos = self.ano_fim_final - self.ano_fim_primeira_janela + 1
        ciclo_atual = 0
 
        if verbose:
            print(f"[MJD] Iniciando: {total_ciclos} ciclos, janela de {self.tamanho_janela} anos, "
                  f"de {self.ano_inicio_primeira_janela}-{self.ano_fim_primeira_janela} "
                  f"deslizando até terminar em {self.ano_fim_final}")
 
        for ano_inicio_win, ano_fim_win, janela in self._banco.extrairMJD(
                self.ano_inicio_primeira_janela, self.tamanho_janela, self.ano_fim_final):
            ciclo_atual += 1
 
            resumo = self.executarCiclo(ano_inicio_win, ano_fim_win, janela, n_anos_simulacao, n_series,
                                         semente_base, verbose, executor=executor)
            resumo['ciclo'] = ciclo_atual
            self.log.append(resumo)
 
            if verbose:
                ciclos_restantes = total_ciclos - ciclo_atual
                if ciclos_restantes > 0:
                    estimativa = sams.estimarTempo(resumo['tempo'], ciclos_restantes)
                    print(f"[MJD] Ciclo {ciclo_atual}/{total_ciclos} concluído. "
                          f"Estimativa para os {ciclos_restantes} restantes: {estimativa}")
 
        return self.log

    def salvarLog(self, pasta: str = PASTA_RESULTADO, nome_arquivo: str = "MJD_log.csv") -> str:
        if not self.log:
            raise RuntimeError("[ERRO MJD] Executar executarTodos() antes de salvar o log.")
 
        os.makedirs(pasta, exist_ok=True)
        caminho_completo = os.path.join(pasta, nome_arquivo)
 
        colunas = ['ciclo', 'janela', 'ano_inicio_win', 'ano_fim_win', 'dias_janela', 'modelo', 'aic', 'bic',
                   'tempo', 'arquivo']
        df_log = pd.DataFrame(self.log)[colunas]
        # Arredonda só na hora de salvar; self.log mantém a precisão completa em memória
        df_log['aic'] = df_log['aic'].round(2)
        df_log['bic'] = df_log['bic'].round(2)
        df_log['tempo'] = df_log['tempo'].round(1)
        df_log.to_csv(caminho_completo, index=False, sep=';', decimal=',')
 
        return caminho_completo
    