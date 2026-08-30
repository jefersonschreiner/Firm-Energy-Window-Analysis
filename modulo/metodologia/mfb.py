# MFB - Modelo Fixo Base

# Bibliotecas e Classes
import os
import time
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from ..sams import sams

PASTA_METODOLOGIA = os.path.dirname(os.path.abspath(__file__))
PASTA_MODULO = os.path.dirname(PASTA_METODOLOGIA)
PASTA_RAIZ = os.path.dirname(PASTA_MODULO)
PASTA_RESULTADO = os.path.join(PASTA_RAIZ, "resultado", "mfb")

class mfb:

    ANO_INICIO_PADRAO = 1931
    ANO_QUEBRA_PADRAO = 1968 # fim do período estacionário (quebra de Pettitt)
    N_ANOS_SIMULACAO_PADRAO = 50
    N_SERIES_PADRAO = 500
    SEMENTE_BASE_PADRAO = 13

    def __init__(self, serie: pd.Series, ano_inicio: int = ANO_INICIO_PADRAO, ano_quebra: int = ANO_QUEBRA_PADRAO):
        self.serie = serie
        self.ano_inicio = ano_inicio
        self.ano_quebra = ano_quebra
        self.janela = self._extrairJanela()
 
        self.sams = None
        self.ensemble = None
        self.tempo_execucao = None

    def _extrairJanela(self) -> pd.Series:
        # Fatia a série completa no período fixo pré-quebra (1931-1968)
        data_inicio = f"{self.ano_inicio}-01-01"
        data_fim = f"{self.ano_quebra}-12-31"
 
        janela = self.serie.loc[data_inicio:data_fim]
 
        if janela.empty:
            raise ValueError(f"[ERRO MFB] Nenhum dado encontrado no intervalo base: {data_inicio} até {data_fim}")
 
        return janela

    def executar(self, n_anos_simulacao: int = N_ANOS_SIMULACAO_PADRAO, n_series: int = N_SERIES_PADRAO,
                 semente_base: int = SEMENTE_BASE_PADRAO, verbose: bool = True, executor: ProcessPoolExecutor = None) -> pd.DataFrame:
        # Ajusta o modelo uma única vez na janela base e gera o ensemble completo
        tempo_inicio = time.time()
 
        if verbose:
            print(f"[MFB] Ajustando modelo na janela base {self.ano_inicio}-{self.ano_quebra} "
                  f"({len(self.janela)} dias)...")
 
        self.sams = sams(self.janela)
        modelo_vencedor = self.sams.selecionar(executor=executor, verbose=verbose)
        self.ensemble = self.sams.gerar(n_anos=n_anos_simulacao, n_series=n_series, semente_base=semente_base)
 
        self.tempo_execucao = time.time() - tempo_inicio
 
        if verbose:
            print(f"[MFB] Concluído: modelo {modelo_vencedor}, {n_series} séries de {n_anos_simulacao} anos "
                  f"em {self.tempo_execucao:.1f}s")
 
        return self._montarDataFrame()

    def _montarDataFrame(self) -> pd.DataFrame:
        # Monta o DataFrame final 
        if self.ensemble is None:
            raise RuntimeError("[ERRO MFB] Executar executar() antes de montar o DataFrame.")
 
        n_series, n_dias = self.ensemble.shape
        n_anos = n_dias // sams.DIAS_ANO
 
        ano_simulado = np.repeat(np.arange(1, n_anos + 1), sams.DIAS_ANO)
        dia_ano = np.tile(np.arange(1, sams.DIAS_ANO + 1), n_anos)
 
        dados = {'ano_simulado': ano_simulado, 'dia_ano': dia_ano}
        for i in range(n_series):
            dados[f'serie_{i + 1:04d}'] = np.round(self.ensemble[i], 4)
 
        return pd.DataFrame(dados)

    def salvar(self, pasta: str = PASTA_RESULTADO, nome_arquivo: str = None) -> str:
        # Salva o ensemble em um único CSV 
        if self.ensemble is None:
            raise RuntimeError("[ERRO MFB] Executar executar() antes de salvar().")
 
        os.makedirs(pasta, exist_ok=True)
 
        if nome_arquivo is None:
            nome_arquivo = f"MFB_{self.ano_inicio}-{self.ano_quebra}.csv"
 
        caminho_completo = os.path.join(pasta, nome_arquivo)
        df = self._montarDataFrame()
        df.to_csv(caminho_completo, index=False, sep=';', decimal=',')
 
        return caminho_completo

    def resumo(self) -> dict:
        # Retorna um resumo da execução
        if self.sams is None or self.sams.modelo_vencedor is None:
            raise RuntimeError("[ERRO MFB] Executar executar() antes de pedir o resumo.")
 
        return {
            'metodologia': 'MFB',
            'janela': f"{self.ano_inicio}-{self.ano_quebra}",
            'modelo': self.sams.modelo_vencedor,
            'aic': self.sams.aic_vencedor,
            'bic': self.sams.bic_vencedor,
            'tempo': self.tempo_execucao,
        }