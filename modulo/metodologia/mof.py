# MOF - Modelo Origem Fixa

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
PASTA_RESULTADO = os.path.join(PASTA_RAIZ, "resultado", "mof")

class mof:

    ANO_INICIO_FIXO_PADRAO = 1931
    ANO_INICIO_SIMULACAO_PADRAO = 1969   # primeiro ano_fim processado (pós-quebra de Pettitt)
    N_ANOS_SIMULACAO_PADRAO = 50
    N_SERIES_PADRAO = 500
    SEMENTE_BASE_PADRAO = 13

    def __init__(self, serie: pd.Series, ano_inicio_fixo: int = ANO_INICIO_FIXO_PADRAO,
                 ano_inicio_simulacao: int = ANO_INICIO_SIMULACAO_PADRAO, ano_fim_final: int = None):
        self.serie = serie
        self.ano_inicio_fixo = ano_inicio_fixo
        self.ano_inicio_simulacao = ano_inicio_simulacao
        self.ano_fim_final = ano_fim_final if ano_fim_final is not None else int(serie.index.year.max())
 
        if self.ano_inicio_simulacao < self.ano_inicio_fixo:
            raise ValueError("[ERRO MOF] ano_inicio_simulacao não pode ser menor que ano_inicio_fixo")
 
        self._banco = banco(carregar_ao_iniciar=False)
        self._banco.serie = serie
 
        self.log = []

    def _extrairJanelas(self):
        for ano_fim, janela in self._banco.extrairMOF(self.ano_inicio_fixo, self.ano_fim_final):
            if ano_fim < self.ano_inicio_simulacao:
                continue
            yield ano_fim, janela

    def executarCiclo(self, ano_fim: int, janela: pd.Series, n_anos_simulacao: int, n_series: int,
                       semente_base: int, verbose: bool, executor: ProcessPoolExecutor = None) -> dict:
        # Ajusta o modelo e gera o ensemble para UMA INTERAÇÃO (um ciclo do MOF)
        tempo_inicio = time.time()
 
        modelo_sams = sams(janela)
        modelo_vencedor = modelo_sams.selecionar(executor=executor, verbose=verbose)
        ensemble = modelo_sams.gerar(n_anos=n_anos_simulacao, n_series=n_series, semente_base=semente_base)
 
        tempo_ciclo = time.time() - tempo_inicio
 
        caminho = self._salvarEnsemble(ano_fim, ensemble)
 
        resumo = {
            'metodologia': 'MOF',
            'janela': f"{self.ano_inicio_fixo}-{ano_fim}",
            'ano_fim': ano_fim,
            'dias_janela': len(janela),
            'modelo': modelo_vencedor,
            'aic': modelo_sams.aic_vencedor,
            'bic': modelo_sams.bic_vencedor,
            'tempo': tempo_ciclo,
            'arquivo': caminho,
        }
 
        if verbose:
            print(f"[MOF] Janela {resumo['janela']} | Modelo: {modelo_vencedor} | "
                  f"Tempo: {tempo_ciclo:.1f}s | Salvo em: {caminho}")
 
        return resumo

    def _salvarEnsemble(self, ano_fim: int, ensemble: np.ndarray, pasta: str = PASTA_RESULTADO) -> str:
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
        nome_arquivo = f"MOF_{self.ano_inicio_fixo}-{ano_fim}.csv"
        caminho_completo = os.path.join(pasta, nome_arquivo)
        df.to_csv(caminho_completo, index=False, sep=';', decimal=',')
 
        return caminho_completo

    def executarTodos(self, n_anos_simulacao: int = N_ANOS_SIMULACAO_PADRAO, n_series: int = N_SERIES_PADRAO,
                       semente_base: int = SEMENTE_BASE_PADRAO, verbose: bool = True, executor: ProcessPoolExecutor = None) -> list:
        # Roda todos os ciclos do MOF em sequência: 1931-1969, 1931-1970
        # (não acumula tudo em memória) e imprime uma estimativa de tempo
        self.log = []
        total_ciclos = self.ano_fim_final - self.ano_inicio_simulacao + 1
        ciclo_atual = 0
 
        if verbose:
            print(f"[MOF] Iniciando: {total_ciclos} ciclos, de {self.ano_inicio_fixo}-{self.ano_inicio_simulacao} "
                  f"até {self.ano_inicio_fixo}-{self.ano_fim_final}")
 
        for ano_fim, janela in self._extrairJanelas():
            ciclo_atual += 1
 
            resumo = self.executarCiclo(ano_fim, janela, n_anos_simulacao, n_series, semente_base, verbose, executor=executor)
            resumo['ciclo'] = ciclo_atual
            self.log.append(resumo)
 
            if verbose:
                ciclos_restantes = total_ciclos - ciclo_atual
                if ciclos_restantes > 0:
                    estimativa = sams.estimarTempo(resumo['tempo'], ciclos_restantes)
                    print(f"[MOF] Ciclo {ciclo_atual}/{total_ciclos} concluído. "
                          f"Estimativa para os {ciclos_restantes} restantes: {estimativa}")
 
        return self.log

    def salvarLog(self, pasta: str = PASTA_RESULTADO, nome_arquivo: str = "MOF_log.csv") -> str:
        # Salva um CSV resumo com uma linha por ciclo (janela, modelo, AIC, BIC, tempo),
        if not self.log:
            raise RuntimeError("[ERRO MOF] Executar executarTodos() antes de salvar o log.")
 
        os.makedirs(pasta, exist_ok=True)
        caminho_completo = os.path.join(pasta, nome_arquivo)
 
        colunas = ['ciclo', 'janela', 'ano_fim', 'dias_janela', 'modelo', 'aic', 'bic', 'tempo', 'arquivo']
        df_log = pd.DataFrame(self.log)[colunas]
        df_log['aic'] = df_log['aic'].round(2)
        df_log['bic'] = df_log['bic'].round(2)
        df_log['tempo'] = df_log['tempo'].round(1)
        df_log.to_csv(caminho_completo, index=False, sep=';', decimal=',')
 
        return caminho_completo