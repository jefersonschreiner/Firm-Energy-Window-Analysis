# Análise, Modelagem e Simulação Estocástica (SAMS)

# Bibliotecas e Classes
import os
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .hymodel import Stoch

warnings.filterwarnings('ignore')

def _ajustar_combo(tarefa: tuple) -> dict:
    
    dados, ordem, ordem_sazonal = tarefa
    try:
        modelo = SARIMAX(
            dados,
            order=ordem,
            seasonal_order=ordem_sazonal,
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        ajuste = modelo.fit(disp=False, low_memory=True)
    except Exception:
        return None

    if not np.isfinite(ajuste.aic):
        return None

    return {
        'ordem': ordem,
        'ordem_sazonal': ordem_sazonal,
        'ar_params': list(ajuste.arparams) if ordem[0] > 0 else [],
        'ma_params': list(ajuste.maparams) if ordem[2] > 0 else [],
        'sar_params': list(ajuste.seasonalarparams) if ordem_sazonal[0] > 0 else [],
        'sma_params': list(ajuste.seasonalmaparams) if ordem_sazonal[2] > 0 else [],
        'aic': ajuste.aic,
        'bic': ajuste.bic,
    }

class sams:

    DIAS_ANO = 365
    PERIODO_SAZONAL = 7 # período sazonal residual testado no SARIMA
    ORDEM_MAX_ARMA = 3 # p, q em 0..3 para o ARMA
    ORDEM_MAX_SARIMA_REGULAR = 2  # p, q em 0..2 para a parte regular do SARIMA
    ORDEM_MAX_SARIMA_SAZONAL = 1  # P, Q em 0..1 para a parte sazonal do SARIMA

    def __init__(self, serie: pd.Series):
        self.serie_vazao = serie.sort_index().copy()
 
        if np.any(self.serie_vazao.values <= 0):
            raise ValueError("[ERRO SAMS] Série contém valores <= 0. Verifique os dados (log exige vazão > 0).")
 
        self.serie_log = np.log(self.serie_vazao)
        self.dia_ano = np.array([self.calcularDiaAno(data) for data in self.serie_vazao.index])
 
        self.media_dia, self.desvio_dia = self._calcularEstatisticasDia()
        self.serie_padronizada = self._padronizar(self.serie_log.values, self.dia_ano)
 
        # Variáveis de resultado
        self.modelo_vencedor = None
        self.aic_vencedor = np.inf
        self.bic_vencedor = np.inf
        self.params_vencedor = None
        self.resultados = {}

    @staticmethod
    def calcularDiaAno(data: pd.Timestamp) -> int:
        # Ignora o dia do ano bissexto (mesma lógica usada em todo o projeto)
        dia = data.timetuple().tm_yday
        if data.is_leap_year and dia > 59:
            dia -= 1
        return dia

    def _calcularEstatisticasDia(self) -> tuple:
        # Média e desvio-padrão de log(vazão) para cada dia do ano
        # usados para padronizar/despadronizar a série (remover/reaplicar sazonalidade)
        df = pd.DataFrame({'dia_ano': self.dia_ano, 'log_vazao': self.serie_log.values})
        agrupado = df.groupby('dia_ano')['log_vazao'].agg(['mean', 'std'])
 
        media_dia = agrupado['mean'].reindex(range(1, self.DIAS_ANO + 1))
        desvio_dia = agrupado['std'].reindex(range(1, self.DIAS_ANO + 1))
 
        desvio_global = float(np.std(self.serie_log.values, ddof=1))
        media_global = float(np.mean(self.serie_log.values))
 
        desvio_dia = desvio_dia.fillna(desvio_global).replace(0, desvio_global)
        media_dia = media_dia.fillna(media_global)

        return media_dia.to_dict(), desvio_dia.to_dict()

    def _padronizar(self, log_valores: np.ndarray, dias_ano: np.ndarray) -> np.ndarray:
        # log(vazão) -> anomalia padronizada: (log - média_dia) / desvio_dia
        media = np.array([self.media_dia[d] for d in dias_ano])
        desvio = np.array([self.desvio_dia[d] for d in dias_ano])
        return (log_valores - media) / desvio

    def _despadronizar(self, serie_padronizada: np.ndarray, dias_ano: np.ndarray) -> np.ndarray:
        # Caminho inverso: reaplica a sazonalidade diária e volta para vazão real
        media = np.array([self.media_dia[d] for d in dias_ano])
        desvio = np.array([self.desvio_dia[d] for d in dias_ano])
        log_valores = serie_padronizada * desvio + media
        return np.exp(log_valores)

    def _combinacoesArma(self) -> list:
        # Grade ARMA(p,q), p,q em 0..ORDEM_MAX_ARMA, sem sazonalidade
        return [
            ('ARMA', (p, 0, q), (0, 0, 0, 0))
            for p in range(self.ORDEM_MAX_ARMA + 1)
            for q in range(self.ORDEM_MAX_ARMA + 1)
            if not (p == 0 and q == 0)
        ]

    def _combinacoesSarima(self) -> list:
        # Grade SARIMA(p,q)(P,Q)_s, s = PERIODO_SAZONAL
        return [
            ('SARIMA', (p, 0, q), (P, 0, Q, self.PERIODO_SAZONAL))
            for p in range(self.ORDEM_MAX_SARIMA_REGULAR + 1)
            for q in range(self.ORDEM_MAX_SARIMA_REGULAR + 1)
            for P in range(self.ORDEM_MAX_SARIMA_SAZONAL + 1)
            for Q in range(self.ORDEM_MAX_SARIMA_SAZONAL + 1)
            if not (p == 0 and q == 0 and P == 0 and Q == 0)
        ]

    def selecionar(self, executor: ProcessPoolExecutor = None, verbose: bool = True) -> str:

        combinacoes = self._combinacoesArma() + self._combinacoesSarima()
        tarefas = [(self.serie_padronizada, ordem, ordem_sazonal) for _, ordem, ordem_sazonal in combinacoes]

        gerenciar_pool = executor is None
        if gerenciar_pool:
            executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 1)

        try:
            resultados_brutos = list(executor.map(_ajustar_combo, tarefas))
        finally:
            if gerenciar_pool:
                executor.shutdown()

        melhores = {'ARMA': None, 'SARIMA': None}
        for (nome, _, _), resultado in zip(combinacoes, resultados_brutos):
            if resultado is None:
                continue
            if melhores[nome] is None or resultado['aic'] < melhores[nome]['aic']:
                melhores[nome] = resultado

        for nome, resultado in melhores.items():
            if resultado is None:
                continue
            entrada = {
                'modelo': nome,
                'ordem': resultado['ordem'],
                'ordem_sazonal': resultado['ordem_sazonal'],
                'ar_params': resultado['ar_params'],
                'ma_params': resultado['ma_params'],
                'sar_params': resultado['sar_params'] if nome == 'SARIMA' else [],
                'sma_params': resultado['sma_params'] if nome == 'SARIMA' else [],
                'aic': resultado['aic'],
                'bic': resultado['bic'],
            }
            self.resultados[nome] = entrada
            if entrada['aic'] < self.aic_vencedor:
                self.aic_vencedor = entrada['aic']
                self.bic_vencedor = entrada['bic']
                self.modelo_vencedor = nome
                self.params_vencedor = entrada

        if self.modelo_vencedor is None:
            raise RuntimeError("[ERRO SAMS] Nenhum modelo convergiu para esta janela.")

        if verbose:
            self.imprimirResumo()

        return self.modelo_vencedor
 
    def gerar(self, n_anos: int, n_series: int, semente_base: int = 42) -> np.ndarray:
        # Gera o ensemble de séries sintéticas, já em vazão real (com sazonalidade)
        if self.modelo_vencedor is None:
            raise RuntimeError("[ERRO SAMS] Executar selecionar() antes de gerar().")
 
        n_dias = n_anos * self.DIAS_ANO
        ensemble = np.zeros((n_series, n_dias))
        for i in range(n_series):
            np.random.seed(semente_base + i)
            ensemble[i] = self.gerarSerie(n_anos)
 
        return ensemble
 
    def gerarSerie(self, n_anos: int) -> np.ndarray:
        # Gera uma única série sintética de vazão real (n_anos * 365 dias)
        n_dias = n_anos * self.DIAS_ANO
        params = self.params_vencedor
        mean = float(np.mean(self.serie_padronizada))
        std_dev = float(np.std(self.serie_padronizada, ddof=1))
 
        if self.modelo_vencedor == 'ARMA':
            serie_padronizada_sintetica = Stoch.arma(
                n_steps=n_dias,
                mean=mean,
                std_dev=std_dev,
                ar_params=params['ar_params'],
                ma_params=params['ma_params'],
                init_value=self.serie_padronizada,
            )
        elif self.modelo_vencedor == 'SARIMA':
            serie_padronizada_sintetica = Stoch.sarima(
                n_steps=n_dias,
                mean=mean,
                std_dev=std_dev,
                ar_params=params['ar_params'],
                ma_params=params['ma_params'],
                seasonal_ar_params=params['sar_params'],
                seasonal_ma_params=params['sma_params'],
                seasonal_period=self.PERIODO_SAZONAL,
                d=0, D=0,
                init_value=self.serie_padronizada,
            )
        else:
            raise RuntimeError(f"[ERRO SAMS] Modelo vencedor desconhecido: {self.modelo_vencedor}")
 
        # Dia do ano cíclico para reaplicar a sazonalidade na série gerada
        dias_ano_sinteticos = np.tile(np.arange(1, self.DIAS_ANO + 1), n_anos)
        return self._despadronizar(serie_padronizada_sintetica, dias_ano_sinteticos)
 
    def imprimirResumo(self) -> None:
        # Imprime a tabela comparativa ARMA vs SARIMA
        print("   SELEÇÃO DO MODELO - SAMS (ARMA vs SARIMA)   ")
        print(f"{'Modelo':<10} {'Ordem':<18} {'AIC':>12} {'BIC':>12}")
        for nome, res in self.resultados.items():
            marca = " ←" if nome == self.modelo_vencedor else ""
            if nome == 'SARIMA':
                ordem_str = f"{res['ordem']}x{res['ordem_sazonal']}"
            else:
                ordem_str = f"{res['ordem']}"
            print(f" {nome:<10} {ordem_str:<18} {res['aic']:>12.2f} {res['bic']:>12.2f}{marca}")
        print(f"Vencedor: {self.modelo_vencedor} (AIC = {self.aic_vencedor:.2f})")
 
    @staticmethod
    def registrarLog(log: list, ciclo: int, janela: str, modelo: str, aic: float, bic: float, tempo: float) -> None:
        # Registra o resultado de cada calibração em uma lista de log
        log.append({
            'ciclo': ciclo,
            'janela': janela,
            'modelo': modelo,
            'aic': aic,
            'bic': bic,
            'tempo': tempo,
        })
 
    @staticmethod
    def estimarTempo(tempo_ciclo: float, ciclos_restante: int) -> str:
        # Estimativa de tempo restante com base no tempo do último ciclo
        segundos = tempo_ciclo * ciclos_restante
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        return f"{horas}h {minutos}min"
    
    
    