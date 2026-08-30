# Módulo de análise estatística da série histórica observada

# Roda uma única vez sobre a série histórica completa para:
# 1. Calcular estatística básicas (média, desvio-padrão, variância);
# 2. Verificar não estacionariedade (Mann-Kendall, Spearman Rho e Pettitt recursivo);
# 3. Exportar todos os resultados em um CSV dentro da pasta 'resultado';

# Bibliotecas e Classes
import os
import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy import stats

NIVEL_SIGNIFICANCIA = 0.05
TAMANHO_MINIMO_SEGMENTO_PETTITT = 4

PASTA_MODULO = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_MODULO)
PASTA_RESULTADO = os.path.join(PASTA_RAIZ, "resultado")

class estatistica:

    def __init__(self, dados):
        self.serieDiaria = self._normalizar_entrada(dados)
        self.dadosAnuais = self.serieDiaria.resample('YE').mean()
        self.anoInicioSerie = int(self.serieDiaria.index.year.min())
        self.anoFimSerie = int(self.serieDiaria.index.year.max())
        self.resultadoEstatisticasBasicas = None
        self.resultadoMannKendall = None
        self.resultadoRho = None
        self.resultadoPettitt = None

    def _normalizar_entrada(self, dados) -> pd.Series:

        if isinstance(dados, pd.Series):
            serie = dados.copy()
            serie.index = pd.to_datetime(serie.index)
            serie.name = 'Vazao'
            return serie.sort_index()

        if isinstance(dados, pd.DataFrame):
            colunas = {coluna.lower(): coluna for coluna in dados.columns}
            colunaData = colunas.get('data')
            colunaVazao = colunas.get('vazao')

            if colunaData is None or colunaVazao is None:
                raise ValueError(
                    "[ERRO ESTATISTICAS] DataFrame precisa ter colunas 'Data' e 'Vazao' (ou 'data'/'vazao')."
                )

            serie = dados.set_index(pd.to_datetime(dados[colunaData]))[colunaVazao]
            serie.name = 'Vazao'
            return serie.sort_index()

        raise TypeError("[ERRO ESTATISTICAS] 'dados' precisa ser um pd.Series ou pd.DataFrame.")

    def executar_analise(self) -> None:
        # Executa todas as análises, nessa ordem
        self._calcular_estatistica_basicas()
        self._testar_mann_kendall()
        self._testar_rho_spearman()
        self._testar_pettitt()

    def _calcular_estatistica_basicas(self) -> None:
        # Estatisticas descritivas sobre a série diária completa
        valores = self.serieDiaria.values

        self.resultadoEstatisticasBasicas = {
            'media': float(np.mean(valores)),
            'desvio_padrao': float(np.std(valores, ddof=1)),
            'variancia': float(np.var(valores, ddof=1)),
        }

    def _testar_mann_kendall(self) -> None:
        # Detecta tendência monotônica na série anual
        resultado = mk.original_test(self.dadosAnuais.values)

        self.resultadoMannKendall = {
            'tendencia': resultado.trend,
            'p_valor': resultado.p,
            'significativo': resultado.p < NIVEL_SIGNIFICANCIA
        }

    def _testar_rho_spearman(self) -> None:
        # Avalia correlação monotônica entre tempo e vazão anual
        tempo = np.arange(len(self.dadosAnuais))
        correlacao, pValor = stats.spearmanr(tempo, self.dadosAnuais.values)

        self.resultadoRho = {
            'correlacao': correlacao,
            'p_valor': pValor,
            'significativo': pValor < NIVEL_SIGNIFICANCIA
        }

    def _testar_pettitt_segmento(self, indices: np.ndarray):
        """
        Aplica o teste de Pettitt (via rankings) em um segmento da
        série definido por índices. Retorna None se o segmento for
        pequeno demais para o teste ter poder estatístico.
        """
        segmento = self.dadosAnuais.values[indices]
        tamanhoSegmento = len(segmento)
 
        if tamanhoSegmento < TAMANHO_MINIMO_SEGMENTO_PETTITT:
            return None
 
        ranks = stats.rankdata(segmento)
 
        # Estatística U calculada de forma incremental (evita montar
        # a matriz completa de comparações par a par)
        estatisticaU = np.zeros(tamanhoSegmento)
        for t in range(1, tamanhoSegmento):
            estatisticaU[t] = estatisticaU[t - 1] + (2 * ranks[t] - tamanhoSegmento - 1)
 
        indiceLocal = int(np.argmax(np.abs(estatisticaU)))
        kStat = np.max(np.abs(estatisticaU))
 
        # Aproximação assintótica do p-valor do teste de Pettitt
        pValor = 2 * np.exp((-6 * kStat**2) / (tamanhoSegmento**3 + tamanhoSegmento**2))
        pValor = min(pValor, 1.0)
 
        # Converte o índice local do segmento para o índice global da
        # série completa, necessário para localizar a data da quebra
        indiceGlobal = indices[indiceLocal]
 
        return indiceGlobal, kStat, pValor

    def _testar_pettitt_recursivo(self, indices: np.ndarray, quebras: list) -> None:
        """
        Aplica o teste de Pettitt recursivamente, dividindo a série em
        dois subsegmentos a cada quebra significativa encontrada, até
        que nenhum novo ponto de quebra seja detectado.
        """
        resultado = self._testar_pettitt_segmento(indices)
        if resultado is None:
            return
 
        indiceGlobal, kStat, pValor = resultado
 
        if pValor < NIVEL_SIGNIFICANCIA:
            dataQuebra = self.dadosAnuais.index[indiceGlobal].date()
 
            quebras.append({
                'ponto_quebra' : indiceGlobal,
                'data_quebra'  : dataQuebra,
                'k_stat'       : kStat,
                'p_valor'      : pValor,
                'significativo': True
            })
 
            self._testar_pettitt_recursivo(indices[:indiceGlobal - indices[0] + 1], quebras)
            self._testar_pettitt_recursivo(indices[indiceGlobal - indices[0] + 1:], quebras)
 
    def _testar_pettitt(self) -> None:
        # Identifica todos os pontos de ruptura significativos na série
        indices = np.arange(len(self.dadosAnuais))
        quebras = []
        self._testar_pettitt_recursivo(indices, quebras)
 
        quebras = sorted(quebras, key=lambda quebra: quebra['ponto_quebra'])
 
        self.resultadoPettitt = {
            'quebras': quebras,
            'significativo': len(quebras) > 0
        }

    def verificar_estacionariedade(self) -> bool:
        # Retorna True se a série for considerada estacionária pelos três testes
        if self.resultadoMannKendall is None or self.resultadoRho is None or self.resultadoPettitt is None:
            raise RuntimeError("Rode executar_analise() antes de verificar a estacionariedade.")
 
        mannKendallEstacionario = not self.resultadoMannKendall['significativo']
        rhoEstacionario = not self.resultadoRho['significativo']
        pettittEstacionario = not self.resultadoPettitt['significativo']
 
        return mannKendallEstacionario and rhoEstacionario and pettittEstacionario
 
    def _formatar_valor(self, valor):

        if isinstance(valor, bool):
            return valor  # bool é subclasse de int, então checa antes
        if isinstance(valor, (float, np.floating)):
            return f"{valor:.6f}".replace('.', ',')
        return valor

    def _montar_linhas_resultado(self) -> list:
        # Achata todos os resultados em uma lista de linhas (formato longo),
        # já que os testes produzem quantidades diferentes de valores cada um
        linhas = []
 
        linhas.append({'categoria': 'Periodo Analisado', 'metrica': 'Ano Inicial', 'valor': self.anoInicioSerie})
        linhas.append({'categoria': 'Periodo Analisado', 'metrica': 'Ano Final', 'valor': self.anoFimSerie})
 
        linhas.append({'categoria': 'Estatisticas Basicas', 'metrica': 'Media', 'valor': self.resultadoEstatisticasBasicas['media']})
        linhas.append({'categoria': 'Estatisticas Basicas', 'metrica': 'Desvio Padrao', 'valor': self.resultadoEstatisticasBasicas['desvio_padrao']})
        linhas.append({'categoria': 'Estatisticas Basicas', 'metrica': 'Variancia', 'valor': self.resultadoEstatisticasBasicas['variancia']})
 
        linhas.append({'categoria': 'Mann-Kendall', 'metrica': 'Tendencia', 'valor': self.resultadoMannKendall['tendencia']})
        linhas.append({'categoria': 'Mann-Kendall', 'metrica': 'p-valor', 'valor': self.resultadoMannKendall['p_valor']})
        linhas.append({'categoria': 'Mann-Kendall', 'metrica': 'Significativo', 'valor': self.resultadoMannKendall['significativo']})
 
        linhas.append({'categoria': 'Spearman Rho', 'metrica': 'Correlacao', 'valor': self.resultadoRho['correlacao']})
        linhas.append({'categoria': 'Spearman Rho', 'metrica': 'p-valor', 'valor': self.resultadoRho['p_valor']})
        linhas.append({'categoria': 'Spearman Rho', 'metrica': 'Significativo', 'valor': self.resultadoRho['significativo']})
 
        linhas.append({'categoria': 'Pettitt', 'metrica': 'Total de Quebras', 'valor': len(self.resultadoPettitt['quebras'])})
        for indice, quebra in enumerate(self.resultadoPettitt['quebras'], 1):
            categoria = f'Pettitt - Quebra {indice}'
            linhas.append({'categoria': categoria, 'metrica': 'Data', 'valor': quebra['data_quebra']})
            linhas.append({'categoria': categoria, 'metrica': 'K Estatistico', 'valor': quebra['k_stat']})
            linhas.append({'categoria': categoria, 'metrica': 'p-valor', 'valor': quebra['p_valor']})
 
        linhas.append({'categoria': 'Conclusao', 'metrica': 'Serie Estacionaria', 'valor': self.verificar_estacionariedade()})

        # Aplica a formatação de vírgula decimal em todos os valores numéricos
        for linha in linhas:
            linha['valor'] = self._formatar_valor(linha['valor'])
 
        return linhas
 
    def _criar_pasta_resultado(self, pasta: str) -> None:
        os.makedirs(pasta, exist_ok=True)
 
    def salvarResultados(self, pasta: str = PASTA_RESULTADO, nome_arquivo: str = "estatisticas_resultado.csv") -> str:

        if self.resultadoMannKendall is None or self.resultadoRho is None or self.resultadoPettitt is None or self.resultadoEstatisticasBasicas is None:
            raise RuntimeError("Rode executar_analise() antes de salvar os resultados.")
 
        self._criar_pasta_resultado(pasta)
 
        linhas = self._montar_linhas_resultado()
        df_resultado = pd.DataFrame(linhas)
 
        caminho_completo = os.path.join(pasta, nome_arquivo)
        df_resultado.to_csv(caminho_completo, index=False, sep=';', decimal=',')
 
        return caminho_completo

    def imprimir_resumo(self) -> None:
        # Imprime um resumo legível dos resultados das análises
        if self.resultadoMannKendall is None:
            raise RuntimeError("Rode executar_analise() antes de imprimir o resumo.")
 
        print("Análise Estatística da Série Histórica")
        print(f"Período: {self.anoInicioSerie} - {self.anoFimSerie}")
 
        print("\n[ Estatísticas Básicas ]")
        print(f"  Média        : {self.resultadoEstatisticasBasicas['media']:.4f}")
        print(f"  Desvio Padrão: {self.resultadoEstatisticasBasicas['desvio_padrao']:.4f}")
        print(f"  Variância    : {self.resultadoEstatisticasBasicas['variancia']:.4f}")
 
        print("\n[ Mann-Kendall ]")
        print(f"  Tendência    : {self.resultadoMannKendall['tendencia']}")
        print(f"  p-valor      : {self.resultadoMannKendall['p_valor']:.4f}")
        print(f"  Significativo: {self.resultadoMannKendall['significativo']}")
 
        print("\n[ Spearman Rho ]")
        print(f"  Correlação   : {self.resultadoRho['correlacao']:.4f}")
        print(f"  p-valor      : {self.resultadoRho['p_valor']:.4f}")
        print(f"  Significativo: {self.resultadoRho['significativo']}")
 
        print("\n[ Pettitt ]")
        if not self.resultadoPettitt['significativo']:
            print("  Nenhum ponto de quebra significativo encontrado.")
        else:
            print(f"  Total de quebras encontradas: {len(self.resultadoPettitt['quebras'])}")
            for indice, quebra in enumerate(self.resultadoPettitt['quebras'], 1):
                print(f"\n  Quebra {indice}:")
                print(f"    Data         : {quebra['data_quebra']}")
                print(f"    K estatístico: {quebra['k_stat']:.4f}")
                print(f"    p-valor      : {quebra['p_valor']:.4f}")
 
        if self.verificar_estacionariedade():
            print("\nSérie ESTACIONÁRIA")
        else:
            print("\nSérie NÃO ESTACIONÁRIA")