import os
import pandas as pd
import numpy as np

PASTA_MODULO = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_MODULO)
PASTA_DADOS = os.path.join(PASTA_RAIZ, "dados")

CAMINHO_BANCO_CSV = os.path.join(PASTA_DADOS, "Banco.csv")
CAMINHO_BANCO_PROCESSADO = os.path.join(PASTA_DADOS, "Banco_processado.csv")

class banco:

    def __init__(self, filepath=CAMINHO_BANCO_CSV, carregar_ao_iniciar=True):
        self.filepath = filepath
        self.serie = None
        if carregar_ao_iniciar:
            self.serie = self.carregarDados(filepath)

    def carregarDados(self, filepath=None):
        filepath = filepath or self.filepath
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")

        coluna_data = df.columns[0]
        colunas_vazao = [f"Vazao{str(i).zfill(2)}" for i in range(1, 32) if f"Vazao{str(i).zfill(2)}" in df.columns]
        df[coluna_data] = pd.to_datetime(df[coluna_data], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=[coluna_data])

        # Transforma as colunas em formato long
        df_long = df.melt(
            id_vars=[coluna_data],
            value_vars=colunas_vazao,
            var_name='coluna_dia',
            value_name='vazao_raw'
        )

        # Extrai o dia a partir do nome da coluna
        df_long['dia'] = df_long['coluna_dia'].str.replace('Vazao', '', regex=False).astype(int)
        df_long['mes'] = df_long[coluna_data].dt.month
        df_long['ano'] = df_long[coluna_data].dt.year

        # Ignorar ano bissexto
        df_long = df_long[~((df_long['mes'] == 2) & (df_long['dia'] == 29))]

        # Monta a data do dia, descarta dias que não existe
        df_long['Data'] = pd.to_datetime(
            dict(year=df_long['ano'], month=df_long['mes'], day=df_long['dia']),
            errors='coerce'
        )
        df_long = df_long.dropna(subset=['Data'])

        # Limpa strings com espaços/vírgulas
        vazao_limpa = (
            df_long['vazao_raw']
            .astype(str)
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df_long['Vazao'] = pd.to_numeric(vazao_limpa, errors='coerce')
        df_long = df_long.dropna(subset=['Vazao'])
        df_serie = df_long[['Data', 'Vazao']].sort_values('Data').set_index('Data')
        self.serie = df_serie['Vazao']

        return self.serie

    def salvarSerieProcessada(self, filepath=CAMINHO_BANCO_PROCESSADO):

        if self.serie is None:
            raise ValueError("[ERRO BANCO] Nenhuma série carregada ainda")

        self._criar_pasta_se_necessario(filepath)

        serie_saida = self.serie.rename('Vazao')
        serie_saida.index = serie_saida.index.rename('Data')
        df_saida = serie_saida.reset_index()
 
        if filepath.endswith('.parquet'):
            df_saida.to_parquet(filepath, index=False)
        else:
            # sep=';' e decimal=',' -> mesmo padrão do CSV original (ANA / Excel BR)
            df_saida.to_csv(filepath, index=False, sep=';', decimal=',')

    def carregarSerieProcessada(self, filepath=CAMINHO_BANCO_PROCESSADO):

        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, sep=';', decimal=',', parse_dates=['Data'])
 
        df = df.sort_values('Data').set_index('Data')
        self.serie = df['Vazao']
        return self.serie

    def extrairMOF(self, ano_init, ano_fim_final=None):

        serie = self._serie_carregada()
 
        if ano_fim_final is None:
            ano_fim_final = int(serie.index.year.max())
 
        if ano_fim_final < ano_init:
            raise ValueError(f"[ERRO BANCO] ano_fim_final ({ano_fim_final}) menor que ano_init ({ano_init})")
 
        for ano_fim in range(ano_init, ano_fim_final + 1):
            data_inicio = f"{ano_init}-01-01"
            data_fim = f"{ano_fim}-12-31"
 
            dados_filtrados = serie.loc[data_inicio:data_fim]
 
            # Validação de segurança para não quebrar a calibração estocástica
            if dados_filtrados.empty:
                raise ValueError(f"[ERRO BANCO] Nenhum dado encontrado no intervalo MOF: {data_inicio} até {data_fim}")
 
            yield ano_fim, dados_filtrados

    def extrairMJD(self, ano_init_win, size_win=30, ano_fim_final=None):

        serie = self._serie_carregada()
 
        if ano_fim_final is None:
            ano_fim_final = int(serie.index.year.max())
 
        ano_fim_win = ano_init_win + size_win - 1
 
        if ano_fim_win > ano_fim_final:
            raise ValueError(
                f"[ERRO BANCO] Janela MJD inicial ({ano_init_win}-{ano_fim_win}) "
                f"já ultrapassa ano_fim_final ({ano_fim_final})"
            )
 
        while ano_fim_win <= ano_fim_final:
            data_inicio = f"{ano_init_win}-01-01"
            data_fim = f"{ano_fim_win}-12-31"
 
            dados_filtrados = serie.loc[data_inicio:data_fim]
 
            # Validação de segurança para não quebrar a calibração estocástica
            if dados_filtrados.empty:
                raise ValueError(f"[ERRO BANCO] Nenhum dado encontrado no intervalo MJD: {data_inicio} até {data_fim}")
 
            yield ano_init_win, ano_fim_win, dados_filtrados
 
            ano_init_win += 1
            ano_fim_win += 1

    def _serie_carregada(self):
        if self.serie is None:
            raise ValueError("[ERRO BANCO] Nenhuma série carregada. Chame carregarDados() ou carregarSerieProcessada() primeiro.")
        return self.serie

    def _criar_pasta_se_necessario(self, filepath):
        pasta = os.path.dirname(filepath)
        if pasta:
            os.makedirs(pasta, exist_ok=True)