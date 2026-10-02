import matplotlib
matplotlib.use('Agg')
from pipeline import PipelineSimulacaoFPE

if __name__ == "__main__":
    pipeline = PipelineSimulacaoFPE()
    pipeline.executar_poco_duplo()
    pipeline.executar_benchmark_problema2(ordem_bdf=2)
    print("\nSimulacoes e diagnosticos concluidos com sucesso!")