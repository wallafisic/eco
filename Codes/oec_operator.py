import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from fpe_solver import FPEChangCooper

class OECOperator:
    """
    Operador de Evolucao Computacional (OEC).
    Mapeia a matriz geradora discreta A para o seu semigrupo formal exp(A * dt)
    e calcula a analise espectral associada.
    """
    def __init__(self, fpe_solver: FPEChangCooper):
        self.solver = fpe_solver
        if self.solver.A is None:
            self.solver.construir_matriz_operador()
        self.A = self.solver.A


    def obter_propagador_exato(self, dt: float) -> np.ndarray:
        """
        Computa o operador semigrupo de transicao formal: E(dt) = exp(A * dt).
        """
        return expm(self.A * dt)


    def evoluir_semigrupo_exato(self, p0: np.ndarray, t_eval: list) -> np.ndarray:
        """
        Evolui o estado p0 ao longo dos instantes em t_eval utilizando
        aplicacoes sucessivas ou diretas do propagador exponencial E(dt).
        """
        solucao_oec = np.zeros((len(self.solver.x), len(t_eval)))
        solucao_oec[:, 0] = p0

        for i in range(1, len(t_eval)):
            dt = t_eval[i] - t_eval[i - 1]
            E_dt = self.obter_propagador_exato(dt)
            p_next = E_dt @ solucao_oec[:, i - 1]
            
            # Normalizacao conservativa
            p_next /= (np.sum(p_next) * self.solver.dx)
            solucao_oec[:, i] = p_next

        return solucao_oec


    def calcular_lacuna_espectral(self, tol_metaestavel=1e-10):
        """
        Analisa os autovalores de A para determinar a taxa de relaxamento ao equilibrio,
        desconsiderando autovalores metaestaveis/de conservacao proximos de zero.
        """
        autovalores = np.linalg.eigvals(self.A)
        partes_reais = np.real(autovalores)
        
        idx = np.argsort(partes_reais)[::-1]
        evals_ordenados = autovalores[idx]
        reais_ordenados = partes_reais[idx]

        # Filtra autovalores estritamente menores que a tolerancia de metaestabilidade (-tol_metaestavel)
        modos_relaxamento = reais_ordenados[reais_ordenados < -tol_metaestavel]

        if len(modos_relaxamento) > 0:
            spectral_gap = np.abs(modos_relaxamento[0])
        else:
            spectral_gap = 0.0

        return evals_ordenados, spectral_gap