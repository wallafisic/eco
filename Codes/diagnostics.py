import numpy as np

class DiagnosticosFPE:
    """
    Ferramentas de analise quantitativa e diagnostico de erro.
    """
    @staticmethod
    def calcular_erros_normas(w_num, w_exato):
        """
        Calcula as normas de erro L2 relativo e L_infinity no instante final.
        """
        erro_pontual = np.abs(w_num[-1, :] - w_exato[-1, :])
        erro_linf = np.max(erro_pontual)
        erro_l2 = np.sqrt(np.sum(erro_pontual**2)) / np.sqrt(np.sum(w_exato[-1, :]**2))
        return erro_l2, erro_linf


    @staticmethod
    def calcular_conservacao_massa(sol_ivp, t_eval_list, dx):
        """
        Gera a serie temporal do valor integrado da massa e o erro maximo.
        """
        massas = [np.sum(sol_ivp.y[:, i]) * dx for i in range(len(t_eval_list))]
        erros_massa = np.abs(np.array(massas) - 1.0)
        erro_max = np.max(erros_massa)
        return massas, erro_max


    @staticmethod
    def validar_principio_I_oec(sol_ivp_y: np.ndarray, sol_oec: np.ndarray):
        """
        Principio I: Fidelidade da Solucao Formal (IVP vs OEC).
        Computa os erros L2 relativo, L_inf absoluto e o vetor de residuo pontual no tempo final.
        """
        p_final_ivp = sol_ivp_y[:, -1]
        p_final_oec = sol_oec[:, -1]
        
        # Erro Relativo L2
        erro_l2 = np.linalg.norm(p_final_ivp - p_final_oec) / np.linalg.norm(p_final_ivp)
        
        # Erro Maximo L_inf (Pontual)
        erro_linf = np.max(np.abs(p_final_ivp - p_final_oec))

        # Vetor de residuo absoluto pontual para plotagem
        residuo_pontual = np.abs(p_final_ivp - p_final_oec)
        
        return erro_l2, erro_linf, residuo_pontual


class OECValidator:
    """
    Executa a auditoria numerica e computa as metricas formais dos 3 Principios do OEC.
    """
    @staticmethod
    def validar_principio_I(sol_ivp_y: np.ndarray, sol_oec: np.ndarray):
        """
        Principio I: Fidelidade da Solucao Formal (IVP vs OEC).
        Computa os erros L2 relativo e L_inf absoluto no tempo final.
        """
        p_final_ivp = sol_ivp_y[:, -1]
        p_final_oec = sol_oec[:, -1]
        
        # Erro Relativo L2
        erro_l2 = np.linalg.norm(p_final_ivp - p_final_oec) / np.linalg.norm(p_final_ivp)
        
        # Erro Maximo L_inf (Pontual)
        erro_linf = np.max(np.abs(p_final_ivp - p_final_oec))

        # Vetor de residuo absoluto pontual para plotagem
        residuo_pontual = np.abs(p_final_ivp - p_final_oec)
        
        return erro_l2, erro_linf, residuo_pontual


    @staticmethod
    def validar_principio_II(sol_oec: np.ndarray, dx: float):
        """
        Principio II: Conservacao de Constantes de Movimento (Massa L1).
        Verifica a variacao da probabilidade total integrada em cada instante.
        """
        massa_tempo = np.sum(sol_oec, axis=0) * dx
        desvio_max_massa = np.max(np.abs(massa_tempo - 1.0))
        return massa_tempo, desvio_max_massa


    @staticmethod
    def validar_principio_III(A_mat: np.ndarray, sol_oec: np.ndarray, t_eval: list, tol_metaestavel: float = 1e-6):
        """
        Principio III: Estabilidade Espectral e Taxa de Relaxamento Local (Intrapoço).
        """
        autovalores = np.linalg.eigvals(A_mat)
        partes_reais = np.real(autovalores)
        
        idx = np.argsort(partes_reais)[::-1]
        evals_ordenados = autovalores[idx]
        reais_ordenados = partes_reais[idx]

        lambda_0 = reais_ordenados[0]
        lambda_1 = reais_ordenados[1]

        modos_fisicos = reais_ordenados[reais_ordenados < -tol_metaestavel]
        gap_fisico = np.abs(modos_fisicos[0]) if len(modos_fisicos) > 0 else 0.0

        # Referencia de equilibrio local (estado assintotico t_final)
        p_estacionario_local = sol_oec[:, -1]

        # Residuos de relaxamento L2
        residuos_relaxamento = [
            np.linalg.norm(sol_oec[:, k] - p_estacionario_local) for k in range(len(t_eval))
        ]

        return evals_ordenados, lambda_0, lambda_1, gap_fisico, residuos_relaxamento


    @staticmethod
    def validar_principio_IV(sol_oec: np.ndarray, tol_positividade: float = -1e-15):
        """
        Principio IV: Preservacao da Positividade (Operador de Markov).
        Verifica se a densidade calculada viola o cone positivo X_+.
        Retorna o valor minimo global encontrado e um booleano de conformidade.
        """
        val_minimo = np.min(sol_oec)

        # Tolerancia em relacao ao zero da precisao de maquina para descartar ruído de arredondamento
        is_positivo = val_minimo >= tol_positividade
        
        return val_minimo, is_positivo


    @staticmethod
    def validar_principio_V(solver_cc_builder, param_nome: str, param_val: float, d_param: float = 1e-5):
        """
        Princípio V: Compatibilidade e Suporte Paramétrico.
        Verifica a continuidade e a sensibilidade do operador A(theta) e a derivada
        da densidade em relação a perturbações no parâmetro físico theta.
        """
        from oec_operator import OECOperator

        # Estado Nominal (theta)
        solver_nom = solver_cc_builder(param_nome, param_val)
        A_nom = solver_nom.A.copy()
        
        # Estado Perturbado (theta + d_param)
        solver_pert = solver_cc_builder(param_nome, param_val + d_param)
        A_pert = solver_pert.A.copy()

        # Derivada Direcional do Operador: dA / d_theta
        dA_dparam = (A_pert - A_nom) / d_param
        norma_sensibilidade_A = np.linalg.norm(dA_dparam, ord=2)

        # Avaliação da variação da densidade p_OEC(t) no tempo t=1.0
        t_eval = [0.0, 1.0]
        p0 = solver_nom.condicao_inicial_gaussiana()
        
        oec_nom = OECOperator(solver_nom)
        oec_pert = OECOperator(solver_pert)

        sol_nom = oec_nom.evoluir_semigrupo_exato(p0, t_eval)[:, -1]
        sol_pert = oec_pert.evoluir_semigrupo_exato(p0, t_eval)[:, -1]
        
        # Variação da densidade: ||p_{theta + dt} - p_{theta}||_2 / d_theta
        variacao_densidade = np.linalg.norm(sol_pert - sol_nom, ord=2) / d_param

        # Compatibilidade: o operador e a densidade devem apresentar variações finitas (bem-postas)
        is_compativel = np.isfinite(norma_sensibilidade_A) and np.isfinite(variacao_densidade)

        return norma_sensibilidade_A, variacao_densidade, is_compativel


    @staticmethod
    def validar_principio_VI(solver_cc, oec, p0, k_passos: int = 10, dt_step: float = 0.1):
        """
        Principio VI: Reutilizacao Computacional e Amortizacao de Custo.
        Verifica a consistencia da composicao do semigrupo [E(dt)]^k p0 e mensura
        o ganho de desempenho computacional (speedup) do reuso do OEC em relacao ao IVP.
        """
        import time
        from scipy.linalg import expm

        # 1. Benchmark de Tempo: Solucionador de EDP (IVP - Recalculo contínuo)
        t_eval = np.linspace(0.0, k_passos * dt_step, k_passos + 1)
        
        start_ivp = time.perf_counter()
        sol_ivp = solver_cc.resolver(t_eval=t_eval, p0=p0)
        p_ivp_final = sol_ivp.y[:, -1]
        tempo_ivp = time.perf_counter() - start_ivp

        # 2. Benchmark de Tempo: Reuso do OEC (Composicao do Semigrupo [E(dt)]^k)
        start_oec = time.perf_counter()
        # Matriz de passo unico E(dt) = exp(A * dt) pre-calculada off-line
        E_dt = expm(solver_cc.A * dt_step)
        
        # Propagacao recursiva por produto matriz-vetor on-line (Reuso puro de Kernel)
        p_oec_reuso = p0.copy()
        for _ in range(k_passos):
            p_oec_reuso = E_dt @ p_oec_reuso
        tempo_oec = time.perf_counter() - start_oec

        # 3. Metricas de Validacao
        speedup = tempo_ivp / tempo_oec if tempo_oec > 0 else np.inf
        erro_composicao = np.linalg.norm(p_oec_reuso - p_ivp_final, ord=2) / np.linalg.norm(p_ivp_final, ord=2)
        is_reutilizavel = speedup > 1.0 and erro_composicao < 1e-6

        return speedup, tempo_ivp, tempo_oec, erro_composicao, is_reutilizavel