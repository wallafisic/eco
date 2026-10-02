import numpy as np
import matplotlib.pyplot as plt
from fpe_solver import FPEChangCooper, FPEBenchmark
from diagnostics import DiagnosticosFPE, OECValidator
from visualization import VisualizadorFPE
from oec_operator import OECOperator

class PipelineSimulacaoFPE:
    """
    Orquestrador do fluxo de simulacao, integrando Solvers, OEC e Diagnosticos.
    """

    def __init__(self, xmin=-4.0, xmax=4.0, N=200, D=1.0, kB_T=1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.N = N
        self.D = D
        self.kB_T = kB_T
        self.solver_cc = FPEChangCooper(xmin=xmin, xmax=xmax, N=N, D=D, kB_T=kB_T)


    def executar_poco_duplo(self, t_eval=[0.0, 0.1, 0.5, 2.0, 5.0, 15.0]):
        print("===========================================================")
        print(" 1. SIMULACAO FPE: POTENCIAL DE POCO DUPLO (CHANG-COOPER) ")
        print("===========================================================")
        print(f"Malha Total (N+1): {len(self.solver_cc.x)} pontos")
        print(f"Passo Espacial dx: {self.solver_cc.dx:.4f}")

        # 1. Grafico Potencial
        curvas_p = [{'y': self.solver_cc.G_prime(self.solver_cc.x), 'color': 'red', 'label': r'Derivada $\frac{dG}{dx} = x^3 - x$'}]
        curvas_s = [{'y': self.solver_cc.G(self.solver_cc.x), 'color': 'k', 'linestyle': '--', 'alpha': 0.3, 'label': r'Potencial $G(x) = \frac{x^4}{4} - \frac{x^2}{2}$'}]
        fig, _ = VisualizadorFPE.criar_grafico_potencial(self.solver_cc.x, curvas_p, curvas_s, "Potencial Quártico de Poço Duplo e sua Derivada")
        fig.savefig("../Figures/grafico_potencial.png")
        plt.close(fig)

        # 2. Resolucao Temporal via IVP Integrator (Codigo Original)
        self.solver_cc.construir_matriz_operador()
        p0 = self.solver_cc.condicao_inicial_gaussiana()
        sol_ivp = self.solver_cc.resolver(t_eval=t_eval, p0=p0)

        # 3. Diagnostico da Discretizacao Espacial
        VisualizadorFPE.diagnostico_discretizacao(
            self.solver_cc.x_int, self.solver_cc.a_coef, self.solver_cc.b_coef, self.solver_cc.c_coef,
            self.solver_cc.A, self.solver_cc.dx, self.solver_cc.D
        )

        # 4. Evolucao Temporal Padrao
        VisualizadorFPE.evolucao_temporal(
            self.solver_cc.x, sol_ivp, t_eval, G_func=self.solver_cc.G
        )

        # 5. Conservacao de Massa
        massas, erro_max = DiagnosticosFPE.calcular_conservacao_massa(sol_ivp, t_eval, self.solver_cc.dx)
        VisualizadorFPE.conservacao_massa(t_eval, massas, erro_max)

        # =========================================================
        # 6. IMPLEMENTACAO E VALIDACAO DO OEC (PRINCIPIO I - SEÇÃO 5.6)
        # =========================================================
        print("\n-----------------------------------------------------------")
        print(" VALIDACAO DO OEC: PRINCIPIO I (FIDELIDADE DA SOLUCAO FORMAL)")
        print("-----------------------------------------------------------")
        oec = OECOperator(self.solver_cc)
        
        # Evolucao Formal via Semigrupo Matricial exp(A * dt)
        sol_oec = oec.evoluir_semigrupo_exato(p0, t_eval)
        
        # Auditoria Princípio I: Fidelidade da Solução Formal (IVP vs OEC)
        erro_l2, erro_linf, residuo_pontual = DiagnosticosFPE.validar_principio_I_oec(
            sol_ivp.y, sol_oec
        )

        # Auditoria Princípio II: Conservação de Constantes de Movimento (Massa L1 no OEC)
        massas_oec, desvio_max_massa_oec = OECValidator.validar_principio_II(
            sol_oec, self.solver_cc.dx
        )

        # Auditoria Princípio III: Estabilidade Espectral e Taxa de Relaxamento
        evals, lambda_0, lambda_1, gap, residuos_relax = OECValidator.validar_principio_III(
            self.solver_cc.A, sol_oec, t_eval
        )

        # Auditoria Princípio IV: Preservacao da Positividade (Operador de Markov)
        val_min_oec, is_positivo_oec = OECValidator.validar_principio_IV(sol_oec)


        # ---------------------------------------------------------------------
        # Auditoria Princípio V: Compatibilidade e Suporte Parametrico
        # ---------------------------------------------------------------------
        def builder_solver(p_name, p_val):
            kwargs = {
                'N': self.N, 
                'xmin': self.xmin, 
                'xmax': self.xmax, 
                'D': self.D,
                'kB_T': self.kB_T
            }
            kwargs[p_name] = p_val
            solver = FPEChangCooper(**kwargs)
            solver.construir_matriz_operador()  # <--- Adicionado para preencher solver.A
            return solver

        # Execucao da validacao do Principio V em relacao ao parametro de difusao D
        sensibilidade_A, var_densidade, is_param_compativel = OECValidator.validar_principio_V(
            builder_solver, param_nome='D', param_val=self.D, d_param=1e-5
        )


        # ---------------------------------------------------------------------
        # Auditoria Principio VI: Reutilizacao Computacional e Amortizacao
        # ---------------------------------------------------------------------
        speedup, tempo_ivp, tempo_oec, erro_comp, is_reutilizavel = OECValidator.validar_principio_VI(
            self.solver_cc, oec, p0, k_passos=100, dt_step=0.01
        )


        # ---------------------------------------------------------------------
        # Exibição Consolidada dos 5 Princípios do OEC no Terminal
        # ---------------------------------------------------------------------
        print("\n-----------------------------------------------------------")
        print(" AUDITORIA CONSOLIDADA DOS 5 PRINCIPIOS DO OEC")
        print("-----------------------------------------------------------")
        print(f"Princípio I   - Erro Relativo L2 (IVP vs OEC): {erro_l2:.6e}")
        print(f"Princípio I   - Erro Máximo L_inf (IVP vs OEC): {erro_linf:.6e}")
        print(f"Princípio II  - Desvio Máximo de Massa L1 (OEC): {desvio_max_massa_oec:.6e}")
        print(f"Princípio III - Autovalor Estacionário Re(lambda_0): {np.real(lambda_0):.6e}")
        print(f"Princípio III - Modo Metaestável Re(lambda_1): {np.real(lambda_1):.6e}")
        print(f"Princípio III - Spectral Gap Físico (gamma): {gap:.6f}")
        print(f"Princípio IV  - Mínimo Global p(x,t): {val_min_oec:.6e} (Preserva Positividade: {is_positivo_oec})")
        print(f"Princípio V   - Sensibilidade ||dA/dD||_2       : {sensibilidade_A:.6e}")
        print(f"Princípio V   - Derivada Densidade ||dp/dD||_2  : {var_densidade:.6e} (Compatível: {is_param_compativel})")
        print(f"Princípio VI  - Speedup de Amortização (Reuso)  : {speedup:.2f}x (Erro Composição: {erro_comp:.6e})")
        print("-----------------------------------------------------------\n")

        # Renderizacao Grafica do Painel Duplo (visualization.py)
        VisualizadorFPE.comparacao_fidelidade_oec(
            self.solver_cc.x, sol_ivp.y, sol_oec, t_eval, erro_l2, erro_linf, residuo_pontual
        )

        # Renderizacao Grafica do Principio II: Conservacao de Massa do OEC (Princípio II)
        VisualizadorFPE.conservacao_massa(
            t_eval, massas_oec, desvio_max_massa_oec, nome_arquivo="../Figures/conservacao_massa_oec.png"
        )

        # Renderização Grafica do Principio III
        VisualizadorFPE.validacao_espectral_oec(
            t_eval, residuos_relax, gap, np.real(lambda_1)
        )


    def executar_benchmark_problema2(self, ordem_bdf=2):
        print("\n===========================================================")
        print(" 2. VALIDACAO DO PROBLEMA 2 (FPE) E ANALISE DE ERRO       ")
        print("===========================================================")

        x_fpe, t_fpe, W_num = FPEBenchmark.resolver(M=10, N=100, T_final=1.0, ordem_bdf=ordem_bdf)
        X_grid, T_grid = np.meshgrid(x_fpe, t_fpe)
        W_exato = X_grid * np.exp(T_grid)
        erro_rms = np.sqrt(np.sum((W_num[-1, :] - W_exato[-1, :])**2)) / np.sqrt(np.sum(W_exato[-1, :]**2))
        print(f"Erro RMS no tempo final (BDF{ordem_bdf}): {erro_rms:.6e}")

        print(f"\n--- Gerando Estudo de Convergencia Grafica (BDF{ordem_bdf}) ---")
        lista_M = [10, 20, 40, 80, 160]
        lista_N = [100, 200, 400, 800, 1600]
        
        erros_l2, erros_linf, h_valores = [], [], []

        for M, N in zip(lista_M, lista_N):
            x_i, t_i, W_n = FPEBenchmark.resolver(M=M, N=N, T_final=1.0, ordem_bdf=ordem_bdf)
            X_g, T_g = np.meshgrid(x_i, t_i)
            W_e = X_g * np.exp(T_g)
            
            e_l2, e_linf = DiagnosticosFPE.calcular_erros_normas(W_n, W_e)
            erros_l2.append(e_l2)
            erros_linf.append(e_linf)
            h_valores.append(1.0 / M)
            print(f"M={M:3d}, N={N:4d} | Erro L2: {e_l2:.4e} | Erro L_inf: {e_linf:.4e}")

        x_ref, t_ref, W_ref = FPEBenchmark.resolver(M=40, N=400, T_final=1.0, ordem_bdf=ordem_bdf)
        W_exato_ref = x_ref * np.exp(1.0)
        erro_espacial = np.abs(W_ref[-1, :] - W_exato_ref)

        VisualizadorFPE.convergencia_estudo(h_valores, erros_l2, erros_linf, x_ref, erro_espacial, ordem_bdf=ordem_bdf)