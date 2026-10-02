import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

class VisualizadorFPE:
    """Gerenciador de graficos e geracao de figuras de diagnostico."""

    @staticmethod
    def criar_grafico_potencial(x_eixo, curvas_eixo_principal=None, curvas_eixo_secundario=None, 
                                 titulo="Validação do Potencial", xlabel=r"Posição $x$"):
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
        linhas, labels = [], []

        if curvas_eixo_principal:
            for c_item in curvas_eixo_principal:
                l, = ax1.plot(x_eixo, c_item['y'], color=c_item.get('color', 'blue'), 
                              linestyle=c_item.get('linestyle', '-'), 
                              linewidth=c_item.get('linewidth', 1.5), 
                              alpha=c_item.get('alpha', 1.0), 
                              label=c_item.get('label', ''))
                linhas.append(l)
                labels.append(c_item.get('label', ''))
            ax1.set_ylabel("Derivada / Força", fontsize=15)
            ax1.tick_params(axis='y', labelsize=12)

        if curvas_eixo_secundario:
            ax2 = ax1.twinx()
            for c_item in curvas_eixo_secundario:
                l, = ax2.plot(x_eixo, c_item['y'], color=c_item.get('color', 'gray'), 
                              linestyle=c_item.get('linestyle', '--'), 
                              linewidth=c_item.get('linewidth', 1.5), 
                              alpha=c_item.get('alpha', 0.25), 
                              label=c_item.get('label', ''))
                linhas.append(l)
                labels.append(c_item.get('label', ''))
            ax2.set_ylabel(r'Energia Potencial $G(x)$', color='gray', fontsize=15)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=12)
            ax2.grid(False)

        ax1.set_xlabel(xlabel, fontsize=15)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_title(titulo, fontsize=12, pad=10)
        ax1.grid(True, linestyle=':', alpha=0.6)

        if labels:
            ax1.legend(linhas, labels, loc='upper left', frameon=True)

        plt.tight_layout()
        return fig, ax1

    @staticmethod
    def diagnostico_discretizacao(x_eixo, a_coef, b_coef, c_coef, A_mat, dx, D, nome_arquivo="../Figures/diagnostico_sistema.png"):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 12), dpi=120)

        ax1.plot(x_eixo[:len(a_coef)], a_coef, label=r'Subdiagonal $a_i$', color='navy', lw=1.8)
        ax1.plot(x_eixo[:len(c_coef)], c_coef, label=r'Superdiagonal $c_i$', color='crimson', lw=1.8, linestyle='--')
        ax1.axhline(D / (dx**2), color='gray', linestyle=':', label='Difusão pura')
        ax1.set_title("1. Coeficientes de Transporte ($a_i, c_i$)", fontsize=11)
        ax1.set_xlabel(r"Posição $x$", fontsize=15)
        ax1.set_ylabel("Valor do Coeficiente", fontsize=15)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='best', fontsize=9, frameon=True)

        ax2.plot(x_eixo[:len(b_coef)], b_coef, color='darkgreen', lw=1.8, label=r'Diagonal $b_i$')
        ax2.set_title("2. Diagonal Principal ($b_i$)", fontsize=11)
        ax2.set_xlabel(r"Posição $x$", fontsize=15)
        ax2.set_ylabel("Valor Diagonal", fontsize=15)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='best', fontsize=9, frameon=True)

        centro = A_mat.shape[0] // 2
        janela = slice(max(0, centro - 6), min(A_mat.shape[0], centro + 6))
        sub_A = A_mat[janela, janela]

        mascara = np.zeros_like(sub_A)
        for i in range(sub_A.shape[0]):
            for j in range(sub_A.shape[1]):
                if i == j:
                    mascara[i, j] = 1       
                elif i == j + 1:
                    mascara[i, j] = 2       
                elif j == i + 1:
                    mascara[i, j] = 3       

        cores = ['#F0F0F0', 'darkgreen', 'navy', 'crimson']
        cmap_custom = ListedColormap(cores)
        ax3.imshow(mascara, cmap=cmap_custom, aspect='equal', vmin=0, vmax=3)

        legend_elements = [
            mpatches.Patch(facecolor='navy', label=r'Subdiagonal $a_i$'),
            mpatches.Patch(facecolor='darkgreen', label=r'Diagonal $b_i$'),
            mpatches.Patch(facecolor='crimson', label=r'Superdiagonal $c_i$'),
            mpatches.Patch(facecolor='#F0F0F0', edgecolor='gray', label='Fora da banda (0)')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8.5, frameon=True)
        ax3.set_title("3. Acoplamento Local", fontsize=11)

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Diagnostico do sistema salvo em '{nome_arquivo}'!")

    @staticmethod
    def evolucao_temporal(x_eixo, sol_ivp, t_eval_list, G_func=None, nome_arquivo="../Figures/evolucao_temporal_p.png"):
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

        if G_func is not None:
            ax2 = ax1.twinx()
            ax2.set_ylim(-0.5, 2)
            ax2.plot(x_eixo, G_func(x_eixo), color='gray', linestyle=':', alpha=0.35, lw=1.8, label=r'Potencial $G(x)$')
            ax2.set_ylabel(r'Energia Potencial $G(x)$', color='gray', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.grid(False)

        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0.1, 0.9, len(t_eval_list)))

        for idx, t_val in enumerate(t_eval_list):
            p_t = sol_ivp.y[:, idx]
            ls = '--' if idx == 0 else '-'
            lbl = f'$t = {t_val:.1f}$ (C.I.)' if idx == 0 else (f'$t = {t_val:.1f}$ (Estacionário)' if idx == len(t_eval_list)-1 else f'$t = {t_val:.1f}$')
            ax1.plot(x_eixo, p_t, color=colors[idx], linestyle=ls, linewidth=1.8, label=lbl)

        ax1.set_xlim(-3.5, 3.5)
        ax1.set_xlabel(r"Posição $x$", fontsize=13)
        ax1.set_ylim(-0.5, 2)
        ax1.set_ylabel(r"Densidade de Probabilidade $p(x,t)$", fontsize=13)
        ax1.set_title(r"Evolução Temporal da Densidade $p(x,t)$ — Poço Duplo", fontsize=13)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Grafico de evolucao temporal salvo em '{nome_arquivo}'!")

    @staticmethod
    def conservacao_massa(t_eval_list, massas, erro_max, nome_arquivo="../Figures/conservacao_massa.png"):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.plot(t_eval_list, massas, 'o-', color='darkgreen', linewidth=1.8, label='Massa Integrada (Discreta)')
        ax.axhline(1.0, color='crimson', linestyle='--', label='Massa Teórica (1.0)')

        ax.set_xlabel(r"Tempo $t$", fontsize=12)
        ax.set_ylabel(r"Massa Total $\sum p_i \Delta x$", fontsize=12)
        ax.set_title(r"Validacao Numerica: Conservacao da Massa Total", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', frameon=True)

        ax.text(0.55, 0.20, f"Erro Maximo: {erro_max:.2e}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Grafico de conservacao de massa salvo em '{nome_arquivo}'! Erro max: {erro_max:.2e}")

    @staticmethod
    def convergencia_estudo(h_valores, erros_l2, erros_linf, x_ref, erro_espacial, ordem_bdf=2, nome_arquivo="../Figures/estudo_convergencia_p2.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        ax1.loglog(h_valores, erros_l2, 'o-', color='navy', lw=1.8, ms=6, label=r'Erro $L_2$ Relativo')
        ax1.loglog(h_valores, erros_linf, 's--', color='crimson', lw=1.8, ms=6, label=r'Erro $L_{\infty}$')
        
        guia_o2 = [erros_l2[0] * (h / h_valores[0])**2 for h in h_valores]
        ax1.loglog(h_valores, guia_o2, 'k:', alpha=0.5, label=r'Inclinação Teórica $\mathcal{O}(h^2)$')

        ax1.set_xlabel(r'Tamanho do Elemento Espacial $h = 1/M$', fontsize=15)
        ax1.set_ylabel('Erro Absoluto / Relativo', fontsize=15)
        ax1.set_title(f'Taxa de Convergência Numérica (BDF{ordem_bdf})', fontsize=11)
        ax1.grid(True, which="both", ls=":", alpha=0.6)
        ax1.legend(loc='upper left', frameon=True, fontsize=9)

        ax2.plot(x_ref, erro_espacial, color='darkgreen', lw=1.8, label=r'$|w_{\mathrm{num}} - w_{\mathrm{exato}}|$ em $t=1.0$')
        ax2.set_xlabel(r'Posição Espacial $x$', fontsize=15)
        ax2.set_ylabel(r'Erro Absoluto Pontual', fontsize=15)
        ax2.set_title(r'Distribuição Espacial do Erro ($M=40, N=400$)', fontsize=11)
        ax2.grid(True, ls=":", alpha=0.6)
        ax2.legend(loc='upper left', frameon=True, fontsize=9)

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Grafico de validacao e convergencia salvo em '{nome_arquivo}'!")


    @staticmethod
    def comparacao_fidelidade_oec(
        x_eixo: np.ndarray, 
        sol_ivp_y: np.ndarray, 
        sol_oec: np.ndarray, 
        t_eval_list: list, 
        erro_l2: float, 
        erro_linf: float, 
        residuo_pontual: np.ndarray,
        nome_arquivo: str = "../Figures/fidelidade_oec_validacao.png"
    ):
        """
        Renderiza o painel duplo de validacao do Principio I do OEC.
        Recebe as metricas de erro e o vetor de residuo ja calculados pelo OECValidator.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=120, sharex=True)

        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0.1, 0.9, len(t_eval_list)))

        # Painel 1: Perfil de Densidade (Solver IVP vs OEC Exato)
        for idx, t_val in enumerate(t_eval_list):
            p_ivp = sol_ivp_y[:, idx]
            p_oec = sol_oec[:, idx]
            
            ax1.plot(
                x_eixo, p_ivp, color=colors[idx], linestyle='-', lw=2.0, alpha=0.8, 
                label=f'IVP (BDF) t={t_val:.1f}' if idx in [0, len(t_eval_list)//2, len(t_eval_list)-1] else ""
            )
            ax1.plot(
                x_eixo, p_oec, color='black', linestyle=':', lw=1.5, 
                label=r'OEC $\exp(\mathbf{A}\Delta t)$' if idx == 0 else ""
            )

        ax1.set_xlim(-3.5, 3.5)
        ax1.set_ylabel(r"Densidade $p(x,t)$", fontsize=12)
        ax1.set_title(r"Princípio I: Fidelidade entre Solver Temporal (BDF) e OEC Formal $e^{\mathbf{A}\Delta t}$", fontsize=12)
        ax1.set_ylim(-0.5, 2)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True)

        # Painel 2: Resíduo / Diferença Pontual Fornecida pelo Validador
        ax2.plot(
            x_eixo, residuo_pontual, color='crimson', lw=1.8, 
            label=f'Diferença Absoluta em t={t_eval_list[-1]:.1f}'
        )
        ax2.set_xlabel(r"Posição $x$", fontsize=12)
        ax2.set_ylabel(r"$|p_{\mathrm{IVP}} - p_{\mathrm{OEC}}|$", fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', frameon=True)

        # Quadro de Texto com Metricas do OECValidator
        ax2.text(
            0.03, 0.75, 
            f"Erro Relativo L2: {erro_l2:.2e}\nErro Máximo L_inf: {erro_linf:.2e}", 
            transform=ax2.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Gráfico de validação de fidelidade OEC salvo em '{nome_arquivo}'!")


    @staticmethod
    def validacao_espectral_oec(
        t_eval: list, 
        residuos_relaxamento: list, 
        gap_fisico: float, 
        lambda_1: float,
        nome_arquivo: str = "../Figures/estabilidade_espectral_oec.png"
    ):
        """
        Renderiza a curva de relaxamento do OEC descartando o ponto final identicamente nulo.
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        t_arr = np.array(t_eval)
        res_arr = np.array(residuos_relaxamento)

        # Filtra pontos onde o resíduo é estritamente positivo (elimina o t_final onde res = 0)
        mask = res_arr > 1e-14
        t_filtrado = t_arr[mask]
        res_filtrado = res_arr[mask]

        # Plot da curva de relaxamento numerico OEC
        ax.semilogy(t_filtrado, res_filtrado, 'o-', color='indigo', lw=2.0, ms=6, 
                    label=r'Decaimento Numérico $\|p(t) - p_{\mathrm{est, local}}\|_2$')

        # Curva Guia Teorica: exp(-gap * t) baseada no primeiro transiente
        if len(t_filtrado) > 1:
            guia_teorica = res_filtrado[0] * np.exp(-gap_fisico * (t_filtrado - t_filtrado[0]))
            ax.semilogy(t_filtrado, guia_teorica, 'k--', lw=1.5, alpha=0.7, 
                        label=rf'Taxa Teórica $\mathcal{{O}}(e^{{-\gamma t}})$, $\gamma = {gap_fisico:.4f}$')

        ax.set_xlabel(r"Tempo $t$", fontsize=12)
        ax.set_ylabel(r"Resíduo de Relaxamento $\|p(t) - p_{\mathrm{est}}\|_2$", fontsize=12)
        ax.set_title(r"Princípio III: Estabilidade Espectral e Relaxamento Local", fontsize=12)
        ax.grid(True, which="both", linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True)

        ax.text(
            0.05, 0.15, 
            f"Modo Estacionário $\\lambda_0 = 0.00$\nModo Metaestável $\\lambda_1 = {lambda_1:.4e}$\nSpectral Gap $\\gamma = {gap_fisico:.4f}$", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

        plt.tight_layout()
        fig.savefig(nome_arquivo)
        plt.close(fig)
        print(f"Gráfico de estabilidade espectral OEC salvo em '{nome_arquivo}'!")