import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ===========================================================
# 1. Parametros fisicos e discretizacao espacial
# ===========================================================
D = 1.0         # Constante de difusao
kB_T = 1.0      # Energia termica
xmin, xmax = -4.0, 4.0  # Dominio expandido para zerar densidade nas bordas
N = 200

x = np.linspace(xmin, xmax, N + 1)
dx = (xmax - xmin) / N

x_int = x[1:-1]
N_int = len(x_int)
N_pontos = len(x)


# ===========================================================
# 2. Potencial de Poco Duplo
# ===========================================================
def G(x_val):
    return (x_val**4) / 4.0 - (x_val**2) / 2.0

def G_prime(x_val):
    return x_val**3 - x_val


# ===========================================================
# 3. Discretizacao de Chang-Cooper (Conservacao Exata)
# ===========================================================
def bernoulli(z):
    z = np.asarray(z)
    small_z = np.abs(z) < 1e-6
    res = np.zeros_like(z)
    res[small_z] = 1.0 - z[small_z] / 2.0 + (z[small_z]**2) / 12.0
    res[~small_z] = z[~small_z] / (np.exp(z[~small_z]) - 1.0)
    return res

G_nodes = G(x)
delta_G = G_nodes[1:] - G_nodes[:-1]
z = delta_G / kB_T

W_plus = (D / dx) * bernoulli(-z)   # Fluxo i -> i+1
W_minus = (D / dx) * bernoulli(z)    # Fluxo i+1 -> i

a = W_plus[:-1] / dx
c = W_minus[1:] / dx
b = -(W_minus[:-1] + W_plus[1:]) / dx

A = np.zeros((N_pontos, N_pontos))

for i in range(N_pontos - 1):
    flux_right = W_plus[i] / dx
    flux_left = W_minus[i] / dx
    
    A[i, i]         -= flux_right
    A[i + 1, i]     += flux_right
    
    A[i + 1, i + 1] -= flux_left
    A[i, i + 1]     += flux_left


# ===========================================================
# 4. Condicao Inicial e Integracao Temporal
# ===========================================================
mu, sigma = -1.0, 0.25
p0 = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((x - mu)**2) / (2.0 * sigma**2))

# Normalizacao consistente com a soma de volumes da malha: sum(p0)*dx = 1.0
p0 = p0 / (np.sum(p0) * dx)

def rhs(t, p_vec):
    return A @ p_vec

t_eval = [0.0, 0.1, 0.5, 2.0, 5.0, 15.0]
t_span = (0.0, max(t_eval))

sol = solve_ivp(
    rhs, 
    t_span, 
    p0, 
    t_eval=t_eval, 
    method='BDF',   
    rtol=1e-8, 
    atol=1e-10
)


# ===========================================================
# 5. Funcoes de Graficos e Diagnosticos
# ===========================================================
def criar_grafico_potencial(x_eixo, curvas_eixo_principal=None, curvas_eixo_secundario=None, 
                             titulo="Validação do Potencial", xlabel=r"Posição $x$"):

    """
        Gera um grafico flexivel permitindo plotar multiplas curvas no eixo principal 
        e/ou no eixo secundario (twinx).

        Parâmetros:
        -----------
        x : array-like
            Eixo horizontal (independente).
        curvas_eixo_principal : list of dict, optional
            Lista de dicionarios configurando as curvas do eixo esquerdo (ax1).
        curvas_eixo_secundario : list of dict, optional
            Lista de dicionários configurando as curvas do eixo direito (ax2).
        titulo : str
            Título do grafico.
        xlabel : str
            Rotulo do eixo X.
        """

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


def visualizar_diagnostico_discretizacao(x_eixo, a_coef, b_coef, c_coef, A_mat, nome_arquivo="../Figures/diagnostico_sistema.png"):
    """
    Gera um painel com 3 graficos limpos para diagnosticar a discretizacao:
    1. Assimetria dos coeficientes de transporte (a_i e c_i)
    2. Perfil da diagonal principal (b_i)
    3. Mapa de calor focado na matriz A
    """
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

    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

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


def visualizar_evolucao_temporal(x_eixo, sol_ivp, t_eval_list, G_func=None, nome_arquivo="../Figures/evolucao_temporal_p.png"):
    """
    Gera o grafico da evolucao temporal da densidade p(x,t) sobreposta ao potencial G(x).
    
    Parametros:
    -----------
    x_int : array-like
        Pontos da malha interna.
    sol : OdeResult (scipy)
        Objeto retornado pelo solve_ivp.
    t_eval : list or array
        Tempos em que a solucao foi avaliada.
    G_func : callable, optional
        Funcao do potencial G(x) para plotar em marca d'agua no eixo secundario.
    nome_arquivo : str
        Caminho para salvar a figura.
    """

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


def visualizar_conservacao_massa(x_eixo, sol_ivp, t_eval_list, nome_arquivo="../Figures/conservacao_massa.png"):
    """
    Gera o grafico do erro de conservacao da massa integrando estritamente
    sobre x_int, pela matriz de Chang-Cooper.
    """
    # Massa discreta exata conservada pela matriz de Chang-Cooper
    massas = [np.sum(sol_ivp.y[:, i]) * dx for i in range(len(t_eval_list))]
    erros_massa = np.abs(np.array(massas) - 1.0)
    erro_max = np.max(erros_massa)

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


def estudo_convergencia_problema2(ordem_bdf=2, nome_arquivo="../Figures/estudo_convergencia_p2.png"):
    """
    Realiza o estudo numerico de convergencia (L2 e L_infinity) 
    e gera graficos log-log para validacao estilo Neena et al.
    """
    print(f"\n--- Gerando Estudo de Convergencia Grafica (BDF{ordem_bdf}) ---")
    
    # Conjunto de refinamentos de malha para análise
    lista_M = [10, 20, 40, 80, 160]
    lista_N = [100, 200, 400, 800, 1600]
    
    erros_l2 = []
    erros_linf = []
    h_valores = []  # Passo espacial h = 1/M

    for M, N in zip(lista_M, lista_N):
        x_fpe, t_fpe, W_num = resolver_fpe_problema2(M=M, N=N, T_final=1.0, ordem_bdf=ordem_bdf)
        X_grid, T_grid = np.meshgrid(x_fpe, t_fpe)
        W_exato = X_grid * np.exp(T_grid)
        
        e_l2, e_linf = calcular_erros_normas(W_num, W_exato)
        
        erros_l2.append(e_l2)
        erros_linf.append(e_linf)
        h_valores.append(1.0 / M)
        
        print(f"M={M:3d}, N={N:4d} | Erro L2: {e_l2:.4e} | Erro L_inf: {e_linf:.4e}")

    # --- Construcao da Figura com Solucao Grafica ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # 1. Grafico Log-Log de Convergencia das Normas
    ax1.loglog(h_valores, erros_l2, 'o-', color='navy', lw=1.8, ms=6, label=r'Erro $L_2$ Relativo')
    ax1.loglog(h_valores, erros_linf, 's--', color='crimson', lw=1.8, ms=6, label=r'Erro $L_{\infty}$')
    
    # Linha guia teorica de ordem 2 (O(h^2)) para comparacao
    guia_o2 = [erros_l2[0] * (h / h_valores[0])**2 for h in h_valores]
    ax1.loglog(h_valores, guia_o2, 'k:', alpha=0.5, label=r'Inclinação Teórica $\mathcal{O}(h^2)$')

    ax1.set_xlabel(r'Tamanho do Elemento Espacial $h = 1/M$', fontsize=15)
    ax1.set_ylabel('Erro Absoluto / Relativo', fontsize=15)
    ax1.set_title(f'Taxa de Convergência Numérica (BDF{ordem_bdf})', fontsize=11)
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)

    # 2. Perfil Espacial do Erro Pontual no Instante Final (M=40, N=400)
    x_ref, t_ref, W_ref = resolver_fpe_problema2(M=40, N=400, T_final=1.0, ordem_bdf=ordem_bdf)
    W_exato_ref = x_ref * np.exp(1.0)  # t = 1.0
    erro_espacial = np.abs(W_ref[-1, :] - W_exato_ref)

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


# ===========================================================
# 6. Problema 2 (FPE)
# ===========================================================
def resolver_fpe_problema2(M=10, N=100, T_final=1.0, ordem_bdf=2):
    x_grid = np.linspace(0, 1, M + 1)
    h = 1.0 / M
    t_grid, k = np.linspace(0, T_final, N + 1, retstep=True)
    
    x_internos = x_grid[1:-1]
    num_internos = len(x_internos)
    
    # Coeficientes do operador espacial
    p_coef = x_internos**2 - h * x_internos
    q_coef = -2.0 * x_internos**2
    r_coef = x_internos**2 + h * x_internos
    
    A_mat = np.diag(q_coef) + np.diag(p_coef[1:], k=-1) + np.diag(r_coef[:-1], k=1)
    escala = 1.0 / (2.0 * h**2)
    I = np.eye(num_internos)
    
    W = np.zeros((N + 1, num_internos))
    W[0, :] = x_internos  # Condicao inicial: w(x,0) = x
    
    for n in range(N):
        t_prox = t_grid[n + 1]
        B_prox = np.zeros(num_internos)
        B_prox[-1] = escala * r_coef[-1] * np.exp(t_prox)
        
        # Selecao dinamica do metodo e bootstrap de ordem
        # BDF1 (Usado se solicitado, ou no primeiro passo n=0 de ordens superiores)
        if ordem_bdf == 1 or n == 0:
            Matriz_Sistema = I - k * escala * A_mat
            Lado_Direito = W[n, :] + k * B_prox
            
        # BDF2 (Usado se solicitado para n>=1, ou no segundo passo n=1 de BDF3)
        elif ordem_bdf == 2 or (ordem_bdf == 3 and n == 1):
            Matriz_Sistema = I - (2.0 / 3.0) * k * escala * A_mat
            Lado_Direito = (4.0 / 3.0) * W[n, :] - (1.0 / 3.0) * W[n - 1, :] + (2.0 / 3.0) * k * B_prox
            
        # BDF3 (Usado para n>=2 quando ordem_bdf=3)
        elif ordem_bdf == 3 and n >= 2:
            Matriz_Sistema = I - (6.0 / 11.0) * k * escala * A_mat
            Lado_Direito = (18.0 / 11.0) * W[n, :] - (9.0 / 11.0) * W[n - 1, :] + (2.0 / 11.0) * W[n - 2, :] + (6.0 / 11.0) * k * B_prox
            
        else:
            raise ValueError("ordem_bdf deve ser 1, 2 ou 3.")

        W[n + 1, :] = np.linalg.solve(Matriz_Sistema, Lado_Direito)
            
    # Reconstrucao da malha completa incluindo fronteiras
    W_completo = np.zeros((N + 1, M + 1))
    for idx_t in range(N + 1):
        W_completo[idx_t, 0] = 0.0                     # Condicao Dirichlet x=0
        W_completo[idx_t, 1:-1] = W[idx_t, :]          
        W_completo[idx_t, -1] = np.exp(t_grid[idx_t])  # Condicao Dirichlet x=1
        
    return x_grid, t_grid, W_completo


# ===========================================================
# 7. Wrappers de Execucao
# ===========================================================
def resolve_potencial(plot_G=True, plot_G_prime=True, nome_arquivo="../Figures/grafico_potencial.png"):
    """
        Prepara os dados e gera o grafico permitindo ativar/desativar 
        o potencial G(x) e sua derivada G'(x).
        
        Parâmetros:
        -----------
        plot_G : bool (default: True)
            Se True, plota o potencial G(x) no eixo secundário.
        plot_G_prime : bool (default: True)
            Se True, plota a derivada G'(x) no eixo principal.
        nome_arquivo : str
            Caminho e nome do arquivo de saída.
    """
    curvas_principais, curvas_secundarias = [], []
    if plot_G_prime:
        curvas_principais.append({'y': G_prime(x), 'color': 'red', 'label': r'Derivada $\frac{dG}{dx} = x^3 - x$'})
    if plot_G:
        curvas_secundarias.append({'y': G(x), 'color': 'k', 'linestyle': '--', 'alpha': 0.3, 'label': r'Potencial $G(x) = \frac{x^4}{4} - \frac{x^2}{2}$'})

    fig, _ = criar_grafico_potencial(x, curvas_principais, curvas_secundarias, "Potencial Quártico de Poço Duplo e sua Derivada")
    fig.savefig(nome_arquivo)
    plt.close(fig)
    print(f"Grafico potencial gerado e salvo em '{nome_arquivo}'!")

def resolve_evolucao_temporal(sol_ivp, t_eval_list, plot_G=True, nome_arquivo="../Figures/evolucao_temporal_p.png"):
    G_func = G if plot_G else None
    visualizar_evolucao_temporal(x, sol_ivp, t_eval_list, G_func=G_func, nome_arquivo=nome_arquivo)

def resolve_conservacao_massa(sol_ivp, t_eval_list, nome_arquivo="../Figures/conservacao_massa.png"):
    visualizar_conservacao_massa(x, sol_ivp, t_eval_list, nome_arquivo=nome_arquivo)

def validar_problema2(M=10, N=100, T_final=1.0, ordem_bdf=2, verbose=True):
    """
    Executar e calcular a validacao do Problema 2 (FPE) 
    frente a solucao analitica exata w(x,t) = x * exp(t).
    """
    if verbose:
        print(f"\n--- Validacao do Problema 2 (FPE) - BDF{ordem_bdf} ---")
    
    x_fpe, t_fpe, W_num = resolver_fpe_problema2(M=M, N=N, T_final=T_final, ordem_bdf=ordem_bdf)

    X_grid, T_grid = np.meshgrid(x_fpe, t_fpe)
    W_exato = X_grid * np.exp(T_grid)

    erro_rms = np.sqrt(np.sum((W_num[-1, :] - W_exato[-1, :])**2)) / np.sqrt(np.sum(W_exato[-1, :]**2))
    
    if verbose:
        print(f"Erro RMS no tempo final (BDF{ordem_bdf}): {erro_rms:.6e}")

    return erro_rms, x_fpe, t_fpe, W_num, W_exato


def calcular_erros_normas(w_num, w_exato):
    """
    Calcula as normas de erro L2 relativo e L_infinity no instante final.
    """
    erro_pontual = np.abs(w_num[-1, :] - w_exato[-1, :])
    
    # Erro L_infinity (Norma do Maximo)
    erro_linf = np.max(erro_pontual)
    
    # Erro L2 Relativo
    erro_l2 = np.sqrt(np.sum(erro_pontual**2)) / np.sqrt(np.sum(w_exato[-1, :]**2))
    
    return erro_l2, erro_linf



# ===========================================================
# 8. Bloco Principal
# ===========================================================
if __name__ == "__main__":
    print("===========================================================")
    print(" 1. SIMULACAO FPE: POTENCIAL DE POCO DUPLO (CHANG-COOPER) ")
    print("===========================================================")
    print(f"Malha Total (N+1): {len(x)} pontos")
    print(f"Passo Espacial dx: {dx:.4f}")

    # Executa diagnosticos e simulacao do Poco Duplo
    resolve_potencial(plot_G=True, plot_G_prime=True)
    visualizar_diagnostico_discretizacao(x_int, a, b, c, A)
    resolve_evolucao_temporal(sol, t_eval, plot_G=True)
    resolve_conservacao_massa(sol, t_eval)

    print("\n===========================================================")
    print(" 2. VALIDACAO DO PROBLEMA 2 (FPE) E ANALISE DE ERRO       ")
    print("===========================================================")

    # 2.1 Validacao pontual com print direto
    validar_problema2(M=10, N=100, T_final=1.0, ordem_bdf=2)

    # 2.2 Estudo completo de convergencia grafica (Normas L2, L_inf e taxa O(h^2))
    estudo_convergencia_problema2(ordem_bdf=2)
    
    print("\nSimulacoes e diagnosticos concluidos com sucesso!")