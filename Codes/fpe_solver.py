import numpy as np
from scipy.integrate import solve_ivp

class FPEChangCooper:
    """
    Solver para a Equaçao de Fokker-Planck usando a discretizacao espacial
    conservativa de Chang-Cooper e integracao temporal adaptativa (BDF).
    """
    def __init__(self, xmin=-4.0, xmax=4.0, N=200, D=1.0, kB_T=1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.N = N
        self.D = D
        self.kB_T = kB_T

        self.x = np.linspace(self.xmin, self.xmax, self.N + 1)
        self.dx = (self.xmax - self.xmin) / self.N
        self.x_int = self.x[1:-1]
        
        self.A = None
        self.a_coef = None
        self.b_coef = None
        self.c_coef = None
        self.sol = None


    @staticmethod
    def G(x_val):
        """
        Potencial Quartico de Poco Duplo.
        """
        return (x_val**4) / 4.0 - (x_val**2) / 2.0


    @staticmethod
    def G_prime(x_val):
        """
        Derivada do Potencial (Forca).
        """
        return x_val**3 - x_val


    @staticmethod
    def bernoulli(z):
        """
        Funcao de Bernoulli estabilizada para o esquema de Chang-Cooper.
        """
        z = np.asarray(z)
        small_z = np.abs(z) < 1e-6
        res = np.zeros_like(z)
        res[small_z] = 1.0 - z[small_z] / 2.0 + (z[small_z]**2) / 12.0
        res[~small_z] = z[~small_z] / (np.exp(z[~small_z]) - 1.0)
        return res


    def construir_matriz_operador(self):
        """
        Monta a matriz geradora A de Chang-Cooper.
        """
        G_nodes = self.G(self.x)
        delta_G = G_nodes[1:] - G_nodes[:-1]
        z = delta_G / self.kB_T

        W_plus = (self.D / self.dx) * self.bernoulli(-z)   # Fluxo i -> i+1
        W_minus = (self.D / self.dx) * self.bernoulli(z)    # Fluxo i+1 -> i

        # Mapeamento exato da equação semidiscreta
        self.a_coef = W_plus[:-1] / self.dx
        self.c_coef = W_minus[1:] / self.dx
        self.b_coef = -(W_minus[:-1] + W_plus[1:]) / self.dx

        N_pontos = len(self.x)
        self.A = np.zeros((N_pontos, N_pontos))

        for i in range(N_pontos - 1):
            flux_right = W_plus[i] / self.dx
            flux_left = W_minus[i] / self.dx
            
            self.A[i, i]         -= flux_right
            self.A[i + 1, i]     += flux_right
            self.A[i + 1, i + 1] -= flux_left
            self.A[i, i + 1]     += flux_left

        return self.A


    def condicao_inicial_gaussiana(self, mu=-1.0, sigma=0.25):
        """
        Gera e normaliza a distribuicao inicial Gaussiana.
        """
        p0 = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((self.x - mu)**2) / (2.0 * sigma**2))
        p0 = p0 / (np.sum(p0) * self.dx)
        return p0


    def resolver(self, t_eval, p0=None, method='BDF', rtol=1e-8, atol=1e-10):
        """
        Realiza a integracao temporal do sistema de EDOs.
        """
        if self.A is None:
            self.construir_matriz_operador()

        if p0 is None:
            p0 = self.condicao_inicial_gaussiana()

        def rhs(t, p_vec):
            return self.A @ p_vec

        t_span = (0.0, max(t_eval))
        self.sol = solve_ivp(
            rhs, 
            t_span, 
            p0, 
            t_eval=t_eval, 
            method=method,   
            rtol=rtol, 
            atol=atol
        )
        return self.sol


class FPEBenchmark:
    """
    Solver para o Problema 2 (Benchmark de Neena et al.) com BDF1, BDF2 e BDF3.
    """
    @staticmethod
    def resolver(M=10, N=100, T_final=1.0, ordem_bdf=2):
        x_grid = np.linspace(0, 1, M + 1)
        h = 1.0 / M
        t_grid, k = np.linspace(0, T_final, N + 1, retstep=True)
        
        x_internos = x_grid[1:-1]
        num_internos = len(x_internos)
        
        p_coef = x_internos**2 - h * x_internos
        q_coef = -2.0 * x_internos**2
        r_coef = x_internos**2 + h * x_internos
        
        A_mat = np.diag(q_coef) + np.diag(p_coef[1:], k=-1) + np.diag(r_coef[:-1], k=1)
        escala = 1.0 / (2.0 * h**2)
        I = np.eye(num_internos)
        
        W = np.zeros((N + 1, num_internos))
        W[0, :] = x_internos  # w(x,0) = x
        
        for n in range(N):
            t_prox = t_grid[n + 1]
            B_prox = np.zeros(num_internos)
            B_prox[-1] = escala * r_coef[-1] * np.exp(t_prox)
            
            if ordem_bdf == 1 or n == 0:
                Matriz_Sistema = I - k * escala * A_mat
                Lado_Direito = W[n, :] + k * B_prox
            elif ordem_bdf == 2 or (ordem_bdf == 3 and n == 1):
                Matriz_Sistema = I - (2.0 / 3.0) * k * escala * A_mat
                Lado_Direito = (4.0 / 3.0) * W[n, :] - (1.0 / 3.0) * W[n - 1, :] + (2.0 / 3.0) * k * B_prox
            elif ordem_bdf == 3 and n >= 2:
                Matriz_Sistema = I - (6.0 / 11.0) * k * escala * A_mat
                Lado_Direito = (18.0 / 11.0) * W[n, :] - (9.0 / 11.0) * W[n - 1, :] + (2.0 / 11.0) * W[n - 2, :] + (6.0 / 11.0) * k * B_prox
            else:
                raise ValueError("ordem_bdf deve ser 1, 2 ou 3.")

            W[n + 1, :] = np.linalg.solve(Matriz_Sistema, Lado_Direito)
                
        W_completo = np.zeros((N + 1, M + 1))
        for idx_t in range(N + 1):
            W_completo[idx_t, 0] = 0.0                     # Dirichlet x=0
            W_completo[idx_t, 1:-1] = W[idx_t, :]          
            W_completo[idx_t, -1] = np.exp(t_grid[idx_t])  # Dirichlet x=1
            
        return x_grid, t_grid, W_completo