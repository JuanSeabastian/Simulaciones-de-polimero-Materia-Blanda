#%% ===========================================================
# RANDOM WALK 2D - POLÍMERO ESTADÍSTICO SELF-AVOIDED
# ===========================================================
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

#%% Parámetros generales 
L = 1000
N_mono_max = [10**3]     # Puedes agregar más valores
n_sim = 10**7             
l = 1                   

os.makedirs("datos_polimero_selfavoided", exist_ok=True)



#%% ===========================================================
# HISTOGRAMAS 2D - SELF AVOIDED
# ===========================================================
for N_mono in N_mono_max:
    print(f"📊 Graficando histogramas para Nmax = {N_mono}...")

    data = np.loadtxt(
        f"datos_polimero_selfavoided/resultados_2D_Nmax{N_mono}.txt",
        skiprows=1
    )

    Lreal = data[:,0]
    Rg2D = data[:,1]
    DeltaR2D = data[:,3]

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(f"Distribuciones Self-Avoided 2D (Nmax = {N_mono})", fontsize=16)

    axs[0].hist(Rg2D, bins=20, edgecolor='black', alpha=0.7)
    axs[0].set_title("$R_g$")

    axs[1].hist(DeltaR2D, bins=20, edgecolor='black', alpha=0.7)
    axs[1].set_title("$\\Delta R$")

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()



#%% ===========================================================
# PROMEDIO POR LONGITUD REAL
# ===========================================================

def promedio_por_longitud(Lreal, valores):
    L_unicos = np.unique(Lreal)
    valores_prom = np.zeros_like(L_unicos, dtype=float)

    for i, L in enumerate(L_unicos):
        valores_prom[i] = np.mean(valores[Lreal == L])

    return L_unicos, valores_prom



#%% ===========================================================
# ANALISIS DE ESCALAMIENTO: <R> vs N (REAL)
# ===========================================================

Ls_totales = []
promRg_tot = []
promDR_tot = []

for N_mono in N_mono_max:

    data = np.loadtxt(
        f"datos_polimero_selfavoided/resultados_2D_Nmax{N_mono}.txt",
        skiprows=1
    )

    Lreal = data[:,0]
    Rg2D = data[:,1]
    DeltaR2D = data[:,3]

    # Caso 1: todas las simulaciones tienen el mismo N real
    if len(np.unique(Lreal)) == 1:
        Ls_totales.append(Lreal[0])
        promRg_tot.append(np.mean(Rg2D))
        promDR_tot.append(np.mean(DeltaR2D))

    # Caso 2: múltiples valores de N_real → promediar
    else:
        Ls, Rg_prom = promedio_por_longitud(Lreal, Rg2D)
        _, DR_prom = promedio_por_longitud(Lreal, DeltaR2D)

        Ls_totales.extend(Ls)
        promRg_tot.extend(Rg_prom)
        promDR_tot.extend(DR_prom)


# Convertir a arrays
Ls_totales = np.array(Ls_totales)
promRg_tot = np.array(promRg_tot)
promDR_tot = np.array(promDR_tot)


#%% Ajuste a ley de potencia en log–log
def ley_potencia(N, a, nu):
    return a * N**nu

popt_Rg, _ = curve_fit(ley_potencia, Ls_totales, promRg_tot)
popt_DR, _ = curve_fit(ley_potencia, Ls_totales, promDR_tot)



#%% ===========================================================
# GRÁFICOS DE ESCALAMIENTO
# ===========================================================

plt.figure(figsize=(8,6))
plt.loglog(Ls_totales, promDR_tot, 'o', label=f"$\\Delta R$  (ν={popt_DR[1]:.3f})")
plt.loglog(Ls_totales, ley_potencia(Ls_totales, *popt_DR), '--')
plt.title("Escalamiento de ⟨$\\Delta R$⟩ con la longitud real N")
plt.xlabel("N real")
plt.ylabel("⟨$\\Delta R$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
plt.loglog(Ls_totales, promRg_tot, 'o', label=f"$R_g$  (ν={popt_Rg[1]:.3f})")
plt.loglog(Ls_totales, ley_potencia(Ls_totales, *popt_Rg), '--')
plt.title("Escalamiento de ⟨$R_g$⟩ con la longitud real N")
plt.xlabel("N real")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()


#%% Resultados
print("\n=== RESULTADOS ===")
print("DeltaR2D: Amplitud =", popt_DR[0], ", ν =", popt_DR[1])
print("Rg2D:     Amplitud =", popt_Rg[0], ", ν =", popt_Rg[1])
