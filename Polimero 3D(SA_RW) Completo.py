#%% ===========================================================
# SELF-AVOIDED RANDOM WALK 3D – ARCHIVO ÚNICO
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

#%% ===========================================================
# CREAR CARPETA PARA DATOS
# ===========================================================

os.makedirs("datos_polimero_selfavoided", exist_ok=True)

#%% ===========================================================
# CAMINATA SELF-AVOIDED 3D
# ===========================================================

def self_avoided_random_walk(Nmax, dim=3):
    """
    Genera un self-avoiding walk de máximo Nmax pasos.
    Si se bloquea antes, devuelve la cadena truncada.
    """
    trayectoria = np.zeros((Nmax, dim), dtype=int)
    visitadas = {tuple(trayectoria[0])}

    for i in range(1, Nmax):
        posibles_pasos = []
        for eje in range(dim):
            for signo in [-1, 1]:
                paso = np.zeros(dim, dtype=int)
                paso[eje] = signo
                nueva_pos = tuple(trayectoria[i - 1] + paso)
                if nueva_pos not in visitadas:
                    posibles_pasos.append(paso)

        if not posibles_pasos:
            return trayectoria[:i]

        paso = posibles_pasos[np.random.randint(len(posibles_pasos))]
        trayectoria[i] = trayectoria[i - 1] + paso
        visitadas.add(tuple(trayectoria[i]))

    return trayectoria

#%% ===========================================================
# PROPIEDADES DEL POLÍMERO
# ===========================================================

def propiedades(trayectoria):
    r_cm = np.mean(trayectoria, axis=0)
    Rg2 = np.mean(np.sum((trayectoria - r_cm)**2, axis=1))
    Rg = np.sqrt(Rg2)
    deltaR = np.linalg.norm(trayectoria[-1] - trayectoria[0])
    R0 = deltaR**2
    return Rg, R0, deltaR

#%% ===========================================================
# PARÁMETROS GENERALES
# ===========================================================

N_mono_max = [10**3,10**6]   # máximo largo del intento de la cadena
n_sim = 100        # número total de simulaciones
l = 1

#%% ===========================================================
# SIMULACIONES 3D
# ===========================================================

for Nmax in N_mono_max:
    print(f"🔹 Simulando polímeros 3D con Nmax = {Nmax}...")

    Rg3D, R03D, DeltaR3D, Longitudes3D = [], [], [], []

    for _ in range(n_sim):
        tr = self_avoided_random_walk(Nmax, 3)
        Rg, R0, dR = propiedades(tr)

        Longitudes3D.append(len(tr))
        Rg3D.append(Rg)
        R03D.append(R0)
        DeltaR3D.append(dR)

    np.savetxt(f"datos_polimero_selfavoided/resultados_3D_Nmax{Nmax}.txt",
               np.column_stack((Longitudes3D, Rg3D, R03D, DeltaR3D)),
               header="LargoReal\tRg\tR0\tDeltaR", fmt="%.6f")

print("✅ Datos 3D guardados.")

#%% ===========================================================
# HISTOGRAMAS 3D
# ===========================================================

for Nmax in N_mono_max:
    data = np.loadtxt(f"datos_polimero_selfavoided/resultados_3D_Nmax{Nmax}.txt",
                      skiprows=1)

    Lreal = data[:,0]
    Rg3 = data[:,1]
    DeltaR3 = data[:,3]

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(f"Distribuciones Self-Avoided 3D (Nmax = {Nmax})", fontsize=16)

    axs[0].hist(Rg3, bins=20, edgecolor='black', alpha=0.7)
    axs[0].set_title("$R_g$ 3D")

    axs[1].hist(DeltaR3, bins=20, edgecolor='black', alpha=0.7)
    axs[1].set_title("$\\Delta R$ 3D")

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

#===========================================================
# ESCALAMIENTO 3D
# ===========================================================

Ls_tot, promRg_tot, promDR_tot = [], [], []

for Nmax in N_mono_max:

    data = np.loadtxt(f"datos_polimero_selfavoided/resultados_3D_Nmax{Nmax}.txt",
                      skiprows=1)

    Lreal = data[:,0]
    Rg3 = data[:,1]
    DR3 = data[:,3]

    if len(np.unique(Lreal)) == 1:
        Ls_tot.append(Lreal[0])
        promRg_tot.append(np.mean(Rg3))
        promDR_tot.append(np.mean(DR3))
    else:
        Ls, Rg_prom = promedio_por_longitud(Lreal, Rg3)
        _, DR_prom = promedio_por_longitud(Lreal, DR3)

        Ls_tot.extend(Ls)
        promRg_tot.extend(Rg_prom)
        promDR_tot.extend(DR_prom)

Ls_tot = np.array(Ls_tot)
promRg_tot = np.array(promRg_tot)
promDR_tot = np.array(promDR_tot)

#%% ===========================================================
# AJUSTE DE LEY DE POTENCIA
# ===========================================================

def ley_potencia(N, A, nu):
    return A * N**nu

popt_Rg, _ = curve_fit(ley_potencia, Ls_tot, promRg_tot)
popt_DR, _ = curve_fit(ley_potencia, Ls_tot, promDR_tot)

#%% ===========================================================
# GRAFICOS DE ESCALAMIENTO
# ===========================================================

plt.figure(figsize=(8,6))
plt.loglog(Ls_tot, promDR_tot, 'o', label=f"$\\Delta R$ 3D (ν={popt_DR[1]:.3f})")
plt.loglog(Ls_tot, ley_potencia(Ls_tot, *popt_DR), '--')
plt.title("Escalamiento de ⟨$\\Delta R$⟩ en 3D")
plt.xlabel("N real")
plt.ylabel("⟨$\\Delta R$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.loglog(Ls_tot, promRg_tot, 'o', label=f"$R_g$ 3D (ν={popt_Rg[1]:.3f})")
plt.loglog(Ls_tot, ley_potencia(Ls_tot, *popt_Rg), '--')
plt.title("Escalamiento de ⟨$R_g$⟩ en 3D")
plt.xlabel("N real")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()

#%% ===========================================================
# RESULTADOS
# ===========================================================

print("\n=== RESULTADOS 3D ===")
print("DeltaR3D:  ν =", popt_DR[1])
print("Rg3D:   ν =", popt_Rg[1])
print("==========================================")

# %%
