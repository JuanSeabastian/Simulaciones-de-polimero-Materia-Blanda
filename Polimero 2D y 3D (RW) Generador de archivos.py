#%%
# ===========================================================
# RANDOM WALK 2D Y 3D - POLÍMERO ESTADÍSTICO CON BARRIDO EN N
# ===========================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import os

# %% Parámetros generales
L = 1000                     # Tamaño del espacio (solo para graficar)
N_mono_list = [100, 1000, 10000]  # Números de monómeros a barrer
n_sim = 500                  # Número de simulaciones por caso
l = 1                        # Longitud de cada monómero

# Crear carpeta para datos
os.makedirs("datos_polimero", exist_ok=True)

# %% Función general de random walk
def random_walk(N, dim):
    trayectoria = np.zeros((N, dim), dtype=int)
    for i in range(1, N):
        paso = np.zeros(dim, dtype=int)
        eje = int(np.random.random() * dim)
        paso[eje] = np.random.choice([-1, 1])
        trayectoria[i] = trayectoria[i - 1] + paso
    return trayectoria

# %% Cálculo de propiedades del polímero
def propiedades(trayectoria):
    r_cm = np.mean(trayectoria, axis=0)
    Rg2 = np.mean(np.sum((trayectoria - r_cm)**2, axis=1))
    Rg = np.sqrt(Rg2)
    deltaR = np.linalg.norm(trayectoria[-1] - trayectoria[0])
    R0 = deltaR**2
    return Rg, R0, deltaR

#%%
# ===========================================================
# BARRIDO ESTADÍSTICO 2D Y 3D
# ===========================================================
for N_mono in N_mono_list:
    print(f"🔹 Simulando polímeros con N = {N_mono} monómeros...")

    # --- 2D ---
    Rg2D, R02D, DeltaR2D = [], [], []
    for _ in range(n_sim):
        trayectoria = random_walk(N_mono, 2)
        Rg, R0, deltaR = propiedades(trayectoria)
        Rg2D.append(Rg)
        R02D.append(R0)
        DeltaR2D.append(deltaR)

    np.savetxt(f"datos_polimero/resultados_2D_N{N_mono}.txt",
               np.column_stack((Rg2D, R02D, DeltaR2D)),
               header="Rg\tR0\tDeltaR", fmt="%.6f")

    # --- 3D ---
    Rg3D, R03D, DeltaR3D = [], [], []
    for _ in range(n_sim):
        trayectoria = random_walk(N_mono, 3)
        Rg, R0, deltaR = propiedades(trayectoria)
        Rg3D.append(Rg)
        R03D.append(R0)
        DeltaR3D.append(deltaR)

    np.savetxt(f"datos_polimero/resultados_3D_N{N_mono}.txt",
               np.column_stack((Rg3D, R03D, DeltaR3D)),
               header="Rg\tR0\tDeltaR", fmt="%.6f")

print("✅ Todos los datos guardados en 'datos_polimero/'")
