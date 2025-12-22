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

#%%
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
# HISTOGRAMAS COMPARATIVOS PARA CADA N
# ===========================================================
for N_mono in N_mono_list:
    print(f"📊 Graficando histogramas para N = {N_mono}...")

    # --- Lectura ---
    data2D = np.loadtxt(f"datos_polimero/resultados_2D_N{N_mono}.txt", skiprows=1)
    data3D = np.loadtxt(f"datos_polimero/resultados_3D_N{N_mono}.txt", skiprows=1)

    Rg2D, DeltaR2D = data2D[:,0],data2D[:,2]
    Rg3D,  DeltaR3D = data3D[:,0], data3D[:,2]

    # --- Gráficos ---
    fig, axs = plt.subplots(2, 2, figsize=(15,8))
    fig.suptitle(f"Distribuciones para N = {N_mono}", fontsize=16)

    # 2D
    axs[0,0].hist(Rg2D, bins=15, color='royalblue', edgecolor='black', alpha=0.7)
    axs[0,0].set_title("2D - $R_g$")
    axs[0,1].hist(DeltaR2D, bins=15, color='darkorange', edgecolor='black', alpha=0.7)
    axs[0,1].set_title("2D - $\\Delta R$")

    # 3D
    axs[1,0].hist(Rg3D, bins=15, color='royalblue', edgecolor='black', alpha=0.7)
    axs[1,0].set_title("3D - $R_g$")
    axs[1,1].hist(DeltaR3D, bins=15, color='darkorange', edgecolor='black', alpha=0.7)
    axs[1,1].set_title("3D - $\\Delta R$")

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

#%%
# ===========================================================
# ANÁLISIS DE ESCALAMIENTO: <R> vs N
# ===========================================================
def ley_potencia(N, a, nu):
    return a * N**nu

promRg2D, promRg3D = [], []
promDeltaR2D, promDeltaR3D = [], []

for N_mono in N_mono_list:
    data2D = np.loadtxt(f"datos_polimero/resultados_2D_N{N_mono}.txt", skiprows=1)
    data3D = np.loadtxt(f"datos_polimero/resultados_3D_N{N_mono}.txt", skiprows=1)
    promRg2D.append(np.mean(data2D[:,0]))
    promRg3D.append(np.mean(data3D[:,0]))
    promDeltaR2D.append(np.mean(data2D[:,2]))
    promDeltaR3D.append(np.mean(data3D[:,2]))

#%% Ajuste ley de potencias para deltaR

# Ajuste ley de potencias
popt_DeltaR2D, _ = curve_fit(ley_potencia, N_mono_list, promDeltaR2D)
popt_DeltaR3D, _ = curve_fit(ley_potencia, N_mono_list, promDeltaR3D)

# ===========================================================
# GRÁFICOS DE ESCALAMIENTO
# ===========================================================
plt.figure(figsize=(8,6))
plt.loglog(N_mono_list, promDeltaR2D, 'o', color= "red",label=f"2D  (ν={popt_DeltaR2D[1]:.3f})")
plt.loglog(N_mono_list, ley_potencia(np.array(N_mono_list), *popt_DeltaR2D), color='blue', alpha=0.5)

plt.title("Escalamiento de ⟨$R_g$⟩ con el número de monómeros")
plt.xlabel("N")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()




plt.figure(figsize=(8,6))
plt.loglog(N_mono_list, promDeltaR3D, 's',  color= "red",label=f"3D  (ν={popt_DeltaR3D[1]:.3f})")
plt.loglog(N_mono_list, ley_potencia(np.array(N_mono_list), *popt_DeltaR3D),  color='blue', alpha=0.5)
plt.title("Escalamiento de ⟨$R_g$⟩ con el número de monómeros")
plt.xlabel("N")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()


#%%
# Ajuste ley de potencias para Rg
popt_Rg2D, _ = curve_fit(ley_potencia, N_mono_list, promRg2D)
popt_Rg3D, _ = curve_fit(ley_potencia, N_mono_list, promRg3D)

# ===========================================================
# GRÁFICOS DE ESCALAMIENTO
# ===========================================================
plt.figure(figsize=(8,6))
plt.loglog(N_mono_list, promRg2D, 'o', color= "red",label=f"2D  (ν={popt_Rg2D[1]:.3f})")
plt.loglog(N_mono_list, ley_potencia(np.array(N_mono_list), *popt_Rg2D), color='blue', alpha=0.5)

plt.title("Escalamiento de ⟨$R_g$⟩ con el número de monómeros")
plt.xlabel("N")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()




plt.figure(figsize=(8,6))
plt.loglog(N_mono_list, promRg3D, 's',  color= "red",label=f"3D  (ν={popt_Rg3D[1]:.3f})")
plt.loglog(N_mono_list, ley_potencia(np.array(N_mono_list), *popt_Rg3D),  color='blue', alpha=0.5)
plt.title("Escalamiento de ⟨$R_g$⟩ con el número de monómeros")
plt.xlabel("N")
plt.ylabel("⟨$R_g$⟩")
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend()
plt.show()
#%% Resultados


print("DeltaR2D: Amplitud ",popt_DeltaR2D[0],"potencia",popt_DeltaR2D[1])
print("DeltaR2D: Amplitud ",popt_DeltaR3D[0],"potencia",popt_DeltaR3D[1])
print("Rg2D: Amplitud ",popt_Rg2D[0],"potencia",popt_Rg2D[1])
print("Rg3D: Amplitud ",popt_Rg3D[0],"potencia",popt_Rg3D[1])



#%%
# ===========================================================
# VISUALIZACIÓN DE UN EJEMPLO 2D Y 3D
# ===========================================================
N_demo = 1000
tray2D = random_walk(N_demo, 2)
tray3D = random_walk(N_demo, 3)

# 2D
plt.figure(figsize=(6,6))
plt.plot(tray2D[:,0], tray2D[:,1], '-o', color='navy', linewidth=2)
plt.scatter(tray2D[0,0], tray2D[0,1], color='green', s=100, label='Inicio')
plt.scatter(tray2D[-1,0], tray2D[-1,1], color='red', s=100, label='Fin')
plt.title("Ejemplo de polímero 2D (N=1000)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(tray3D[:,0], tray3D[:,1], tray3D[:,2], '-o', color='navy', linewidth=2)
ax.scatter(tray3D[0,0], tray3D[0,1], tray3D[0,2], color='green', s=100, label='Inicio')
ax.scatter(tray3D[-1,0], tray3D[-1,1], tray3D[-1,2], color='red', s=100, label='Fin')
ax.set_title("Ejemplo de polímero 3D (N=1000)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()
# %%
