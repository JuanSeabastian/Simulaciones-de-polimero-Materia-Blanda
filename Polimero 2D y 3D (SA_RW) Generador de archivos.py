#%%
import numpy as np
import os

#%% Parámetros generales 
L = 1000  # Tamaño del espacio (solo para graficar)
N_mono_max = [10**3]  # Máximo número de monómeros
n_sim = 10         # Número total de simulaciones por caso
l = 1                 # Longitud de cada monómero

# Crear carpeta para datos 
os.makedirs("datos_polimero_selfavoided", exist_ok=True)

#%%
def self_avoided_random_walk(Nmax, dim):
    """
    Genera un self-avoiding walk hasta un máximo de Nmax pasos.
    Si la caminata queda bloqueada antes, se corta.
    Retorna la trayectoria real obtenida.
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

        # Se quedó sin movimientos → cadena truncada
        if not posibles_pasos:
            trayectoria = trayectoria[:i]
            return trayectoria

        # Elegir un paso posible al azar
        paso = posibles_pasos[np.random.randint(len(posibles_pasos))]
        trayectoria[i] = trayectoria[i - 1] + paso
        visitadas.add(tuple(trayectoria[i]))

    # Si llega a Nmax sin bloqueo → trayectoria completa
    return trayectoria


# %% Propiedades del polímero
def propiedades(trayectoria):
    """
    Calcula Rg, R0 y DeltaR para una trayectoria dada.
    """
    r_cm = np.mean(trayectoria, axis=0)
    Rg2 = np.mean(np.sum((trayectoria - r_cm)**2, axis=1))
    Rg = np.sqrt(Rg2)
    deltaR = np.linalg.norm(trayectoria[-1] - trayectoria[0])
    R0 = deltaR**2
    return Rg, R0, deltaR


#%%
# ===========================================================
# BARRIDO ESTADÍSTICO 2D 
# ===========================================================
for Nmax in N_mono_max:
    print(f"🔹 Simulando polímeros con máximo N = {Nmax} monómeros...")

    Rg2D, R02D, DeltaR2D, Longitudes = [], [], [], []

    for _ in range(n_sim):
        trayectoria = self_avoided_random_walk(Nmax, 2)
        Rg, R0, deltaR = propiedades(trayectoria)

        Rg2D.append(Rg)
        R02D.append(R0)
        DeltaR2D.append(deltaR)
        Longitudes.append(len(trayectoria))  # largo real

    np.savetxt(f"datos_polimero_selfavoided/resultados_2D_Nmax{Nmax}.txt",
               np.column_stack((Longitudes, Rg2D, R02D, DeltaR2D)),
               header="LargoReal\tRg\tR0\tDeltaR", fmt="%.6f")

print("✅ Todos los datos guardados en 'datos_polimero_selfavoided/'")


# %%
