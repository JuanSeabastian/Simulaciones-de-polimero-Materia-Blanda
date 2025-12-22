#%%
import mathplot.lib as plt
import numpy as np
import os

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
Nmax = 1000        # Largo máximo del polímero
n_sim = 50         
n_polimeros = 1000   # Número de polímeros en el brush

# Lista de separaciones (Enteros >= 1 para modelo de retícula)
separaciones = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 50, 100]

# Crear carpeta si no existe
os.makedirs("datos_brush", exist_ok=True)

# ------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------
def propiedades(tray):
    """Calcula Rg y DeltaR (distancia extremo-extremo)"""
    if len(tray) < 10: return None, None # Ignorar cadenas muy cortas/truncadas
    
    r_cm = np.mean(tray, axis=0)
    Rg2 = np.mean(np.sum((tray - r_cm)**2, axis=1))
    Rg = np.sqrt(Rg2)
    deltaR = np.linalg.norm(tray[-1] - tray[0])
    return Rg, deltaR

def self_avoided_brush(Nmax, separation, n_pol):
    starts = np.array([[int(i * separation), 0] for i in range(n_pol)], dtype=int)
    trayectorias = [np.zeros((Nmax, 2), dtype=int) for _ in range(n_pol)]
    
    for i in range(n_pol):
        trayectorias[i][0] = starts[i]

    visitadas = {tuple(starts[i]) for i in range(n_pol)}
    pasos = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]) # Arriba, Abajo, Der, Izq
    
    activos = [True] * n_pol
    longitudes = [1] * n_pol

    for step in range(1, Nmax):
        for p in range(n_pol):
            if not activos[p]: continue

            pos_actual = trayectorias[p][step-1]
            candidatos = []

            for dv in pasos:
                nueva = pos_actual + dv
                if nueva[1] < 0: continue # Suelo duro
                if tuple(nueva) not in visitadas:
                    candidatos.append(nueva)

            if not candidatos:
                trayectorias[p] = trayectorias[p][:step]
                activos[p] = False
                longitudes[p] = step
                continue

            nueva = candidatos[np.random.randint(len(candidatos))]
            trayectorias[p][step] = nueva
            visitadas.add(tuple(nueva))
            longitudes[p] = step + 1

    for p in range(n_pol):
        trayectorias[p] = trayectorias[p][:longitudes[p]]

    return trayectorias

# ------------------------------------------------------------
# BARRIDO Y GUARDADO
# ------------------------------------------------------------
resultados = [] # Aquí acumularemos [separacion, mean_Rg, std_Rg, mean_DR, std_DR]

print(f"Iniciando simulación para {len(separaciones)} distancias...")

for sep in separaciones:
    rg_acum = []
    dr_acum = []
    
    for sim in range(n_sim):
        trays = self_avoided_brush(Nmax, separation=sep, n_pol=n_polimeros)
        
        # Medir propiedades de CADA polímero en la simulación
        for p in range(n_polimeros):
            rg, dr = propiedades(trays[p])
            if rg is not None:
                rg_acum.append(rg)
                dr_acum.append(dr)

    # Calcular estadísticas para esta separación
    if len(rg_acum) > 0:
        mean_rg = np.mean(rg_acum)
        std_rg = np.std(rg_acum)
        mean_dr = np.mean(dr_acum)
        std_dr = np.std(dr_acum)
    else:
        mean_rg, std_rg, mean_dr, std_dr = 0, 0, 0, 0

    # Guardar en la lista
    resultados.append([sep, mean_rg, std_rg, mean_dr, std_dr])
    print(f" -> Sep {sep}: <Rg> = {mean_rg:.2f}")

# Convertir a numpy array y guardar
resultados = np.array(resultados)
header = "Separacion\tRg_mean\tRg_std\tDeltaR_mean\tDeltaR_std"
np.savetxt("datos_brush/resultados_transicion.txt", resultados, header=header, fmt="%.4f")

print("\n✅ Archivo guardado: datos_brush/resultados_transicion.txt")







#---------------------
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------

#---------------------
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------------------------------------------
# 1. CARGAR DATOS ESTADÍSTICOS
# ------------------------------------------------------------
archivo = "datos_brush/resultados_transicion.txt"
try:
    datos = np.loadtxt(archivo)
    separaciones = datos[:, 0]
    rg_mean      = datos[:, 1]
    rg_std       = datos[:, 2]
    dr_mean      = datos[:, 3]
    dr_std       = datos[:, 4]
    print(f"✅ Datos cargados. Puntos analizados: {len(separaciones)}")
except OSError:
    print("⚠️ No se encontró el archivo .txt. Corre la simulación primero.")
    # Datos de prueba para que no falle si copias y pegas sin archivo
    separaciones, rg_mean, rg_std, dr_mean = [1,10], [10,2], [1,0.1], [20,4]

# ============================================================
# GRÁFICO 1: Radio de Giro vs Separación
# ============================================================
plt.figure(figsize=(7, 5))  # Figura nueva independiente

plt.errorbar(separaciones, rg_mean, yerr=rg_std, fmt='-o', 
             color='royalblue', ecolor='lightblue', capsize=3, label='Datos Simulación')

plt.ylim(bottom=0)
plt.xlabel("Distancia entre anclajes (d)")
plt.ylabel(r"Radio de Giro $\langle R_g \rangle$")
plt.title("Transición de Fase: $R_g$ vs Separación")
plt.grid(True, which="both", alpha=0.3)

# Zonas teóricas
plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
plt.text(2, np.max(rg_mean)*0.8, "BRUSH", color='red', ha='center', fontweight='bold')
plt.text(np.max(separaciones)*0.7, np.max(rg_mean)*0.8, "MUSHROOM", color='green', ha='center', fontweight='bold')

plt.show() # Muestra el gráfico y cierra la figura


# ============================================================
# GRÁFICO 2: Relación de Forma
# ============================================================
plt.figure(figsize=(7, 5))  # Figura nueva independiente

# Evitar división por cero
mask = rg_mean > 0
ratio = np.zeros_like(rg_mean)
ratio[mask] = dr_mean[mask] / rg_mean[mask]

plt.plot(separaciones, ratio, '-s', color='darkorange', label='Simulación')
plt.axhline(y=2.45, color='k', linestyle=':', label='Ideal (Gaussiano)')

plt.xlabel("Distancia entre anclajes (d)")
plt.ylabel(r"Relación $\langle \Delta R \rangle / \langle R_g \rangle$")
plt.title("Cambio de Conformación (Estiramiento)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# PREPARACIÓN DE VISUALIZACIONES
# ============================================================
# Parámetros visuales
N_vis = 400      
n_pol_vis = 15   
colors = cm.jet(np.linspace(0, 1, n_pol_vis))

print("\n--- Generando Visualizaciones ---")

# --- CASO BRUSH ---
try:
    print("Simulando caso BRUSH (d=1)...")
    trays_tight = self_avoided_brush(Nmax=N_vis, separation=1, n_pol=n_pol_vis)
    largo_promedio_tight = np.mean([len(t) for t in trays_tight])
    print(f" -> Largo promedio logrado: {largo_promedio_tight:.1f}")
except NameError:
    trays_tight = []
    largo_promedio_tight = 0

# --- CASO MUSHROOM ---
try:
    print("Simulando caso MUSHROOM (d=15)...")
    trays_loose = self_avoided_brush(Nmax=N_vis, separation=15, n_pol=n_pol_vis)
    largo_promedio_loose = np.mean([len(t) for t in trays_loose])
    print(f" -> Largo promedio logrado: {largo_promedio_loose:.1f}")
except NameError:
    trays_loose = []
    largo_promedio_loose = 0


# ============================================================
# GRÁFICO 3: Visualización BRUSH
# ============================================================
plt.figure(figsize=(7, 7)) # Figura cuadrada para ver mejor la geometría

for i, tr in enumerate(trays_tight):
    plt.plot(tr[:,0], tr[:,1], lw=1.5, alpha=0.8, color=colors[i])
    plt.scatter(tr[0,0], tr[0,1], color='black', s=15, zorder=10)

plt.title(f"Régimen BRUSH (d=1)\nLargo real promedio: {largo_promedio_tight:.1f} monómeros")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)
plt.axis('equal') # Importante

plt.show()


# ============================================================
# GRÁFICO 4: Visualización MUSHROOM
# ============================================================
plt.figure(figsize=(7, 7)) 

for i, tr in enumerate(trays_loose):
    plt.plot(tr[:,0], tr[:,1], lw=1.5, alpha=0.8, color=colors[i])
    plt.scatter(tr[0,0], tr[0,1], color='black', s=15, zorder=10)

plt.title(f"Régimen MUSHROOM (d=15)\nLargo real promedio: {largo_promedio_loose:.1f} monómeros")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.3)

# Opcional: forzar misma escala Y que el anterior para comparar altura
if len(trays_tight) > 0:
    ymax_val = max([np.max(t[:,1]) for t in trays_tight])
    plt.ylim(-5, ymax_val + 5)

plt.show()
#%%


# Vamos a graficar el area bajo la curva de la relacion de forma vs separacion
#%%
plt.figure(figsize=(7, 5))
plt.plot(separaciones, ratio, '-s', color='darkorange', label='Simulación')
plt.axhline(y=2.45, color='k', linestyle=':', label='Ideal (Gaussiano)')
plt.xlabel("Distancia entre anclajes (d)")
plt.ylabel(r"Relación $\langle \Delta R \rangle / \langle R_g \rangle$")
plt.title("Cambio de Conformación (Estiramiento)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%


