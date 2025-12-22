#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ===========================================================
# 1. PARÁMETROS DEL ESTUDIO
# ===========================================================
L = 80                 # Tamaño de la red (LxL)
N_mon = 100             # Largo de cada polímero
N_polimeros = 10        # Cantidad de cadenas
N_pasos = 10000        # Pasos Monte Carlo por simulación
Snapshots = [0, 100, 1000,2500, 5000,8000, 10000] # Momentos para guardar fotos

separaciones_a_probar = [10,5]

# Carpeta de salida
carpeta_base = "comparativa_brush"
os.makedirs(carpeta_base, exist_ok=True)

# Diccionario para guardar historias de Rg
historia_Rg_global = {}

# ===========================================================
# 2. FUNCIONES (Verdier-Stockmayer)
# ===========================================================

def inicializar_verticales(L, N_mon, N_pol, sep):
    """Inicializa polímeros como varillas verticales."""
    grid = np.zeros((L, L), dtype=int)
    chains = []
    
    # Calcular ancho total para centrar
    ancho_total = (N_pol - 1) * sep
    start_x = (L - ancho_total) // 2
    
    for p in range(N_pol):
        x = start_x + p * sep
        chain = []
        for y in range(N_mon):
            if y < L and 0 <= x < L: 
                grid[y, x] = p + 1 
                chain.append([y, x])
        chains.append(np.array(chain))
    return grid, chains

def movimiento_monte_carlo(grid, chains):
    """Intenta mover un monómero (End-Rotation o Kink-Jump)"""
    p_idx = np.random.randint(len(chains))
    chain = chains[p_idx]
    m_idx = np.random.randint(len(chain))
    
    L_grid = grid.shape[0]
    old_pos = chain[m_idx]
    
    # --- Extremos ---
    if m_idx == 0: return False # Anclado
    elif m_idx == len(chain) - 1:
        vecino = chain[m_idx - 1]
        moves = [[0,1], [0,-1], [1,0], [-1,0]]
        np.random.shuffle(moves)
        for move in moves:
            new_pos = vecino + np.array(move)
            if not (0 <= new_pos[0] < L_grid and 0 <= new_pos[1] < L_grid): continue
            if new_pos[0] < 0: continue # Suelo
            if grid[new_pos[0], new_pos[1]] == 0:
                grid[old_pos[0], old_pos[1]] = 0
                grid[new_pos[0], new_pos[1]] = p_idx + 1
                chain[m_idx] = new_pos
                return True
    # --- Interior (Kink-Jump) ---
    else:
        prev_mon = chain[m_idx - 1]
        next_mon = chain[m_idx + 1]
        # Distancia al cuadrado (diagonal = 2)
        if 1.9 < np.sum((prev_mon - next_mon)**2) < 2.1:
            new_pos = prev_mon + next_mon - old_pos
            if not (0 <= new_pos[0] < L_grid and 0 <= new_pos[1] < L_grid): return False
            if new_pos[0] < 0: return False
            if grid[new_pos[0], new_pos[1]] == 0:
                grid[old_pos[0], old_pos[1]] = 0
                grid[new_pos[0], new_pos[1]] = p_idx + 1
                chain[m_idx] = new_pos
                return True
    return False

def calcular_Rg_promedio(chains):
    Rgs = []
    for chain in chains:
        cm = np.mean(chain, axis=0)
        rg2 = np.mean(np.sum((chain - cm)**2, axis=1))
        Rgs.append(np.sqrt(rg2))
    return np.mean(Rgs)

#%%
# ===========================================================
# 3. BUCLE PRINCIPAL (Iterar sobre separaciones)
# ===========================================================

for sep in separaciones_a_probar:
    print(f"\n🌀 --- INICIANDO SIMULACIÓN: d = {sep} ---")
    
    # 1. Setup Inicial
    Grid, Chains = inicializar_verticales(L, N_mon, N_polimeros, sep)
    Rg_run = []
    
    # 2. Evolución Temporal
    for t in range(N_pasos + 1):
        
        # 1 Monte Carlo Sweep (~ N intentos)
        intentos = N_mon * N_polimeros
        for _ in range(intentos):
            movimiento_monte_carlo(Grid, Chains)
            
        # Guardar datos
        if t % 50 == 0:
            Rg_run.append(calcular_Rg_promedio(Chains))
            # ... (Todo lo anterior igual hasta el bucle de Snapshots)

        # Guardar Snapshots
        if t in Snapshots:
            plt.figure(figsize=(5,5))
            
            masked_grid = np.ma.masked_where(Grid == 0, Grid)
            cmap = cm.get_cmap('nipy_spectral', N_polimeros + 1)
            
            # --- MODIFICACIÓN: CENTRAR COORDENADAS ---
            # Definimos que el eje X va de -L/2 a L/2 para que el 0 esté en el centro
            mitad_L = L / 2
            extension = [-mitad_L, mitad_L, 0, L] 
            
            plt.imshow(masked_grid, cmap=cmap, vmin=1, vmax=N_polimeros+1, 
                       origin='lower', extent=extension) # Agregamos extent
            
            # --- MODIFICACIÓN: ZOOM SOLICITADO ---
            plt.xlim(-sep*7, sep*7)
            
            # Opcional: Ajustar Y para ver mejor la base (ej. de 0 a 60)
            # plt.ylim(0, 60) 

            plt.title(f"Sep={sep} | t={t}")
            
            # Quitamos axis('off') para que puedas ver los números del eje X y confirmar el zoom
            # plt.axis('off') 
            plt.xlabel("Posición X (Centrada)")
            plt.ylabel("Altura Y")
            plt.grid(alpha=0.3, linestyle='--')
            
            fname = f"{carpeta_base}/sim_d{sep}_t{t:05d}.png"
            plt.savefig(fname, dpi=80, bbox_inches='tight')
            plt.close()

    # Guardar la curva de este experimento
    historia_Rg_global[sep] = Rg_run
    print(f"✅ Terminada d={sep}. Final <Rg> = {Rg_run[-1]:.2f}")


# ===========================================================
# 4. GUARDADO DE DATOS (.TXT)  <--- ¡AQUÍ ESTÁ LO AGREGADO!
# ===========================================================
print("\n💾 Guardando datos en archivo de texto...")

# Reconstruimos el eje de tiempo (asumiendo guardado cada 50 pasos)
# Usamos la primera serie de datos para medir el largo
pasos_guardados = np.arange(len(historia_Rg_global[separaciones_a_probar[0]])) * 50

# Lista de columnas: [Tiempo, Rg_d9, Rg_d3, Rg_d1...]
datos_a_guardar = [pasos_guardados]
header_txt = "Tiempo"

for sep in separaciones_a_probar:
    datos_a_guardar.append(historia_Rg_global[sep])
    header_txt += f"\tRg_d{sep}"

# Transponer y apilar
matriz_final = np.column_stack(datos_a_guardar)

# Guardar
ruta_txt = f"{carpeta_base}/dinamica_relajacion.txt"
np.savetxt(ruta_txt, matriz_final, header=header_txt, fmt="%.4f")

print(f"✅ Datos guardados exitosamente en: {ruta_txt}")


#%%
# ===========================================================
# 5. GRÁFICO COMPARATIVO FINAL (LEYENDO DESDE EL TXT)
# ===========================================================
print("\n📊 Generando gráfico leyendo el archivo guardado...")

# 1. Cargar el archivo que acabamos de guardar
ruta_txt = f"{carpeta_base}/dinamica_relajacion.txt"
datos_leidos = np.loadtxt(ruta_txt)

# La columna 0 es el tiempo
tiempo = datos_leidos[:, 0]

plt.figure(figsize=(10, 6))

colores = {9: 'green', 3: 'orange', 1: 'red'}
labels = {9: 'd=9 (Mushroom)', 3: 'd=3 (Intermedio)', 1: 'd=1 (Brush)'}

# 2. Graficar columnas
# Sabemos que guardamos en orden: Tiempo, d=9, d=3, d=1
# Así que d=9 es col 1, d=3 es col 2, d=1 es col 3
idx_columna = 1

for sep in separaciones_a_probar:
    data = datos_leidos[:, idx_columna] # Leemos la columna correspondiente
    
    plt.plot(tiempo, data, label=labels[sep], color=colores[sep], lw=2)
    
    idx_columna += 1 # Avanzamos a la siguiente columna para el próximo d

plt.xlabel("Pasos Monte Carlo (Tiempo)")
plt.ylabel("Radio de Giro Promedio <Rg>")
plt.title("Cinética de Relajación (Datos leídos del TXT)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
# %%
