import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

st.title('🔮 Predicciones La Liga 2025/2026')

# Leer datos
@st.cache_data
def load_data():
    data_2024_2025 = pd.read_html('https://fbref.com/en/comps/12/2024-2025/schedule/2024-2025-La-Liga-Scores-and-Fixtures',
                                 attrs={"id": "sched_2024-2025_12_1"})[0]
    data_2025_2026 = pd.read_html('https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
                                 attrs={"id": "sched_2025-2026_12_1"})[0]
    return data_2024_2025, data_2025_2026

data_2024_2025, data_2025_2026 = load_data()

# --- PREPROCESAMIENTO ---
with st.spinner('Procesando datos...'):
    # Temporada pasada (entrenamiento)
    df_train = data_2024_2025.dropna(subset=['Wk']).copy()
    df_train['Score'] = df_train['Score'].fillna('0–0')
    df_train[['Score_home', 'Score_away']] = df_train['Score'].str.split('–', expand=True)
    df_train['Score_home'] = df_train['Score_home'].astype(int)
    df_train['Score_away'] = df_train['Score_away'].astype(int)

    # Temporada actual (predicción)
    df_test = data_2025_2026.copy()
    df_test = df_test[df_test['Match Report'] == 'Head-to-Head'].copy().reset_index(drop=True)
    df_test['Match_ID'] = df_test['Home'] + ' vs ' + df_test['Away']

    # Equipos únicos
    teams = pd.unique(df_test[['Home', 'Away']].values.ravel())

    # Estadísticas de goles promedio
    home_goals = df_train.groupby('Home')['Score_home'].mean()
    away_goals = df_train.groupby('Away')['Score_away'].mean()

    # Simulación
    n_simulations = st.slider('Número de simulaciones', 100, 10000, 1000, 100)
    position_counts = {team: np.zeros(len(teams)) for team in teams}

    # Guardar acumulación de goles por partido
    simulated_goals = {match_id: {'home': [], 'away': []} for match_id in df_test['Match_ID']}

    progress_bar = st.progress(0)
    for sim in range(n_simulations):
        points = defaultdict(int)
        goal_diff = defaultdict(int)
        goals_for = defaultdict(int)

        for i, row in df_test.iterrows():
            home = row['Home']
            away = row['Away']
            match_id = row['Match_ID']

            lambda_home = home_goals.get(home, 1.2)
            lambda_away = away_goals.get(away, 1.0)

            sim_home = np.random.poisson(lambda_home)
            sim_away = np.random.poisson(lambda_away)

            # Guardar para promedio por partido
            simulated_goals[match_id]['home'].append(sim_home)
            simulated_goals[match_id]['away'].append(sim_away)

            # Puntos
            if sim_home > sim_away:
                points[home] += 3
            elif sim_home < sim_away:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

            # Goles
            goal_diff[home] += sim_home - sim_away
            goal_diff[away] += sim_away - sim_home
            goals_for[home] += sim_home
            goals_for[away] += sim_away

        # Tabla de posiciones
        standings = pd.DataFrame({
            'Team': list(points.keys()),
            'Points': list(points.values()),
            'GD': [goal_diff[t] for t in points.keys()],
            'GF': [goals_for[t] for t in points.keys()]
        })

        standings.sort_values(by=['Points', 'GD', 'GF'], ascending=False, inplace=True)
        standings.reset_index(drop=True, inplace=True)

        for pos, team in enumerate(standings['Team']):
            position_counts[team][pos] += 1
        
        if sim % 100 == 0:
            progress_bar.progress((sim + 1) / n_simulations)

    progress_bar.empty()

    # Crear matriz de posiciones
    position_matrix = pd.DataFrame(position_counts)
    position_matrix = position_matrix.div(n_simulations) * 100
    position_matrix.index = [f'{i+1}°' for i in range(len(teams))]

    # Generar resultados promedio por partido
    final_results = []
    for i, row in df_test.iterrows():
        match_id = row['Match_ID']
        avg_home = np.mean(simulated_goals[match_id]['home'])
        avg_away = np.mean(simulated_goals[match_id]['away'])

        # Determinar resultado
        if avg_home > avg_away:
            result = f"{row['Home']} win"
        elif avg_home < avg_away:
            result = f"{row['Away']} win"
        else:
            result = "Draw"

        final_results.append({
            'Date': row['Date'] if 'Date' in row else None,
            'Home': row['Home'],
            'Away': row['Away'],
            'Avg_Home_Goals': round(avg_home, 2),
            'Avg_Away_Goals': round(avg_away, 2),
            'Result_Predicted': result
        })

    df_predicted_scores = pd.DataFrame(final_results)
    df_predicted_scores.sort_values(by='Date', inplace=True, ignore_index=True)

    # Crear diccionarios para almacenar puntos, GF y GC
    points = defaultdict(int)
    goals_for = defaultdict(float)
    goals_against = defaultdict(float)

    for _, row in df_predicted_scores.iterrows():
        home = row['Home']
        away = row['Away']
        gh = row['Avg_Home_Goals']
        ga = row['Avg_Away_Goals']

        # Asignar puntos según resultado promedio
        if gh > ga:
            points[home] += 3
        elif gh < ga:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

        # Goles a favor y en contra
        goals_for[home] += gh
        goals_against[home] += ga
        goals_for[away] += ga
        goals_against[away] += gh

    # Crear DataFrame final
    final_table = pd.DataFrame({
        'Team': list(points.keys()),
        'Points': [points[t] for t in points.keys()],
        'GF': [goals_for[t] for t in points.keys()],
        'GA': [goals_against[t] for t in points.keys()],
    })

    final_table['GD'] = final_table['GF'] - final_table['GA']

    # Ordenar tabla por puntos, GD y GF
    final_table.sort_values(by=['Points', 'GD', 'GF'], ascending=False, inplace=True)
    final_table.reset_index(drop=True, inplace=True)
    final_table.index = final_table.index + 1

# Mostrar resultados
st.success('¡Simulación completada!')

st.header('📊 Matriz de posiciones simuladas')
fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(position_matrix, annot=True, fmt=".1f", cmap='Blues', cbar_kws={'label': '% de veces'})
plt.title("Probabilidad de terminar en cada posición (%) - Temporada 2025/2026")
plt.ylabel("Posición final")
plt.xlabel("Equipo")
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

st.header('🏆 Tabla final simulada')
st.dataframe(final_table.style.background_gradient(cmap='Blues', subset=['Points', 'GD', 'GF']), 
             use_container_width=True)

st.header('⚽ Resultados por partido')
st.dataframe(df_predicted_scores, use_container_width=True)

# Mostrar estadísticas adicionales
st.sidebar.header('📈 Estadísticas')
st.sidebar.metric("Número de partidos simulados", len(df_test))
st.sidebar.metric("Número de simulaciones realizadas", n_simulations)
st.sidebar.metric("Equipo con más puntos", final_table.iloc[0]['Team'])
st.sidebar.metric("Puntos del campeón", final_table.iloc[0]['Points'])

st.caption("Datos obtenidos de FBref.com | Modelo basado en distribución de Poisson")