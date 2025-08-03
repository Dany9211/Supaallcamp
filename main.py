import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import ast

# Configurazione della pagina
st.set_page_config(page_title="Analisi Squadre Combinate", layout="wide")
st.title("Analisi Statistiche Combinate per Squadra")

# --- Funzione di connessione al database (cacheata per efficienza) ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL sul database PostgreSQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database ad ogni aggiornamento.
    """
    try:
        # st.secrets["postgres"] contiene le credenziali del database
        conn = psycopg2.connect(**st.secrets["postgres"], sslmode="require")
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        return pd.DataFrame()

# --- Caricamento dati iniziali dal database ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta di colonne calcolate e pulizia dati per facilitare le analisi ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["totale_gol_ft"] = pd.to_numeric(df["gol_home_ft"], errors='coerce').fillna(0) + pd.to_numeric(df["gol_away_ft"], errors='coerce').fillna(0)
    df['risultato_ft'] = df.apply(
        lambda row: 'Vince Casa' if row['gol_home_ft'] > row['gol_away_ft']
        else ('Vince Ospite' if row['gol_away_ft'] > row['gol_home_ft']
              else 'Pareggio'), axis=1
    )
    df['over_15_ft'] = df['totale_gol_ft'].apply(lambda x: 1 if x > 1.5 else 0)
    df['over_25_ft'] = df['totale_gol_ft'].apply(lambda x: 1 if x > 2.5 else 0)
    df['over_35_ft'] = df['totale_gol_ft'].apply(lambda x: 1 if x > 3.5 else 0)
    df['under_35_ft'] = df['totale_gol_ft'].apply(lambda x: 1 if x < 3.5 else 0)
    df['under_25_ft'] = df['totale_gol_ft'].apply(lambda x: 1 if x < 2.5 else 0)

# Calcolo dei gol nel secondo tempo
if all(col in df.columns for col in ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]):
    df["gol_home_sh"] = pd.to_numeric(df["gol_home_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_home_ht"], errors='coerce').fillna(0)
    df["gol_away_sh"] = pd.to_numeric(df["gol_away_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_away_ht"], errors='coerce').fillna(0)
    df["risultato_sh"] = df["gol_home_sh"].astype(int).astype(str) + "-" + df["gol_away_sh"].astype(int).astype(str)
else:
    st.sidebar.warning("Colonne mancanti per il calcolo delle statistiche del Secondo Tempo.")

# Converti le colonne dei minutaggi gol da stringhe a liste di interi
for col in ['minutaggio_gol', 'minutaggio_gol_away']:
    if col in df.columns:
        # Sostituisci stringhe vuote con '[]' e poi valuta
        df[col] = df[col].apply(lambda x: ast.literal_eval(f"[{x.replace(';', ',')}]") if isinstance(x, str) and x else [])

# Identifica la colonna della data per il filtro delle ultime N partite
date_col_name = "data" if "data" in df.columns else "date" if "date" in df.columns else None
has_date_column = date_col_name is not None
if has_date_column:
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    # Filtro aggiuntivo: solo le partite dopo il 1° agosto 2023
    df = df[df[date_col_name] > datetime.datetime(2023, 8, 1)]
else:
    st.sidebar.warning("Le colonne 'data' o 'date' non sono presenti. Il filtro per le ultime N partite non sarà disponibile.")

squadre_unite = sorted(list(set(df['home_team']).union(set(df['away_team']))))

# --- Selettori nella Sidebar per la configurazione dell'analisi ---
st.sidebar.header("Seleziona Partita")

# 1. Selettore Campionato
leagues = ["Seleziona..."] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues, key="league_select")

df_filtered_by_league = df if selected_league == "Seleziona..." else df[df["league"] == selected_league]
all_teams = sorted(list(set(df_filtered_by_league['home_team'].dropna().unique()) | set(df_filtered_by_league['away_team'].dropna().unique())))

# 2. Selettori Squadre
home_team_selected = st.sidebar.selectbox("Seleziona Squadra CASA", ["Seleziona..."] + all_teams, key="home_team_select")
away_team_selected = st.sidebar.selectbox("Seleziona Squadra TRASFERTA", ["Seleziona..."] + all_teams, key="away_team_select")

if home_team_selected != "Seleziona..." and away_team_selected != "Seleziona...":
    
    # --- Selettore per le ultime N partite ---
    num_to_filter = None
    if has_date_column:
        st.sidebar.header("Filtra Partite per Numero")
        num_partite_options = ["Tutte", "Ultime 5", "Ultime 10", "Ultime 15", "Ultime 20", "Ultime 30", "Ultime 40", "Ultime 50"]
        selected_num_partite_str = st.sidebar.selectbox("Numero di partite da analizzare", num_partite_options, key="num_partite_select")
        
        if selected_num_partite_str != "Tutte":
            num_to_filter = int(selected_num_partite_str.split(' ')[1])

    # --- FILTRAGGIO E COMBINAZIONE DATI PRE-ANALISI ---
    
    # Filtra le partite in casa della squadra selezionata
    df_home = df_filtered_by_league[df_filtered_by_league["home_team"] == home_team_selected]
    
    # Filtra le partite in trasferta della squadra selezionata
    df_away = df_filtered_by_league[df_filtered_by_league["away_team"] == away_team_selected]
    
    # Ordina per data e applica il filtro per il numero di partite
    if has_date_column:
        df_home = df_home.sort_values(by=date_col_name, ascending=False)
        df_away = df_away.sort_values(by=date_col_name, ascending=False)

    if num_to_filter is not None:
        df_home = df_home.head(num_to_filter)
        df_away = df_away.head(num_to_filter)

    # Combina i due DataFrame per le statistiche pre-partita
    df_combined = pd.concat([df_home, df_away], ignore_index=True)
    
    st.header(f"Analisi Combinata: {home_team_selected} (Casa) vs {away_team_selected} (Trasferta)")
    st.write(f"Basata su **{len(df_home)}** partite casalinghe di '{home_team_selected}' e **{len(df_away)}** partite in trasferta di '{away_team_selected}'.")
    st.write(f"**Totale Partite Analizzate:** {len(df_combined)}")

    if not df_combined.empty:
        
        # --- INIZIO FUNZIONI STATISTICHE ---
        
        def get_outcome(home_score, away_score):
            """Determina l'esito (1, X, 2) da un punteggio."""
            if home_score > away_score:
                return '1'
            elif home_score < away_score:
                return '2'
            else:
                return 'X'

        def calcola_winrate(df_to_analyze, col_risultato, title, df_home_matches, df_away_matches):
            """
            Calcola e mostra il WinRate basato sui risultati complessivi.
            """
            home_count = len(df_home_matches)
            away_count = len(df_away_matches)
            total_count = len(df_to_analyze)
            
            st.subheader(f"WinRate {title} - Totale: {total_count} partite (Home: {home_count}, Away: {away_count})")
            df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))].copy()
            
            risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
            for ris in df_valid[col_risultato]:
                try:
                    home, away = map(int, ris.split("-"))
                    outcome = get_outcome(home, away)
                    if outcome == '1':
                        risultati["1 (Casa)"] += 1
                    elif outcome == '2':
                        risultati["2 (Trasferta)"] += 1
                    else:
                        risultati["X (Pareggio)"] += 1
                except ValueError:
                    continue
            
            totale = len(df_valid)
            stats = []
            for esito, count in risultati.items():
                perc = round((count / totale) * 100, 2) if totale > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                stats.append((esito, count, perc, odd_min))
            
            df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))
        
        def calcola_doppia_chance(df_to_analyze, col_risultato, title, df_home_matches, df_away_matches):
            """
            Calcola e mostra le probabilità per la doppia chance (1X, 12, X2).
            """
            home_count = len(df_home_matches)
            away_count = len(df_away_matches)
            total_count = len(df_to_analyze)
            
            st.subheader(f"Doppia Chance {title} - Totale: {total_count} partite (Home: {home_count}, Away: {away_count})")
            
            df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))].copy()
            
            risultati = {"1X": 0, "X2": 0, "12": 0}
            
            for ris in df_valid[col_risultato]:
                try:
                    home, away = map(int, ris.split("-"))
                    outcome = get_outcome(home, away)
                    if outcome == '1' or outcome == 'X':
                        risultati["1X"] += 1
                    if outcome == 'X' or outcome == '2':
                        risultati["X2"] += 1
                    if outcome == '1' or outcome == '2':
                        risultati["12"] += 1
                except ValueError:
                    continue
            
            total = len(df_valid)
            stats = []
            for esito, count in risultati.items():
                perc = round((count / total) * 100, 2) if total > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                stats.append((esito, count, perc, odd_min))
            
            df_stats = pd.DataFrame(stats, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def mostra_risultati_esatti(df_to_analyze, col_risultato, titolo, df_home_matches, df_away_matches):
            """Mostra la distribuzione dei risultati esatti."""
            home_count = len(df_home_matches)
            away_count = len(df_away_matches)
            total_count = len(df_to_analyze)
            
            st.subheader(f"Risultati Esatti {titolo} - Totale: {total_count} partite (Home: {home_count}, Away: {away_count})")
            risultati_interessanti = [
                "0-0", "0-1", "0-2", "0-3", "1-0", "1-1", "1-2", "1-3",
                "2-0", "2-1", "2-2", "2-3", "3-0", "3-1", "3-2", "3-3"
            ]
            df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))].copy()

            def classifica_risultato(ris):
                if ris in risultati_interessanti: return ris
                try:
                    home, away = map(int, ris.split("-"))
                    if home > away: return "Altro risultato casa vince"
                    elif home < away: return "Altro risultato ospite vince"
                    else: return "Altro pareggio"
                except: return "Altro"

            df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
            distribuzione = df_valid["classificato"].value_counts().reset_index()
            distribuzione.columns = [titolo, "Conteggio"]
            distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
            distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_over_goals(df_to_analyze, col_gol_home, col_gol_away, title, df_home_matches, df_away_matches):
            """Calcola la distribuzione Over/Under goals."""
            home_count = len(df_home_matches)
            away_count = len(df_away_matches)
            total_count = len(df_to_analyze)
            
            st.subheader(f"Over/Under Goals {title} - Totale: {total_count} partite (Home: {home_count}, Away: {away_count})")
            df_copy = df_to_analyze.copy()
            df_copy["tot_goals"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) + pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0)
            
            over_data = []
            for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                count = (df_copy["tot_goals"] > t).sum()
                perc = round((count / len(df_copy)) * 100, 2)
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                over_data.append([f"Over {t}", count, perc, odd_min])
            
            df_over = pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_over.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_btts(df_to_analyze, col_gol_home, col_gol_away, title, df_home_matches, df_away_matches):
            """Calcola la statistica GG/NG."""
            home_count = len(df_home_matches)
            away_count = len(df_away_matches)
            total_count = len(df_to_analyze)
            
            st.subheader(f"Entrambe le Squadre a Segno (GG/NG) {title} - Totale: {total_count} partite (Home: {home_count}, Away: {away_count})")
            df_copy = df_to_analyze.copy()
            df_copy["home_scored"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) > 0
            df_copy["away_scored"] = pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0) > 0
            
            gg_count = (df_copy["home_scored"] & df_copy["away_scored"]).sum()
            ng_count = (~(df_copy["home_scored"] & df_copy["away_scored"])).sum()
            
            total = len(df_copy)
            gg_perc = round((gg_count / total) * 100, 2) if total > 0 else 0
            ng_perc = round((ng_count / total) * 100, 2) if total > 0 else 0

            data = {
                "Esito": ["GG (Sì)", "NG (No)"],
                "Conteggio": [gg_count, ng_count],
                "Percentuale %": [gg_perc, ng_perc]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_clean_sheets(df_home_to_analyze, df_away_to_analyze, home_team_name, away_team_name, col_gol_home, col_gol_away, title):
            """Calcola le Clean Sheets e il Fallimento a Segnare per le squadre selezionate."""
            total_home_matches = len(df_home_to_analyze)
            total_away_matches = len(df_away_to_analyze)
            total_matches = total_home_matches + total_away_matches
            
            st.subheader(f"Clean Sheets / Fail to Score {title} - Totale: {total_matches} partite (Home: {total_home_matches}, Away: {total_away_matches})")
            
            # Calcolo per la squadra di casa (in casa)
            home_clean_sheets = (pd.to_numeric(df_home_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
            home_fail_to_score = (pd.to_numeric(df_home_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
            
            # Calcolo per la squadra in trasferta (fuori casa)
            away_clean_sheets = (pd.to_numeric(df_away_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
            away_fail_to_score = (pd.to_numeric(df_away_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
            
            data = {
                "Squadra": [home_team_name, away_team_name],
                "Clean Sheets": [home_clean_sheets, away_clean_sheets],
                "Fail to Score": [home_fail_to_score, away_fail_to_score]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["% Clean Sheets"] = [
                round((home_clean_sheets / total_home_matches) * 100, 2) if total_home_matches > 0 else 0,
                round((away_clean_sheets / total_away_matches) * 100, 2) if total_away_matches > 0 else 0
            ]
            df_stats["% Fail to Score"] = [
                round((home_fail_to_score / total_home_matches) * 100, 2) if total_home_matches > 0 else 0,
                round((away_fail_to_score / total_away_matches) * 100, 2) if total_away_matches > 0 else 0
            ]
            
            st.dataframe(df_stats.style.background_gradient(cmap='Blues', subset=['% Clean Sheets']).background_gradient(cmap='Reds', subset=['% Fail to Score']))

        def calcola_media_gol(df_home, df_away, home_team_name, away_team_name, title, col_home_goals, col_away_goals):
            """Calcola la media gol fatti e subiti per le squadre selezionate in un intervallo specifico (HT, FT, etc)."""
            total_home_matches = len(df_home)
            total_away_matches = len(df_away)
            total_matches = total_home_matches + total_away_matches
            st.subheader(f"Media Gol Fatti e Subiti {title} - Totale: {total_matches} partite (Home: {total_home_matches}, Away: {total_away_matches})")
            home_goals_mean = pd.to_numeric(df_home[col_home_goals], errors='coerce').mean()
            home_conceded_mean = pd.to_numeric(df_home[col_away_goals], errors='coerce').mean()
            away_goals_mean = pd.to_numeric(df_away[col_away_goals], errors='coerce').mean()
            away_conceded_mean = pd.to_numeric(df_away[col_home_goals], errors='coerce').mean()
            data = {
                "Squadra": [home_team_name, away_team_name],
                "Gol Fatti": [f"{home_goals_mean:.2f}", f"{away_goals_mean:.2f}"],
                "Gol Subiti": [f"{home_conceded_mean:.2f}", f"{away_conceded_mean:.2f}"]
            }
            df_stats = pd.DataFrame(data)
            st.dataframe(df_stats)
        
        # Funzione per calcolare le statistiche sul 'Next Goal' in una timeband
        def calcola_timeband_stats(df_to_analyze, time_bands, title, team1, team2, current_minute):
            st.subheader(f"Statistiche {title}")
            data = defaultdict(lambda: defaultdict(int))

            for _, match in df_to_analyze.iterrows():
                home_goals = [(m, 'home') for m in match.get('minutaggio_gol', [])]
                away_goals = [(m, 'away') for m in match.get('minutaggio_gol_away', [])]
                
                all_goals = sorted(home_goals + away_goals, key=lambda x: x[0])
                first_goal_after_current = next((g for g in all_goals if g[0] > current_minute), None)

                if first_goal_after_current:
                    minuto, tipo_squadra = first_goal_after_current
                    gol_team = match['home_team'] if tipo_squadra == 'home' else match['away_team']
                    
                    for start, end in time_bands:
                        if start <= minuto <= end:
                            data[f"{start}-{end}"][gol_team] += 1
                            data[f"{start}-{end}"]["Totale Partite"] += 1
                            break
                else:
                    for start, end in time_bands:
                        if current_minute < end:
                            data[f"{start}-{end}"]["Nessun Gol"] += 1
                            data[f"{start}-{end}"]["Totale Partite"] += 1
                            break

            results = []
            for band in sorted(data.keys(), key=lambda x: int(x.split('-')[0])):
                totale = data[band]["Totale Partite"]
                if totale > 0:
                    percentuale_team1 = (data[band].get(team1, 0) / totale) * 100
                    percentuale_team2 = (data[band].get(team2, 0) / totale) * 100
                    percentuale_nessun_gol = (data[band].get("Nessun Gol", 0) / totale) * 100
                    results.append({
                        "Banda Minuti": band,
                        f"Gol {team1}": data[band].get(team1, 0),
                        f"Gol {team2}": data[band].get(team2, 0),
                        "Nessun Gol": data[band].get("Nessun Gol", 0),
                        f"% Gol {team1}": f"{percentuale_team1:.2f}%",
                        f"% Gol {team2}": f"{percentuale_team2:.2f}%",
                        "% Nessun Gol": f"{percentuale_nessun_gol:.2f}%",
                        "Totale": totale
                    })
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.set_index("Banda Minuti")
                st.dataframe(df_results, use_container_width=True)
            else:
                st.info("Nessun dato disponibile per le bande temporali specificate.")

        # Funzione per calcolare lo score a un minuto specifico
        def get_score_at_minute(row, target_minute):
            home_goals_in_match = [g for g in row.get('minutaggio_gol', []) if g <= target_minute]
            away_goals_in_match = [g for g in row.get('minutaggio_gol_away', []) if g <= target_minute]
            return len(home_goals_in_match), len(away_goals_in_match)

        # --- Interfaccia utente per i parametri dinamici ---
        st.header(f"Analisi Dinamica: {home_team_selected} vs {away_team_selected}")

        score_change_options = st.radio(
            "Vuoi usare un punteggio fisso o dinamico?",
            ("Punteggio Dinamico", "Punteggio Fisso"),
            key='score_options'
        )

        if score_change_options == "Punteggio Dinamico":
            st.info("Questa sezione non è ancora disponibile per l'uso.")
        else: # Punteggio Fisso
            st.subheader("Parametri Punteggio Fisso")
            col1, col2 = st.columns(2)
            with col1:
                current_minute_dynamic_ft = st.number_input("Minuto attuale (dal 0 al 90)", min_value=0, max_value=90, value=60, key='minute_input')
            with col2:
                col3, col4 = st.columns(2)
                with col3:
                    current_score_home_ft = st.number_input(f"Punteggio {home_team_selected}", min_value=0, max_value=20, value=1, key='score_home_input')
                with col4:
                    current_score_away_ft = st.number_input(f"Punteggio {away_team_selected}", min_value=0, max_value=20, value=1, key='score_away_input')

            # Filtra il DataFrame combinato in base ai parametri dinamici
            df_combined_dynamic_ft = df_combined[
                df_combined.apply(
                    lambda row: get_score_at_minute(row, current_minute_dynamic_ft) == (current_score_home_ft, current_score_away_ft),
                    axis=1
                )
            ]

            if not df_combined_dynamic_ft.empty:
                st.subheader("Analisi 'Next Goal' sulle partite trovate")
                with st.expander("Bande Temporali Fisse FT", expanded=False):
                    # Bande temporali fisse di 5 minuti: 0-5, 6-10, ...
                    # Nota: le bande sono fisse, ma l'analisi si basa sui dati filtrati dinamicamente
                    fixed_time_bands_5min_ft = [(i * 5 + 1, (i + 1) * 5) for i in range(18)]
                    fixed_time_bands_5min_ft[0] = (0, 5) # Correggi la prima banda
                    
                    calcola_timeband_stats(
                        df_combined_dynamic_ft, 
                        fixed_time_bands_5min_ft, 
                        f"Fisse 5 min", 
                        home_team_selected, 
                        away_team_selected,
                        current_minute_dynamic_ft
                    )
                    
                    # Bande temporali fisse di 15 minuti: 0-15, 16-30, ...
                    fixed_time_bands_15min_ft = [(i * 15 + 1, (i + 1) * 15) for i in range(6)]
                    fixed_time_bands_15min_ft[0] = (0, 15) # Correggi la prima banda
                    
                    calcola_timeband_stats(
                        df_combined_dynamic_ft, 
                        fixed_time_bands_15min_ft, 
                        f"Fisse 15 min", 
                        home_team_selected, 
                        away_team_selected,
                        current_minute_dynamic_ft
                    )
            else:
                st.warning("Nessuna partita storica trovata con i parametri dinamici specificati. Prova a modificare minuto e punteggio.")
    else:
        st.warning("Seleziona una squadra 'CASA' e una squadra 'OSPITE' valide per l'analisi.")

