import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Squadre Combinate", layout="wide")
st.title("Analisi Statistiche Combinate per Squadra")

# --- Funzione connessione al database ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database
    ogni volta che l'applicazione si aggiorna.
    """
    try:
        conn = psycopg2.connect(**st.secrets["postgres"], sslmode="require")
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        return pd.DataFrame()

# --- Caricamento dati iniziali ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta colonne calcolate ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

# Calcolo dei gol e del risultato del Secondo Tempo
if all(col in df.columns for col in ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]):
    df["gol_home_sh"] = pd.to_numeric(df["gol_home_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_home_ht"], errors='coerce').fillna(0)
    df["gol_away_sh"] = pd.to_numeric(df["gol_away_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_away_ht"], errors='coerce').fillna(0)
    df["risultato_sh"] = df["gol_home_sh"].astype(int).astype(str) + "-" + df["gol_away_sh"].astype(int).astype(str)
else:
    st.sidebar.warning("Colonne mancanti per il calcolo delle statistiche del Secondo Tempo.")

# Controllo se la colonna 'data' o 'date' esiste prima di usarla
date_col_name = None
if "data" in df.columns:
    date_col_name = "data"
elif "date" in df.columns:
    date_col_name = "date"

has_date_column = date_col_name is not None
if has_date_column:
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
else:
    st.sidebar.warning("Le colonne 'data' o 'date' non sono presenti nel database. Il filtro per le ultime N partite non sarà disponibile.")


# --- Selettori Sidebar ---
st.sidebar.header("Seleziona Squadre")

# 1. Selettore Campionato
leagues = ["Seleziona..."] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)

if selected_league != "Seleziona...":
    # Filtra le squadre in base al campionato selezionato
    league_teams_df = df[df["league"] == selected_league]
    all_teams = sorted(list(set(league_teams_df['home_team'].dropna().unique()) | set(league_teams_df['away_team'].dropna().unique())))

    # 2. Selettori Squadre
    home_team_selected = st.sidebar.selectbox("Seleziona Squadra CASA", ["Seleziona..."] + all_teams)
    away_team_selected = st.sidebar.selectbox("Seleziona Squadra TRASFERTA", ["Seleziona..."] + all_teams)

    if home_team_selected != "Seleziona..." and away_team_selected != "Seleziona...":
        
        # --- Selettore per le ultime N partite, mostrato solo se la colonna 'data' o 'date' esiste ---
        num_to_filter = None
        if has_date_column:
            st.sidebar.header("Filtra Partite per Numero")
            num_partite_options = ["Tutte", "Ultime 5", "Ultime 10", "Ultime 15", "Ultime 20", "Ultime 30", "Ultime 40", "Ultime 50"]
            selected_num_partite_str = st.sidebar.selectbox("Numero di partite da analizzare", num_partite_options)
            
            # Converti la selezione in un numero intero
            if selected_num_partite_str != "Tutte":
                num_to_filter = int(selected_num_partite_str.split(' ')[1])

        # --- FILTRAGGIO E COMBINAZIONE DATI ---
        
        # Prendi tutte le partite in casa della squadra di casa
        df_home = df[(df["home_team"] == home_team_selected) & (df["league"] == selected_league)]
        
        # Prendi tutte le partite in trasferta della squadra in trasferta
        df_away = df[(df["away_team"] == away_team_selected) & (df["league"] == selected_league)]
        
        # Ordina per data decrescente e filtra per il numero di partite, solo se la colonna esiste
        if has_date_column:
            df_home = df_home.sort_values(by=date_col_name, ascending=False)
            df_away = df_away.sort_values(by=date_col_name, ascending=False)

        if num_to_filter is not None:
            df_home = df_home.head(num_to_filter)
            df_away = df_away.head(num_to_filter)

        # Combina i due DataFrame
        df_combined = pd.concat([df_home, df_away], ignore_index=True)
        
        st.header(f"Analisi Combinata: {home_team_selected} (Casa) vs {away_team_selected} (Trasferta)")
        st.write(f"Basata su **{len(df_home)}** partite casalinghe di '{home_team_selected}' e **{len(df_away)}** partite in trasferta di '{away_team_selected}'.")
        st.write(f"**Totale Partite Analizzate:** {len(df_combined)}")

        if not df_combined.empty:
            
            # --- Funzione per calcolare il punteggio al minuto 'end_minute' ---
            def get_scores_at_minute(df_row, selected_min, home_team_name):
                """Calcola il punteggio al minuto specificato per una singola partita."""
                gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                # Se la squadra selezionata è quella in casa, usa i suoi gol_home, altrimenti i gol_away
                is_home_selected_at_home = df_row["home_team"] == home_team_name
                
                if is_home_selected_at_home:
                    home_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                    away_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                else: # Se stiamo analizzando una partita dove la "squadra di casa selezionata" è in realtà la squadra in trasferta
                    home_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min) # Qui i gol della home_team_selected sono i gol_away della riga
                    away_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min) # Qui i gol della away_team_selected sono i gol_home della riga
                
                return f"{home_goals_count}-{away_goals_count}"

            # --- INIZIO FUNZIONI STATISTICHE ---
            def get_outcome(home_score, away_score):
                """Determina l'esito (1, X, 2) da un punteggio."""
                if home_score > away_score:
                    return '1'
                elif home_score < away_score:
                    return '2'
                else:
                    return 'X'

            def calcola_winrate(df_to_analyze, col_risultato, title):
                st.subheader(f"WinRate {title} ({len(df_to_analyze)} partite)")
                df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))]
                risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
                for ris in df_valid[col_risultato]:
                    try:
                        home, away = map(int, ris.split("-"))
                        if home > away:
                            risultati["1 (Casa)"] += 1
                        elif home < away:
                            risultati["2 (Trasferta)"] += 1
                        else:
                            risultati["X (Pareggio)"] += 1
                    except:
                        continue
                totale = len(df_valid)
                stats = []
                for esito, count in risultati.items():
                    perc = round((count / totale) * 100, 2) if totale > 0 else 0
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    stats.append((esito, count, perc, odd_min))
                
                df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))

            def mostra_risultati_esatti(df_to_analyze, col_risultato, titolo):
                st.subheader(f"Risultati Esatti {titolo} ({len(df_to_analyze)} partite)")
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
            
            def calcola_over_goals(df_to_analyze, col_gol_home, col_gol_away, title):
                st.subheader(f"Over/Under Goals {title} ({len(df_to_analyze)} partite)")
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

            def calcola_btts(df_to_analyze, col_gol_home, col_gol_away, title):
                st.subheader(f"Entrambe le Squadre a Segno (GG/NG) {title} ({len(df_to_analyze)} partite)")
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
                st.subheader(f"Clean Sheets / Fail to Score {title} (Home: {len(df_home_to_analyze)} partite, Away: {len(df_away_to_analyze)} partite)")
                
                # Calcolo per la squadra di casa (in casa)
                home_clean_sheets = (pd.to_numeric(df_home_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
                home_fail_to_score = (pd.to_numeric(df_home_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
                total_home_matches = len(df_home_to_analyze)
                
                # Calcolo per la squadra in trasferta (fuori casa)
                away_clean_sheets = (pd.to_numeric(df_away_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
                away_fail_to_score = (pd.to_numeric(df_away_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
                total_away_matches = len(df_away_to_analyze)
                
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
            
            def calcola_first_to_score(df_to_analyze, home_team_name, away_team_name, timeframe_label, start_min=1, end_min=150):
                st.subheader(f"Prima Squadra a Segnare {timeframe_label} ({len(df_to_analyze)} partite)")
                if df_to_analyze.empty: return

                risultati = {f"{home_team_name}": 0, f"{away_team_name}": 0, "Nessun Gol": 0}
                
                for _, row in df_to_analyze.iterrows():
                    # Identifica correttamente i gol della squadra selezionata di casa e di trasferta
                    if row["home_team"] == home_team_name:
                        selected_home_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                        selected_away_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    else: # Questo caso si verifica quando analizziamo le partite in trasferta della away_team_selected
                        selected_home_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                        selected_away_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                        
                    # Filtra i gol per il timeframe
                    gol_home_in_range = [g for g in selected_home_goals_minutes if start_min <= g <= end_min]
                    gol_away_in_range = [g for g in selected_away_goals_minutes if start_min <= g <= end_min]
                    
                    min_home_goal = min(gol_home_in_range) if gol_home_in_range else float('inf')
                    min_away_goal = min(gol_away_in_range) if gol_away_in_range else float('inf')
                    
                    if min_home_goal < min_away_goal:
                        risultati[f"{home_team_name}"] += 1
                    elif min_away_goal < min_home_goal:
                        risultati[f"{away_team_name}"] += 1
                    elif min_home_goal == float('inf') and min_away_goal == float('inf'):
                        risultati["Nessun Gol"] += 1

                stats = []
                totale_partite = len(df_to_analyze)
                
                for esito, count in risultati.items():
                    perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    stats.append((esito, count, perc, odd_min))
                
                df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def mostra_distribuzione_timeband(df_to_analyze, title, home_team_name, away_team_name, timeframe=5):
                st.subheader(f"Distribuzione Gol per Timeframe {title} ({len(df_to_analyze)} partite)")
                if df_to_analyze.empty: return
                
                if timeframe == 15:
                    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
                    label_intervalli = [f"{start}-{end}" for start, end in intervalli]
                else:
                    start_mins = [0] + [i + 1 for i in range(5, 90, 5)]
                    end_mins = list(range(5, 91, 5))
                    intervalli = list(zip(start_mins, end_mins))
                    label_intervalli = [f"{start}-{end}" for start, end in intervalli]

                intervalli.append((91, 150))
                label_intervalli.append("90+")
                
                risultati = []
                totale_partite = len(df_to_analyze)
                
                for (start, end), label in zip(intervalli, label_intervalli):
                    partite_con_gol = 0
                    partite_con_almeno_due_gol = 0
                    
                    home_selected_goals_scored = 0
                    home_selected_goals_conceded = 0
                    away_selected_goals_scored = 0
                    away_selected_goals_conceded = 0
                    
                    for _, row in df_to_analyze.iterrows():
                        gol_home_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                        gol_away_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                        
                        goals_in_interval_home = sum(1 for g in gol_home_minutes if start <= g <= end)
                        goals_in_interval_away = sum(1 for g in gol_away_minutes if start <= g <= end)

                        # Calcolo gol segnati e subiti per le squadre selezionate
                        if row["home_team"] == home_team_name:
                            home_selected_goals_scored += goals_in_interval_home
                            home_selected_goals_conceded += goals_in_interval_away
                        elif row["away_team"] == home_team_name:
                            home_selected_goals_scored += goals_in_interval_away
                            home_selected_goals_conceded += goals_in_interval_home

                        if row["home_team"] == away_team_name:
                            away_selected_goals_scored += goals_in_interval_home
                            away_selected_goals_conceded += goals_in_interval_away
                        elif row["away_team"] == away_team_name:
                            away_selected_goals_scored += goals_in_interval_away
                            away_selected_goals_conceded += goals_in_interval_home
                        
                        if (goals_in_interval_home + goals_in_interval_away) >= 1:
                            partite_con_gol += 1
                        if (goals_in_interval_home + goals_in_interval_away) >= 2:
                            partite_con_almeno_due_gol += 1
                    
                    perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    
                    risultati.append([
                        label, 
                        partite_con_gol, 
                        partite_con_almeno_due_gol, 
                        perc, 
                        odd_min, 
                        f"Segnati: {home_selected_goals_scored}, Subiti: {home_selected_goals_conceded}",
                        f"Segnati: {away_selected_goals_scored}, Subiti: {away_selected_goals_conceded}"
                    ])
                    
                df_result = pd.DataFrame(risultati, columns=[
                    "Timeframe", 
                    "Partite con 1+ Gol", 
                    "Partite con 2+ Gol", 
                    "Percentuale % (1+ Gol)", 
                    "Odd Minima (1+ Gol)", 
                    f"Statistiche {home_team_selected}",
                    f"Statistiche {away_team_selected}"
                ])
                st.dataframe(df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale % (1+ Gol)']))
                
            def calcola_media_gol(df_home, df_away, home_team_name, away_team_name):
                st.subheader(f"Media Gol Fatti e Subiti (Home: {len(df_home)} partite, Away: {len(df_away)} partite)")
                
                # Medie per il Primo Tempo (HT)
                home_goals_ht = pd.to_numeric(df_home["gol_home_ht"], errors='coerce').mean()
                home_conceded_ht = pd.to_numeric(df_home["gol_away_ht"], errors='coerce').mean()
                away_goals_ht = pd.to_numeric(df_away["gol_away_ht"], errors='coerce').mean()
                away_conceded_ht = pd.to_numeric(df_away["gol_home_ht"], errors='coerce').mean()

                # Medie per il Fine Partita (FT)
                home_goals_ft = pd.to_numeric(df_home["gol_home_ft"], errors='coerce').mean()
                home_conceded_ft = pd.to_numeric(df_home["gol_away_ft"], errors='coerce').mean()
                away_goals_ft = pd.to_numeric(df_away["gol_away_ft"], errors='coerce').mean()
                away_conceded_ft = pd.to_numeric(df_away["gol_home_ft"], errors='coerce').mean()

                # Medie per il Secondo Tempo (SH)
                home_goals_sh = pd.to_numeric(df_home["gol_home_sh"], errors='coerce').mean()
                home_conceded_sh = pd.to_numeric(df_home["gol_away_sh"], errors='coerce').mean()
                away_goals_sh = pd.to_numeric(df_away["gol_away_sh"], errors='coerce').mean()
                away_conceded_sh = pd.to_numeric(df_away["gol_home_sh"], errors='coerce').mean()


                data = {
                    "Squadra": [home_team_name, away_team_name],
                    "Gol Fatti (PT)": [f"{home_goals_ht:.2f}", f"{away_goals_ht:.2f}"],
                    "Gol Subiti (PT)": [f"{home_conceded_ht:.2f}", f"{away_conceded_ht:.2f}"],
                    "Gol Fatti (ST)": [f"{home_goals_sh:.2f}", f"{away_goals_sh:.2f}"],
                    "Gol Subiti (ST)": [f"{home_conceded_sh:.2f}", f"{away_conceded_sh:.2f}"],
                    "Gol Fatti (FT)": [f"{home_goals_ft:.2f}", f"{away_goals_ft:.2f}"],
                    "Gol Subiti (FT)": [f"{away_conceded_ft:.2f}", f"{away_conceded_ft:.2f}"]
                }
                
                df_media = pd.DataFrame(data)
                st.table(df_media)
            
            def calcola_ht_ft_combo(df_to_analyze):
                st.subheader(f"Risultato Parziale/Finale (HT/FT) ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()

                df_copy["ht_outcome"] = df_copy.apply(lambda row: get_outcome(row["gol_home_ht"], row["gol_away_ht"]), axis=1)
                df_copy["ft_outcome"] = df_copy.apply(lambda row: get_outcome(row["gol_home_ft"], row["gol_away_ft"]), axis=1)

                df_copy["combo"] = df_copy["ht_outcome"] + "/" + df_copy["ft_outcome"]
                combo_counts = df_copy["combo"].value_counts().reset_index()
                combo_counts.columns = ["Risultato", "Conteggio"]
                
                total = len(df_to_analyze)
                combo_counts["Percentuale %"] = (combo_counts["Conteggio"] / total * 100).round(2)
                combo_counts["Odd Minima"] = combo_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(combo_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def calcola_margine_vittoria(df_to_analyze, col_gol_home, col_gol_away, title):
                st.subheader(f"Margine di Vittoria {title} ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()
                df_copy["gol_diff"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) - pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0)
                
                def classify_margin(diff):
                    if diff > 0:
                        if diff == 1: return "Casa vince di 1"
                        elif diff == 2: return "Casa vince di 2"
                        else: return "Casa vince di 3+"
                    elif diff < 0:
                        if diff == -1: return "Trasferta vince di 1"
                        elif diff == -2: return "Trasferta vince di 2"
                        else: return "Trasferta vince di 3+"
                    else:
                        return "Pareggio"
                
                df_copy["margine"] = df_copy["gol_diff"].apply(classify_margin)
                
                margin_counts = df_copy["margine"].value_counts().reset_index()
                margin_counts.columns = ["Margine", "Conteggio"]
                total = len(df_to_analyze)
                margin_counts["Percentuale %"] = (margin_counts["Conteggio"] / total * 100).round(2)
                margin_counts["Odd Minima"] = margin_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(margin_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def mostra_progressione_combinata(df_combined, home_team_selected, away_team_selected, start_min, end_min):
                st.subheader(f"Progressione Risultati (Combinata tra min {start_min} e {end_min})")
                total_matches = len(df_combined)

                if total_matches == 0:
                    st.write("Nessun dato disponibile per l'analisi della progressione.")
                    return

                home_went_1_0 = 0
                home_went_1_1_after_1_0 = 0
                home_went_2_0 = 0

                away_went_0_1 = 0
                away_went_1_1_after_0_1 = 0
                away_went_0_2 = 0
                
                total_matches_with_goals = 0

                for _, row in df_combined.iterrows():
                    gol_home_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                    gol_away_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    
                    # Filtra i gol nel range di tempo
                    gol_home_range = [m for m in gol_home_minutes if start_min <= m <= end_min]
                    gol_away_range = [m for m in gol_away_minutes if start_min <= m <= end_min]

                    if gol_home_range or gol_away_range:
                        total_matches_with_goals += 1
                        
                        is_home_selected_at_home = row["home_team"] == home_team_selected
                        
                        goal_events = []
                        if is_home_selected_at_home:
                            goal_events = [(m, "home") for m in gol_home_range] + [(m, "away") for m in gol_away_range]
                        else:
                            goal_events = [(m, "away") for m in gol_away_range] + [(m, "home") for m in gol_home_range]
                        
                        goal_events.sort(key=lambda x: x[0])

                        current_home_score_analysis = 0
                        current_away_score_analysis = 0
                        
                        has_been_1_0 = False
                        has_been_1_1_after_1_0 = False
                        has_been_2_0 = False
                        
                        has_been_0_1 = False
                        has_been_1_1_after_0_1 = False
                        has_been_0_2 = False

                        for _, team in goal_events:
                            if team == "home":
                                current_home_score_analysis += 1
                            else:
                                current_away_score_analysis += 1

                            if is_home_selected_at_home:
                                if current_home_score_analysis == 1 and current_away_score_analysis == 0:
                                    has_been_1_0 = True
                                if has_been_1_0 and current_home_score_analysis == 1 and current_away_score_analysis == 1:
                                    has_been_1_1_after_1_0 = True
                                if current_home_score_analysis == 2 and current_away_score_analysis == 0:
                                    has_been_2_0 = True
                            else:
                                if current_home_score_analysis == 0 and current_away_score_analysis == 1:
                                    has_been_0_1 = True
                                if has_been_0_1 and current_home_score_analysis == 1 and current_away_score_analysis == 1:
                                    has_been_1_1_after_0_1 = True
                                if current_home_score_analysis == 0 and current_away_score_analysis == 2:
                                    has_been_0_2 = True
                        
                        if is_home_selected_at_home:
                            if has_been_1_0: home_went_1_0 += 1
                            if has_been_1_1_after_1_0: home_went_1_1_after_1_0 += 1
                            if has_been_2_0: home_went_2_0 += 1
                        else:
                            if has_been_0_1: away_went_0_1 += 1
                            if has_been_1_1_after_0_1: away_went_1_1_after_0_1 += 1
                            if has_been_0_2: away_went_0_2 += 1
                
                denominator = total_matches_with_goals if total_matches_with_goals > 0 else 1
                
                home_progression_data = {
                    "Scenario": [
                        f"{home_team_selected} va in vantaggio 1-0",
                        f"Il punteggio diventa 1-1 dopo l'1-0",
                        f"{home_team_selected} va in vantaggio 2-0"
                    ],
                    "Conteggio": [
                        home_went_1_0,
                        home_went_1_1_after_1_0,
                        home_went_2_0
                    ],
                    "Percentuale su partite con gol %": [
                        round((home_went_1_0 / denominator) * 100, 2),
                        round((home_went_1_1_after_1_0 / denominator) * 100, 2),
                        round((home_went_2_0 / denominator) * 100, 2)
                    ]
                }
                df_home_prog = pd.DataFrame(home_progression_data)
                df_home_prog["Odd Minima"] = df_home_prog["Percentuale su partite con gol %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                away_progression_data = {
                    "Scenario": [
                        f"{away_team_selected} va in vantaggio 0-1",
                        f"Il punteggio diventa 1-1 dopo lo 0-1",
                        f"{away_team_selected} va in vantaggio 0-2"
                    ],
                    "Conteggio": [
                        away_went_0_1,
                        away_went_1_1_after_0_1,
                        away_went_0_2
                    ],
                    "Percentuale su partite con gol %": [
                        round((away_went_0_1 / denominator) * 100, 2),
                        round((away_went_1_1_after_0_1 / denominator) * 100, 2),
                        round((away_went_0_2 / denominator) * 100, 2)
                    ]
                }
                df_away_prog = pd.DataFrame(away_progression_data)
                df_away_prog["Odd Minima"] = df_away_prog["Percentuale su partite con gol %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{home_team_selected}**")
                    st.dataframe(df_home_prog.style.background_gradient(cmap='RdYlGn', subset=['Percentuale su partite con gol %']))
                with col2:
                    st.write(f"**{away_team_selected}**")
                    st.dataframe(df_away_prog.style.background_gradient(cmap='RdYlGn', subset=['Percentuale su partite con gol %']))

            
            # --- ESECUZIONE E VISUALIZZAZIONE STATS PRE-PARTITA (FISSE) ---
            
            st.markdown("---")
            st.header("Statistiche Pre-partita")
            
            # Media Gol Totali
            st.subheader(f"Media Gol Totali per Partita ({len(df_combined)} partite)")
            avg_ht_goals = (pd.to_numeric(df_combined["gol_home_ht"], errors='coerce').fillna(0) + pd.to_numeric(df_combined["gol_away_ht"], errors='coerce').fillna(0)).mean()
            avg_sh_goals = (pd.to_numeric(df_combined["gol_home_sh"], errors='coerce').fillna(0) + pd.to_numeric(df_combined["gol_away_sh"], errors='coerce').fillna(0)).mean()
            avg_ft_goals = (pd.to_numeric(df_combined["gol_home_ft"], errors='coerce').fillna(0) + pd.to_numeric(df_combined["gol_away_ft"], errors='coerce').fillna(0)).mean()
            st.table(pd.DataFrame({
                "Periodo": ["Primo Tempo (PT)", "Secondo Tempo (ST)", "Finale (FT)"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_sh_goals:.2f}", f"{avg_ft_goals:.2f}"]
            }))

            # Media gol fatti e subiti per singola squadra
            calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)

            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_winrate(df_combined, "risultato_ht", "PT")
            with col2:
                calcola_winrate(df_combined, "risultato_sh", "ST")
            with col3:
                calcola_winrate(df_combined, "risultato_ft", "FT")

            col1, col2, col3 = st.columns(3)
            with col1:
                mostra_risultati_esatti(df_combined, "risultato_ht", "PT")
            with col2:
                mostra_risultati_esatti(df_combined, "risultato_sh", "ST")
            with col3:
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")

            # Nuove stats
            calcola_ht_ft_combo(df_combined)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_margine_vittoria(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_margine_vittoria(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_margine_vittoria(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            st.markdown("---")
            st.header("Statistiche sui Gol Pre-partita")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_over_goals(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_btts(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")
            
            st.markdown("---")
            st.header("Analisi Temporale dei Gol Pre-partita")

            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "PT", start_min=1, end_min=45)
            with col2:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "ST", start_min=46, end_min=90)
            with col3:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "FT", start_min=1, end_min=150)

            
            # Analisi per intervalli di 5 minuti
            mostra_distribuzione_timeband(df_combined, "(5 Min)", home_team_selected, away_team_selected)

            # Analisi per intervalli di 15 minuti
            mostra_distribuzione_timeband(df_combined, "(15 Min)", home_team_selected, away_team_selected, timeframe=15)
            
            # --- ESECUZIONE E VISUALIZZAZIONE STATS DINAMICHE ---
            st.markdown("---")
            st.header("Statistiche Dinamiche (con Minutaggio e Risultato di Partenza)")

            col_sliders_1, col_sliders_2 = st.columns(2)
            with col_sliders_1:
                # Modifica del cursore in un range di minutaggio
                minutaggio_range = st.slider("Intervallo di minutaggio", 0, 90, (45, 90), key="minute_range_slider")
                start_minute = minutaggio_range[0]
                end_minute = minutaggio_range[1]
            with col_sliders_2:
                starting_score_str = st.text_input("Risultato di Partenza (es. 1-0)", "0-0", key="score_input")

            # Validazione e parsing del risultato di partenza
            try:
                if "-" in starting_score_str:
                    starting_score_home, starting_score_away = map(int, starting_score_str.split('-'))
                    # Filtra il DataFrame in base al minutaggio e al risultato di partenza
                    df_dynamic_filtered = df_combined[df_combined.apply(
                        lambda row: get_scores_at_minute(row, end_minute, home_team_selected) == starting_score_str,
                        axis=1
                    )]
                else:
                    st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                    df_dynamic_filtered = pd.DataFrame()
            except ValueError:
                st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                df_dynamic_filtered = pd.DataFrame()
            
            if not df_dynamic_filtered.empty:
                st.write(f"Analisi basata su **{len(df_dynamic_filtered)}** partite in cui il punteggio era **{starting_score_str}** al minuto **{end_minute}**.")
                
                # Le funzioni dinamiche usano ora il DataFrame filtrato
                calcola_winrate(df_dynamic_filtered, "risultato_ft", "Finale (in partite che avevano punteggio specificato al minuto selezionato)")
                mostra_risultati_esatti(df_dynamic_filtered, "risultato_ft", "Finale (in partite che avevano punteggio specificato al minuto selezionato)")
                calcola_over_goals(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale (in partite che avevano punteggio specificato al minuto selezionato)")
                calcola_btts(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale (in partite che avevano punteggio specificato al minuto selezionato)")
                calcola_margine_vittoria(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale (in partite che avevano punteggio specificato al minuto selezionato)")

            else:
                if starting_score_str and "-" in starting_score_str:
                    st.warning(f"Nessuna partita trovata in cui il punteggio era {starting_score_str} al minuto {end_minute}.")
                else:
                    st.info("Inserisci un risultato di partenza valido per avviare l'analisi dinamica.")


        else:
            st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Per iniziare, seleziona un campionato dalla barra laterale.")

