import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict

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

# --- Aggiunta di colonne calcolate per facilitare le analisi ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

# Calcolo dei gol e del risultato del Secondo Tempo (SH)
if all(col in df.columns for col in ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]):
    df["gol_home_sh"] = pd.to_numeric(df["gol_home_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_home_ht"], errors='coerce').fillna(0)
    df["gol_away_sh"] = pd.to_numeric(df["gol_away_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_away_ht"], errors='coerce').fillna(0)
    df["risultato_sh"] = df["gol_home_sh"].astype(int).astype(str) + "-" + df["gol_away_sh"].astype(int).astype(str)
else:
    st.sidebar.warning("Colonne mancanti per il calcolo delle statistiche del Secondo Tempo.")

# Identifica la colonna della data per il filtro delle ultime N partite
date_col_name = "data" if "data" in df.columns else "date" if "date" in df.columns else None
has_date_column = date_col_name is not None
if has_date_column:
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
else:
    st.sidebar.warning("Le colonne 'data' o 'date' non sono presenti. Il filtro per le ultime N partite non sarà disponibile.")


# --- Selettori nella Sidebar per la configurazione dell'analisi ---
st.sidebar.header("Seleziona Squadre")

# 1. Selettore Campionato
leagues = ["Seleziona..."] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)

df_filtered_by_league = df if selected_league == "Seleziona..." else df[df["league"] == selected_league]
all_teams = sorted(list(set(df_filtered_by_league['home_team'].dropna().unique()) | set(df_filtered_by_league['away_team'].dropna().unique())))

# 2. Selettori Squadre
home_team_selected = st.sidebar.selectbox("Seleziona Squadra CASA", ["Seleziona..."] + all_teams)
away_team_selected = st.sidebar.selectbox("Seleziona Squadra TRASFERTA", ["Seleziona..."] + all_teams)

if home_team_selected != "Seleziona..." and away_team_selected != "Seleziona...":
    
    # --- Selettore per le ultime N partite ---
    num_to_filter = None
    if has_date_column:
        st.sidebar.header("Filtra Partite per Numero")
        num_partite_options = ["Tutte", "Ultime 5", "Ultime 10", "Ultime 15", "Ultime 20", "Ultime 30", "Ultime 40", "Ultime 50"]
        selected_num_partite_str = st.sidebar.selectbox("Numero di partite da analizzare", num_partite_options)
        
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

        def calcola_winrate(df_to_analyze, col_risultato, title):
            """
            Calcola e mostra il WinRate basato sui risultati complessivi.
            Per l'analisi dinamica, questo si riferisce al risultato FT delle partite filtrate.
            """
            st.subheader(f"WinRate {title} ({len(df_to_analyze)} partite)")
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

        def mostra_risultati_esatti(df_to_analyze, col_risultato, titolo):
            """Mostra la distribuzione dei risultati esatti."""
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
            """Calcola la distribuzione Over/Under goals."""
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
            """Calcola la statistica GG/NG."""
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
            """Calcola le Clean Sheets e il Fallimento a Segnare per le squadre selezionate."""
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

        def calcola_media_gol(df_home, df_away, home_team_name, away_team_name, title, col_home_goals, col_away_goals):
            """Calcola la media gol fatti e subiti per le squadre selezionate in un intervallo specifico (HT, FT, etc)."""
            st.subheader(f"Media Gol Fatti e Subiti {title} (Home: {len(df_home)} partite, Away: {len(df_away)} partite)")
            
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
            
        def calcola_primo_gol_stats(df_combined, home_team_name, away_team_name, title):
            """Calcola le probabilità di chi segna il primo gol della partita/periodo."""
            st.subheader(f"Chi segna il primo gol? {title} ({len(df_combined)} partite)")
            
            first_goal_stats = defaultdict(int)
            
            total_matches = len(df_combined)
            
            for _, row in df_combined.iterrows():
                home_goals = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                away_goals = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                first_goal_minute_home = min(home_goals) if home_goals else float('inf')
                first_goal_minute_away = min(away_goals) if away_goals else float('inf')
                
                # Chi ha segnato per primo?
                if first_goal_minute_home < first_goal_minute_away:
                    if row["home_team"] == home_team_name:
                        first_goal_stats[f"{home_team_name}"] += 1
                    else:
                        first_goal_stats[f"{away_team_name}"] += 1
                elif first_goal_minute_away < first_goal_minute_home:
                    if row["away_team"] == home_team_name:
                        first_goal_stats[f"{home_team_name}"] += 1
                    else:
                        first_goal_stats[f"{away_team_name}"] += 1
                else:
                    first_goal_stats["Nessun gol"] += 1
            
            data = []
            for team, count in first_goal_stats.items():
                perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                data.append([team, count, perc, odd_min])
            
            df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_timeband_stats(df_to_analyze, time_bands, title, home_team_name, away_team_name):
            """
            Calcola le statistiche dei gol segnati per intervalli di tempo specifici.
            Inclusa la percentuale di partite con almeno 1 e 2 gol e colorazione a gradiente.
            """
            total_matches = len(df_to_analyze)
            st.subheader(f"Statistiche per intervallo: {title} ({total_matches} partite)")
            
            stats = []
            
            for start, end in time_bands:
                
                home_scored_count = 0
                home_conceded_count = 0
                away_scored_count = 0
                away_conceded_count = 0
                matches_at_least_1_goal = 0
                matches_at_least_2_goals = 0
                
                for _, row in df_to_analyze.iterrows():
                    
                    # Estrae i minuti dei gol, gestendo i valori NaN e stringhe vuote
                    home_goal_minutes_str = str(row.get("minutaggio_gol", ""))
                    away_goal_minutes_str = str(row.get("minutaggio_gol_away", ""))

                    home_goals_in_match = [int(x) for x in home_goal_minutes_str.split(";") if x.isdigit()]
                    away_goals_in_match = [int(x) for x in away_goal_minutes_str.split(";") if x.isdigit()]
                    
                    goals_by_home_team_in_band = [g for g in home_goals_in_match if start <= g < end]
                    goals_by_away_team_in_band = [g for g in away_goals_in_match if start <= g < end]
                    
                    # Calcola i gol fatti e subiti in base alla squadra selezionata
                    if row["home_team"] == home_team_name:
                        home_scored_count += len(goals_by_home_team_in_band)
                        home_conceded_count += len(goals_by_away_team_in_band)
                        away_scored_count += len(goals_by_away_team_in_band)
                        away_conceded_count += len(goals_by_home_team_in_band)
                    else: # Squadra selezionata gioca fuori casa
                        home_scored_count += len(goals_by_away_team_in_band)
                        home_conceded_count += len(goals_by_home_team_in_band)
                        away_scored_count += len(goals_by_home_team_in_band)
                        away_conceded_count += len(goals_by_away_team_in_band)
                        
                    # Calcola le metriche di gol totali per la banda temporale
                    total_goals_in_band = len(goals_by_home_team_in_band) + len(goals_by_away_team_in_band)
                    if total_goals_in_band >= 1:
                        matches_at_least_1_goal += 1
                    if total_goals_in_band >= 2:
                        matches_at_least_2_goals += 1

                if total_matches > 0:
                    perc_1_goal = round((matches_at_least_1_goal / total_matches) * 100, 2)
                    odd_min_1_goal = round(100 / perc_1_goal, 2) if perc_1_goal > 0 else "-"
                    perc_2_goals = round((matches_at_least_2_goals / total_matches) * 100, 2)
                    odd_min_2_goals = round(100 / perc_2_goals, 2) if perc_2_goals > 0 else "-"
                else:
                    perc_1_goal, odd_min_1_goal = 0, "-"
                    perc_2_goals, odd_min_2_goals = 0, "-"
                
                stats.append([
                    f"{start}'-{end}'", 
                    f"{home_scored_count} / {home_conceded_count}", 
                    f"{away_scored_count} / {away_conceded_count}", 
                    perc_1_goal,
                    odd_min_1_goal,
                    perc_2_goals,
                    odd_min_2_goals
                ])

            df_stats = pd.DataFrame(stats, columns=[
                "Intervallo", 
                f"Gol Fatti/Subiti {home_team_name}", 
                f"Gol Fatti/Subiti {away_team_name}", 
                "% Winrate >= 1 Gol", 
                "Odd Minima >= 1 Gol",
                "% Winrate >= 2 Gol",
                "Odd Minima >= 2 Gol"
            ])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=[
                '% Winrate >= 1 Gol', '% Winrate >= 2 Gol'
            ]))
            
        # --- SEZIONE ANALISI PRE-PARTITA ---
        st.header("Analisi Pre-Partita")
        st.markdown("---")

        with st.expander("Statistiche Primo Tempo (HT)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "HT", "gol_home_ht", "gol_away_ht")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                calcola_primo_gol_stats(df_combined, home_team_selected, away_team_selected, "HT")
                calcola_winrate(df_combined, "risultato_ht", "HT")
                mostra_risultati_esatti(df_combined, "risultato_ht", "HT")
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "HT")
        
        with st.expander("Statistiche Fine Partita (FT)", expanded=True):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "FT", "gol_home_ft", "gol_away_ft")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")
                calcola_primo_gol_stats(df_combined, home_team_selected, away_team_selected, "FT")
                calcola_winrate(df_combined, "risultato_ft", "FT")
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")
        
        # --- Nuova sezione per le time bands pre-partita (5 e 15 min) ---
        with st.expander("Analisi per Bande Temporali (Pre-Partita)", expanded=True):
            if not df_combined.empty:
                st.subheader("Bande temporali ogni 5 minuti (0-90)")
                time_bands_5min = [(i, i + 5) for i in range(0, 90, 5)]
                calcola_timeband_stats(df_combined, time_bands_5min, f"Pre-partita 5 min", home_team_selected, away_team_selected)
                
                st.subheader("Bande temporali ogni 15 minuti (0-90)")
                time_bands_15min = [(i, i + 15) for i in range(0, 90, 15)]
                calcola_timeband_stats(df_combined, time_bands_15min, f"Pre-partita 15 min", home_team_selected, away_team_selected)


        # --- SEZIONE ANALISI DINAMICA IN-MATCH ---
        st.header("Analisi Dinamica In-Match")
        st.markdown("---")
        
        # Filtri dinamici
        st.subheader("Filtri Dinamici")
        col1_dyn, col2_dyn = st.columns(2)
        with col1_dyn:
            all_partial_results = [f"{h}-{a}" for h in range(10) for a in range(10)]
            selected_start_result = st.selectbox("Risultato di Partenza", all_partial_results)
        
        with col2_dyn:
            start_min, end_min = st.slider(
                'Seleziona intervallo di tempo (minuti)',
                0, 90, (45, 90)
            )
        
        st.markdown("---")

        def get_scores_at_minute(df_row, selected_min, home_team_name, away_team_name):
            """
            Calcola il punteggio di una partita a un minuto specifico.
            Restituisce una stringa nel formato 'gol_home-gol_away'.
            """
            gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            if df_row["home_team"] == home_team_name:
                home_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                away_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
            else:
                home_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                away_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
            
            return f"{home_goals_count}-{away_goals_count}"

        filtered_df_dynamic = pd.DataFrame()
        if start_min < end_min:
            for _, row in df_combined.iterrows():
                risultato_attuale = get_scores_at_minute(row, start_min, home_team_selected, away_team_selected)
                if risultato_attuale == selected_start_result:
                    row_copy = row.copy()
                    row_copy["risultato_ft"] = row["risultato_ft"]
                    row_copy["gol_home_ft"] = row["gol_home_ft"]
                    row_copy["gol_away_ft"] = row["gol_away_ft"]
                    filtered_df_dynamic = pd.concat([filtered_df_dynamic, row_copy.to_frame().T], ignore_index=True)

        if not filtered_df_dynamic.empty:
            st.subheader(f"Statistiche basate su risultato {selected_start_result} al {start_min}° minuto ({len(filtered_df_dynamic)} partite)")
            
            # Filtra il dataframe dinamico per le partite in casa e in trasferta
            filtered_df_dynamic_home = filtered_df_dynamic[filtered_df_dynamic["home_team"] == home_team_selected]
            filtered_df_dynamic_away = filtered_df_dynamic[filtered_df_dynamic["away_team"] == away_team_selected]
            
            # Calcolo next goal
            risultati = {"Next Gol: Home": 0, "Next Gol: Away": 0, "Nessun prossimo gol": 0}
            totale_partite = len(filtered_df_dynamic)
            
            for _, row in filtered_df_dynamic.iterrows():
                gol_home_dinamici = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit() and start_min <= int(x) <= end_min]
                gol_away_dinamici = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit() and start_min <= int(x) <= end_min]
                
                next_home_goal = min(gol_home_dinamici) if gol_home_dinamici else float('inf')
                next_away_goal = min(gol_away_dinamici) if gol_away_dinamici else float('inf')

                if row["home_team"] == home_team_selected:
                    if next_home_goal < next_away_goal:
                        risultati["Next Gol: Home"] += 1
                    elif next_away_goal < next_home_goal:
                        risultati["Next Gol: Away"] += 1
                    else:
                        if next_home_goal == float('inf'):
                            risultati["Nessun prossimo gol"] += 1
                else: # Squadra selezionata gioca fuori casa
                    if next_away_goal < next_home_goal:
                        risultati["Next Gol: Home"] += 1
                    elif next_home_goal < next_away_goal:
                        risultati["Next Gol: Away"] += 1
                    else:
                        if next_away_goal == float('inf'):
                            risultati["Nessun prossimo gol"] += 1

            stats = []
            for esito, count in risultati.items():
                perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                stats.append((esito, count, perc, odd_min))
            
            df_next_goal = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_next_goal.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            with st.expander(f"Statistiche Primo Tempo (HT) per le partite filtrate", expanded=False):
                if not filtered_df_dynamic.empty:
                    calcola_media_gol(filtered_df_dynamic_home, filtered_df_dynamic_away, home_team_selected, away_team_selected, "HT", "gol_home_ht", "gol_away_ht")
                    calcola_clean_sheets(filtered_df_dynamic_home, filtered_df_dynamic_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                    calcola_winrate(filtered_df_dynamic, "risultato_ht", f"con risultato {selected_start_result} al {start_min}° min")
                    mostra_risultati_esatti(filtered_df_dynamic, "risultato_ht", f"con risultato {selected_start_result} al {start_min}° min")
                    calcola_over_goals(filtered_df_dynamic, "gol_home_ht", "gol_away_ht", f"con risultato {selected_start_result} al {start_min}° min")
                    calcola_btts(filtered_df_dynamic, "gol_home_ht", "gol_away_ht", f"con risultato {selected_start_result} al {start_min}° min")
            
            with st.expander(f"Statistiche Fine Partita (FT) per le partite filtrate", expanded=True):
                if not filtered_df_dynamic.empty:
                    calcola_media_gol(filtered_df_dynamic_home, filtered_df_dynamic_away, home_team_selected, away_team_selected, "FT", "gol_home_ft", "gol_away_ft")
                    calcola_clean_sheets(filtered_df_dynamic_home, filtered_df_dynamic_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")
                    # La logica seguente calcola il winrate FT solo per le partite che corrispondono ai filtri dinamici
                    calcola_winrate(filtered_df_dynamic, "risultato_ft", f"con risultato {selected_start_result} al {start_min}° min")
                    mostra_risultati_esatti(filtered_df_dynamic, "risultato_ft", f"con risultato {selected_start_result} al {start_min}° min")
                    calcola_over_goals(filtered_df_dynamic, "gol_home_ft", "gol_away_ft", f"con risultato {selected_start_result} al {start_min}° min")
                    calcola_btts(filtered_df_dynamic, "gol_home_ft", "gol_away_ft", f"con risultato {selected_start_result} al {start_min}° min")
                else:
                    st.warning("Nessuna partita trovata per le statistiche con i filtri selezionati.")
            
            # --- Nuova sezione per le time bands dinamiche (5 e 15 min) ---
            with st.expander("Analisi per Bande Temporali (Dinamica)", expanded=True):
                if not filtered_df_dynamic.empty:
                    
                    st.subheader(f"Bande temporali ogni 5 minuti (dal {start_min}° min)")
                    time_bands_5min_dynamic = [(i, i + 5) for i in range(start_min, 90, 5)]
                    if time_bands_5min_dynamic:
                        calcola_timeband_stats(filtered_df_dynamic, time_bands_5min_dynamic, f"Dinamica 5 min", home_team_selected, away_team_selected)
                    else:
                        st.info("Nessun intervallo di 5 minuti dopo il minuto iniziale selezionato.")

                    st.subheader(f"Bande temporali ogni 15 minuti (dal {start_min}° min)")
                    time_bands_15min_dynamic = [(i, i + 15) for i in range(start_min, 90, 15)]
                    if time_bands_15min_dynamic:
                        calcola_timeband_stats(filtered_df_dynamic, time_bands_15min_dynamic, f"Dinamica 15 min", home_team_selected, away_team_selected)
                    else:
                        st.info("Nessun intervallo di 15 minuti dopo il minuto iniziale selezionato.")
                else:
                    st.warning("Nessuna partita trovata per l'analisi dinamica per bande temporali.")
            
        else:
            st.warning("Nessuna partita trovata per l'analisi dinamica con i filtri selezionati.")

    else:
        st.warning("Seleziona una squadra 'CASA' e una 'TRASFERTA' per avviare l'analisi.")
else:
    st.info("Seleziona un campionato e due squadre per iniziare.")
