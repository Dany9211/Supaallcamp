import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

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
            """Calcola e mostra il WinRate basato sui risultati complessivi."""
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

        def calcola_media_gol(df_home, df_away, home_team_name, away_team_name):
            """Calcola la media gol fatti e subiti per le squadre selezionate."""
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
                "Gol Subiti (FT)": [f"{home_conceded_ft:.2f}", f"{away_conceded_ft:.2f}"]
            }
            
            df_stats = pd.DataFrame(data)
            st.dataframe(df_stats)

        def calcola_first_to_score(df_to_analyze, home_team_name, away_team_name):
            """Calcola quale squadra ha segnato per prima."""
            st.subheader(f"First to Score ({len(df_to_analyze)} partite)")
            
            home_first_score_count = 0
            away_first_score_count = 0
            no_score_count = 0
            
            for _, row in df_to_analyze.iterrows():
                # Estrai i minuti dei gol di casa e trasferta
                home_goals = [int(m) for m in str(row.get("minutaggio_gol")).split(';') if m.isdigit()]
                away_goals = [int(m) for m in str(row.get("minutaggio_gol_away")).split(';') if m.isdigit()]
                
                # Trova il primo gol
                first_home_goal = min(home_goals) if home_goals else float('inf')
                first_away_goal = min(away_goals) if away_goals else float('inf')

                if first_home_goal < first_away_goal:
                    home_first_score_count += 1
                elif first_away_goal < first_home_goal:
                    away_first_score_count += 1
                else: # Se i minuti sono uguali o non ci sono gol
                    no_score_count += 1

            total_matches = len(df_to_analyze)
            
            data = {
                "Esito": [f"Primo gol {home_team_name}", f"Primo gol {away_team_name}", "Nessun gol"],
                "Conteggio": [home_first_score_count, away_first_score_count, no_score_count],
                "Percentuale %": [
                    round((home_first_score_count / total_matches) * 100, 2) if total_matches > 0 else 0,
                    round((away_first_score_count / total_matches) * 100, 2) if total_matches > 0 else 0,
                    round((no_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
                ]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_progressioni_gol(df_to_analyze, home_team_name, away_team_name):
            """Analizza le progressioni di punteggio comuni come 1-0, 1-1, 2-0."""
            st.subheader("Analisi Progressioni Gol (Risultato Finale)")
            
            # Filtra le partite con punteggio valido
            df_valid = df_to_analyze[
                df_to_analyze["risultato_ft"].notna() & 
                (df_to_analyze["risultato_ft"].str.contains("-"))
            ].copy()
            
            total_matches = len(df_valid)
            if total_matches == 0:
                st.info("Nessuna partita valida per analizzare le progressioni di gol.")
                return

            stats = {
                "Final Score 1-0": 0,
                "Final Score 0-1": 0,
                "Final Score 1-1": 0,
                "Final Score 2-0": 0,
                "Final Score 0-2": 0
            }
            
            for _, row in df_valid.iterrows():
                home_score, away_score = map(int, row["risultato_ft"].split('-'))
                
                if home_score == 1 and away_score == 0:
                    stats["Final Score 1-0"] += 1
                elif home_score == 0 and away_score == 1:
                    stats["Final Score 0-1"] += 1
                elif home_score == 1 and away_score == 1:
                    stats["Final Score 1-1"] += 1
                elif home_score == 2 and away_score == 0:
                    stats["Final Score 2-0"] += 1
                elif home_score == 0 and away_score == 2:
                    stats["Final Score 0-2"] += 1
                    
            prog_data = []
            for esito, count in stats.items():
                perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                prog_data.append([esito, count, perc, odd_min])
            
            df_prog = pd.DataFrame(prog_data, columns=["Progressione", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_prog.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_to_score_ht_ft(df_home_to_analyze, df_away_to_analyze, home_team_name, away_team_name):
            """Calcola quante volte una squadra ha segnato sia HT che FT."""
            st.subheader(f"To Score HT & FT")
            
            # Calcolo per la squadra di casa
            home_ht_ft_score_count = 0
            total_home_matches = len(df_home_to_analyze)
            for _, row in df_home_to_analyze.iterrows():
                if pd.to_numeric(row["gol_home_ht"], errors='coerce').fillna(0) > 0 and pd.to_numeric(row["gol_home_sh"], errors='coerce').fillna(0) > 0:
                    home_ht_ft_score_count += 1
            home_perc = round((home_ht_ft_score_count / total_home_matches) * 100, 2) if total_home_matches > 0 else 0

            # Calcolo per la squadra in trasferta
            away_ht_ft_score_count = 0
            total_away_matches = len(df_away_to_analyze)
            for _, row in df_away_to_analyze.iterrows():
                if pd.to_numeric(row["gol_away_ht"], errors='coerce').fillna(0) > 0 and pd.to_numeric(row["gol_away_sh"], errors='coerce').fillna(0) > 0:
                    away_ht_ft_score_count += 1
            away_perc = round((away_ht_ft_score_count / total_away_matches) * 100, 2) if total_away_matches > 0 else 0

            data = {
                "Squadra": [home_team_name, away_team_name],
                "Scored HT & FT": [home_ht_ft_score_count, away_ht_ft_score_count],
                "Percentuale %": [home_perc, away_perc]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        
        # --- FUNZIONE DI SUPPORTO PER CALCOLARE I GOL AL MINUTO X ---
        def get_scores_at_minute(df_row, selected_min, home_team_name, away_team_name):
            """
            Calcola il punteggio di una partita a un minuto specifico.
            Restituisce una stringa nel formato 'gol_home-gol_away'.
            """
            gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            
            # Calcola i gol segnati dalla squadra di casa e da quella in trasferta selezionate
            # fino al minuto specificato.
            if df_row["home_team"] == home_team_name:
                home_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                away_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
            else:
                home_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                away_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
            
            return f"{home_goals_count}-{away_goals_count}"
        
        # --- FUNZIONE PER ANALISI TIMEBAND AGGIORNATA ---
        def calcola_stats_timeband(df_to_analyze, start_min, end_min, timeband_length, home_team_name, away_team_name):
            """
            Analizza le statistiche dei gol in timeband specifici, mostrando la percentuale
            di partite con almeno 1 o 2 gol, e i gol fatti/subiti per squadra.
            """
            st.subheader(f"Analisi per Timeband di {timeband_length} minuti")
            
            timeband_data = []
            
            num_partite = len(df_to_analyze)
            if num_partite == 0:
                st.warning("Nessuna partita trovata per questa analisi di timeband.")
                return
            
            for t_start in range(start_min, end_min, timeband_length):
                t_end = t_start + timeband_length
                if t_end > end_min:
                    t_end = end_min
                
                # Inizializza i contatori per l'intervallo corrente
                total_home_goals_in_interval = 0
                total_away_goals_in_interval = 0
                games_with_1_plus_goals = 0
                games_with_2_plus_goals = 0
                
                for _, row in df_to_analyze.iterrows():
                    all_home_goal_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                    all_away_goal_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    
                    # Controlla se la riga è una partita di casa o trasferta per la squadra selezionata
                    if row["home_team"] == home_team_name:
                        goals_home_selected = sum(1 for g in all_home_goal_minutes if t_start < g <= t_end)
                        goals_away_selected = sum(1 for g in all_away_goal_minutes if t_start < g <= t_end)
                    else:
                        goals_home_selected = sum(1 for g in all_away_goal_minutes if t_start < g <= t_end)
                        goals_away_selected = sum(1 for g in all_home_goal_minutes if t_start < g <= t_end)
                    
                    total_goals_in_interval = goals_home_selected + goals_away_selected
                    
                    if total_goals_in_interval >= 1:
                        games_with_1_plus_goals += 1
                    if total_goals_in_interval >= 2:
                        games_with_2_plus_goals += 1
                        
                    total_home_goals_in_interval += goals_home_selected
                    total_away_goals_in_interval += goals_away_selected

                home_goals_stats = f"Fatti: {total_home_goals_in_interval} / Subiti: {total_away_goals_in_interval}"
                away_goals_stats = f"Fatti: {total_away_goals_in_interval} / Subiti: {total_home_goals_in_interval}"
                
                perc_1_plus_goals = round((games_with_1_plus_goals / num_partite) * 100, 2)
                perc_2_plus_goals = round((games_with_2_plus_goals / num_partite) * 100, 2)
                
                timeband_data.append([
                    f"{t_start}-{t_end}'",
                    perc_1_plus_goals,
                    perc_2_plus_goals,
                    home_goals_stats,
                    away_goals_stats
                ])
                
            df_timeband_stats = pd.DataFrame(timeband_data, columns=[
                "Intervallo",
                "Almeno 1 Gol %",
                "Almeno 2 Gol %",
                f"Gol Fatti/Subiti ({home_team_name})",
                f"Gol Fatti/Subiti ({away_team_name})"
            ])
            
            # Applica una formattazione condizionale per evidenziare le percentuali
            st.dataframe(
                df_timeband_stats.style.background_gradient(
                    cmap='RdYlGn', 
                    subset=["Almeno 1 Gol %", "Almeno 2 Gol %"]
                ).format(
                    {'Almeno 1 Gol %': "{:.2f}%", 'Almeno 2 Gol %': "{:.2f}%"}
                )
            )
        
        # --- SEZIONE ANALISI PRE-PARTITA ---
        st.header("Analisi Pre-Partita")
        st.markdown("---")

        with st.expander("Statistiche Primo Tempo (HT)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                calcola_winrate(df_combined, "risultato_ht", "HT")
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected)
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                mostra_risultati_esatti(df_combined, "risultato_ht", "HT")
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "HT")
        
        with st.expander("Statistiche Fine Partita (FT)", expanded=True):
            if not df_combined.empty:
                calcola_winrate(df_combined, "risultato_ft", "FT")
                calcola_to_score_ht_ft(df_home, df_away, home_team_selected, away_team_selected)
                calcola_progressioni_gol(df_combined, home_team_selected, away_team_selected)
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")

        with st.expander("Analisi Timeband Pre-Partita", expanded=False):
            if not df_combined.empty:
                calcola_stats_timeband(df_combined, 0, 90, 5, home_team_selected, away_team_selected)
                st.markdown("---")
                calcola_stats_timeband(df_combined, 0, 90, 15, home_team_selected, away_team_selected)

        # --- SEZIONE ANALISI DINAMICA IN-MATCH ---
        st.header("Analisi Dinamica In-Match")
        st.markdown("---")
        
        # Filtri dinamici
        st.subheader("Filtri Dinamici")
        col1_dyn, col2_dyn = st.columns(2)
        with col1_dyn:
            # Seleziona tutti i possibili risultati parziali per l'input utente
            all_partial_results = [f"{h}-{a}" for h in range(10) for a in range(10)]
            selected_start_result = st.selectbox("Risultato di Partenza", all_partial_results)
        
        with col2_dyn:
            start_min, end_min = st.slider(
                'Seleziona intervallo di tempo (minuti)',
                0, 90, (45, 90)
            )
        
        st.markdown("---")


        # Filtra il DataFrame combinato in base all'intervallo di tempo e al risultato di partenza
        filtered_df_dynamic = pd.DataFrame()
        if start_min < end_min:
            for _, row in df_combined.iterrows():
                # Calcolo del risultato al minuto 'start_min'
                risultato_attuale = get_scores_at_minute(row, start_min, home_team_selected, away_team_selected)
                if risultato_attuale == selected_start_result:
                    # Crea un nuovo DataFrame con i risultati "dinamici" per ogni riga
                    home_goals_at_end = sum(1 for g in [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()] if g <= end_min)
                    away_goals_at_end = sum(1 for g in [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()] if g <= end_min)
                    
                    row_copy = row.copy()
                    
                    if row["home_team"] == home_team_selected:
                        row_copy["gol_home_ft"] = home_goals_at_end
                        row_copy["gol_away_ft"] = away_goals_at_end
                    else: # Quando la squadra selezionata gioca fuori casa
                        row_copy["gol_home_ft"] = away_goals_at_end
                        row_copy["gol_away_ft"] = home_goals_at_end

                    row_copy["risultato_ft"] = f"{int(row_copy['gol_home_ft'])}-{int(row_copy['gol_away_ft'])}"
                    filtered_df_dynamic = pd.concat([filtered_df_dynamic, row_copy.to_frame().T], ignore_index=True)


        if not filtered_df_dynamic.empty:
            st.subheader(f"Statistiche basate su risultato {selected_start_result} al {start_min}° minuto")
            # Calcolo next goal
            risultati = {"Next Gol: Home": 0, "Next Gol: Away": 0, "Nessun prossimo gol": 0}
            totale_partite = len(filtered_df_dynamic)
            
            for _, row in filtered_df_dynamic.iterrows():
                # Calcola il primo gol segnato nell'intervallo [start_min, end_min]
                gol_home_dinamici = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit() and start_min < int(x) <= end_min]
                gol_away_dinamici = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit() and start_min < int(x) <= end_min]
                
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
        
        else:
            st.warning("Nessuna partita trovata per l'analisi dinamica con i filtri selezionati.")

        # Visualizzazione delle statistiche in base alla struttura a scomparsa
        with st.expander(f"Statistiche Fine Partita (FT) basate sul risultato al {end_min}° minuto", expanded=True):
            if not filtered_df_dynamic.empty:
                calcola_winrate(filtered_df_dynamic, "risultato_ft", f"al {end_min}° min")
                mostra_risultati_esatti(filtered_df_dynamic, "risultato_ft", f"al {end_min}° min")
                calcola_over_goals(filtered_df_dynamic, "gol_home_ft", "gol_away_ft", f"al {end_min}° min")
                calcola_btts(filtered_df_dynamic, "gol_home_ft", "gol_away_ft", f"al {end_min}° min")
            else:
                st.warning("Nessuna partita trovata per le statistiche con i filtri selezionati.")
            
        with st.expander("Statistiche Primo Tempo (HT)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                calcola_winrate(df_combined, "risultato_ht", "HT")
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected)
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                mostra_risultati_esatti(df_combined, "risultato_ht", "HT")
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "HT")
            else:
                st.warning("Nessuna partita trovata per le statistiche del Primo Tempo.")

        # --- NUOVE SEZIONI DI ANALISI TIMEBAND ---
        if not filtered_df_dynamic.empty:
            st.markdown("---")
            st.subheader("Analisi Dettagliata per Intervalli di Tempo")
            
            with st.expander("Timeband 5 Minuti", expanded=True):
                calcola_stats_timeband(filtered_df_dynamic, start_min, end_min, 5, home_team_selected, away_team_selected)
                
            with st.expander("Timeband 15 Minuti", expanded=False):
                calcola_stats_timeband(filtered_df_dynamic, start_min, end_min, 15, home_team_selected, away_team_selected)
            
    else:
        st.warning("Seleziona una squadra 'CASA' e una 'TRASFERTA' per avviare l'analisi.")
else:
    st.info("Seleziona un campionato e due squadre per iniziare.")
