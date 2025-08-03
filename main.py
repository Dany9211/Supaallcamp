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

if selected_league != "Seleziona...":
    # Filtra le squadre disponibili in base al campionato selezionato
    league_teams_df = df[df["league"] == selected_league]
    all_teams = sorted(list(set(league_teams_df['home_team'].dropna().unique()) | set(league_teams_df['away_team'].dropna().unique())))

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
        df_home = df[(df["home_team"] == home_team_selected) & (df["league"] == selected_league)]
        
        # Filtra le partite in trasferta della squadra selezionata
        df_away = df[(df["away_team"] == away_team_selected) & (df["league"] == selected_league)]
        
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
            
            # --- Funzione di supporto per ottenere il punteggio a un minuto specifico ---
            def get_scores_at_minute(df_row, selected_min, home_team_name, away_team_name):
                """
                Calcola il punteggio di una partita a un minuto specificato.
                Restituisce una stringa nel formato 'gol_home-gol_away'.
                
                NOTA IMPORTANTE: questa funzione viene usata SOLO per filtrare le partite.
                Tutte le statistiche successive si baseranno sui risultati FINALI delle
                partite filtrate, non su questo punteggio parziale.
                """
                gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                # Calcola i gol segnati dalla squadra di casa e da quella in trasferta selezionate
                # fino al minuto specificato.
                if df_row["home_team"] == home_team_name:
                    home_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                    away_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                else: # Questo caso si verifica quando la squadra "di casa" selezionata gioca in trasferta nella riga attuale
                    home_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                    away_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                
                return f"{home_goals_count}-{away_goals_count}"
            
            # --- Funzione di supporto per calcolare i gol del secondo tempo dopo un minuto specifico ---
            def get_sh_scores_dynamic(df_row, start_minute, home_team_name, away_team_name):
                """
                Calcola i gol segnati nel secondo tempo, a partire da un minuto specifico,
                per un'analisi dinamica.
                """
                gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                # Calcola i gol segnati dopo il minuto di partenza
                if df_row["home_team"] == home_team_name:
                    home_goals_sh = sum(1 for g in gol_home_minutes if g > start_minute)
                    away_goals_sh = sum(1 for g in gol_away_minutes if g > start_minute)
                else:
                    home_goals_sh = sum(1 for g in gol_away_minutes if g > start_minute)
                    away_goals_sh = sum(1 for g in gol_home_minutes if g > start_minute)

                return f"{home_goals_sh}-{away_goals_sh}"


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
                
                df_media = pd.DataFrame(data)
                st.table(df_media)

            def calcola_ht_ft_combo(df_to_analyze):
                """Calcola la distribuzione dei risultati Parziale/Finale (HT/FT)."""
                st.subheader(f"Risultato Parziale/Finale (HT/FT) ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()

                df_copy["ht_outcome"] = df_copy.apply(lambda row: get_outcome(pd.to_numeric(row["gol_home_ht"]), pd.to_numeric(row["gol_away_ht"])), axis=1)
                df_copy["ft_outcome"] = df_copy.apply(lambda row: get_outcome(pd.to_numeric(row["gol_home_ft"]), pd.to_numeric(row["gol_away_ft"])), axis=1)

                df_copy["combo"] = df_copy["ht_outcome"] + "/" + df_copy["ft_outcome"]
                combo_counts = df_copy["combo"].value_counts().reset_index()
                combo_counts.columns = ["Risultato", "Conteggio"]
                
                total = len(df_to_analyze)
                combo_counts["Percentuale %"] = (combo_counts["Conteggio"] / total * 100).round(2)
                combo_counts["Odd Minima"] = combo_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(combo_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def calcola_risultato_esatto_ht_ft(df_to_analyze):
                """Calcola la distribuzione dei risultati esatti Parziale/Finale (HT/FT)."""
                st.subheader(f"Risultato Esatto Parziale/Finale (HT/FT) ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()
                
                df_copy["exact_ht_ft"] = df_copy["risultato_ht"].astype(str) + "/" + df_copy["risultato_ft"].astype(str)
                
                exact_counts = df_copy["exact_ht_ft"].value_counts().reset_index()
                exact_counts.columns = ["Risultato Esatto HT/FT", "Conteggio"]
                
                total = len(df_to_analyze)
                exact_counts["Percentuale %"] = (exact_counts["Conteggio"] / total * 100).round(2)
                exact_counts["Odd Minima"] = exact_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(exact_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))


            def calcola_margine_vittoria(df_to_analyze, col_gol_home, col_gol_away, title):
                """Calcola il margine di vittoria (es. vittoria di 1, 2, 3+ gol)."""
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

            def calcola_first_to_score(df_to_analyze, home_team_name, away_team_name, timeframe_label, start_min=1, end_min=150):
                """Determina la prima squadra a segnare in un intervallo di tempo specifico."""
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

            def mostra_distribuzione_timeband(df_to_analyze, title, home_team_name, away_team_name, timeframe=5, start_minute=0, end_minute=90):
                """Mostra la distribuzione dei gol per intervalli di tempo in un range specifico."""
                st.subheader(f"Distribuzione Gol per Timeframe {title} ({len(df_to_analyze)} partite)")
                if df_to_analyze.empty: return

                intervalli = []
                label_intervalli = []
                
                current_start = start_minute
                while current_start < end_minute:
                    interval_end = min(current_start + timeframe -1, end_minute) if timeframe > 1 else current_start
                    intervalli.append((current_start, interval_end))
                    label_intervalli.append(f"{current_start}-{interval_end}")
                    current_start += timeframe
                
                if end_minute >= 90:
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

            def calcola_next_goal(df_to_analyze, start_minute, home_team_name, away_team_name):
                """Calcola la probabilità che la prossima squadra a segnare sia quella di casa, quella in trasferta o nessuna delle due."""
                st.subheader(f"Prossimo Gol dopo il minuto {start_minute} ({len(df_to_analyze)} partite)")
                if df_to_analyze.empty:
                    st.warning("Nessuna partita analizzabile per questa statistica.")
                    return

                risultati = {f"{home_team_name}": 0, f"{away_team_name}": 0, "Nessun altro gol": 0}

                for _, row in df_to_analyze.iterrows():
                    # Identifica correttamente i gol della squadra selezionata di casa e di trasferta
                    if row["home_team"] == home_team_name:
                        selected_home_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                        selected_away_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    else:
                        selected_home_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                        selected_away_goals_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]

                    # Trova i gol segnati dopo il minuto di partenza
                    next_home_goal_minutes = [g for g in selected_home_goals_minutes if g > start_minute]
                    next_away_goal_minutes = [g for g in selected_away_goals_minutes if g > start_minute]

                    min_next_home_goal = min(next_home_goal_minutes) if next_home_goal_minutes else float('inf')
                    min_next_away_goal = min(next_away_goal_minutes) if next_away_goal_minutes else float('inf')

                    if min_next_home_goal < min_next_away_goal:
                        risultati[f"{home_team_name}"] += 1
                    elif min_next_away_goal < min_next_home_goal:
                        risultati[f"{away_team_name}"] += 1
                    else:
                        risultati["Nessun altro gol"] += 1
                
                stats = []
                total = len(df_to_analyze)
                for esito, count in risultati.items():
                    perc = round((count / total) * 100, 2) if total > 0 else 0
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    stats.append((esito, count, perc, odd_min))
                
                df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))



            # --- ESECUZIONE E VISUALIZZAZIONE STATS PRE-PARTITA (FISSE) ---
            
            st.markdown("---")
            st.header("Statistiche Pre-partita")
            
            # Media gol fatti e subiti per singola squadra
            calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)
            
            # WinRate PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_winrate(df_combined, "risultato_ht", "PT")
            with col2:
                calcola_winrate(df_combined, "risultato_sh", "ST")
            with col3:
                calcola_winrate(df_combined, "risultato_ft", "FT")

            # Risultati Esatti PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                mostra_risultati_esatti(df_combined, "risultato_ht", "PT")
            with col2:
                mostra_risultati_esatti(df_combined, "risultato_sh", "ST")
            with col3:
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")

            # Combo HT/FT
            calcola_ht_ft_combo(df_combined)
            
            # Risultato Esatto HT/FT
            calcola_risultato_esatto_ht_ft(df_combined)

            # Margine di Vittoria PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_margine_vittoria(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_margine_vittoria(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_margine_vittoria(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            st.markdown("---")
            st.header("Statistiche sui Gol Pre-partita")
            
            # Over/Under PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_over_goals(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            # GG/NG PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_btts(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")
            
            # Clean Sheets / Fail to Score
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")

            st.markdown("---")
            st.header("Analisi Temporale dei Gol Pre-partita")

            # Prima Squadra a Segnare per timeframe
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "PT", start_min=1, end_min=45)
            with col2:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "ST", start_min=46, end_min=90)
            with col3:
                calcola_first_to_score(df_combined, home_team_selected, away_team_selected, "FT", start_min=1, end_min=150)

            # Distribuzione per intervalli di 5 e 15 minuti
            mostra_distribuzione_timeband(df_combined, "(5 Min)", home_team_selected, away_team_selected, timeframe=5)
            mostra_distribuzione_timeband(df_combined, "(15 Min)", home_team_selected, away_team_selected, timeframe=15)
            
            # --- ESECUZIONE E VISUALIZZAZIONE STATS DINAMICHE ---
            st.markdown("---")
            st.header("Statistiche Dinamiche (basate su un intervallo di minutaggio)")

            col_sliders_1, col_sliders_2 = st.columns(2)
            with col_sliders_1:
                # Cursore con due maniglie per l'intervallo di minutaggio
                start_minute = st.slider("Minuto di Riferimento", 0, 90, 45, key="minute_slider")
            with col_sliders_2:
                starting_score_str = st.text_input("Risultato di Partenza (es. 1-0)", "0-0", key="score_input")
            
            # Validazione e parsing del risultato di partenza
            try:
                if "-" in starting_score_str:
                    # FILTRAGGIO CRITICO: trova le partite che avevano il punteggio desiderato al minuto specificato
                    df_dynamic_filtered = df_combined[df_combined.apply(
                        lambda row: get_scores_at_minute(row, start_minute, home_team_selected, away_team_selected) == starting_score_str,
                        axis=1
                    )]
                    # Per la Clean Sheets, suddivido il dataframe filtrato
                    df_dynamic_home = df_dynamic_filtered[df_dynamic_filtered['home_team'] == home_team_selected]
                    df_dynamic_away = df_dynamic_filtered[df_dynamic_filtered['away_team'] == away_team_selected]
                else:
                    st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                    df_dynamic_filtered = pd.DataFrame()
                    df_dynamic_home = pd.DataFrame()
                    df_dynamic_away = pd.DataFrame()
            except ValueError:
                st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                df_dynamic_filtered = pd.DataFrame()
                df_dynamic_home = pd.DataFrame()
                df_dynamic_away = pd.DataFrame()
            
            if not df_dynamic_filtered.empty:
                st.write(f"Analisi basata su **{len(df_dynamic_filtered)}** partite in cui il punteggio era **{starting_score_str}** al minuto **{start_minute}**.")
                
                # Prova visiva che la logica è corretta
                with st.expander("Visualizzazione Partite Filtrate"):
                    st.write("Di seguito sono mostrate le partite che corrispondono al filtro dinamico e i loro risultati finali. Tutte le statistiche successive si basano su questi risultati.")
                    st.dataframe(df_dynamic_filtered[['home_team', 'away_team', 'risultato_ft', 'minutaggio_gol', 'minutaggio_gol_away']])

                # --- STATISTICHE DINAMICHE ---
                st.markdown("---")
                st.subheader("Statistiche sul Risultato Finale")
                calcola_winrate(df_dynamic_filtered, "risultato_ft", "Finale")
                mostra_risultati_esatti(df_dynamic_filtered, "risultato_ft", "Finale")
                calcola_margine_vittoria(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale")
                
                st.markdown("---")
                st.subheader("Statistiche sui Gol del Secondo Tempo (dopo il minuto di riferimento)")
                # Calcola dinamicamente i gol del secondo tempo per il DataFrame filtrato
                df_dynamic_filtered["risultato_sh_dynamic"] = df_dynamic_filtered.apply(
                    lambda row: get_sh_scores_dynamic(row, start_minute, home_team_selected, away_team_selected), axis=1)

                calcola_winrate(df_dynamic_filtered, "risultato_sh_dynamic", "Secondo Tempo")
                mostra_risultati_esatti(df_dynamic_filtered, "risultato_sh_dynamic", "Secondo Tempo")
                calcola_margine_vittoria(df_dynamic_filtered, "gol_home_sh", "gol_away_sh", "Secondo Tempo")


                st.markdown("---")
                st.subheader("Statistiche sui Gol Totali")
                calcola_over_goals(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale")
                calcola_btts(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale")
                calcola_clean_sheets(df_dynamic_home, df_dynamic_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "Finale")

                st.markdown("---")
                st.subheader("Analisi Temporale Dinamica")
                calcola_first_to_score(df_dynamic_filtered, home_team_selected, away_team_selected, f"Dopo il Minuto {start_minute}", start_min=start_minute + 1, end_min=150)
                
                # Selettore per timeframe dinamico
                timeframe_dynamic_options = [5, 15]
                timeframe_dynamic = st.selectbox("Seleziona l'intervallo di minutaggio per la distribuzione dei gol", timeframe_dynamic_options, key="timeframe_dynamic")
                mostra_distribuzione_timeband(df_dynamic_filtered, f"(Dopo il Minuto {start_minute})", home_team_selected, away_team_selected, timeframe=timeframe_dynamic, start_minute=start_minute + 1, end_minute=90)


            else:
                if starting_score_str and "-" in starting_score_str:
                    st.warning(f"Nessuna partita trovata in cui il punteggio era {starting_score_str} al minuto {start_minute}.")
                else:
                    st.info("Inserisci un risultato di partenza valido per avviare l'analisi dinamica.")

        else:
            st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Per iniziare, seleziona un campionato dalla barra laterale.")
