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
        
        # --- FILTRAGGIO E COMBINAZIONE DATI ---
        
        # Prendi tutte le partite in casa della squadra di casa
        df_home = df[(df["home_team"] == home_team_selected) & (df["league"] == selected_league)]
        
        # Prendi tutte le partite in trasferta della squadra in trasferta
        df_away = df[(df["away_team"] == away_team_selected) & (df["league"] == selected_league)]
        
        # Combina i due DataFrame
        df_combined = pd.concat([df_home, df_away], ignore_index=True)
        
        st.header(f"Analisi Combinata: {home_team_selected} (Casa) vs {away_team_selected} (Trasferta)")
        st.write(f"Basata su **{len(df_home)}** partite casalinghe di '{home_team_selected}' e **{len(df_away)}** partite in trasferta di '{away_team_selected}'.")
        st.write(f"**Totale Partite Analizzate:** {len(df_combined)}")

        if not df_combined.empty:
            
            # --- INIZIO FUNZIONI STATISTICHE (copiate e adattate) ---

            def calcola_winrate(df_to_analyze, col_risultato, title):
                st.subheader(f"WinRate {title}")
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
                st.subheader(f"Over/Under Goals {title}")
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

            def calcola_btts_squadre(df_home, df_away, col_gol_home, col_gol_away, title):
                st.subheader(f"Entrambe le Squadre a Segno (GG/NG) {title}")
                
                total_home_matches = len(df_home)
                total_away_matches = len(df_away)
                
                # Calcolo per la squadra di casa
                df_home_copy = df_home.copy()
                df_home_copy[col_gol_home] = pd.to_numeric(df_home_copy[col_gol_home], errors='coerce').fillna(0)
                df_home_copy[col_gol_away] = pd.to_numeric(df_home_copy[col_gol_away], errors='coerce').fillna(0)
                btts_home_count = ((df_home_copy[col_gol_home] > 0) & (df_home_copy[col_gol_away] > 0)).sum()
                btts_home_perc = round((btts_home_count / total_home_matches) * 100, 2) if total_home_matches > 0 else 0
                
                # Calcolo per la squadra in trasferta
                df_away_copy = df_away.copy()
                df_away_copy[col_gol_home] = pd.to_numeric(df_away_copy[col_gol_home], errors='coerce').fillna(0)
                df_away_copy[col_gol_away] = pd.to_numeric(df_away_copy[col_gol_away], errors='coerce').fillna(0)
                btts_away_count = ((df_away_copy[col_gol_home] > 0) & (df_away_copy[col_gol_away] > 0)).sum()
                btts_away_perc = round((btts_away_count / total_away_matches) * 100, 2) if total_away_matches > 0 else 0

                # Calcolo per la media combinata
                total_combined_matches = len(df_home) + len(df_away)
                btts_combined_count = btts_home_count + btts_away_count
                btts_combined_perc = round((btts_combined_count / total_combined_matches) * 100, 2) if total_combined_matches > 0 else 0
                
                data = {
                    "Squadra": [f"GG (Sì) {home_team_selected}", f"GG (Sì) {away_team_selected}", "GG (Sì) Media Combinata"],
                    "Conteggio": [btts_home_count, btts_away_count, btts_combined_count],
                    "Percentuale %": [btts_home_perc, btts_away_perc, btts_combined_perc]
                }
                
                df_stats = pd.DataFrame(data)
                df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def calcola_first_to_score(df_to_analyze, title):
                st.subheader(f"Prima Squadra a Segnare {title}")
                if df_to_analyze.empty: return
                
                risultati = {"Squadra Casa": 0, "Squadra Trasferta": 0, "Nessun Gol": 0}
                for _, row in df_to_analyze.iterrows():
                    gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                    gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    min_home_goal = min(gol_home) if gol_home else float('inf')
                    min_away_goal = min(gol_away) if gol_away else float('inf')
                    
                    if min_home_goal < min_away_goal: risultati["Squadra Casa"] += 1
                    elif min_away_goal < min_home_goal: risultati["Squadra Trasferta"] += 1
                    elif min_home_goal == float('inf'): risultati["Nessun Gol"] += 1

                stats = []
                totale_partite = len(df_to_analyze)
                for esito, count in risultati.items():
                    perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    stats.append((esito, count, perc, odd_min))
                
                df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def mostra_distribuzione_timeband(df_to_analyze, title, timeframe=5):
                st.subheader(f"Distribuzione Gol per Timeframe {title}")
                if df_to_analyze.empty: return
                
                intervalli = [(i, i + timeframe - 1) for i in range(0, 90, timeframe)] + [(90, 150)]
                label_intervalli = [f"{start}-{end}" for start, end in intervalli[:-1]] + ["90+"]
                
                risultati = []
                totale_partite = len(df_to_analyze)
                
                for (start, end), label in zip(intervalli, label_intervalli):
                    partite_con_gol = 0
                    partite_con_almeno_due_gol = 0
                    gol_fatti = 0
                    gol_subiti = 0
                    
                    for _, row in df_to_analyze.iterrows():
                        gol_home_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                        gol_away_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                        
                        goals_in_interval_home = sum(1 for g in gol_home_minutes if start <= g <= end)
                        goals_in_interval_away = sum(1 for g in gol_away_minutes if start <= g <= end)
                        
                        gol_fatti += goals_in_interval_home
                        gol_subiti += goals_in_interval_away
                        
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
                        f"Fatti: {gol_fatti}, Subiti: {gol_subiti}"
                    ])
                    
                df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con 1+ Gol", "Partite con 2+ Gol", "Percentuale % (1+ Gol)", "Odd Minima (1+ Gol)", "Gol Fatti/Subiti"])
                st.dataframe(df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale % (1+ Gol)']))
                
            def calcola_media_gol(df_home, df_away, home_team_name, away_team_name):
                st.subheader("Media Gol Fatti e Subiti")
                
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

                data = {
                    "Squadra": [home_team_name, away_team_name],
                    "Gol Fatti (PT)": [f"{home_goals_ht:.2f}", f"{away_goals_ht:.2f}"],
                    "Gol Subiti (PT)": [f"{home_conceded_ht:.2f}", f"{away_conceded_ht:.2f}"],
                    "Gol Fatti (FT)": [f"{home_goals_ft:.2f}", f"{away_goals_ft:.2f}"],
                    "Gol Subiti (FT)": [f"{home_conceded_ft:.2f}", f"{away_conceded_ft:.2f}"]
                }
                
                df_media = pd.DataFrame(data)
                st.table(df_media)

            # --- ESECUZIONE E VISUALIZZAZIONE STATS ---
            
            st.markdown("---")
            st.header("Statistiche Generali e Risultati")
            
            # Media Gol Totali
            st.subheader("Media Gol Totali per Partita")
            avg_ht_goals = (pd.to_numeric(df_combined["gol_home_ht"], errors='coerce').fillna(0) + pd.to_numeric(df_combined["gol_away_ht"], errors='coerce').fillna(0)).mean()
            avg_ft_goals = (pd.to_numeric(df_combined["gol_home_ft"], errors='coerce').fillna(0) + pd.to_numeric(df_combined["gol_away_ft"], errors='coerce').fillna(0)).mean()
            avg_sh_goals = avg_ft_goals - avg_ht_goals
            st.table(pd.DataFrame({
                "Periodo": ["Primo Tempo (PT)", "Finale (FT)", "Secondo Tempo (ST)"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
            }))

            # Media gol fatti e subiti per singola squadra
            calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)

            col1, col2 = st.columns(2)
            with col1:
                calcola_winrate(df_combined, "risultato_ht", "PT")
            with col2:
                calcola_winrate(df_combined, "risultato_ft", "FT")

            col1, col2 = st.columns(2)
            with col1:
                mostra_risultati_esatti(df_combined, "risultato_ht", "PT")
            with col2:
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")
            
            st.markdown("---")
            st.header("Statistiche sui Gol")

            col1, col2 = st.columns(2)
            with col1:
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            calcola_btts_squadre(df_home, df_away, "gol_home_ft", "gol_away_ft", "FT")
            
            calcola_first_to_score(df_combined, "FT")

            st.markdown("---")
            st.header("Analisi Temporale dei Gol")
            mostra_distribuzione_timeband(df_combined, "(5 Min)")

        else:
            st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Per iniziare, seleziona un campionato dalla barra laterale.")
