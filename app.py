import streamlit as st
import pandas as pd
import data_fetcher
import analysis_logic
import time
import ai_models
import json # Import json for parsing LLM output

from data_fetcher import TickerNotFoundError, NoDataError, TimeoutError
from analysis_logic import is_uptrend, identify_pivots, get_fibo_targets, calculate_fibonacci_levels, find_fibonacci_confluences, check_last_d1_low_against_confluences

def main():
    # --- 1. KONFIGURACJA I SIDEBAR ---
    st.set_page_config(layout='wide', page_title='FiboZone')

    st.sidebar.title('Ustawienia Analizy')
    ticker = st.sidebar.text_input('Wprowadź ticker akcji zgodny z Yahoo Finance', 'PKO.WA')
    analyze_button = st.sidebar.button('Analizuj')

    # --- 2. STRONA GŁÓWNA: TOP ---
    st.title('FiboZone - Analiza Akcji Giełdowych')
    
    st.error('Zastrzeżenie: FiboZone jest narzędziem eksperymentalnym służącym wyłącznie celom edukacyjnym. Nie stanowi porady finansowej. Wszelkie informacje dostarczone przez to narzędzie powinny zostać zweryfikowane niezależnie, a użytkownicy powinni przeprowadzić własne analizy przed podjęciem jakichkolwiek decyzji inwestycyjnych.')
    
    with st.expander('Opis Projektu FiboZone'):
        st.markdown('''
        **FiboZone** to zaawansowany system analizy techniczno-fundamentalnej akcji giełdowych. Jego głównym celem jest automatyczne wyznaczanie stref konfluencji opartych na poziomach Fibonacciego oraz weryfikacja, czy aktualna cena akcji znajduje się w tych strefach. Projekt integruje pobieranie danych historycznych, zaawansowane algorytmy analizy technicznej (w tym identyfikację punktów zwrotnych, obliczanie poziomów Fibonacciego i wykrywanie konfluencji), a także analizę fundamentalną wspieraną przez modele AI (GPT-4o i GPT-4o-mini). Interfejs użytkownika został zbudowany przy użyciu Streamlit, co zapewnia intuicyjny sposób interakcji z narzędziem.
        ''')

    if analyze_button:
        if ticker:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            financial_data = {} 
            signal_present = False
            signal_details = None
            confluences_results = []
            d1_trend_up, w1_trend_up = False, False 
            fundamental_analysis_response = None 
            json_ai_response = None 

            try:
                progress_text.text('Krok 1/6: Pobieranie danych historycznych...')
                daily_data = data_fetcher.fetch_historical_data(ticker)
                weekly_data = data_fetcher.convert_to_weekly(daily_data)
                progress_bar.progress(15)

                progress_text.text('Krok 2/6: Weryfikacja trendu...')
                d1_trend_up, w1_trend_up = is_uptrend(ticker)
                progress_bar.progress(25)

                progress_text.text('Krok 3/6: Analiza techniczna...')
                pivots_df = identify_pivots(weekly_data, daily_data)
                fibo_targets_dict = None
                peak_price = None

                if not pivots_df.empty:
                    fibo_targets_dict = get_fibo_targets(pivots_df)
                    if fibo_targets_dict and fibo_targets_dict.get('peak') is not None:
                        peak_price = fibo_targets_dict['peak']['Price']
                        confluences_results = find_fibonacci_confluences(peak_price, fibo_targets_dict['troughs'])
                        signal_present, signal_details = check_last_d1_low_against_confluences(daily_data, confluences_results)

                # --- 3. DASHBOARD: KLUCZOWE METRYKI (Z IKONAMI) ---
                st.subheader(f'Status Analizy: {ticker}')
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    icon_d1 = '📈' if d1_trend_up else '📉'
                    st.metric('Trend D1', f"{icon_d1} {'Wzrostowy' if d1_trend_up else 'Spadkowy/Boczny'}")
                with m_col2:
                    icon_w1 = '📈' if w1_trend_up else '📉'
                    st.metric('Trend W1', f"{icon_w1} {'Wzrostowy' if w1_trend_up else 'Spadkowy/Boczny'}")
                with m_col3:
                    status_text = '🟢 W STREFIE' if signal_present else '🔴 POZA STREFĄ'
                    st.metric('Sygnał Fibo', status_text)
                st.divider()

                progress_bar.progress(50)
                progress_text.text('Krok 4/6: Pobieranie danych fundamentalnych...')
                if ai_models.client_initialized:
                    financial_data = ai_models.fetch_yfinance_data(ticker)
                
                progress_bar.progress(75)
                progress_text.text('Krok 5/6: Generowanie raportów AI...')
                
                is_overall_uptrend = d1_trend_up and w1_trend_up
                if is_overall_uptrend and signal_present and ai_models.client_initialized and financial_data:
                    fundamental_analysis_response = ai_models.analyze_fundamental_with_gpt4o(ticker, financial_data, [])
                    
                    relevant_confluence_details = {
                        'last_d1_low': signal_details.get('last_d1_low'), 
                        'confluence_min_level': signal_details.get('confluence_min_level'), 
                        'confluence_max_level': signal_details.get('confluence_max_level'), 
                        'confluence_center': signal_details.get('confluence_center'), 
                        'upper_signal_limit': signal_details.get('upper_signal_limit')
                    }
                    json_ai_response = ai_models.analyze_technical_with_gpt4o_mini(relevant_confluence_details, {'is_uptrend': is_overall_uptrend})

                progress_text.text('Krok 6/6: Renderowanie widoku końcowego...')
                progress_bar.progress(95)

                # --- 4. ZINTEGROWANY RAPORT AI (CENTRUM) ---
                st.subheader('🤖 Wynik Zintegrowanej Analizy AI')
                
                if fundamental_analysis_response and json_ai_response:
                    combined_analysis_json_str = ai_models.synthesize_combined_analysis(fundamental_analysis_response, json_ai_response)
                    combined_data = json.loads(combined_analysis_json_str)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('**Punkty zgodności:**')
                        for p in combined_data.get('agreement_points', []): st.write(f'- {p}')
                    with c2:
                        st.markdown('**Punkty rozbieżności:**')
                        for p in combined_data.get('disagreement_points', []): st.write(f'- {p}')
                    
                    st.info(f"**Widok zintegrowany:**\n\n{combined_data.get('integrated_view', '')}")
                    st.success(f"**Skonsolidowana rekomendacja AI:** {combined_data.get('overall_recommendation', 'Brak')}")
                else:
                    st.info('Zintegrowany raport AI dostępny po potwierdzonym trendzie i sygnale technicznym.')

                # --- 5. PEŁNE SZCZEGÓŁY TECHNICZNE (NAD JSON) ---
                st.divider()
                with st.expander('🔍 Szczegóły Analizy Technicznej (Pivots, Fibo, Konfluencje)'):
                    if not pivots_df.empty:
                        st.subheader('Punkty zwrotne (Pivots)')
                        st.dataframe(pivots_df, use_container_width=True)
                        
                        if fibo_targets_dict and fibo_targets_dict.get('peak') is not None:
                            st.write('**Absolutny Szczyt (Reference High):**')
                            st.dataframe(pd.DataFrame([fibo_targets_dict['peak']]), use_container_width=True)

                            if fibo_targets_dict.get('troughs') is not None and not fibo_targets_dict['troughs'].empty:
                                st.subheader('Wyliczone Zniesienia Fibonacciego')
                                fibo_display = []
                                for idx, row in fibo_targets_dict['troughs'].iterrows():
                                    lvls = calculate_fibonacci_levels(peak_price, row['Price'])
                                    fibo_display.append({
                                        'Data Dołka': row['Date'].strftime('%Y-%m-%d'),
                                        'Cena Dołka': row['Price'],
                                        '38.2%': lvls.get('38.2%'),
                                        '50.0%': lvls.get('50%'),
                                        '61.8%': lvls.get('61.8%'),
                                        '78.6%': lvls.get('78.6%')
                                    })
                                st.dataframe(pd.DataFrame(fibo_display), use_container_width=True)

                                if confluences_results:
                                    st.subheader('Tabela Konfluencji Fibonacciego')
                                    conf_flat = []
                                    for c in confluences_results:
                                        for l in c['levels']:
                                            conf_flat.append({'Punkty': c['total_score'], 'Poziom': l['label'], 'Cena': f"{l['level_value']:.2f}", 'Z dnia': l['trough_date'].strftime('%Y-%m-%d')})
                                    st.dataframe(pd.DataFrame(conf_flat), use_container_width=True)
                    
                    if signal_present:
                        st.success(f"✅ Sygnał: Low ostatniej świecy ({signal_details['last_d1_low']:.2f}) w strefie.")
                    else:
                        st.info('ℹ️ Brak sygnału w strefie konfluencji.')

                # --- 6. SUROWE DANE JSON (DEBUG) ---
                with st.expander('🛠️ Surowe Dane JSON'):
                    st.write('**Dane Fundamentalne (Input):**')
                    st.json(financial_data)
                    st.write('**Analiza Fundamentalna (Output AI):**')
                    st.json(fundamental_analysis_response)
                    st.write('**Analiza Techniczna (Output AI):**')
                    st.json(json_ai_response)

                # --- KONIEC PROCESU ---
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                progress_text.empty()

            except Exception as e:
                progress_bar.empty()
                progress_text.empty()
                st.error(f'Wystąpił nieoczekiwany błąd: {e}')
        else:
            st.warning('Wprowadź ticker w panelu bocznym.')

if __name__ == '__main__':
    main()