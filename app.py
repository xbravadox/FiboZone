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
    st.set_page_config(layout="wide")

    st.sidebar.title("Ustawienia Analizy")
    ticker = st.sidebar.text_input("Wprowadź ticker akcji zgodny z Yahoo Finance", "PKO.WA")

    st.title("FiboZone - Analiza Akcji Giełdowych")
    st.markdown("""
    FiboZone to system analizy techniczno-fundamentalnej akcji giełdowych, który automatycznie wyznacza strefy konfluencji Fibonacciego i sprawdza, czy cena aktualnie się w nich znajduje.
    """)

    if st.button("Analizuj"):
        if ticker:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            # Initialize variables to prevent UnboundLocalError and ensure proper flow
            financial_data = {} 
            signal_present = False
            signal_details = None
            confluences_results = []
            d1_trend_up, w1_trend_up = False, False # Initialize trend status
            fundamental_analysis_response = None # Initialize to None
            json_ai_response = None # Initialize to None

            try:
                progress_text.text("Krok 1/5: Pobieranie danych historycznych i przetwarzanie...")
                daily_data = data_fetcher.fetch_historical_data(ticker)
                weekly_data = data_fetcher.convert_to_weekly(daily_data)
                progress_bar.progress(10)

                progress_text.text("Krok 2/5: Weryfikacja trendu...")
                d1_trend_up, w1_trend_up = is_uptrend(ticker)
                trend_status = 'Trend wzrostowy potwierdzony' if (d1_trend_up and w1_trend_up) else 'Trend wzrostowy niepotwierdzony'
                st.write(f"D1 Trend: {'Wzrostowy' if d1_trend_up else 'Nie-wzrostowy'} | W1 Trend: {'Wzrostowy' if w1_trend_up else 'Nie-wzrostowy'} | Status Trendu: {trend_status}")
                progress_bar.progress(20)

                progress_text.text("Krok 3/5: Analiza struktury technicznej (Pivots i Konfluencje)...")
                fibo_targets_dict = None
                peak_price = None
                absolute_peak_info = None
                
                with st.expander("Szczegóły Analizy Technicznej"):
                    st.subheader("Punkty zwrotne (Pivots)")
                    pivots_df = identify_pivots(weekly_data, daily_data)
                    if not pivots_df.empty:
                        st.dataframe(pivots_df)
                        fibo_targets_dict = get_fibo_targets(pivots_df)

                        if fibo_targets_dict and fibo_targets_dict.get('peak') is not None:
                            absolute_peak_info = fibo_targets_dict['peak']
                            peak_price = absolute_peak_info['Price']
                            st.subheader("Absolutny Szczyt (Odniesienie dla Fibonacciego)")
                            st.dataframe(pd.DataFrame([absolute_peak_info]))

                            if fibo_targets_dict.get('troughs') is not None and not fibo_targets_dict['troughs'].empty:
                                st.write("Wybrane Dołki Fibonacciego (podstawa obliczeń):")
                                st.dataframe(fibo_targets_dict['troughs'])
                            else:
                                st.warning("Nie znaleziono odpowiednich dołków Fibonacciego do obliczenia poziomów.")
                        else:
                            st.warning("Nie znaleziono absolutnego szczytu do obliczenia poziomów Fibonacciego.")
                    else:
                        st.warning("Nie udało się zidentyfikować punktów zwrotnych.")

                    if fibo_targets_dict and absolute_peak_info is not None and fibo_targets_dict.get('troughs') is not None and not fibo_targets_dict['troughs'].empty:
                        with st.container(): 
                            with st.expander("Zniesienia Fibonacciego"):
                                all_fibo_levels_for_display = []
                                for index, trough_row in fibo_targets_dict['troughs'].iterrows():
                                    trough_price = trough_row['Price']
                                    calculated_levels = calculate_fibonacci_levels(peak_price, trough_price)
                                    row_data = {
                                        'Trough Date': trough_row['Date'].strftime('%Y-%m-%d'),
                                        'Trough Price': trough_row['Price'],
                                        'Fibo 38.2%': calculated_levels.get('38.2%'),
                                        'Fibo 50%': calculated_levels.get('50%'),
                                        'Fibo 61.8%': calculated_levels.get('61.8%'),
                                        'Fibo 78.6%': calculated_levels.get('78.6%')
                                    }
                                    all_fibo_levels_for_display.append(row_data)
                                fibo_levels_df = pd.DataFrame(all_fibo_levels_for_display)
                                st.dataframe(fibo_levels_df)

                            with st.expander("Konfluencje"):
                                confluences_results = find_fibonacci_confluences(peak_price, fibo_targets_dict['troughs'])
                                if confluences_results:
                                    st.subheader("Znalezione Konfluencje Fibonacciego:")
                                    confluence_details_list = []
                                    for i, confluence in enumerate(confluences_results):
                                        total_score = confluence['total_score']
                                        reference_peak_price = peak_price
                                        for level in confluence['levels']:
                                            confluence_details_list.append({
                                                'Suma Punktów Konfluencji': total_score,
                                                'Cena Szczytu': f'{reference_peak_price:.2f}' if reference_peak_price is not None else 'N/A',
                                                'Etykieta Poziomu': level['label'],
                                                'Cena Poziomu': f'{level["level_value"]:.2f}',
                                                'Data Dołka': level['trough_date'].strftime('%Y-%m-%d'),
                                                'Cena Dołka': f'{level["trough_price"]:.2f}'
                                            })
                                    if confluence_details_list:
                                        confluences_df = pd.DataFrame(confluence_details_list)
                                        confluences_df = confluences_df[['Suma Punktów Konfluencji', 'Cena Szczytu', 'Etykieta Poziomu', 'Cena Poziomu', 'Data Dołka', 'Cena Dołka']]
                                        st.dataframe(confluences_df)
                                    else:
                                        st.warning("Nie znaleziono konfluencji Fibonacciego.")
                                else:
                                    st.warning("Nie znaleziono konfluencji Fibonacciego.")

                    with st.expander("Analiza ostatniej świecy D1"):
                        if not daily_data.empty:
                            signal_present, signal_details = check_last_d1_low_against_confluences(daily_data, confluences_results)

                            if signal_present:
                                st.success(f"✅ Sygnał: Low ostatniej świecy D1 ({signal_details['last_d1_low']:.2f}) znajduje się w strefie konfluencji.")
                                st.markdown(f"""
                                **Szczegóły sygnału:**
                                - Strefa konfluencji: od {signal_details['confluence_min_level']:.2f} do {signal_details['confluence_max_level']:.2f}
                                - Środek strefy: {signal_details['confluence_center']:.2f}
                                - Górna granica sygnału (+5% od środka): {signal_details['upper_signal_limit']:.2f}
                                """)
                            else:
                                st.info("ℹ️ Low ostatniej świecy D1 nie znajduje się w zdefiniowanej strefie sygnału konfluencji.")
                                if confluences_results:
                                    last_d1_low_val = daily_data['Low'].iloc[-1] if not daily_data.empty else 'N/A'
                                    st.markdown(f"""
                                    **Aktualne dane:**
                                    - Low ostatniej świecy D1: {last_d1_low_val:.2f}
                                    """)
                                    # Use the first confluence if available for context
                                    if confluences_results and 'levels' in confluences_results[0] and confluences_results[0]['levels']:
                                        first_conf = confluences_results[0]
                                        min_lvl = min(item['level_value'] for item in first_conf['levels'])
                                        max_lvl = max(item['level_value'] for item in first_conf['levels'])
                                        center = (min_lvl + max_lvl) / 2
                                        upper_lim = center * 1.05
                                        st.markdown(f"""
                                        **Kontekst (pierwsza konfluencja):**
                                        - Strefa konfluencji: od {min_lvl:.2f} do {max_lvl:.2f}
                                        - Środek strefy: {center:.2f}
                                        - Górna granica sygnału (+5% od środka): {upper_lim:.2f}
                                        """)
                                    else:
                                        st.warning("Nie znaleziono żadnych konfluencji do porównania.")
                                else:
                                    st.warning("Nie znaleziono konfluencji do porównania.")
                        else:
                            st.warning("Brak danych dziennych do analizy ostatniej świecy.")
                progress_bar.progress(50)

                # --- Krok 4: Pobieranie danych fundamentalnych ---
                progress_text.text("Krok 4/5: Pobieranie danych fundamentalnych...")
                
                if ai_models.client_initialized:
                    try:
                        financial_data = ai_models.fetch_yfinance_data(ticker)
                    except Exception as e:
                        st.error(f"Błąd podczas pobierania danych dla analizy fundamentalnej: {e}")
                else:
                    st.warning("Nie można pobrać danych fundamentalnych, ponieważ połączenie z OpenAI nie zostało nawiązane.")

                progress_bar.progress(65)

                # --- Krok 12: Integracja z AI - Analiza Fundamentalna ---
                st.subheader("Analiza Fundamentalna AI")
                
                is_overall_uptrend = d1_trend_up and w1_trend_up
                # The condition for calling fundamental analysis now relies only on financial_data.
                if is_overall_uptrend and signal_present and ai_models.client_initialized and financial_data:
                    try:
                        # Calling analyze_fundamental_with_gpt4o without news data
                        fundamental_analysis_response = ai_models.analyze_fundamental_with_gpt4o(
                            ticker=ticker,
                            financial_data=financial_data,
                            news=[] # Explicitly pass empty list for news
                        )
                        st.write("Analiza Fundamentalna AI:")
                        st.json(fundamental_analysis_response) 

                    except Exception as e:
                        st.error(f"Błąd podczas analizy fundamentalnej AI: {e}")
                elif not ai_models.client_initialized:
                    st.info("Warunki nie zostały spełnione: Połączenie z OpenAI nie jest aktywne.")
                elif not is_overall_uptrend or not signal_present:
                    st.info("Warunki nie zostały spełnione: Wymagany jest potwierdzony trend wzrostowy i sygnał techniczny.")
                elif not financial_data:
                    st.info("Warunki nie zostały spełnione: Brak wystarczających danych fundamentalnych do przeprowadzenia analizy.")

                # --- Krok 13: Integracja z AI - Analiza Techniczna ---
                st.subheader("Analiza Techniczna AI")
                relevant_confluence_details = {}
                if signal_present and signal_details:
                    relevant_confluence_details = {
                        'last_d1_low': signal_details.get('last_d1_low'),
                        'confluence_min_level': signal_details.get('confluence_min_level'),
                        'confluence_max_level': signal_details.get('confluence_max_level'),
                        'confluence_center': signal_details.get('confluence_center'),
                        'upper_signal_limit': signal_details.get('upper_signal_limit')
                    }
                
                is_overall_uptrend = d1_trend_up and w1_trend_up 
                if is_overall_uptrend and signal_present and ai_models.client_initialized and relevant_confluence_details:
                    try:
                        trend_data_for_ai = {'is_uptrend': is_overall_uptrend}
                        pivots_data_for_ai = relevant_confluence_details
                        json_ai_response = ai_models.analyze_technical_with_gpt4o_mini(
                            pivots_data=pivots_data_for_ai,
                            trend_data=trend_data_for_ai
                        )
                        st.json(json_ai_response)
                    except Exception as e:
                        st.error(f"Błąd podczas analizy technicznej AI: {e}")
                elif not ai_models.client_initialized:
                    st.info("Warunki nie zostały spełnione: Połączenie z OpenAI nie jest aktywne.")
                else:
                    st.info("Warunki do analizy technicznej AI nie zostały spełnione (wymagany trend wzrostowy i sygnał konfluencji).")

                progress_text.text("Krok 5/5: Analiza zakończona.")
                time.sleep(1)
                progress_bar.progress(100)

            except (TimeoutError, NoDataError, TickerNotFoundError) as e:
                progress_text.text("Błąd pobierania danych.")
                st.error(f"Błąd pobierania danych: {e}")
                progress_bar.progress(0)
            except Exception as e:
                progress_text.text("Nieoczekiwany błąd.")
                st.error(f"Wystąpił nieoczekiwany błąd: {e}")
                progress_bar.progress(0)
            finally:
                pass

            # --- Sekcja podsumowująca wyniki analizy ---
            st.subheader("Podsumowanie Analizy")
            
            # --- NOWA SEKCJA: Synteza Połączonych Analiz ---
            st.subheader("Zintegrowana Analiza Techniczna i Fundamentalna")
            
            # Check if both analyses were successfully performed and are valid JSON strings
            fundamental_analysis_json_str = fundamental_analysis_response if isinstance(fundamental_analysis_response, str) and 'error' not in fundamental_analysis_response else None
            technical_analysis_json_str = json_ai_response if isinstance(json_ai_response, str) and 'error' not in json_ai_response else None

            if fundamental_analysis_json_str and technical_analysis_json_str:
                try:
                    # Call the new synthesis function
                    combined_analysis_json_str = ai_models.synthesize_combined_analysis(
                        fundamental_analysis_json=fundamental_analysis_json_str,
                        technical_analysis_json=technical_analysis_json_str
                    )
                    
                    # Display the combined analysis results
                    st.write("Wynik zintegrowanej analizy AI:")
                    try:
                        combined_analysis_data = json.loads(combined_analysis_json_str)
                        if "error" in combined_analysis_data:
                            st.error(f"Błąd w zintegrowanej analizie: {combined_analysis_data['error']}")
                            if "raw_output" in combined_analysis_data:
                                st.json(combined_analysis_data["raw_output"])
                        else:
                            # Display individual components for clarity
                            st.markdown(f"**Punkty zgodności:**")
                            if combined_analysis_data.get('agreement_points'):
                                for point in combined_analysis_data['agreement_points']:
                                    st.write(f"- {point}")
                            else:
                                st.write("- Brak zidentyfikowanych punktów zgodności.")

                            st.markdown(f"**Punkty rozbieżności:**")
                            if combined_analysis_data.get('disagreement_points'):
                                for point in combined_analysis_data['disagreement_points']:
                                    st.write(f"- {point}")
                            else:
                                st.write("- Brak zidentyfikowanych punktów rozbieżności.")

                            st.markdown(f"**Zintegrowany widok:**")
                            st.write(combined_analysis_data.get('integrated_view', 'Brak zintegrowanego widoku.'))
                            
                            st.markdown(f"**Skonsolidowana rekomendacja:**")
                            st.write(combined_analysis_data.get('overall_recommendation', 'Brak skonsolidowanej rekomendacji.'))

                    except json.JSONDecodeError:
                        st.error("Nie udało się sparsować odpowiedzi zintegrowanej analizy jako JSON.")
                        st.json(combined_analysis_json_str) # Display raw output if parsing fails

                except Exception as e:
                    st.error(f"Wystąpił błąd podczas syntezy połączonych analiz: {e}")
            else:
                st.info("Nie można przeprowadzić zintegrowanej analizy. Wymagane są oba wyniki analizy fundamentalnej i technicznej.")
            # --- Koniec sekcji Syntezy Połączonych Analiz ---

        else:
            st.warning("Proszę wprowadzić ticker akcji w panelu bocznym.")

if __name__ == "__main__":
    main()
