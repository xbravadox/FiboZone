import streamlit as st
import pandas as pd
import data_fetcher
import analysis_logic
import time
import ai_models

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

            try:
                progress_text.text("Krok 1/5: Pobieranie i przetwarzanie danych...")
                daily_data = data_fetcher.fetch_historical_data(ticker)
                weekly_data = data_fetcher.convert_to_weekly(daily_data)
                progress_bar.progress(20)

                progress_text.text("Krok 2/5: Weryfikacja trendu...")
                d1_trend_up, w1_trend_up = is_uptrend(ticker)
                trend_status = 'Trend wzrostowy potwierdzony' if (d1_trend_up and w1_trend_up) else 'Trend wzrostowy niepotwierdzony'
                st.write(f"D1 - {'trend wzrostowy' if d1_trend_up else 'inny'} | W1 - {'trend wzrostowy' if w1_trend_up else 'inny'} | {trend_status}")
                progress_bar.progress(30)

                progress_text.text("Krok 3/5: Analiza struktury (Pivots)...")
                fibo_targets_dict = None
                peak_price = None
                absolute_peak_info = None
                confluences_results = []

                with st.expander("Struktura setupu"):
                    st.subheader("Punkty zwrotne (Pivots)")
                    pivots_df = identify_pivots(weekly_data, daily_data)
                    if not pivots_df.empty:
                        st.dataframe(pivots_df)
                        fibo_targets_dict = get_fibo_targets(pivots_df)

                        if fibo_targets_dict.get('peak') is not None:
                            absolute_peak_info = fibo_targets_dict['peak']
                            peak_price = absolute_peak_info['Price']
                            st.subheader("Absolutny Szczyt (Odniesienie dla Fibonacciego)")
                            st.dataframe(pd.DataFrame([absolute_peak_info]))

                            if not fibo_targets_dict['troughs'].empty:
                                st.write("Wybrane Dołki Fibonacciego (podstawa obliczeń):")
                                st.dataframe(fibo_targets_dict['troughs'])
                            else:
                                st.warning("Nie znaleziono odpowiednich dołków Fibonacciego do obliczenia poziomów.")
                        else:
                            st.warning("Nie znaleziono absolutnego szczytu do obliczenia poziomów Fibonacciego.")
                    else:
                        st.warning("Nie udało się zidentyfikować punktów zwrotnych.")
                progress_bar.progress(50)

                if fibo_targets_dict and absolute_peak_info is not None and not fibo_targets_dict['troughs'].empty:
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
                                first_conf = confluences_results[0]
                                if 'levels' in first_conf and first_conf['levels']:
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
                        st.warning("Brak danych dziennych do analizy ostatniej świecy.")

                progress_bar.progress(70)

                # --- Krok 12: Integracja z AI - Analiza Techniczna ---
                st.subheader("Analiza Techniczna AI")

                # Prepare data for AI analysis
                is_overall_uptrend = d1_trend_up and w1_trend_up

                relevant_confluence_details = {}
                # Check if a signal is present and details are available from the previous step
                if signal_present and signal_details:
                    # Extract relevant details for the single closest confluence for the LLM
                    # These keys are confirmed from previous codebase_investigator output
                    relevant_confluence_details = {
                        'last_d1_low': signal_details.get('last_d1_low'),
                        'confluence_min_level': signal_details.get('confluence_min_level'),
                        'confluence_max_level': signal_details.get('confluence_max_level'),
                        'confluence_center': signal_details.get('confluence_center'),
                        'upper_signal_limit': signal_details.get('upper_signal_limit')
                    }

                # Define the condition for sending data to LLM:
                # Overall uptrend must be confirmed AND a relevant signal (confluence near/in) must be present.
                # The presence of relevant_confluence_details is ensured by signal_present and signal_details check.
                if is_overall_uptrend and signal_present:
                    if ai_models.client_initialized:
                        try:
                            # Prepare data for the AI model
                            # trend_data expects a dictionary
                            trend_data_for_ai = {'is_uptrend': is_overall_uptrend}

                            # pivots_data is the relevant confluence details
                            pivots_data_for_ai = relevant_confluence_details if relevant_confluence_details else {}

                            # Ensure we only call if we have data for the AI
                            if pivots_data_for_ai:
                                # This will now be a string response
                                raw_ai_response_text = ai_models.analyze_technical_with_gpt4o_mini(
                                    pivots_data=pivots_data_for_ai,
                                    trend_data=trend_data_for_ai
                                )

                                # Display the raw LLM response directly
                                st.write("Analiza Techniczna AI:")
                                st.write(raw_ai_response_text)

                            else:
                                st.warning("Nie udało się przygotować danych o konfluencji do analizy AI.")

                        except Exception as e:
                            st.error(f"Błąd podczas analizy technicznej AI: {e}")
                    else:
                        st.warning("Nie można wykonać analizy AI, ponieważ połączenie z OpenAI nie zostało nawiązane.")
                else:
                    # Display a message indicating that AI analysis was skipped due to unmet conditions.
                    st.info("Warunki do analizy technicznej AI nie zostały spełnione (wymagany trend wzrostowy i sygnał konfluencji).")

                # --- Koniec sekcji integracji z AI ---

                progress_text.text("Krok 5/5: Analiza zakończona.")
                time.sleep(1)
                progress_bar.progress(100)

            except (TimeoutError, NoDataError, TickerNotFoundError) as e:
                progress_text.text("Błąd podczas pobierania danych.")
                st.error(f"Błąd pobierania danych: {e}")
                progress_bar.progress(0)
            except Exception as e:
                progress_text.text("Nieoczekiwany błąd.")
                st.error(f"Wystąpił nieoczekiwany błąd: {e}")
                progress_bar.progress(0)
            finally:
                pass

            st.subheader("Wyniki Analizy:")
            st.write("Wyniki analizy technicznej i fundamentalnej pojawią się tutaj po uruchomieniu.")
        else:
            st.warning("Proszę wprowadzić ticker akcji w panelu bocznym.")

if __name__ == "__main__":
    main()