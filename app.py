import streamlit as st
import pandas as pd
import data_fetcher
import time
from analysis_logic import is_uptrend, identify_pivots, get_fibo_targets, calculate_fibonacci_levels, find_fibonacci_confluences

def main():
    st.set_page_config(layout='wide')

    # --- Sidebar ---
    st.sidebar.title('Ustawienia Analizy')
    ticker = st.sidebar.text_input('Wprowadź ticker akcji zgodny z Yahoo Finance', 'PKO.WA')

    # --- Główna część aplikacji ---
    st.title('FiboZone - Analiza Akcji Giełdowych')
    st.markdown('''
    FiboZone to system analizy techniczno-fundamentalnej akcji giełdowych, który automatycznie wyznacza strefy konfluencji Fibonacciego i sprawdza, czy cena aktualnie się w nich znajduje.
    ''')

    if st.button('Analizuj'):
        if ticker:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            try:
                # --- Krok 1/5: Pobieranie i przetwarzanie danych ---
                progress_text.text('Krok 1/5: Pobieranie i przetwarzanie danych...')
                daily_data = data_fetcher.fetch_historical_data(ticker)
                weekly_data = data_fetcher.convert_to_weekly(daily_data)
                progress_bar.progress(20)

                # --- Krok 2/5: Weryfikacja trendu ---
                progress_text.text('Krok 2/5: Weryfikacja trendu...')
                d1_trend_up, w1_trend_up = is_uptrend(ticker)
                st.write(f"D1 - {'trend wzrostowy' if d1_trend_up else 'inny'} | W1 - {'trend wzrostowy' if w1_trend_up else 'inny'} | {'Trend wzrostowy potwierdzony' if (d1_trend_up and w1_trend_up) else 'Trend wzrostowy niepotwierdzony'}")
                progress_bar.progress(30)

                # --- Krok 3/5: Analiza struktury (Pivots) ---
                progress_text.text('Krok 3/5: Analiza struktury (Pivots)...')
                fibo_targets_dict = None
                peak_price = None

                with st.expander('Struktura setupu'):
                    st.subheader('Punkty zwrotne (Pivots)')
                    pivots_df = identify_pivots(weekly_data, daily_data)
                    if not pivots_df.empty:
                        st.dataframe(pivots_df, width='stretch')
                        fibo_targets_dict = get_fibo_targets(pivots_df)

                        if fibo_targets_dict.get('peak') is not None:
                            peak_price = fibo_targets_dict['peak']['Price']
                            st.subheader('Absolutny Szczyt (Odniesienie dla Fibonacciego)')
                            st.dataframe(pd.DataFrame([fibo_targets_dict['peak']]), width='stretch')

                            if not fibo_targets_dict['troughs'].empty:
                                st.write('Wybrane Dołki Fibonacciego (podstawa obliczeń):')
                                st.dataframe(fibo_targets_dict['troughs'], width='stretch')
                        else:
                            st.warning('Nie znaleziono absolutnego szczytu do obliczenia poziomów Fibonacciego.')
                    else:
                        st.warning('Nie udało się zidentyfikować punktów zwrotnych.')
                progress_bar.progress(50)

                # --- Krok 4/5: Analiza poziomów Fibonacciego i Konfluencji ---
                progress_text.text('Krok 4/5: Analiza poziomów Fibonacciego i Konfluencji...')
                if fibo_targets_dict and peak_price is not None and not fibo_targets_dict['troughs'].empty:
                    # Zniesienia
                    with st.expander('Zniesienia Fibonacciego'):
                        all_fibo_levels = []
                        for index, trough_row in fibo_targets_dict['troughs'].iterrows():
                            trough_price = trough_row['Price']
                            calculated_levels = calculate_fibonacci_levels(peak_price, trough_price)
                            all_fibo_levels.append({
                                'Trough Date': trough_row['Date'].strftime('%Y-%m-%d'),
                                'Trough Price': trough_row['Price'],
                                'Fibo 38.2%': calculated_levels.get('38.2%'),
                                'Fibo 50%': calculated_levels.get('50%'),
                                'Fibo 61.8%': calculated_levels.get('61.8%'),
                                'Fibo 78.6%': calculated_levels.get('78.6%')
                            })
                        st.dataframe(pd.DataFrame(all_fibo_levels), width='stretch')

                    # Konfluencje
                    confluences_results = find_fibonacci_confluences(peak_price, fibo_targets_dict['troughs'])
                    with st.expander('Konfluencje'):
                        if confluences_results:
                            st.subheader('Znalezione Konfluencje Fibonacciego:')
                            confluence_list = []
                            for confluence in confluences_results:
                                score = confluence['total_score']
                                for lvl in confluence['levels']:
                                    confluence_list.append({
                                        'Suma Punktów': score,
                                        'Etykieta': lvl['label'],
                                        'Cena Poziomu': f"{lvl['level_value']:.2f}",
                                        'Data Dołka': lvl['trough_date'].strftime('%Y-%m-%d'),
                                        'Cena Dołka': f"{lvl['trough_price']:.2f}"
                                    })
                            st.dataframe(pd.DataFrame(confluence_list), width='stretch')
                        else:
                            st.warning('Nie znaleziono konfluencji Fibonacciego.')
                
                progress_bar.progress(80)

                # --- Krok 5/5: Analiza zakończona ---
                progress_text.text('Krok 5/5: Analiza zakończona.')
                time.sleep(1)
                progress_bar.progress(100)

            except Exception as e:
                progress_text.text('Wystąpił błąd.')
                st.error(f'Błąd: {e}')
                progress_bar.progress(0)

            st.subheader('Wyniki Analizy:')
            st.write('Analiza techniczna została wyświetlona powyżej w sekcjach rozwijanych.')
        else:
            st.warning('Proszę wprowadzić ticker akcji w panelu bocznym.')

if __name__ == '__main__':
    main()