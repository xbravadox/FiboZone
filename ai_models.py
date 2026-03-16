import openai
import streamlit as st
import re
import pandas as pd
import yfinance as yf
import json
from curl_cffi import requests as cffi_requests


_session = cffi_requests.Session(impersonate="chrome")


# OpenAI client initialization
openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = None
client_initialized = False


if openai_api_key:
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        client.models.list()
        client_initialized = True
    except openai.AuthenticationError:
        st.error("Błąd uwierzytelnienia OpenAI: Klucz API jest nieprawidłowy lub wygasł.")
    except openai.RateLimitError:
        st.error("Limit zapytań OpenAI API został przekroczony. Spróbuj ponownie później.")
    except openai.APIConnectionError as e:
        st.error(f"Błąd połączenia z OpenAI API: {e}")
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas inicjalizacji OpenAI: {e}")
else:
    st.warning("Klucz API OpenAI nie został znaleziony w Streamlit Secrets. Funkcje AI mogą nie działać.")


def test_openai_connection():
    """
    Funkcja testująca połączenie z OpenAI API.
    """
    if client_initialized:
        return "Połączenie z OpenAI API jest aktywne."
    else:
        return "Połączenie z OpenAI API nie jest aktywne. Sprawdź klucz API i konfigurację."


def fetch_yfinance_data(ticker: str) -> dict:
    '''
    Pobiera dane fundamentalne dla spółki z yfinance.
    Obejmuje: wskaźniki finansowe, opis spółki, sektor i branżę.

    Args:
        ticker (str): Symbol tickera spółki.

    Returns:
        dict: Słownik zawierający dane fundamentalne.
                Format: {
                    'indicators': { 'key': value, ... },
                    'description': str,
                    'sector': str,
                    'industry': str
                }
            Zwraca pusty dict w przypadku błędu.
    '''
    if not ticker:
        return {}

    try:
        stock = yf.Ticker(ticker, session=_session)
        info = stock.info

        financial_data = {
            'marketCap': info.get('marketCap'),
            'beta': info.get('beta'),
            'forwardPE': info.get('forwardPE'),
            'dividendYield': info.get('dividendYield'),
            'profitMargins': info.get('profitMargins'),
            'grossMargins': info.get('grossMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'returnOnAssets': info.get('returnOnAssets'),
            'returnOnEquity': info.get('returnOnEquity'),
            'currentRatio': info.get('currentRatio'),
            'debtToEquity': info.get('debtToEquity'),
            'quickRatio': info.get('quickRatio'),
            'longName': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'website': info.get('website'),
            'longBusinessSummary': info.get('longBusinessSummary'),
            'exchange': info.get('exchange'),
            'quoteType': info.get('quoteType'),
        }

        cleaned_data = {}
        cleaned_data['indicators'] = {k: v for k, v in financial_data.items() if v is not None and k not in ['longBusinessSummary', 'sector', 'industry', 'longName', 'website']}
        cleaned_data['description'] = financial_data.get('longBusinessSummary', 'Brak opisu działalności spółki.')
        cleaned_data['sector'] = financial_data.get('sector', 'Brak sektora.')
        cleaned_data['industry'] = financial_data.get('industry', 'Brak branży.')

        return cleaned_data

    except Exception as e:
        st.error(f"Błąd podczas pobierania danych fundamentalnych z yfinance dla {ticker}: {e}")
        return {}


def analyze_fundamental_with_gpt4o(ticker: str, financial_data: dict, news: list[str]) -> str:
    '''
    Przeprowadza analizę fundamentalną spółki przy użyciu GPT-4o, zwracając wynik w formacie JSON.

    Args:
        ticker (str): Symbol tickera spółki.
        financial_data (dict): Słownik zawierający dane fundamentalne pobrane z yfinance.
                                Format: {'indicators': {...}, 'description': ..., 'sector': ..., 'industry': ...}
        news (list[str]): Lista najnowszych wiadomości dotyczących spółki.

    Returns:
        str: Odpowiedź LLM w formacie JSON lub komunikat o błędzie w formacie JSON.
    '''
    if not client_initialized:
        return json.dumps({"error": "Klucz API OpenAI nie jest skonfigurowany lub połączenie nie powiodło się. Nie można wykonać analizy."})

    prompt_parts = [f"Dane dla tickera: {ticker}"]

    if financial_data:
        prompt_parts.append("Dane finansowe i opis spółki:")
        indicators_str = json.dumps(financial_data.get('indicators', {}), indent=2, ensure_ascii=False)
        prompt_parts.append(f"  Wskaźniki: {indicators_str}")
        prompt_parts.append(f"  Opis działalności: {financial_data.get('description', 'Brak opisu działalności spółki.')}")
        prompt_parts.append(f"  Sektor: {financial_data.get('sector', 'Brak sektora.')}")
        prompt_parts.append(f"  Branża: {financial_data.get('industry', 'Brak branży.')}")
    else:
        prompt_parts.append("Brak danych finansowych do analizy.")

    if news:
        prompt_parts.append("Najnowsze wiadomości:")
        for i, new in enumerate(news):
            prompt_parts.append(f"  {i+1}. {new}")
    else:
        prompt_parts.append("Brak najnowszych wiadomości.")

    final_prompt = f"""
{chr(10).join(prompt_parts)}

Przeanalizuj powyższe dane i wygeneruj kompleksową analizę fundamentalną spółki.
Twoja analiza musi być oparta WYŁĄCZNIE na dostarczonych danych (wskaźniki finansowe, opis działalności, sektor, branża, wiadomości). NIE używaj żadnej zewnętrznej wiedzy ani nie wyciągaj wniosków spoza dostarczonego kontekstu.

Wynik musi być w formacie JSON, zawierającym następujące klucze:
- 'overall_sentiment': Ogólny sentyment analizy (np. 'pozytywny', 'neutralny', 'negatywny').
- 'key_strengths': Lista kluczowych mocnych stron spółki.
- 'key_weaknesses': Lista kluczowych słabych stron spółki.
- 'growth_potential': Potencjalne obszary wzrostu.
- 'risks': Główne ryzyka związane ze spółką.
- 'summary': Krótkie podsumowanie analizy.

Przykład struktury JSON:
{{
  "overall_sentiment": "pozytywny",
  "key_strengths": ["Silna pozycja rynkowa", "Stabilny wzrost przychodów"],
  "key_weaknesses": ["Wysokie zadłużenie"],
  "growth_potential": ["Ekspansja na nowe rynki", "Innowacyjne produkty"],
  "risks": ["Zmiany regulacyjne", "Konkurencja"],
  "summary": "Spółka prezentuje stabilne wyniki i potencjał wzrostu, wymaga jednak uwagi w kontekście zadłużenia."
}}

Jeśli brakuje danych do pełnej analizy, użyj odpowiednich komunikatów w polu 'summary' lub odpowiednich kluczy listy (np. klucze mogą być puste).
TWOJE WYJŚCIE MUSI BYĆ CZYSTYM OBIEKTEM JSON BEZ DODATKOWEGO TEKSTU PRZED LUB PO NIM.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od analizy fundamentalnej spółek giełdowych. Twoim zadaniem jest tworzenie zwięzłych analiz w formacie JSON wyłącznie na podstawie dostarczonych danych."},
                {"role": "user", "content": final_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )

        json_output = response.choices[0].message.content.strip()

        try:
            json.loads(json_output)
            return json_output
        except json.JSONDecodeError:
            return json.dumps({"error": "Model nie zwrócił poprawnego formatu JSON.", "raw_output": json_output})

    except Exception as e:
        return json.dumps({"error": f"Wystąpił błąd podczas zapytania do OpenAI API: {e}"})


def analyze_technical_with_gpt4o_mini(pivots_data: dict, trend_data: dict) -> str:
    '''
    Generuje analizę techniczną w formacie JSON przy użyciu GPT-4o-mini.

    Args:
        pivots_data (dict): Dane dotyczące punktów zwrotnych (pivotów).
        trend_data (dict): Dane dotyczące trendu.

    Returns:
        str: Analiza techniczna w formacie JSON lub komunikat o błędzie w formacie JSON.
    '''
    if not client_initialized:
        return json.dumps({"error": "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."})

    try:
        trend_data_str = json.dumps(trend_data)
        pivots_data_str = json.dumps(pivots_data)

        final_prompt = f"""
Przeanalizuj podane dane techniczne akcji giełdowych. Twoja analiza musi być oparta WYŁĄCZNIE na dostarczonych danych.
Wynik musi być w formacie JSON, zawierającym następujące klucze:
- 'technical_analysis': Szczegółowa analiza techniczna.
- 'recommendation': Konkretna rekomendacja (np. 'Kupuj', 'Sprzedaj', 'Czekaj').

Wykorzystaj poniższe dane do analizy:
1. Potwierdzenie trendu wzrostowego (dane: {trend_data_str}).
2. Szczegóły najbliższej strefy konfluencji Fibonacciego (dane: {pivots_data_str}).

Jeśli brakuje danych dla którejś sekcji (np. rekomendacji lub szczegółów analizy), użyj komunikatu "Brak danych.", ale zachowaj ogólną strukturę JSON.
TWOJE WYJŚCIE MUSI BYĆ CZYSTYM OBIEKTEM JSON BEZ DODATKOWEGO TEKSTU PRZED LUB PO NIM.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od analizy technicznej akcji giełdowych. Twoim zadaniem jest generowanie analizy w formacie JSON na podstawie dostarczonych danych. Zwracaj tylko obiekt JSON."},
                {"role": "user", "content": final_prompt}
            ],
            response_format={"type": "json_object"},
        )

        json_output = response.choices[0].message.content.strip()

        try:
            parsed_json = json.loads(json_output)

            if 'technical_analysis' not in parsed_json:
                parsed_json['technical_analysis'] = "Brak danych."
            if 'recommendation' not in parsed_json:
                parsed_json['recommendation'] = "Brak danych."

            return json.dumps(parsed_json, indent=2, ensure_ascii=False)

        except json.JSONDecodeError:
            return json.dumps({"error": "Model nie zwrócił poprawnego formatu JSON.", "raw_output": json_output})
        except Exception as e:
            return json.dumps({"error": f"Błąd przetwarzania odpowiedzi AI: {e}", "raw_output": json_output})

    except Exception as e:
        return json.dumps({"error": f"Wystąpił błąd podczas zapytania do OpenAI API dla analizy technicznej: {e}"})

# NOWA FUNKCJA DO SYNTETYZOWANIA POŁĄCZONYCH ANALIZ
def synthesize_combined_analysis(fundamental_analysis_json: str, technical_analysis_json: str) -> str:
    '''
    Syntetyzuje i porównuje analizę fundamentalną i techniczną, generując zintegrowany raport.

    Args:
        fundamental_analysis_json (str): Wynik analizy fundamentalnej w formacie JSON.
        technical_analysis_json (str): Wynik analizy technicznej w formacie JSON.

    Returns:
        str: Zintegrowany raport w formacie JSON lub komunikat o błędzie.
    '''
    if not client_initialized:
        return json.dumps({"error": "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."})

    try:
        # Walidacja i sparsowanie wejściowych danych JSON
        try:
            fundamental_data = json.loads(fundamental_analysis_json)
            technical_data = json.loads(technical_analysis_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Błąd parsowania wejściowych danych JSON: {e}"})

        # Przygotowanie promptu dla LLM
        final_prompt = f"""
Przeanalizuj poniższe wyniki analizy fundamentalnej (JSON A) i analizy technicznej (JSON B).

JSON A (Analiza Fundamentalna):
{fundamental_analysis_json}

JSON B (Analiza Techniczna):
{technical_analysis_json}

Twoim zadaniem jest porównanie tych dwóch analiz i wygenerowanie zsyntetyzowanego raportu, który uwzględnia oba punkty widzenia. Skup się na następujących aspektach:

1.  **Punkty zgodności:** Zidentyfikuj, w jakich obszarach analizy się uzupełniają lub potwierdzają (np. pozytywny sentyment fundamentalny wspierający trend wzrostowy techniczny).
2.  **Punkty rozbieżności:** Wskaż, gdzie analizy się różnią lub sobie zaprzeczają (np. rekomendacja kupna techniczna pomimo słabych fundamentów, lub silne fundamenty przy neutralnym sygnale technicznym).
3.  **Zintegrowany widok:** Stwórz podsumowanie, które stanowi zbalansowany obraz sytuacji, biorąc pod uwagę zarówno dane techniczne, jak i fundamentalne. Oceń, jak kluczowe mocne strony, słabości, potencjał wzrostu i ryzyka z analizy fundamentalnej wpływają na sygnały techniczne (i odwrotnie).
4.  **Skonsolidowana rekomendacja:** Na podstawie całościowej analizy wygeneruj jedną, spójną rekomendację (np. "Kupuj z ostrożnością", "Czekaj", "Sprzedaj").

Wynik musi być w formacie JSON, zawierającym następujące klucze:
- 'agreement_points': Lista stringów opisujących punkty zgodności.
- 'disagreement_points': Lista stringów opisujących punkty rozbieżności.
- 'integrated_view': Krótkie podsumowanie zintegrowanego widoku sytuacji.
- 'overall_recommendation': Zsyntetyzowana rekomendacja.

Przykład struktury JSON wyjściowej:
{{
  "agreement_points": ["Analiza techniczna potwierdza trend wzrostowy, wspierany potencjałem wzrostu spółki."],
  "disagreement_points": ["Pomimo rekomendacji 'Kupuj' technicznej, analiza fundamentalna wskazuje na wysokie zadłużenie i niską płynność, co rodzi ryzyko."],
  "integrated_view": "Spółka posiada obiecujący potencjał wzrostu i silny trend wzrostowy techniczny, jednak wysokie zadłużenie i niska płynność stanowią istotne ryzyko, które wymaga monitorowania. Sugerowane jest ostrożne podejście.",
  "overall_recommendation": "Kupuj z ostrożnością"
}}

Twoje wyjście musi być CZYSTYM OBIEKTEM JSON BEZ DODATKOWEGO TEKSTU PRZED LUB PO NIM.
Użyj modelu 'gpt-4o-mini' do tej analizy.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Używamy gpt-4o-mini zgodnie z wcześniejszymi ustaleniami
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od analizy finansowej, łączącym wiedzę techniczną i fundamentalną. Twoim zadaniem jest porównanie dwóch analiz i wygenerowanie zintegrowanego raportu w formacie JSON. Zwracaj tylko obiekt JSON."},
                {"role": "user", "content": final_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000 # Zwiększona liczba tokenów dla bardziej złożonych odpowiedzi
        )

        json_output = response.choices[0].message.content.strip()

        # Walidacja wyjściowego JSON-a
        try:
            parsed_output = json.loads(json_output)
            # Sprawdzenie wymaganych kluczy w wyniku
            required_keys = ['agreement_points', 'disagreement_points', 'integrated_view', 'overall_recommendation']
            if not all(key in parsed_output for key in required_keys):
                return json.dumps({"error": "Model nie zwrócił wszystkich wymaganych kluczy w odpowiedzi JSON.", "raw_output": json_output})
            
            return json.dumps(parsed_output, indent=2, ensure_ascii=False)

        except json.JSONDecodeError:
            return json.dumps({"error": "Model nie zwrócił poprawnego formatu JSON.", "raw_output": json_output})
        except Exception as e:
            return json.dumps({"error": f"Błąd przetwarzania odpowiedzi AI: {e}", "raw_output": json_output})

    except Exception as e:
        return json.dumps({"error": f"Wystąpił nieoczekiwany błąd podczas syntezy analiz: {e}"})
