import openai
import streamlit as st
import re
import pandas as pd
import yfinance as yf
import json

# OpenAI client initialization
openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = None
client_initialized = False

if openai_api_key:
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        # Test connection by listing models (or any simple API call)
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
        stock = yf.Ticker(ticker)
        info = stock.info

        # Pobieranie wskaźników finansowych i innych danych
        # Klucze pobrane z yfinance.info, można rozszerzyć w razie potrzeby
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
        
        # Filtrujemy None values, aby uniknąć pustych pól w danych przekazywanych do LLM.
        # Zachowujemy kluczowe pola nawet jeśli są puste, ale z domyślnymi komunikatami.
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

    # Zbudowanie promptu z wszystkimi danymi
    prompt_parts = [f"Dane dla tickera: {ticker}"]
    
    if financial_data:
        prompt_parts.append("Dane finansowe i opis spółki:")
        # Zmieniamy json.dumps dla wskaźników na bardziej czytelny, jeśli jest duży
        indicators_str = json.dumps(financial_data.get('indicators', {}), indent=2, ensure_ascii=False)
        prompt_parts.append(f"  Wskaźniki: {indicators_str}")
        prompt_parts.append(f"  Opis działalności: {financial_data.get('description', 'Brak opisu działalności spółki.')}")
        prompt_parts.append(f"  Sektor: {financial_data.get('sector', 'Brak sektora.')}")
        prompt_parts.append(f"  Branża: {financial_data.get('industry', 'Brak branży.')}")
    else:
        prompt_parts.append("Brak danych finansowych do analizy.")

    if news:
        prompt_parts.append("ajnowsze wiadomości:")
        for i, new in enumerate(news):
            prompt_parts.append(f"  {i+1}. {new}")
    else:
        prompt_parts.append("rak najnowszych wiadomości.")

    # Finalny prompt
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
        # Używamy gpt-4o-mini zgodnie z obecną implementacją. Jeśli chcesz użyć gpt-4o, zmień model na "gpt-4o".
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od analizy fundamentalnej spółek giełdowych. Twoim zadaniem jest tworzenie zwięzłych analiz w formacie JSON wyłącznie na podstawie dostarczonych danych."},
                {"role": "user", "content": final_prompt}
            ],
            # Ustawiamy response_format aby upewnić się, że dostaniemy JSON
            response_format={"type": "json_object"},
            max_tokens=800 # Zwiększamy tokeny dla bardziej szczegółowej analizy JSON
        )

        json_output = response.choices[0].message.content.strip()
        
        # Walidacja czy odpowiedź jest rzeczywiście JSON
        try:
            json.loads(json_output)
            return json_output
        except json.JSONDecodeError:
            # Jeśli model nie zwrócił poprawnego JSON, zwracamy błąd
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

        # Prompt proszący o format JSON, bez informacji o kosztach
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
                # Zaktualizowana rola systemowa dla wyjścia JSON
                {"role": "system", "content": "Jesteś ekspertem od analizy technicznej akcji giełdowych. Twoim zadaniem jest generowanie analizy w formacie JSON na podstawie dostarczonych danych. Zwracaj tylko obiekt JSON."},
                {"role": "user", "content": final_prompt}
            ],
            # Żądanie wyjścia JSON
            response_format={"type": "json_object"},
            # max_tokens może być nadal użyteczny, ale response_format powinien pomóc w strukturze.
            # Pozostawiamy bez max_tokens na razie, aby model miał więcej swobody w generowaniu JSON.
        )

        json_output = response.choices[0].message.content.strip()
        
        # Walidacja czy odpowiedź jest rzeczywiście JSON
        try:
            parsed_json = json.loads(json_output)
            
            # Upewnij się, że wymagane klucze istnieją
            if 'technical_analysis' not in parsed_json:
                parsed_json['technical_analysis'] = "Brak danych."
            if 'recommendation' not in parsed_json:
                parsed_json['recommendation'] = "Brak danych."
            
            return json.dumps(parsed_json, indent=2, ensure_ascii=False) # Zwracamy ładnie sformatowany JSON

        except json.JSONDecodeError:
            # Jeśli model nie zwrócił poprawnego JSON mimo response_format, zwracamy błąd JSON
            return json.dumps({"error": "Model nie zwrócił poprawnego formatu JSON.", "raw_output": json_output})
        except Exception as e: # Obsługa innych błędów podczas przetwarzania
             return json.dumps({"error": f"Błąd przetwarzania odpowiedzi AI: {e}", "raw_output": json_output})

    except Exception as e:
        # Obsługa potencjalnych błędów podczas wywołania API
        return json.dumps({"error": f"Wystąpił błąd podczas zapytania do OpenAI API dla analizy technicznej: {e}"})
