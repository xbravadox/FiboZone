import openai
import streamlit as st

openai_api_key = st.secrets.get("OPENAI_API_KEY")

client = None
client_initialized = False
if openai_api_key:
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        # Testowe zapytanie, aby sprawdzić połączenie i klucz
        test_model = "gpt-3.5-turbo"
        client.models.list() # Sprawdza autentykację
        # st.success("Połączenie z OpenAI API udane. Klucz API jest poprawny.") # Usunięto komunikat sukcesu
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
    '''
    Funkcja testująca połączenie z OpenAI API.
    '''
    if client_initialized:
        return "Połączenie z OpenAI API jest aktywne."
    else:
        return "Połączenie z OpenAI API nie jest aktywne. Sprawdź klucz API i konfigurację."

def analyze_fundamental_with_gpt4o(ticker: str) -> str:
    '''
    Przeprowadza analizę fundamentalną spółki przy użyciu GPT-4o.
    Zadaje proste pytanie o analizę fundamentalną dla podanego tickera.

    Args:
        ticker (str): Symbol tickera spółki.

    Returns:
        str: Odpowiedź LLM dotycząca analizy fundamentalnej.
    '''
    if not client:
        return "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."

    try:
        prompt = f"Podaj krótką analizę fundamentalną dla spółki o tickerze {ticker}."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Używamy GPT-4o zgodnie z projektem
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem udzielającym analiz finansowych."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Wystąpił błąd podczas zapytania do OpenAI: {e}"

def analyze_technical_with_gpt4o_mini(pivots_data: dict, trend_data: dict) -> str:
    '''
    Generuje ustrukturyzowaną analizę techniczną w formacie JSON przy użyciu GPT-4o-mini.

    Args:
        pivots_data (dict): Dane dotyczące punktów zwrotnych (pivotów).
        trend_data (dict): Dane dotyczące trendu.

    Returns:
        str: Ustrukturyzowana analiza techniczna w formacie JSON.
    '''
    if not client:
        return "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."
    return "Analiza techniczna GPT-4o-mini nie została jeszcze zaimplementowana."