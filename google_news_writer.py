import time #Iwish
import os
import json
import requests
import streamlit as st
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

GEMINI_MODEL_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

GEMINI_MODEL_LABELS = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}

GEMINI_GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]


def _exception_message(exc: Exception) -> str:
    return str(exc).lower()


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InternalServerError,
            google_exceptions.TooManyRequests,
        ),
    ):
        return True
    message = _exception_message(exc)
    return any(
        token in message
        for token in ("429", "quota", "rate limit", "overloaded", "503", "timeout")
    )


def _is_fatal_gemini_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            google_exceptions.Unauthenticated,
            google_exceptions.PermissionDenied,
            google_exceptions.InvalidArgument,
        ),
    ):
        return True
    message = _exception_message(exc)
    if not os.getenv("GEMINI_API_KEY"):
        return True
    return any(
        token in message
        for token in (
            "api key",
            "api_key",
            "permission denied",
            "unauthenticated",
            "invalid api key",
        )
    )


def _is_safety_block_error(exc: Exception) -> bool:
    message = _exception_message(exc)
    return any(
        token in message
        for token in ("safety", "blocked", "block_reason", "candidate was blocked")
    )


def _should_fallback_to_next_model(exc: Exception) -> bool:
    if _is_fatal_gemini_error(exc) or _is_safety_block_error(exc):
        return False
    if _is_transient_error(exc):
        return True
    if isinstance(exc, google_exceptions.NotFound):
        return True
    message = _exception_message(exc)
    return any(
        token in message
        for token in ("not found", "not supported", "404")
    )


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Alwrity - AI News Reporter(Beta)",
        layout="wide",
    )
    # Remove the extra spaces from margin top.
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                        ::-webkit-scrollbar-track {
        background: #e1ebf9;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #90CAF9;
            border-radius: 10px;
            border: 3px solid #e1ebf9;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #64B5F6;
        }

        ::-webkit-scrollbar {
            width: 16px;
        }
        div.stButton > button:first-child {
            background: #1565C0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            font-weight: bold;
        }
      </style>
    """
    , unsafe_allow_html=True)

    # Hide top header line
    hide_decoration_bar_style = '<style>header {visibility: hidden;}</style>'
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # Hide footer
    hide_streamlit_footer = '<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>'
    st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

    # Title and description
    st.title("✍️ Alwrity - AI News Report Generator(Beta)")

    with st.expander("**PRO-TIP** - Read the instructions below.", expanded=True):
        news_keywords = st.text_input("Enter News Headlines to search Google News on:", 
                                      placeholder="News item to get latest results on..")
        col1, col2, space = st.columns([5, 5, 0.5])
        with col1:
            # Radio Buttons for Origin Country
            news_country = st.radio("**Select origin country of the News event:**",
                         options=["Spain", "Vietnam", "Pakistan", "India", "Germany", "China"],
                         index=3)
        with col2:
            # Radio Buttons for News Language
            news_language = st.radio("**Select news article language to search for:**",
                         options=["English", "Spanish", "Vietnamese", "Arabic", "Hindi", "German", "Chinese"],
                         index=0)
        
    # Generate Blog FAQ button
    if st.button('**Write News Report**'):
        if not news_keywords or len(news_keywords.split()) < 2:
            st.error("🚫 News keywords should be at least two words long. Least, you can do..")
        
        with st.status("Assigning the News to Virtual Reporter..", expanded=True) as status:
            st.write(f"Reading News articles on {news_keywords} from {news_country}..")
            news_report = perform_serper_news_search(news_keywords, news_country, news_language, status)
            # Clicking without providing data, really ?
            if news_report:
                status.update(label="Found some News aritcles, Creating News report..")
                final_report = write_news_google_search(
                    news_keywords, news_country, news_language, news_report, status
                )
                if final_report:
                    st.subheader(
                        f"**🧕🔬👩 Verify: Alwrity can make mistakes. Your Final News Report on {news_keywords}!**"
                    )
                    st.write(final_report)
                    st.write("\n\n\n\n\n")
                    status.update(
                        label="Done: Scroll Down. Please Verify, Alwrity can make mistakes!",
                        state="complete",
                        expanded=True,
                    )
                else:
                    st.write("💥**Failed to generate News Report. Please try again!**")
            else:
                st.write("💥**Failed to generate News Report. Please try again!**")


def write_news_google_search(news_keywords, news_country, news_language, search_results, status):
    """Combine the given online research and gpt blog content"""
    news_language = get_language_name(news_language)
    news_country = get_country_name(news_country)

    prompt = f"""
        As an experienced {news_language} news journalist and senior editor.
        I will provide you with my 'News keywords' and its 'Google News Results', as context.
        Your Task is to write a detailed {news_language} News report, from the given Google News Results.
        Important, as a news report, its imperative that your content is factually correct and cited.
        
        Follow below guidelines:
        1). Understand and utilize the provided Google News Results.
        2). Always provide in-line citations and provide referance links.
        4). Always include the dates when then news was reported.
        6). Do not explain, describe your response.
        7). Your news report should be highly formatted in markdown style and in {news_language} language.

        \n\nNews Keywords: '''{news_keywords}'''\n\n
        Google News Result: '''{search_results}'''
        """
    status.update(label="Writing News report from Google News search results.")
    response = generate_text_with_exception_handling(prompt, status=status)
    if not response:
        st.error("Failed to get response from LLM. Please try again.")
    return response


def perform_serper_news_search(news_keywords, news_country, news_language, status):
    """ Function for Serper.dev News google search """
    # Get the Serper API key from environment variables
    news_language = get_language_name(news_language)
    news_country = get_country_name(news_country)

    status.update(label=f"Doing Google News Search, Wait for Breaking News.. {news_keywords} - {news_country} - {news_language}")
    serper_api_key = os.getenv('SERPER_API_KEY')

    # Check if the API key is available
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY is missing. Set it in the .env file.")

    # Serper API endpoint URL
    url = "https://google.serper.dev/news"
    payload = json.dumps({
        "q": news_keywords,
        "gl": news_country,
        "hl": news_language
    })
    # Request headers with API key
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    # Send a POST request to the Serper API with progress bar
    with st.spinner("Doing Google News Search, Wait for Breaking News.."):
        response = requests.post(url, headers=headers, data=payload, stream=True)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse and return the JSON response
            #process_search_results(response, "news")
            return response.json()
        else:
            # Print an error message if the request fails
            st.error(f"Error: {response.status_code}, {response.text}")


@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
def _generate_with_gemini_model(prompt: str, model_name: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Set it in the .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=GEMINI_GENERATION_CONFIG,
        safety_settings=GEMINI_SAFETY_SETTINGS,
    )
    convo = model.start_chat(history=[])
    convo.send_message(prompt)
    text = convo.last.text
    if not text or not text.strip():
        raise ValueError("Gemini returned an empty response.")
    return text


def generate_text_with_exception_handling(prompt, status=None):
    """
    Generates text using Gemini 2.5 models with per-model retry and fallback.

    Returns:
        str: The generated text, or None if all models fail.
    """
    last_error = None

    for index, model_name in enumerate(GEMINI_MODEL_CHAIN):
        label = GEMINI_MODEL_LABELS.get(model_name, model_name)
        if status:
            status.update(label=f"Generating report with {label}...")

        try:
            return _generate_with_gemini_model(prompt, model_name)
        except Exception as exc:
            last_error = exc
            if _is_fatal_gemini_error(exc):
                st.error(f"Gemini configuration error: {exc}")
                return None
            if _is_safety_block_error(exc):
                st.error(
                    "Content was blocked by Gemini safety filters. "
                    "Try different keywords or adjust your query."
                )
                return None
            if not _should_fallback_to_next_model(exc):
                st.error(f"Failed to generate report with {label}: {exc}")
                return None

            next_index = index + 1
            if next_index < len(GEMINI_MODEL_CHAIN):
                next_model = GEMINI_MODEL_CHAIN[next_index]
                next_label = GEMINI_MODEL_LABELS.get(next_model, next_model)
                st.warning(f"{label} unavailable ({exc}). Trying {next_label}...")
                if status:
                    status.update(label=f"{label} failed, trying {next_label}...")

    st.error(
        "All Gemini models failed to generate the report. "
        f"Last error: {last_error}"
    )
    return None


def get_language_name(language_code):
    languages = {
            "Spanish": "es",
            "Vietnamese": "vn",
            "English": "en",
            "Arabic": "ar",
            "Hindi": "hi",
            "German": "de",
            "Chinese (Simplified)": "zh-cn"
        # Add more language codes and corresponding names as needed
    }
    return languages.get(language_code, "Unknown")


def get_country_name(country_code):
    countries = {
            "Spain": "es",
            "Vietnam": "vn",
            "Pakistan": "pk",
            "India": "in",
            "Germany": "de",
            "China": "cn"
        # Add more country codes and corresponding names as needed
        }
    return countries.get(country_code, "Unknown")


if __name__ == "__main__":
    main()
