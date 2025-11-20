import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Prefer the official GOOGLE_API_KEY, fall back to GEMINI_API_KEY for legacy setups
_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Configure if available; otherwise we'll handle gracefully at call time
if _api_key:
    genai.configure(api_key=_api_key)


def generate_itinerary(source: str, destination: str, start_date: str, end_date: str, no_of_day: int) -> str:
    """Generate a travel itinerary using Google's Gemini model.

    Returns markdown formatted text suitable for rendering on the dashboard.
    """

    # --- PROMPT HAS BEEN UPDATED BELOW ---
    prompt = (
        f"You are an expert travel consultant specializing in {destination}. Your task is to create a "
        f"comprehensive, highly detailed, day-by-day travel itinerary for a {no_of_day}-day trip "
        f"from {source} to {destination}, starting on {start_date} and ending on {end_date}. "
        f"The itinerary must include the following for each day: \n"
        f"1. **Daily Schedule:** A logical flow of activities for Morning, Afternoon, and Evening. Include a mix of popular landmarks and unique, local 'hidden gem' experiences. \n"
        f"2. **Dining Suggestions:** Recommend specific types of local cuisine for lunch and dinner. Suggest one budget-friendly option and one mid-range restaurant for each day. \n"
        f"3. **Logistics & Travel:** Provide practical tips on local transportation (e.g., best metro lines to use, ride-sharing app advice, walking routes). \n"
        f"4. **Budgeting:** Give an estimated daily budget breakdown in INR, separated into categories: Food, Transport, and Activities. \n"
        f"5. **Pro-Tips:** Include insider advice, like the best time to visit an attraction to avoid crowds or booking passes in advance. \n"
        f"The final output must be in clear, well-structured markdown. Use headings for each day (e.g., 'Day 1: Arrival and Exploration') and bullet points for all details."
    )
    # --- END OF PROMPT UPDATE ---

    # Ensure the API key is present
    if not _api_key:
        return (
            "Missing Gemini API key. Add GOOGLE_API_KEY to your .env or set it in your environment, "
            "then restart the app."
        )

    # Ensure the installed SDK supports GenerativeModel
    if not hasattr(genai, "GenerativeModel"):
        return (
            "Itinerary engine needs an update. Please run: pip install -U google-generativeai "
            "and restart the app."
        )

    try:
        # (Re)configure in case the process started before the key was available
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = getattr(response, "text", "")
        if not text:
            return "No itinerary could be generated at this time. Please try again."
        return text
    except Exception as exc:  # Log and return a helpful message instead of raising
        logging.exception("Gemini itinerary generation failed: %s", exc)
        return (
            "Itinerary generation failed. Check that GOOGLE_API_KEY is set in your .env, "
            "your internet connection is active, and try again. If the issue persists, run: "
            "pip install -U google-generativeai"
        )