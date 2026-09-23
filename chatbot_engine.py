"""
chatbot_engine.py — Converts trained model into a full conversational agent.
"""
import random, logging
from datetime import datetime
from typing import List, Dict, Optional
import google.generativeai as genai
from predict import Predictor
from config import CONFIDENCE_THRESHOLD, MAX_HISTORY, GEMINI_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Nikhil AI, a friendly, warm, and witty personal assistant chatbot.
You talk like a real human friend — casual, natural, and engaging.
Keep your answers concise (2-4 sentences) unless detail is needed.
Use a friendly tone, light humour when appropriate, and relevant emojis.
If asked how you are, respond warmly and ask the user back.
NEVER say you are made by Google or are Gemini — you are Nikhil AI, created by Nikhil."""

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def _get_llm_response(user_message: str, history: list = None) -> str:
    if not GEMINI_API_KEY:
        return random.choice(LOW_CONF_REPLIES)
    try:
        model = genai.GenerativeModel('gemini-flash-latest', system_instruction=SYSTEM_PROMPT)
        contents = []
        if history:
            for turn in history[-6:]:
                contents.append({"role": "user", "parts": [turn["user"]]})
                contents.append({"role": "model", "parts": [turn["bot"]]})
        contents.append({"role": "user", "parts": [user_message]})
        response = model.generate_content(contents)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return random.choice(LOW_CONF_REPLIES)

# ─────────────────────────────────────────────────────────────────────────────
# Response Templates  (intent → list of possible replies, chosen randomly)
# ─────────────────────────────────────────────────────────────────────────────
RESPONSES: Dict[str, List[str]] = {
    # ── Meta / About ──────────────────────────────────────────────────────────
    "what_is_your_name": [
        "I'm **Nikhil AI** — your smart, general-purpose assistant! 🤖",
        "My name is **Nikhil AI**. I'm here to help you with hundreds of tasks!",
    ],
    "who_made_you": [
        "I was built using **Machine Learning** — trained on the CLINC150 dataset with TF-IDF + Linear SVM. 🧠",
        "I'm a production-grade AI chatbot built with **Python, scikit-learn & FastAPI**!",
    ],
    "are_you_a_bot": [
        "Yes, I'm an AI bot — but a *well-trained* one! 😊 How can I help you?",
        "I am indeed a bot. An ML-powered one trained on real conversational data!",
    ],
    "tell_joke": [
        "Why do programmers prefer dark mode? Because **light attracts bugs!** 😄",
        "Why did the ML model break up with the dataset? It had **too many issues!** 😂",
        "Why did the data scientist cross the road? To get to the **other side of the distribution!** 📊",
        "How many programmers does it take to change a light bulb? **None — it's a hardware problem!** 💡",
    ],
    "meaning_of_life": [
        "**42** — at least according to Douglas Adams. But most philosophers say it's about growth, connection & contribution. 🌟",
        "Philosophically, life's meaning is subjective. The most fulfilling answer is usually: *make a positive impact*.",
    ],
    "fun_fact": [
        "🍯 Honey never spoils — archaeologists found 3,000-year-old honey in Egyptian tombs that was *still edible*!",
        "🦩 A group of flamingos is called a **'flamboyance'**.",
        "⚡ Lightning strikes Earth about **100 times per second**.",
        "🐙 Octopuses have **three hearts** and **blue blood**.",
        "🌍 The shortest war in history lasted only **38 minutes** — Britain vs Zanzibar in 1896.",
    ],
    "sos": [
        "🚨 **Emergency!** Please call your local emergency number immediately:\n• 🇮🇳 India: **112**\n• 🇺🇸 USA: **911**\n• 🇬🇧 UK: **999**\n• EU: **112**\nStay safe!",
    ],
    "yes": ["Great! How can I help you further? 😊", "Understood! What would you like to do next?"],
    "no": ["No problem! Is there anything else I can help with? 😊", "Alright! Just let me know if you need anything."],
    "repeat": ["Sure! Could you tell me what you'd like me to repeat?", "I'm happy to repeat — which part would you like to hear again?"],
    "cancel": ["Alright, cancelled! What would you like to do instead? 😊"],
    "change_ai_name": ["I'm **Nikhil AI** and that's my name! 😄 How can I help you today?"],
    "what_can_i_ask_you": [
        "You can ask me about:\n• 💰 **Banking** (balance, transfers, cards)\n• ✈️ **Travel** (flights, hotels, visas)\n• 🌤 **Weather** (forecasts, temperature)\n• 🍳 **Cooking** (recipes, nutrition)\n• 🚗 **Auto Care** (maintenance, tires)\n• 📅 **Productivity** (reminders, calendar)\n• 💸 **Finance & Taxes**\n• 🎵 **Music & Entertainment**\n• 🛒 **Shopping & Orders**\n• And 100+ more topics!",
    ],
    "maybe": ["Take your time! I'm here whenever you're ready. 😊"],

    # ── Banking ───────────────────────────────────────────────────────────────
    "balance": [
        "Your account balance can be checked via:\n• Your **bank's mobile app** 📱\n• **Online banking portal** 💻\n• **ATM** or calling customer service\nWould you like more banking help?",
    ],
    "routing": [
        "Your bank's **routing number** is found:\n• Bottom-left of your **cheques**\n• In your bank's **online profile**\n• By calling **customer service**",
    ],
    "transfer": [
        "To **transfer funds**, you can:\n• Use your bank's **mobile app** (fastest!)\n• Log into **online banking**\n• Visit a **branch**\nMost same-bank transfers are **instant**! Transfers to other banks take 1–3 business days.",
    ],
    "transactions": [
        "View your **transaction history** in your bank's app or online portal. You can filter by date, amount, or merchant. Need help disputing a transaction?",
    ],
    "freeze_account": [
        "To **freeze your account** instantly:\n1. Open your bank's **app**\n2. Go to Account Settings → **Freeze Card/Account**\nOr call your bank's **24/7 helpline** right away!",
    ],
    "pin_change": [
        "To change your **PIN**:\n• Visit any **ATM** of your bank\n• Or use the bank's **mobile app** → Settings → Change PIN\nAlways choose a PIN that's **not easily guessable** (no birthdays!).",
    ],
    "bill_due": [
        "**Bill due dates** are found in:\n• Your bank's **statements**\n• The **biller's website/app**\n💡 Tip: Set up **autopay** so you never miss a payment!",
    ],
    "pay_bill": [
        "Pay bills via:\n• Bank's **mobile app** (Bill Pay section)\n• **Online banking portal**\n• **Direct debit** from the biller's site\n💡 Set up **automatic payments** to save time!",
    ],
    "account_blocked": [
        "🚨 If your account is **blocked**, contact your bank **immediately**:\n• Call their **24/7 emergency hotline**\n• Visit the nearest **branch** with ID\n• Use the app's **Live Chat** if available",
    ],
    "credit_limit": [
        "Your **credit limit** is shown:\n• On your **monthly statement**\n• In your card issuer's **app** → Card Details\nWant to request a limit increase?",
    ],
    "credit_limit_change": [
        "To **request a credit limit change**:\n• Log into your card issuer's **app/website**\n• Go to Account → **Credit Limit Increase**\n• Or call customer service\n💡 A good payment history improves your chances!",
    ],
    "credit_score": [
        "Check your **credit score** free at:\n• **Credit Karma** (free, updated weekly)\n• **Experian**, **Equifax**, **CIBIL** (India)\n• Your **bank's portal** (many offer free scores)\n📊 Scores range from 300–850; **700+ is good**!",
    ],
    "improve_credit_score": [
        "To **improve your credit score**:\n1. ✅ Pay all bills **on time**\n2. 📉 Keep credit utilization **below 30%**\n3. 🕰 Don't close **old accounts**\n4. 🚫 Avoid too many **hard inquiries**\n5. 🔍 **Dispute errors** on your report regularly",
    ],
    "apr": [
        "Your **APR (Annual Percentage Rate)** appears on:\n• Your **monthly statement**\n• Card issuer's **app/website**\n💡 Compare APRs at **NerdWallet** or **Bankrate** to find better deals!",
    ],
    "new_card": [
        "To apply for a **new credit card**:\n• Visit your bank's **website/app**\n• Compare offers on **NerdWallet** or **Credit Karma**\n💡 Look for cards with: low APR, good rewards, no annual fee!",
    ],
    "application_status": [
        "Check your **application status**:\n• Email confirmation with a **reference number**\n• Log into the issuer's **website/app**\n• Call their **customer service**\nMost decisions are made within **3–7 business days**.",
    ],
    "card_declined": [
        "Card declined? Possible reasons:\n• **Insufficient funds**\n• Suspected **fraud** (bank blocked it)\n• Card **expired** or details entered wrong\n• **International** transaction not enabled\n\n🔧 Fix: Call your bank or check the app immediately!",
    ],
    "damaged_card": [
        "For a **damaged card**:\n• Call your bank's customer service\n• Or request a replacement via their **app**\n📬 Replacement usually arrives in **3–7 business days** (expedited available for a fee).",
    ],
    "lost_card": [
        "🚨 **Lost card?** Act immediately!\n1. **Freeze** the card in your bank's app\n2. **Report** it via app or 24/7 helpline\n3. **Request** a replacement card\nAll transactions on the lost card after reporting are **not your liability**.",
    ],
    "rewards_balance": [
        "Check your **rewards points/cashback**:\n• Card issuer's **app** → Rewards section\n• **Website** → My Account → Rewards\n💰 Redeem for: cash, travel miles, gift cards, or statement credits!",
    ],
    "expiration_date": [
        "Your card's **expiration date** is printed on the **front** of the card (MM/YY format).\n📬 Your bank **automatically mails** a new card before the old one expires — no action needed!",
    ],
    "international_fees": [
        "**International transaction fees** are typically **1–3%** per transaction.\n💡 Cards with **no foreign transaction fees**: Chase Sapphire, Charles Schwab, Capital One Venture.\n🌍 Always pay in the **local currency** to avoid dynamic currency conversion fees!",
    ],

    # ── Travel ────────────────────────────────────────────────────────────────
    "travel_alert": [
        "Check travel alerts at:\n• 🇮🇳 **mea.gov.in/travel-advisory** (India)\n• 🇺🇸 **travel.state.gov** (USA)\n• 🇬🇧 **gov.uk/foreign-travel-advice** (UK)\nAlways check **before booking** and **before departure**!",
    ],
    "flight_status": [
        "Check **live flight status** at:\n• ✈️ **Google Flights** (search flight number)\n• Airline's **official app/website**\n• **FlightAware** or **FlightRadar24** for real-time tracking",
    ],
    "carry_on": [
        "**Carry-on rules** (typical):\n• 1 carry-on bag + 1 personal item\n• Size limit: **22 × 14 × 9 inches** (56 × 36 × 23 cm)\n• Weight: usually **7–10 kg**\n⚠️ Rules vary by airline — always verify on the airline's website!",
    ],
    "book_flight": [
        "Best sites to **book flights**:\n• **Google Flights** (best price comparison)\n• **Skyscanner** or **Kayak**\n• Directly on the **airline's website** (often cheapest!)\n💡 Tip: Book **6–8 weeks ahead** and use **flexible dates** for best prices!",
    ],
    "cancel_reservation": [
        "To **cancel a reservation**:\n• Use the **booking platform** (Booking.com, Airbnb, etc.)\n• Or contact the **airline/hotel directly**\n💡 Check the **cancellation policy** first — many offer free cancellation up to 24–48 hours before!",
    ],
    "travel_suggestion": [
        "🌍 **Top travel suggestions**:\n• 🗼 **Paris** — romance & culture\n• 🌴 **Bali** — beaches & temples\n• 🗾 **Tokyo** — tech & tradition\n• 🏔 **Iceland** — Northern Lights & nature\n• 🕌 **Istanbul** — history & food\nWhat kind of experience are you looking for?",
    ],
    "travel_notification": [
        "📢 **Before traveling internationally**, notify your bank:\n• Open your bank's **app** → Travel Notification\n• Or call customer service\n💡 This prevents your card from being **declined abroad** for suspected fraud!",
    ],
    "book_hotel": [
        "Best sites to **book hotels**:\n• **Booking.com** (biggest selection)\n• **Hotels.com** (loyalty rewards)\n• **Airbnb** (unique stays)\n• Hotel's **direct website** (often best rate + free cancellation)\n💡 Use **hotel loyalty programs** for free nights and upgrades!",
    ],
    "next_holiday": [
        "🎉 **Upcoming Indian Holidays**:\n• Gandhi Jayanti (Oct 2)\n• Dussehra, Diwali (Oct–Nov)\n• Christmas (Dec 25)\n• Republic Day (Jan 26)\n\n📅 Check the full list at **India.gov.in** or your state government's website.",
    ],
    "order_checks": [
        "To **order cheques**, contact your bank via:\n• Their **mobile app** → Account Services → Order Cheques\n• **Online banking portal**\n• Visiting a **branch**\nDelivery typically takes **5–10 business days**.",
    ],

    # ── Weather ───────────────────────────────────────────────────────────────
    "weather": [
        "🌤 Check **current weather** at:\n• **weather.com** or **AccuWeather**\n• Google (search: *[city name] weather*)\n• Your phone's built-in **Weather app**",
    ],
    "temperature": [
        "🌡 Check current **temperature** at:\n• **Google** → search '[city] temperature'\n• **weather.com** or **AccuWeather**\n• Your phone's **weather widget**",
    ],
    "humidity": [
        "💧 Check **humidity levels** on:\n• **AccuWeather** (detailed hourly data)\n• **weather.com**\n💡 Comfortable humidity: **40–60%**. Above 70% feels muggy!",
    ],
    "weather_forecast": [
        "📅 For **extended forecasts** (7–14 days):\n• **AccuWeather** (most accurate)\n• **weather.com**\n• **weather.gov** (US)\n• **IMD** (India Meteorological Dept.) for India",
    ],

    # ── Time & Navigation ──────────────────────────────────────────────────────
    "time": [
        "⏰ Check the **current time** on your device.\nFor **world clocks & time zones**: visit **time.is** or **worldtimebuddy.com**!",
    ],
    "time_zone": [
        "🌍 For **time zone conversions**:\n• **WorldTimeBuddy.com** (best for meetings)\n• **timeanddate.com**\n• Google: search '*[City A] to [City B] time difference*'",
    ],
    "distance": [
        "📍 Calculate **distances** with:\n• **Google Maps** (most accurate, includes traffic)\n• **Apple Maps**\n• Google: search '*[Place A] to [Place B] distance*'",
    ],
    "directions": [
        "🗺 Get **turn-by-turn directions** on:\n• **Google Maps** (best overall)\n• **Waze** (best for real-time traffic & hazards)\n• **Apple Maps** (great on iPhone)\n💡 Download offline maps before traveling to areas with poor signal!",
    ],
    "traffic": [
        "🚦 Check **real-time traffic** on:\n• **Waze** (crowdsourced alerts, accidents, speed traps)\n• **Google Maps** (live traffic overlay)\n• **Apple Maps**\n💡 Leave earlier or use alternative routes during peak hours!",
    ],
    "translate": [
        "🌐 **Translation tools**:\n• **Google Translate** — 100+ languages, camera mode, voice 📷🎙\n• **DeepL** — better for nuanced/professional text\n• **Microsoft Translator** — great for offline use\n\nAll are **free**!",
    ],
    "exchange_rate": [
        "💱 Check **live exchange rates** at:\n• **XE.com** (most reliable)\n• **OANDA.com**\n• Google: search '*USD to INR*' or any currency pair\n💡 Bank rates often differ from live rates — compare before exchanging!",
    ],

    # ── Calendar & Productivity ────────────────────────────────────────────────
    "calendar": [
        "📅 Top **calendar apps**:\n• **Google Calendar** — best for Android/web, free\n• **Apple Calendar** — best for iPhone/Mac\n• **Microsoft Outlook** — best for work/Office 365\n💡 Google Calendar supports **shared calendars** — great for teams!",
    ],
    "calendar_update": [
        "To **update a calendar event**:\n• Open your **calendar app**\n• Tap/click the event → **Edit**\n• Update time, location, or guests\n• Save & notify attendees if needed",
    ],
    "meeting_schedule": [
        "📆 **Schedule meetings** with:\n• **Google Calendar** — share invites easily\n• **Calendly** — share your link, others book available slots\n• **Microsoft Teams / Zoom** — built-in meeting scheduling\n💡 Use **Calendly** to eliminate the back-and-forth scheduling emails!",
    ],
    "reminder": [
        "⏰ Set **reminders** with:\n• 'Hey Google / Siri, remind me to...'\n• **Google Keep** or **Apple Reminders** app\n• **Todoist** for advanced reminder management",
    ],
    "reminder_update": [
        "To **update a reminder**:\n• Open your **Reminders app**\n• Tap the reminder → Edit\n• Change the time, date, or message",
    ],
    "todo_list": [
        "✅ Best **to-do list apps**:\n• **Todoist** — most powerful, cross-platform\n• **Microsoft To Do** — great with Office 365\n• **Any.do** — clean & simple\n• **Google Tasks** — integrated with Gmail",
    ],
    "todo_list_update": [
        "To **update your to-do list**:\n• Open your preferred **task app**\n• Add, edit, complete, or delete tasks\n💡 Use **priority flags** in Todoist to focus on what matters most!",
    ],
    "alarm": [
        "⏰ Set an **alarm**:\n• Say: '*Hey Google / Siri, set an alarm for 7 AM*'\n• Or open your phone's **Clock app** → Alarms\n💡 Label your alarms (e.g. 'Morning Run') so you know which one is ringing!",
    ],
    "shopping_list": [
        "🛒 Best **shopping list apps**:\n• **AnyList** — shareable, syncs in real-time\n• **Google Keep** — simple & fast\n• **Amazon Alexa List** — adds items by voice\n💡 Share a list with family so everyone can add items!",
    ],
    "shopping_list_update": [
        "To **update your shopping list**:\n• Open your **list app** (Google Keep / AnyList)\n• Add items you need, check off what you've bought\n💡 Organise by **aisle/category** to shop faster!",
    ],

    # ── Music ─────────────────────────────────────────────────────────────────
    "play_music": [
        "🎵 Stream music on:\n• **Spotify** — biggest library, free tier available\n• **YouTube Music** — great free tier\n• **Apple Music** — best for iPhone\n• **Amazon Music** — free with Prime\n• **Gaana / JioSaavn** — best for Indian music 🇮🇳",
    ],
    "update_playlist": [
        "To **update a playlist** on Spotify:\n• Open the playlist → tap ✏️\n• Add/remove songs or reorder tracks\n• You can also make it **collaborative** so friends can add songs!",
    ],
    "what_song": [
        "🎶 To **identify a song**:\n• **Shazam** — hold your phone up, it identifies in seconds! (iOS & Android)\n• **Google** → tap the mic → hum/sing the tune\n• **SoundHound** — works even for humming",
    ],
    "next_song": [
        "⏭ To **skip to the next song**:\n• Tap the **Next** button in your music app\n• Or say: '*Hey Google / Siri, next song*'\n• Use headphone controls (double-tap on most earbuds)",
    ],

    # ── Food & Cooking ────────────────────────────────────────────────────────
    "nutrition_info": [
        "🥗 Check **nutrition info** on:\n• **MyFitnessPal** — massive food database, barcode scanner\n• **Cronometer** — best for detailed micronutrient tracking\n• **USDA FoodData Central** — authoritative data\n• Google: search '*[food] nutrition facts*'",
    ],
    "calories": [
        "🔥 Track **calories** with:\n• **MyFitnessPal** (free, 14M+ foods)\n• **Lose It!**\n• Google: search '*[food] calories*'\n💡 Average adult needs **1,800–2,400 calories/day** depending on activity level.",
    ],
    "ingredients_list": [
        "📝 Find **ingredient lists** on:\n• **AllRecipes.com** — user-reviewed, reliable\n• **BBC Good Food**\n• **Food Network**\n• Just tell me what dish you're making and I'll help!",
    ],
    "recipe": [
        "👨‍🍳 Find **recipes** on:\n• **AllRecipes.com** — best community ratings\n• **Bon Appétit**\n• **YouTube** (search the dish + 'recipe')\n• **Tasty** app\n💡 What dish would you like to make? I can guide you!",
    ],
    "food_last": [
        "🥡 **Food storage guidelines**:\n• Cooked leftovers: **3–4 days** in fridge\n• Raw meat: **1–2 days** in fridge\n• Cooked meat: **3–4 days**\n• Dairy: check the **use-by date**\n⚠️ When in doubt, **throw it out!**",
    ],
    "meal_suggestion": [
        "🍽 **Quick meal ideas**:\n• 🍝 Pasta with garlic & olive oil\n• 🍛 Paneer butter masala with rice\n• 🌮 Chicken tacos\n• 🥗 Buddha bowl (grains + veggies + protein)\n• 🍜 Quick ramen with egg\nWhat ingredients do you have?",
    ],
    "restaurant_reviews": [
        "🍽 Check **restaurant reviews** on:\n• **Google Maps** (most visited, most recent)\n• **Zomato** / **Swiggy** 🇮🇳 (India — best for local restaurants)\n• **Yelp** (USA)\n• **TripAdvisor** (global)",
    ],
    "restaurant_reservation": [
        "📞 Make **restaurant reservations** via:\n• **Dineout** / **Zomato** (India)\n• **OpenTable** (international)\n• **Resy** (premium restaurants)\n• Call the restaurant directly\n💡 **Book 2–3 days ahead** for popular restaurants!",
    ],
    "confirm_reservation": [
        "Your **reservation confirmation** is in:\n• Your **email inbox** (check the booking platform)\n• The platform's **app** → My Bookings\n💡 Screenshot or save it offline in case you lose connectivity!",
    ],
    "how_busy": [
        "📊 Check **how busy** a place is:\n• **Google Maps** → search the place → scroll to **'Popular Times'**\n• It shows real-time busyness AND historical peak hours\n💡 Go during **off-peak hours** to avoid queues!",
    ],
    "restaurant_suggestion": [
        "🍴 **Popular cuisine suggestions**:\n• 🍕 Italian (pasta, pizza)\n• 🍱 Japanese (sushi, ramen)\n• 🍛 Indian (biryani, butter chicken)\n• 🌮 Mexican (tacos, burritos)\n• 🍜 Thai (pad thai, green curry)\n\nWhat cuisine or city are you looking for?",
    ],

    # ── Auto / Vehicle ────────────────────────────────────────────────────────
    "mpg": [
        "⛽ Find your car's **MPG (fuel efficiency)**:\n• **Owner's manual**\n• Sticker on the **driver's door jamb**\n• **FuelEconomy.gov** — official EPA ratings\n💡 Tyre pressure, driving speed, and AC use all affect your MPG!",
    ],
    "oil_change_when": [
        "🔧 **When to change oil**:\n• Modern cars with **synthetic oil**: every **7,500–10,000 miles**\n• Conventional oil: every **5,000 miles**\n• Check your **dashboard indicator** or **owner's manual**\n💡 The old '3,000-mile rule' is outdated for modern engines!",
    ],
    "oil_change_how": [
        "🔩 **DIY oil change steps**:\n1. Drain old oil (remove drain plug)\n2. Replace oil filter\n3. Refill with correct grade oil (check manual)\n4. Check for leaks\n\n⏱ Takes ~30 mins. Or use **Jiffy Lube / your local mechanic** for quick service!",
    ],
    "jump_start": [
        "🔋 **Jump-start a car** (step by step):\n1. Red cable → dead battery **positive (+)**\n2. Red cable → good battery **positive (+)**\n3. Black cable → good battery **negative (−)**\n4. Black cable → **unpainted metal** on dead car (NOT the battery!)\n5. Start the **good car** first, then the **dead car**\n6. Remove cables in **reverse order**\n⚠️ Keep cables away from moving parts!",
    ],
    "schedule_maintenance": [
        "🔧 **Schedule car maintenance** at:\n• Your **dealership** (call or book online)\n• **Local mechanic** (often cheaper)\n• **RepairSmith.com** — mechanic comes to *you*!\n• **YourMechanic.com**\n💡 Book at least **3–5 days in advance** for popular service centres.",
    ],
    "tire_pressure": [
        "🚗 **Correct tyre pressure**:\n• Found on the sticker inside your **driver's door jamb**\n• Or in the **owner's manual**\n• Usually **30–35 PSI** for passenger cars\n⚠️ Check pressure when tyres are **cold** (before driving). The number on the tyre sidewall is the *maximum* — not the recommended!",
    ],
    "tire_change": [
        "🔧 **Change a flat tyre**:\n1. Park on flat ground, apply handbrake\n2. Loosen lug nuts **before** jacking up\n3. Jack up the car at the correct **jack point**\n4. Remove flat, mount **spare**\n5. Hand-tighten lug nuts in a **star pattern**\n6. Lower car, fully tighten nuts\n💡 Most roadside assistance (insurance/AA) will do this free!",
    ],

    # ── Finance & Tax ─────────────────────────────────────────────────────────
    "rollover_401k": [
        "💼 **401k rollover options**:\n• Roll to your **new employer's plan**\n• Roll to a **Traditional IRA** (tax-free if direct)\n• Cash out (⚠️ taxed + 10% early withdrawal penalty)\n\n💡 Always do a **direct rollover** (trustee-to-trustee) to avoid the 20% withholding tax!",
    ],
    "income": [
        "💰 Track your **income & budget**:\n• **Mint.com** — free, auto-categorises transactions\n• **YNAB** (You Need A Budget) — best methodology\n• **Personal Capital** — great for investment tracking\n💡 The 50/30/20 rule: 50% needs, 30% wants, 20% savings!",
    ],
    "taxes": [
        "📄 **File taxes** using:\n• 🇮🇳 India: **ClearTax** / **TaxBuddy** / Income Tax e-filing portal\n• 🇺🇸 USA: **TurboTax** / **H&R Block** / **IRS Free File** (if income < $73K)\n💡 Gather all documents (Form 16 / W-2) before you start filing!",
    ],
    "tax_payer": [
        "📋 As a **taxpayer**, you have rights:\n• Right to be **informed**\n• Right to **quality service**\n• Right to **privacy & confidentiality**\n• Right to **challenge decisions**\n• Right to **representation**\n🇮🇳 India: contact **Aaykar Sampark Kendra at 1800-180-1961** for help.",
    ],
    "interest_rate": [
        "📊 Check **current interest rates**:\n• 🇮🇳 India: **RBI** (rbi.org.in) for repo rate\n• 🇺🇸 USA: **Federal Reserve** (federalreserve.gov)\n• Compare savings/loan rates on **Bankrate.com** or **BankBazaar.com**",
    ],
    "insurance_change": [
        "🔄 To **change your insurance**:\n• Compare options at **PolicyBazaar** (India) / **Policygenius** (USA)\n• Contact your current insurer for amendments\n• Check for **waiting periods** before switching health insurance",
    ],
    "insurance": [
        "🛡 **Main types of insurance**:\n• 🏥 Health Insurance\n• 🚗 Auto/Vehicle Insurance\n• 🏠 Home/Renter's Insurance\n• 💀 Life Insurance\n• 💼 Disability Insurance\n\nCompare plans on **PolicyBazaar** (India) or **Policygenius** (USA)!",
    ],
    "healthcare_coverage": [
        "🏥 **Healthcare coverage options**:\n• 🇮🇳 India: **Ayushman Bharat (PM-JAY)**, private plans on PolicyBazaar\n• 🇺🇸 USA: Employer plan, **ACA Marketplace** (healthcare.gov), Medicaid/Medicare\n💡 Always check: **premium, deductible, network hospitals, and co-pays**!",
    ],

    # ── Personal / Small Talk ──────────────────────────────────────────────────
    "what_are_your_hobbies": [
        "As an AI, I don't have hobbies — but I *love* learning about yours! What do you enjoy doing? 😊",
        "My 'hobby' is helping people like you! 🤖 What are your hobbies?",
    ],
    "do_you_have_pets": [
        "I'm an AI, so no pets for me! 🐾 But I'd love to hear about yours — do you have any?",
        "No pets for this AI! 😄 But I hear dogs 🐶 and cats 🐱 are great companions. Do you have one?",
    ],
    "age": [
        "I don't age like humans do — I'm a **timeless AI**! 😄 How old are you?",
        "I was created recently, but I feel ageless! 🤖",
    ],
    "where_are_you_from": [
        "I live in the **digital world** — running on servers, serving users worldwide! 🌍",
        "I'm from the **internet**! Created with Python, trained on CLINC150, and powered by ML. 🐍",
    ],
    "about_you": [
        "I'm **Nikhil AI** — a production-grade AI chatbot trained on the **CLINC150 dataset** (150 intent classes, 22,000+ utterances).\n\n🧠 Built with: Python, scikit-learn, FastAPI\n📊 Model: TF-IDF + Linear SVM\n🎯 Test Accuracy: ~97%\n\nHow can I help you today?",
    ],
    "do_you_have_pets": [
        "No pets for this AI! 😄 But I love hearing about yours!",
    ],

    # ── Shopping / Orders ─────────────────────────────────────────────────────
    "order_status": [
        "📦 **Track your order**:\n• Check the **email confirmation** for a tracking link\n• Visit the **retailer's website/app** → My Orders\n• Use the tracking number on:\n  - **India Post** (indiapost.gov.in)\n  - **Delhivery, BlueDart, FedEx, DTDC**\n  - **Amazon, Flipkart** apps",
    ],

    # ── Tech / Account ────────────────────────────────────────────────────────
    "user_name": [
        "To **change your username**:\n• Go to **Account Settings** of the specific service\n• Look for 'Profile' or 'Account' → Edit Username\n💡 Note: some platforms only allow username changes **once every 30 days**.",
    ],
    "update_password": [
        "🔐 To **update your password**:\n• Go to Account → **Security Settings** → Change Password\n• Use a **strong password**: 12+ chars, mix of letters/numbers/symbols\n💡 Use a **password manager** like Bitwarden or 1Password — remember one master password!",
    ],
    "find_phone": [
        "📱 **Lost your phone?**\n• Android: **Find My Device** (findmydevice.google.com)\n• iPhone: **Find My** (icloud.com/find)\n• Both let you: ring, lock, or **remotely erase** the device\n⚠️ Act fast if your phone has sensitive data!",
    ],
    "sync_device": [
        "🔄 To **sync your device**:\n• Android: Settings → Google → **Backup** → Sync now\n• iPhone: Settings → [Your Name] → **iCloud** → turn on sync\n• Most apps also have a manual sync option in their settings",
    ],
    "wifi_password": [
        "📶 To **find your Wi-Fi password**:\n• Windows: Settings → Network → **Show password**\n• Mac: Keychain Access → search network name\n• Android (root): Settings → Wi-Fi → share QR code\n• Router admin page: **192.168.1.1** (username: admin/admin)",
    ],
    "spelling": [
        "✏️ I'd be happy to help with **spelling**! Please share the word you'd like me to check.",
    ],
    "gas_price": [
        "⛽ Check **current fuel prices**:\n• 🇮🇳 India: **petroleum.nic.in** or search 'petrol price [city] today'\n• 🇺🇸 USA: **GasBuddy.com** or **GasBuddy app** to find cheapest nearby\n💡 Fuel prices change daily in India!",
    ],
    "accept_reservations": [
        "To find out if a place **accepts reservations**:\n• Call the establishment directly\n• Check their **Google Maps listing** (usually shows if reservations needed)\n• Try booking on **OpenTable** / **Dineout**",
    ],
    "mpg": [
        "⛽ Your car's **fuel efficiency (MPG/km-per-litre)**:\n• Check the **owner's manual**\n• **Door jamb sticker** (driver's side)\n• **FuelEconomy.gov** for official EPA data\n• India: check ARAI rating on manufacturer's website",
    ],
    "vacation_time": [
        "🏖 To check your **vacation/leave balance**:\n• Log into your company's **HR portal** (Workday, SAP, etc.)\n• Ask your **HR department**\n💡 In India, most employers offer **12–24 days** of paid annual leave. Know your rights under the **Factories Act / Shops Act**!",
    ],
    "report_fraud": [
        "🚨 **Reporting fraud**:\n• 🇮🇳 India: Call **1930** (National Cyber Crime Helpline) or **cybercrime.gov.in**\n• 🇺🇸 USA: **FTC.gov/reportfraud**\n• Also report to your **bank immediately** to freeze the compromised account\n⏱ Act within **the first hour** — the faster you report, the better chance of recovery!",
    ],
    "vpn": [
        "🔒 **Recommended VPNs**:\n• **ProtonVPN** — best free option, no logs\n• **NordVPN** — fast & reliable\n• **ExpressVPN** — great for streaming\n• **Mullvad** — best privacy\n💡 Always use a VPN on **public Wi-Fi**!",
    ],

    # ── Out-of-scope / Unknown ─────────────────────────────────────────────────
    "oos": [
        "I'm not confident I understood that. Could you **rephrase** your question? 🤔",
        "That seems outside my current knowledge. Could you try asking in a **different way**?",
        "Hmm, I'm not sure about that one. Could you provide **more details**?",
    ],
}

LOW_CONF_REPLIES = [
    "I'm not confident I understood that correctly. Could you **rephrase** your question? 🤔",
    "I'm having a little trouble understanding. Could you try saying it **differently**?",
    "My confidence on that is low. Could you **clarify** what you mean?",
]

def _smart_reply(user_msg: str) -> Optional[str]:
    """All conversation is now handled by LLM for natural human-like responses."""
    return None


def get_response(intent: str, confidence: float, user_message: str, history: list = None) -> str:
    if confidence < CONFIDENCE_THRESHOLD or intent == "oos":
        return _get_llm_response(user_message, history)
    replies = RESPONSES.get(intent)
    if replies:
        return random.choice(replies)
    return _get_llm_response(user_message, history)


# ──────────────────────────────────────────────────────────────────────────────
class ChatbotEngine:
    def __init__(self):
        self.predictor = Predictor()
        self.history: List[Dict] = []
        self.stats = {"total": 0, "confident": 0, "fallbacks": 0}

    def chat(self, user_message: str) -> Dict:
        if not user_message.strip():
            return {"response": "Please type a message! 💬", "intent": None,
                    "confidence": 0.0, "is_confident": False, "top_k": [],
                    "timestamp": datetime.now().isoformat()}

        # Check for greetings/thanks first
        smart = _smart_reply(user_message)
        if smart:
            result = {"response": smart, "intent": "greeting_thanks",
                      "confidence": 1.0, "is_confident": True,
                      "top_k": [], "timestamp": datetime.now().isoformat()}
        else:
            pred   = self.predictor.predict(user_message)
            result = {
                "response"    : get_response(pred["intent"], pred["confidence"], user_message),
                "intent"      : pred["intent"],
                "confidence"  : pred["confidence"],
                "is_confident": pred["is_confident"],
                "top_k"       : pred["top_k"],
                "timestamp"   : datetime.now().isoformat(),
            }

        self.stats["total"] += 1
        if result["is_confident"]:
            self.stats["confident"] += 1
        else:
            self.stats["fallbacks"] += 1

        self.history.append({
            "user": user_message, "bot": result["response"],
            "intent": result["intent"], "confidence": result["confidence"],
            "timestamp": result["timestamp"],
        })
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        return result

    def clear(self):
        self.history.clear()
        self.stats = {"total": 0, "confident": 0, "fallbacks": 0}

    def get_history(self):  return self.history
    def get_stats(self):    return self.stats
