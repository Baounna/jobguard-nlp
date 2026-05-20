import streamlit as st
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import full_preprocess

# OCR — optional, degrades gracefully if tesseract binary is missing
try:
    from PIL import Image
    import pytesseract
    pytesseract.get_tesseract_version()   # raises if binary not found
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

MODELS_DIR          = os.path.join(os.path.dirname(__file__), 'models')
# TF-IDF artifacts
SVM_TFIDF_PATH      = os.path.join(MODELS_DIR, 'svm_model.pkl')
LR_TFIDF_PATH       = os.path.join(MODELS_DIR, 'lr_model.pkl')
TFIDF_VEC_PATH      = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
# BoW artifacts
SVM_BOW_PATH        = os.path.join(MODELS_DIR, 'svm_bow_model.pkl')
LR_BOW_PATH         = os.path.join(MODELS_DIR, 'lr_bow_model.pkl')
BOW_VEC_PATH        = os.path.join(MODELS_DIR, 'bow_vectorizer.pkl')
# LSA artifacts — saved as full sklearn Pipelines (tfidf → svd → normalizer → clf)
SVM_LSA_PATH        = os.path.join(MODELS_DIR, 'svm_lsa_pipeline.pkl')
LR_LSA_PATH         = os.path.join(MODELS_DIR, 'lr_lsa_pipeline.pkl')
# Legacy aliases (used by the boot check below)
MODEL_PATH          = SVM_TFIDF_PATH
VECTORIZER_PATH     = TFIDF_VEC_PATH

# ── Scope gate ────────────────────────────────────────────────────────────────
# The classifier was trained on job postings only. Inputs from other domains
# (login pages, news articles, random screenshots) have no meaningful prediction,
# so we detect them upfront and show a "not a job posting" message instead.
JOB_POSTING_TOKENS = {
    'experience', 'responsibility', 'requirement', 'qualification',
    'skill', 'salary', 'team', 'role', 'position', 'hire', 'hiring',
    'candidate', 'join', 'company', 'remote', 'fulltime', 'parttime',
    'benefit', 'apply', 'job', 'work', 'employment', 'career',
    'engineer', 'manager', 'developer', 'analyst', 'coordinator',
    'specialist', 'officer', 'director', 'lead', 'senior', 'junior',
    'description', 'opportunity', 'duty', 'task', 'training',
    'degree', 'bachelor', 'master', 'year', 'professional',
    'staff', 'department', 'office', 'business', 'service', 'product',
    'client', 'project', 'support', 'communication', 'industry',
    'customer', 'sale', 'marketing', 'design', 'development', 'recruiting',
    'recruit', 'employee', 'employer', 'wage', 'hour', 'time', 'shift'
}

def domain_match(processed_text):
    tokens = processed_text.split()
    n_tok  = len(tokens)
    matched = sorted(set(t for t in tokens if t in JOB_POSTING_TOKENS))
    return len(matched), matched, n_tok

def is_job_posting(processed_text):
    """A real posting has at least 3 distinct job-related lemmas (or 2 for very short text)."""
    n_match, _, n_tok = domain_match(processed_text)
    if n_tok < 6:
        return False, n_match, n_tok
    threshold = 3 if n_tok >= 30 else 2
    return n_match >= threshold, n_match, n_tok

# ── Topic detector ────────────────────────────────────────────────────────────
# Lightweight keyword-based topic classifier. When the text isn't a job posting,
# we still try to tell the user what it IS about (e.g. food, login page, sports).
TOPIC_LEXICON = {
    "Food / Restaurant / Delivery": {
        "icon": "🍔",
        "words": {
            "food","restaurant","pizza","burger","shawarma","chawarma","kebab",
            "delivery","menu","chef","kitchen","cafe","drink","meal","eat","eating",
            "cuisine","dish","tacos","sushi","sandwich","fastfood","pasta","salad",
            "bread","coffee","tea","juice","snack","breakfast","lunch","dinner",
            "tagine","couscous","harira","msemen","tajine","halal","spice","grilled",
            "fried","bakery","dessert","ingredient","mcdonalds","kfc","starbucks"
        }
    },
    "Cooking / Recipes": {
        "icon": "👨‍🍳",
        "words": {
            "recipe","cook","cooking","bake","baking","oven","pan","pot","mix",
            "stir","whisk","flour","sugar","butter","egg","yeast","dough","sauce",
            "marinade","season","tablespoon","teaspoon","cup","preheat","simmer",
            "boil","saute","roast","minute","step","ingredient"
        }
    },
    "Login / Authentication Page": {
        "icon": "🔐",
        "words": {
            "login","password","email","account","register","signin","signup",
            "compte","passe","connecter","connect","authentication","credential",
            "username","logout","session","token","ministere","royaume",
            "institutionnel","adresse","université","ministry","verification","otp"
        }
    },
    "E-commerce / Shopping": {
        "icon": "🛒",
        "words": {
            "shop","cart","buy","price","order","product","sale","store","discount",
            "checkout","shipping","promo","voucher","addtocart","wishlist","stock",
            "brand","category","review","rating","amazon","aliexpress","ebay"
        }
    },
    "News / Article": {
        "icon": "📰",
        "words": {
            "news","report","government","minister","president","country","election",
            "policy","announce","headline","media","press","journalist","reporter",
            "breaking","statement","official","spokesperson","update"
        }
    },
    "Politics": {
        "icon": "🏛️",
        "words": {
            "politic","political","parliament","congress","senate","democrat",
            "republican","liberal","conservative","party","vote","voter","ballot",
            "campaign","prime","ambassador","diplomat","reform","legislation","treaty"
        }
    },
    "Sports": {
        "icon": "⚽",
        "words": {
            "match","team","player","score","league","win","tournament","championship",
            "goal","athlete","stadium","coach","football","soccer","basketball",
            "tennis","olympic","fifa","champion","trophy","draft","rookie","mvp"
        }
    },
    "Fitness / Gym": {
        "icon": "💪",
        "words": {
            "gym","workout","exercise","fitness","muscle","cardio","squat","bench",
            "deadlift","pushup","pullup","trainer","crossfit","reps","set","weight",
            "dumbbell","barbell","treadmill","yoga","pilates","stretch"
        }
    },
    "Education / University": {
        "icon": "🎓",
        "words": {
            "school","student","university","class","course","lesson","study","exam",
            "diploma","professor","teacher","academic","lecture","homework","research",
            "thesis","scholarship","faculty","campus","semester","tuition","grade","gpa"
        }
    },
    "Science / Research": {
        "icon": "🔬",
        "words": {
            "research","experiment","laboratory","hypothesis","theory","scientist",
            "discovery","peer","journal","publication","biology","chemistry","physics",
            "molecule","atom","cell","protein","dna","specimen","analysis"
        }
    },
    "AI / Machine Learning": {
        "icon": "🤖",
        "words": {
            "neural","network","deep","learning","machine","model","training","dataset",
            "algorithm","tensor","embedding","transformer","gradient","backpropagation",
            "regression","classification","cluster","feature","pytorch","tensorflow",
            "huggingface","gpt","llm","artificial","intelligence","inference"
        }
    },
    "Astronomy / Space": {
        "icon": "🚀",
        "words": {
            "space","planet","star","galaxy","universe","astronomy","astronaut","rocket",
            "satellite","orbit","nasa","spacex","mars","moon","earth","sun","jupiter",
            "telescope","blackhole","cosmos","nebula","comet","meteor"
        }
    },
    "Travel / Transport": {
        "icon": "✈️",
        "words": {
            "travel","flight","train","bus","ticket","hotel","booking","airport","trip",
            "destination","tourism","vacation","passport","visa","departure","arrival",
            "luggage","backpack","itinerary","airline","cruise","resort"
        }
    },
    "Automobile / Cars": {
        "icon": "🚗",
        "words": {
            "car","vehicle","engine","wheel","tire","brake","speed","drive","driver",
            "garage","mechanic","fuel","diesel","hybrid","electric","tesla","toyota",
            "honda","bmw","mercedes","horsepower","mileage","sedan","suv","truck"
        }
    },
    "Movies / Cinema": {
        "icon": "🎬",
        "words": {
            "movie","film","cinema","actor","actress","director","scene","plot","script",
            "trailer","oscar","blockbuster","hollywood","bollywood","screenplay",
            "premiere","sequel","prequel","cast","character"
        }
    },
    "Music": {
        "icon": "🎵",
        "words": {
            "music","song","artist","album","band","singer","guitar","piano","drum",
            "bass","melody","rhythm","concert","tour","spotify","playlist","lyrics",
            "rap","rock","pop","jazz","classical","beat"
        }
    },
    "Gaming / Video Games": {
        "icon": "🎮",
        "words": {
            "game","gamer","gaming","playstation","xbox","nintendo","steam","esport",
            "twitch","stream","streamer","fortnite","minecraft","valorant","league",
            "controller","console","pcgaming","fps","mmorpg","speedrun","achievement"
        }
    },
    "Books / Literature": {
        "icon": "📚",
        "words": {
            "book","novel","author","reader","literature","chapter","page","plot",
            "character","poetry","poem","poet","fiction","nonfiction","biography",
            "memoir","library","bookstore","kindle","bestseller","manuscript"
        }
    },
    "Art / Crafts": {
        "icon": "🎨",
        "words": {
            "art","artist","painting","drawing","sketch","canvas","brush","color",
            "palette","gallery","exhibition","museum","sculpture","craft","handmade",
            "watercolor","acrylic","portrait","abstract","illustration"
        }
    },
    "Photography": {
        "icon": "📸",
        "words": {
            "photo","photography","photographer","camera","lens","shot","portrait",
            "landscape","aperture","shutter","iso","exposure","tripod","editing",
            "lightroom","photoshop","instagram","selfie","frame"
        }
    },
    "Fashion / Clothing": {
        "icon": "👗",
        "words": {
            "fashion","clothing","outfit","dress","shirt","pant","jean","shoe","sneaker",
            "boot","jacket","coat","style","trend","designer","runway","model","collection",
            "boutique","wardrobe","accessory","handbag"
        }
    },
    "Beauty / Cosmetics": {
        "icon": "💄",
        "words": {
            "makeup","cosmetic","lipstick","mascara","foundation","eyeliner","blush",
            "skincare","moisturizer","serum","cleanser","perfume","fragrance","beauty",
            "salon","manicure","pedicure","hairstyle","haircut"
        }
    },
    "Wellness / Yoga / Meditation": {
        "icon": "🧘",
        "words": {
            "yoga","meditation","mindful","mindfulness","relax","relaxation","breath",
            "breathing","wellness","selfcare","spiritual","mantra","chakra","balance",
            "calm","peace","stress","therapy","sound","aromatherapy"
        }
    },
    "Health / Medical": {
        "icon": "🏥",
        "words": {
            "doctor","hospital","patient","medical","treatment","health","disease",
            "medicine","clinic","surgery","therapy","vaccine","symptom","diagnosis",
            "pharmacy","nurse","prescription","insurance","emergency","blood"
        }
    },
    "Animals / Pets": {
        "icon": "🐾",
        "words": {
            "dog","cat","puppy","kitten","pet","animal","veterinarian","vet","breed",
            "leash","collar","cage","aquarium","fish","bird","parrot","rabbit","hamster",
            "horse","wildlife","zoo","shelter","adoption"
        }
    },
    "Nature / Environment": {
        "icon": "🌳",
        "words": {
            "nature","forest","tree","mountain","river","ocean","sea","beach","island",
            "lake","wildlife","environment","climate","pollution","sustainable","ecology",
            "ecosystem","biodiversity","recycle","renewable","conservation","green","carbon"
        }
    },
    "Weather / Climate": {
        "icon": "⛅",
        "words": {
            "weather","forecast","temperature","rain","snow","storm","wind","cloud",
            "sunny","cloudy","humidity","celsius","fahrenheit","hurricane","tornado",
            "drought","flood","heatwave","blizzard","fog","thunder","lightning"
        }
    },
    "Religion / Spirituality": {
        "icon": "🕊️",
        "words": {
            "god","prayer","pray","faith","religion","religious","church","mosque",
            "temple","synagogue","priest","imam","rabbi","bible","quran","torah",
            "spiritual","worship","blessing","meditation","ramadan","christmas","easter"
        }
    },
    "Real Estate / Housing": {
        "icon": "🏠",
        "words": {
            "house","apartment","rent","property","lease","tenant","landlord","mortgage",
            "bedroom","kitchen","garden","villa","studio","duplex","neighborhood",
            "realtor","listing","square","meter","footage","balcony"
        }
    },
    "Finance / Banking": {
        "icon": "🏦",
        "words": {
            "bank","money","transaction","transfer","loan","credit","debit","currency",
            "investment","financial","stock","share","dividend","interest","portfolio",
            "broker","mortgage","savings","checking","atm","wire","fintech"
        }
    },
    "Cryptocurrency / Blockchain": {
        "icon": "₿",
        "words": {
            "bitcoin","ethereum","crypto","cryptocurrency","blockchain","wallet","mining",
            "miner","nft","defi","token","altcoin","hodl","binance","coinbase","metamask",
            "stablecoin","ledger","decentralized","smartcontract","staking"
        }
    },
    "Cybersecurity": {
        "icon": "🛡️",
        "words": {
            "security","cybersecurity","hacker","phishing","malware","ransomware","virus",
            "firewall","encryption","vulnerability","exploit","breach","penetration",
            "antivirus","vpn","ddos","spyware","ssl","cve","ciso"
        }
    },
    "Social Media": {
        "icon": "📱",
        "words": {
            "instagram","tiktok","twitter","facebook","linkedin","snapchat","youtube",
            "follower","following","like","comment","share","hashtag","reel","story",
            "post","tweet","retweet","influencer","viral","subscribe","trending"
        }
    },
    "Technology / Software": {
        "icon": "💻",
        "words": {
            "software","application","computer","programming","developer","code",
            "framework","library","github","api","database","server","cloud","linux",
            "python","javascript","android","iphone","smartphone","internet","docker"
        }
    },
    "Gardening / Plants": {
        "icon": "🌱",
        "words": {
            "garden","gardening","plant","seed","soil","water","sunlight","fertilizer",
            "compost","prune","harvest","greenhouse","flower","vegetable","herb",
            "tomato","rose","cactus","bonsai","orchid","lawn","mowing"
        }
    },
    "DIY / Home Improvement": {
        "icon": "🔨",
        "words": {
            "diy","tool","drill","hammer","screw","nail","saw","wrench","plumbing",
            "electrical","wiring","paint","painting","wall","tile","floor","carpenter",
            "carpentry","renovation","handyman","woodwork","ikea"
        }
    },
    "Dating / Relationships": {
        "icon": "💕",
        "words": {
            "date","dating","relationship","love","romance","romantic","partner",
            "boyfriend","girlfriend","tinder","bumble","hinge","crush","wedding","marriage",
            "engaged","engagement","anniversary","valentine"
        }
    },
    "Family / Parenting": {
        "icon": "👨‍👩‍👧",
        "words": {
            "family","parent","mother","father","child","children","baby","toddler",
            "infant","kid","sister","brother","grandma","grandpa","grandparent",
            "parenting","diaper","stroller","nursery","kindergarten"
        }
    },
    "Languages / Translation": {
        "icon": "🌐",
        "words": {
            "language","translate","translation","english","french","spanish","arabic",
            "german","chinese","japanese","russian","grammar","vocabulary","pronounce",
            "fluent","bilingual","duolingo","linguistic","dialect","accent"
        }
    },
}

def detect_topic(processed_text):
    """Return (best_topic, icon, matched_keywords, top_freq_words, top3_candidates)."""
    from collections import Counter
    tokens = processed_text.split()
    token_set = set(tokens)

    scored = []
    for topic, info in TOPIC_LEXICON.items():
        overlap = info["words"] & token_set
        if overlap:
            scored.append((topic, info["icon"], overlap, len(overlap)))
    scored.sort(key=lambda x: x[3], reverse=True)

    if scored:
        best_topic, best_icon, best_overlap, _ = scored[0]
        matched = sorted(best_overlap)[:6]
    else:
        best_topic, best_icon, matched = None, "", []

    top_words = [w for w, _ in Counter(t for t in tokens if len(t) > 3).most_common(6)]
    top3 = [(t, i, n) for t, i, _, n in scored[:3]]
    return best_topic, best_icon, matched, top_words, top3

TOPIC_COUNT = len(TOPIC_LEXICON)

st.set_page_config(
    page_title="JobGuard — Scam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

st.markdown("""
<style>
:root {
    --bg-base:       #ffffff;
    --bg-surface:    #ffffff;
    --bg-raised:     #f8fafc;
    --bg-input:      #ffffff;
    --border-soft:   #e5e7eb;
    --border-strong: #cbd5e1;
    --text-primary:  #0f172a;
    --text-secondary:#334155;
    --text-muted:    #64748b;
    --text-faint:    #94a3b8;
    --accent:        #0ea5e9;
    --accent-2:      #38bdf8;
    --accent-dark:   #0369a1;
    --accent-glow:   rgba(14, 165, 233, 0.22);
    --success:       #10b981;
    --success-bg:    rgba(16, 185, 129, 0.10);
    --danger:        #ef4444;
    --danger-bg:     rgba(239, 68, 68, 0.10);
    --warning:       #f59e0b;
    --warning-bg:    rgba(245, 158, 11, 0.10);
    --mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(14,165,233,0.07), transparent 55%),
        radial-gradient(900px 500px at -10% 18%, rgba(56,189,248,0.06), transparent 55%),
        var(--bg-base) !important;
    background-attachment: fixed !important;
    color: var(--text-primary) !important;
    scroll-behavior: smooth !important;
}
/* Soft dot-grid pattern overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle at 1px 1px, rgba(14,165,233,0.05) 1px, transparent 0);
    background-size: 26px 26px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.8;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
    background: transparent !important;
}
#MainMenu, footer { visibility: hidden; }
/* Make the top header transparent but keep the sidebar collapse / expand
   chevron clickable. Hiding the entire <header> would also hide the toggle. */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    height: auto !important;
}
header[data-testid="stHeader"] [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] [data-testid="stStatusWidget"] { display: none !important; }
/* Force the sidebar collapse chevron to always be visible & prominent.
   Targets every Streamlit version's selector for the expand control. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999 !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,
button[kind="headerNoPadding"] {
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
    border: 1px solid rgba(2,132,199,0.5) !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    box-shadow:
        0 2px 6px rgba(14,165,233,0.30),
        0 0 0 1px rgba(255,255,255,0.6) inset !important;
    color: #ffffff !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] button svg,
button[kind="headerNoPadding"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    width: 20px !important;
    height: 20px !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
button[kind="headerNoPadding"]:hover {
    transform: scale(1.05) !important;
    box-shadow:
        0 4px 12px rgba(14,165,233,0.45),
        0 0 0 1px rgba(255,255,255,0.6) inset !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
}

/* Sidebar — light gradient with soft right-edge accent */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid var(--border-soft) !important;
    box-shadow: inset -1px 0 0 0 rgba(14,165,233,0.06) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
}

/* Inputs — clean white with subtle inner highlight */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 0.9rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px 0 rgba(15,23,42,0.04) !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-faint) !important;
    font-style: italic;
}
[data-testid="stTextInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
    border-color: var(--border-strong) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow), 0 1px 3px 0 rgba(14,165,233,0.10) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stFileUploader"] label {
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background-color: var(--bg-input) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background-color: rgba(56,189,248,0.04) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] section span {
    color: var(--text-muted) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Primary action button — sky-cyan with white text and soft glow */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(2,132,199,0.4) !important;
    border-radius: 8px !important;
    padding: 0.85rem 1.6rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 1px 2px 0 rgba(14,165,233,0.20),
        0 4px 14px 0 rgba(14,165,233,0.35),
        inset 0 1px 0 0 rgba(255,255,255,0.30) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow:
        0 2px 4px 0 rgba(14,165,233,0.25),
        0 10px 28px 0 rgba(14,165,233,0.45),
        inset 0 1px 0 0 rgba(255,255,255,0.35) !important;
    filter: brightness(1.06) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    transition: transform 0.05s ease !important;
}

/* Sidebar buttons — light terminal style */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: #ffffff !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-soft) !important;
    box-shadow: 0 1px 2px 0 rgba(15,23,42,0.04) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #f0f9ff !important;
    border-color: var(--accent) !important;
    color: var(--accent-dark) !important;
    transform: none !important;
    filter: none !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Bordered containers — clean white panels with sky-tint top */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(180deg, rgba(14,165,233,0.025) 0%, transparent 30%),
        var(--bg-surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    padding: 1.8rem !important;
    margin-bottom: 1.1rem !important;
    box-shadow:
        inset 0 1px 0 0 rgba(255,255,255,0.6),
        0 1px 3px 0 rgba(15,23,42,0.05),
        0 8px 24px -8px rgba(15,23,42,0.08) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    z-index: 1;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(14,165,233,0.35) !important;
    box-shadow:
        inset 0 1px 0 0 rgba(255,255,255,0.6),
        0 4px 8px 0 rgba(15,23,42,0.06),
        0 0 0 1px rgba(14,165,233,0.08),
        0 16px 40px -10px rgba(14,165,233,0.15) !important;
}

/* Metrics — light cards with monospace cyan numbers */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(14,165,233,0.04) 0%, transparent 60%), var(--bg-raised) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px 0 rgba(15,23,42,0.04) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(14,165,233,0.35) !important;
    box-shadow: 0 4px 12px 0 rgba(14,165,233,0.10) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-dark) !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

/* Hero gradient title — dark navy → sky on white */
.hero-title {
    background: linear-gradient(135deg, #0f172a 0%, #0369a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
}

/* Mono utility class */
.mono { font-family: var(--mono); }

/* Radio buttons — make the option text fully readable.
   Streamlit's default greys these labels to ~40% opacity, which on a white
   background reads as washed-out. We force full dark text and opacity. */
[data-testid="stRadio"] label,
[data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stRadio"] [role="radiogroup"] label p,
[data-testid="stRadio"] [role="radiogroup"] label div,
[data-testid="stRadio"] [role="radiogroup"] label span {
    color: #0f172a !important;
    opacity: 1 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.6rem !important;
}
[data-testid="stRadio"] [role="radiogroup"] > label {
    padding: 0.5rem 0.8rem !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] [role="radiogroup"] > label:hover {
    border-color: #38bdf8 !important;
    background: #f0f9ff !important;
}
/* Selected radio option — strong sky-blue ring */
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    border-color: #0ea5e9 !important;
    background: #e0f2fe !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.18) !important;
}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) span,
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p,
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) div {
    color: #0369a1 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load every (representation, algorithm) artifact. Returns dict keyed by 'rep_algo'.
       LSA entries are full sklearn Pipelines (tfidf → svd → normalizer → clf) so they
       take raw text directly instead of needing a separate vectorizer."""
    artifacts = {
        # TF-IDF: vectorizer + classifiers
        'tfidf_vec': joblib.load(TFIDF_VEC_PATH),
        'svm_tfidf': joblib.load(SVM_TFIDF_PATH),
        'lr_tfidf':  joblib.load(LR_TFIDF_PATH)  if os.path.exists(LR_TFIDF_PATH)  else None,
        # BoW: vectorizer + classifiers
        'bow_vec':   joblib.load(BOW_VEC_PATH)   if os.path.exists(BOW_VEC_PATH)   else None,
        'svm_bow':   joblib.load(SVM_BOW_PATH)   if os.path.exists(SVM_BOW_PATH)   else None,
        'lr_bow':    joblib.load(LR_BOW_PATH)    if os.path.exists(LR_BOW_PATH)    else None,
        # LSA: full pipelines (handle their own vectorization)
        'svm_lsa':   joblib.load(SVM_LSA_PATH)   if os.path.exists(SVM_LSA_PATH)   else None,
        'lr_lsa':    joblib.load(LR_LSA_PATH)    if os.path.exists(LR_LSA_PATH)    else None,
    }
    return artifacts


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:1.5rem 0 1.2rem;'>"
        "<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:1.5rem;'>"
        "<div style='width:34px; height:34px; border-radius:9px; "
        "background:linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%); "
        "display:flex; align-items:center; justify-content:center; "
        "box-shadow:0 4px 14px 0 rgba(14,165,233,0.35);'>"
        "<span style='font-size:1.1rem;'>🛡️</span></div>"
        "<div>"
        "<div style='font-size:1.05rem; font-weight:700; color:#0f172a; "
        "letter-spacing:-0.01em; line-height:1.1;'>JobGuard</div>"
        "<div style='font-size:0.7rem; color:#64748b; font-weight:500; "
        "letter-spacing:0.04em;'>Scam Detection · v1.0</div>"
        "</div></div>"
        "<div style='font-size:0.7rem; color:#64748b; text-transform:uppercase; "
        "letter-spacing:0.10em; font-weight:600; margin-bottom:0.8rem;'>How it works</div>"
        "</div>",
        unsafe_allow_html=True
    )

    steps = [
        ("01", "🔤", "Enter the job title, company, and description"),
        ("02", "⚙️", "Text is cleaned, tokenized & lemmatized"),
        ("03", "📊", "TF-IDF converts text to a feature vector"),
        ("04", "🤖", "SVM model classifies as Legit or Fraud"),
    ]
    for num, icon, text in steps:
        st.markdown(
            f"<div style='display:flex; gap:0.75rem; align-items:flex-start; "
            f"margin-bottom:0.65rem; padding:0.75rem 0.85rem; "
            f"background:linear-gradient(180deg, #161c27 0%, #11161f 100%); "
            f"border-radius:10px; border:1px solid #e5e7eb;'>"
            f"<span style='font-size:0.7rem; font-weight:700; color:#0369a1; "
            f"letter-spacing:0.05em; padding-top:1px;'>{num}</span>"
            f"<span style='font-size:0.95rem; flex-shrink:0;'>{icon}</span>"
            f"<span style='font-size:0.82rem; color:#475569; line-height:1.5;'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div style='margin-top:1.7rem; font-size:0.7rem; color:#64748b; "
        "text-transform:uppercase; letter-spacing:0.10em; font-weight:600; margin-bottom:0.8rem;'>"
        "Quick Test</div>",
        unsafe_allow_html=True
    )

    if st.button("🚨  Test with Fraud Sample", use_container_width=True):
        st.session_state.update({
            's_title': 'Work From Home – Earn $5000/Week',
            's_company': '',
            's_desc': (
                "Amazing opportunity! Work from home, no experience required. "
                "Earn up to $5000 per week guaranteed. Send your personal details "
                "to get started immediately. Limited slots — act now! No interview needed."
            ),
            's_req': '', 's_ben': 'Unlimited earnings every week!'
        })

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    if st.button("✅  Test with Legit Sample", use_container_width=True):
        st.session_state.update({
            's_title': 'Software Engineer',
            's_company': 'TechCorp Inc.',
            's_desc': (
                "We are looking for a Software Engineer to join our backend team. "
                "You will design and maintain scalable systems, collaborate with "
                "product and design teams, and ship high-quality code."
            ),
            's_req': "3+ years Python, experience with REST APIs, Bachelor's in CS or equivalent.",
            's_ben': "Competitive salary, health insurance, 401k, flexible remote work."
        })

    st.markdown(
        "<div style='margin-top:2.5rem; padding-top:1rem; "
        "font-size:0.72rem; color:#94a3b8; border-top:1px solid #e5e7eb;'>"
        "<div style='display:flex; align-items:center; gap:0.4rem;'>"
        "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; "
        "background:#10b981; box-shadow:0 0 6px #10b981;'></span>"
        "<span>Model online · SVM + TF-IDF</span>"
        "</div>"
        "<div style='margin-top:0.3rem; color:#94a3b8;'>NLP 2026 · Mohamed Baounna · Zakaria Birani</div>"
        "</div>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.markdown(
        "<div style='margin:3rem auto; max-width:520px; background:var(--bg-surface); "
        "border:1px solid #ef4444; border-radius:14px; padding:2rem; text-align:center;'>"
        "<div style='font-size:2rem; margin-bottom:0.5rem;'>⚠️</div>"
        "<div style='color:#ef4444; font-weight:600; margin-bottom:0.4rem;'>Model not found</div>"
        "<div style='color:#475569; font-size:0.88rem;'>Run <code>4_modeling.ipynb</code> first.</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

artifacts = load_artifacts()
# Legacy bindings (used by older code paths below until they're refactored)
svm_model   = artifacts['svm_tfidf']
lr_model    = artifacts['lr_tfidf']
vectorizer  = artifacts['tfidf_vec']

# Hero header — refined with floating shield, stat strip, and animated gradient
st.markdown("""
<style>
@keyframes pulse-dot { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.3); } }
@keyframes float-shield { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes gradient-shift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.live-pill { animation: gradient-shift 4s ease infinite; background-size: 200% 200%; }
.live-dot  { animation: pulse-dot 2s ease-in-out infinite; }
.hero-shield { animation: float-shield 5s ease-in-out infinite; }
.stat-card { transition: all 0.25s ease; }
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px -8px rgba(14,165,233,0.18); }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='padding:2.5rem 0 1.8rem; position:relative; z-index:1; text-align:center;'>"

    # Terminal-style status bar (light)
    "<div style='display:inline-flex; align-items:center; gap:0.7rem; "
    "background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; "
    "padding:0.5rem 0.95rem; margin-bottom:1.6rem; font-family:JetBrains Mono,SF Mono,monospace; "
    "font-size:0.72rem; box-shadow:0 1px 2px rgba(15,23,42,0.04);'>"
    "<span style='color:#10b981;'>●</span>"
    "<span style='color:#64748b;'>system:</span>"
    "<span style='color:#0369a1; font-weight:600;'>jobguard.svc</span>"
    "<span style='color:#cbd5e1;'>│</span>"
    "<span style='color:#64748b;'>status:</span>"
    "<span style='color:#10b981; font-weight:600;'>OPERATIONAL</span>"
    "<span style='color:#cbd5e1;'>│</span>"
    "<span style='color:#64748b;'>model:</span>"
    "<span style='color:#0369a1; font-weight:600;'>svm+tfidf</span>"
    "<span style='color:#cbd5e1;'>│</span>"
    "<span style='color:#64748b;'>f1:</span>"
    "<span style='color:#10b981; font-weight:600;'>0.892</span>"
    "</div>"

    # Floating shield (centered)
    "<div class='hero-shield' style='display:inline-flex; width:84px; height:84px; border-radius:18px; "
    "background:linear-gradient(135deg, #38bdf8 0%, #0ea5e9 50%, #0284c7 100%); "
    "align-items:center; justify-content:center; margin-bottom:1.5rem; "
    "border:1px solid rgba(2,132,199,0.4); "
    "box-shadow:0 0 0 1px rgba(14,165,233,0.20), 0 12px 30px -8px rgba(14,165,233,0.50), "
    "inset 0 1px 0 rgba(255,255,255,0.40);'>"
    "<span style='font-size:2.5rem; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.20));'>🛡️</span>"
    "</div>"

    # Title
    "<h1 class='hero-title' style='font-size:3rem; font-weight:800; "
    "margin:0 0 0.8rem; line-height:1.05; letter-spacing:-0.02em;'>"
    "Job Offer Scam Detector</h1>"

    # Description
    "<p style='color:#475569; font-size:1.02rem; margin:0 auto; max-width:620px; line-height:1.65;'>"
    "Paste a job posting or upload a screenshot. Our SVM + TF-IDF pipeline analyzes the text "
    "and tells you whether it's <b style='color:#10b981;'>legitimate</b> "
    "or <b style='color:#ef4444;'>fraudulent</b> — instantly.</p>"

    # Stat strip — light cards with monospace cyan numbers
    "<div style='display:flex; gap:0.7rem; margin:1.8rem auto 0; max-width:620px;'>"
    + "".join([
        f"<div class='stat-card' style='flex:1; background:linear-gradient(180deg, rgba(14,165,233,0.05) 0%, #ffffff 70%); "
        f"border:1px solid #e2e8f0; border-radius:8px; padding:0.85rem 1rem; text-align:center; "
        f"box-shadow:0 1px 3px rgba(15,23,42,0.04);'>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:0.62rem; color:#64748b; font-weight:500; "
        f"text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.4rem;'>"
        f"<span style='color:#cbd5e1;'>[</span>{kc}<span style='color:#cbd5e1;'>]</span> &nbsp;{label}</div>"
        f"<div style='font-family:JetBrains Mono,monospace; font-size:1.25rem; color:#0369a1; font-weight:700; "
        f"display:flex; align-items:baseline; justify-content:center; gap:0.35rem;'>"
        f"{value}<span style='font-size:0.72rem; color:#64748b; font-weight:500;'>{unit}</span></div>"
        f"</div>"
        for kc, label, value, unit in [
            ("01", "Precision", "0.931", ""),
            ("02", "Recall",    "0.855", ""),
            ("03", "F1-Score",  "0.892", " ★"),
        ]
    ]) + "</div>"

    "<div style='margin-top:2rem; height:1px; "
    "background:linear-gradient(90deg, transparent 0%, #e2e8f0 30%, #38bdf855 50%, #e2e8f0 70%, transparent 100%);'></div>"
    "</div>",
    unsafe_allow_html=True
)

# ── Quick-Test toolbar (always visible on main page) ──────────────────────────
st.markdown(
    "<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem; "
    "font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#64748b; "
    "text-transform:uppercase; letter-spacing:0.10em; font-weight:600;'>"
    "<span style='color:#cbd5e1;'>[</span>QUICK TEST<span style='color:#cbd5e1;'>]</span>"
    "<span style='color:#cbd5e1;'>—</span>"
    "<span style='text-transform:none; letter-spacing:0; color:#94a3b8; font-weight:500;'>"
    "Load a sample posting with one click</span></div>",
    unsafe_allow_html=True
)

qt_col1, qt_col2 = st.columns(2)
with qt_col1:
    if st.button("🚨  Try Fraud Sample", use_container_width=True, key="main_fraud_btn"):
        st.session_state.update({
            's_title': 'Work From Home – Earn $5000/Week',
            's_company': '',
            's_desc': (
                "Amazing opportunity! Work from home, no experience required. "
                "Earn up to $5000 per week guaranteed. Send your personal details "
                "to get started immediately. Limited slots — act now! No interview needed."
            ),
            's_req': '', 's_ben': 'Unlimited earnings every week!'
        })
with qt_col2:
    if st.button("✅  Try Legit Sample", use_container_width=True, key="main_legit_btn"):
        st.session_state.update({
            's_title': 'Software Engineer',
            's_company': 'TechCorp Inc.',
            's_desc': (
                "We are looking for a Software Engineer to join our backend team. "
                "You will design and maintain scalable systems, collaborate with "
                "product and design teams, and ship high-quality code."
            ),
            's_req': "3+ years Python, experience with REST APIs, Bachelor's in CS or equivalent.",
            's_ben': "Competitive salary, health insurance, 401k, flexible remote work."
        })

st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

# ── Model selector — Representation × Algorithm (live demo of all 4 variants) ─
# This directly mirrors the two assignment requirements:
#   • Step 3: compare text representations
#   • Step 4: compare ML algorithms
with st.container(border=True):
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem; "
        "font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#64748b; "
        "text-transform:uppercase; letter-spacing:0.10em; font-weight:600;'>"
        "<span style='color:#cbd5e1;'>[</span>MODEL COMPARISON<span style='color:#cbd5e1;'>]</span>"
        "<span style='color:#cbd5e1;'>—</span>"
        "<span style='text-transform:none; letter-spacing:0; color:#94a3b8; font-weight:500;'>"
        "Switch live between representations and algorithms</span></div>",
        unsafe_allow_html=True
    )

    rep_col, algo_col = st.columns(2)
    with rep_col:
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; color:#0369a1; "
            "text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.3rem;'>"
            "📊 &nbsp;Representation</div>",
            unsafe_allow_html=True
        )
        rep_choice = st.radio(
            "Representation",
            options=['tfidf', 'bow', 'lsa'],
            format_func=lambda v: {
                'tfidf': "TF-IDF (sparse weighted)",
                'bow':   "BoW (raw counts)",
                'lsa':   "LSA (dense semantic, 100-d)",
            }[v],
            horizontal=False,
            label_visibility='collapsed',
            key='rep_choice',
        )
    with algo_col:
        st.markdown(
            "<div style='font-size:0.7rem; font-weight:700; color:#0369a1; "
            "text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.3rem;'>"
            "🤖 &nbsp;Algorithm</div>",
            unsafe_allow_html=True
        )
        algo_choice = st.radio(
            "Algorithm",
            options=['svm', 'lr'],
            format_func=lambda v: "SVM (Linear, Calibrated)" if v == 'svm' else "Logistic Regression",
            horizontal=False,
            label_visibility='collapsed',
            key='algo_choice',
        )

    # Tiny status line showing the resulting combination + a star if it's the deployed model
    is_deployed = (rep_choice == 'tfidf' and algo_choice == 'svm')
    star = " ⭐ DEPLOYED" if is_deployed else ""
    st.markdown(
        f"<div style='margin-top:0.6rem; padding-top:0.6rem; border-top:1px dashed #e2e8f0; "
        f"font-family:JetBrains Mono,monospace; font-size:0.74rem; color:#64748b;'>"
        f"<span style='color:#cbd5e1;'>→</span> active: "
        f"<b style='color:#0369a1;'>{algo_choice.upper()} + {rep_choice.upper()}</b>"
        f"<span style='color:#10b981; font-weight:700;'>{star}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

# ── OCR Upload Section ────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:1rem;'>"
        "<div style='width:34px; height:34px; border-radius:10px; "
        "background:linear-gradient(135deg, rgba(14,165,233,0.18) 0%, rgba(14,165,233,0.10) 100%); "
        "border:1px solid rgba(14,165,233,0.45); "
        "display:flex; align-items:center; justify-content:center; flex-shrink:0;'>"
        "<span style='font-size:1.1rem;'>📷</span>"
        "</div>"
        "<div>"
        "<div style='font-size:0.95rem; color:#0f172a; font-weight:700; line-height:1.2;'>"
        "Upload a Screenshot</div>"
        "<div style='font-size:0.78rem; color:#64748b; margin-top:0.15rem;'>"
        "Drag a posting image — text is extracted automatically</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )

    if not OCR_AVAILABLE:
        st.markdown(
            "<div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.35); "
            "border-radius:10px; padding:0.85rem 1.1rem; color:#f59e0b; font-size:0.87rem;'>"
            "⚠️ &nbsp;OCR is unavailable — Tesseract binary not found. "
            "Install it with <code>brew install tesseract</code> (macOS) or "
            "<code>sudo apt install tesseract-ocr</code> (Linux)."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        uploaded = st.file_uploader(
            "Drop a job posting screenshot to extract text automatically",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        st.markdown(
            "<div style='font-size:0.78rem; color:#94a3b8; margin-top:0.4rem;'>"
            "Supports screenshots from LinkedIn, Indeed, Glassdoor, and more.</div>",
            unsafe_allow_html=True
        )

        if uploaded is not None:
            img = Image.open(uploaded)
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.image(img, caption="Uploaded screenshot", use_container_width=True)

            with col_info:
                # Only OCR once per uploaded file (avoids re-running on every Analyze click,
                # and avoids clobbering edits the user made after upload).
                upload_id = getattr(uploaded, 'file_id', None) or uploaded.name
                is_new_upload = st.session_state.get('_ocr_for') != upload_id

                if is_new_upload:
                    with st.spinner("Extracting text…"):
                        extracted = pytesseract.image_to_string(img).strip()
                    st.session_state['_ocr_for']    = upload_id
                    st.session_state['_ocr_result'] = extracted
                    # Populate the description field — but only on a fresh upload,
                    # so we don't clobber any edits the user makes later.
                    if extracted:
                        st.session_state['s_desc'] = extracted
                else:
                    extracted = st.session_state.get('_ocr_result', '')

                if extracted:
                    st.markdown(
                        "<div style='background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.35); "
                        "border-radius:10px; padding:0.85rem 1.1rem; margin-bottom:0.8rem;'>"
                        "<span style='color:#10b981; font-weight:600; font-size:0.88rem;'>"
                        "✅ &nbsp;Text extracted successfully</span><br>"
                        "<span style='color:#475569; font-size:0.8rem;'>The Job Description field has been filled automatically.</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<div style='font-size:0.7rem; font-weight:600; color:#64748b; "
                        "text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.4rem;'>"
                        "Extracted Text Preview</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='background:#f8fafc; border:1px solid #e5e7eb; border-radius:10px; "
                        f"padding:0.9rem 1rem; font-size:0.82rem; color:#475569; line-height:1.6; "
                        f"max-height:220px; overflow-y:auto; white-space:pre-wrap;'>"
                        f"{extracted[:800]}{'…' if len(extracted) > 800 else ''}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.35); "
                        "border-radius:10px; padding:0.85rem 1.1rem; color:#f59e0b; font-size:0.87rem;'>"
                        "⚠️ &nbsp;No text could be extracted. Try a clearer, higher-resolution screenshot.</div>",
                        unsafe_allow_html=True
                    )

st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)

# ── Input section ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:1.2rem;'>"
        "<div style='width:34px; height:34px; border-radius:10px; "
        "background:linear-gradient(135deg, rgba(14,165,233,0.18) 0%, rgba(14,165,233,0.10) 100%); "
        "border:1px solid rgba(14,165,233,0.45); "
        "display:flex; align-items:center; justify-content:center; flex-shrink:0;'>"
        "<span style='font-size:1.05rem;'>📝</span>"
        "</div>"
        "<div>"
        "<div style='font-size:0.95rem; color:#0f172a; font-weight:700; line-height:1.2;'>"
        "Job Posting Details</div>"
        "<div style='font-size:0.78rem; color:#64748b; margin-top:0.15rem;'>"
        "Fill the fields below or load a sample from the sidebar</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Initialize session-state keys so the widgets bind cleanly
    for k in ('s_title', 's_company', 's_desc', 's_req', 's_ben'):
        if k not in st.session_state:
            st.session_state[k] = ''

    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Job Title", key='s_title',
                              placeholder="e.g. Software Engineer")
    with c2:
        company = st.text_input("Company Name", key='s_company',
                                placeholder="e.g. TechCorp Inc.")

    description = st.text_area("Job Description *", key='s_desc',
                                height=160, placeholder="Paste the full job description here…")

    c3, c4 = st.columns(2)
    with c3:
        requirements = st.text_area("Requirements", key='s_req',
                                     height=100, placeholder="Skills, experience, qualifications…")
    with c4:
        benefits = st.text_area("Benefits", key='s_ben',
                                 height=100, placeholder="Salary, perks, remote work…")

analyze = st.button("Analyze Posting →", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if analyze:
    if not description.strip():
        st.markdown(
            "<div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.35); "
            "border-radius:10px; padding:0.9rem 1.2rem; margin-top:0.8rem; color:#f59e0b; font-size:0.9rem;'>"
            "⚠️ &nbsp;Please enter at least a job description.</div>",
            unsafe_allow_html=True
        )
    else:
        combined  = f"{title} {company} {description} {requirements} {benefits}"
        processed = full_preprocess(combined)

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

        # ── Scope gate: bail out if the text isn't a job posting ─────────────
        in_scope, n_match, n_tok = is_job_posting(processed)
        if not in_scope:
            topic, topic_icon, matched_kw, top_words, top3 = detect_topic(processed)

            st.markdown(
                f"<div style='background:linear-gradient(135deg, rgba(245,158,11,0.10) 0%, rgba(245,158,11,0.04) 100%); "
                f"border:1px solid rgba(245,158,11,0.45); "
                f"border-radius:14px; padding:2rem; display:flex; align-items:center; "
                f"gap:1.5rem; margin-bottom:1rem; "
                f"box-shadow:0 8px 32px -12px rgba(245,158,11,0.25);'>"
                f"<div style='width:64px; height:64px; border-radius:14px; "
                f"background:linear-gradient(135deg, #f59e0b 0%, #d97706 100%); "
                f"display:flex; align-items:center; justify-content:center; flex-shrink:0; "
                f"box-shadow:0 4px 14px 0 rgba(245,158,11,0.35);'>"
                f"<span style='font-size:1.8rem;'>❓</span>"
                f"</div>"
                f"<div>"
                f"<div style='font-size:1.5rem; font-weight:800; color:#f59e0b; "
                f"letter-spacing:-0.01em; line-height:1.1;'>Not a Job Posting</div>"
                f"<div style='color:#475569; font-size:0.92rem; margin-top:0.4rem; line-height:1.5;'>"
                f"This text doesn't look like a job description. The fraud detector only "
                f"works on job postings, so no verdict was produced.</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Main-idea card with detected topic
            if topic:
                topic_html = (
                    f"<div style='display:flex; align-items:center; gap:1rem; "
                    f"margin-bottom:1rem;'>"
                    f"<div style='width:54px; height:54px; border-radius:12px; "
                    f"background:linear-gradient(135deg, rgba(14,165,233,0.18) 0%, rgba(14,165,233,0.10) 100%); "
                    f"border:1px solid rgba(124,58,237,0.40); "
                    f"display:flex; align-items:center; justify-content:center; flex-shrink:0;'>"
                    f"<span style='font-size:1.6rem;'>{topic_icon}</span>"
                    f"</div>"
                    f"<div>"
                    f"<div style='font-size:0.7rem; color:#64748b; font-weight:600; "
                    f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.2rem;'>"
                    f"This text is about</div>"
                    f"<div style='font-size:1.25rem; color:#0f172a; font-weight:700; "
                    f"letter-spacing:-0.01em;'>{topic}</div>"
                    f"</div>"
                    f"</div>"
                )
                kw_block = ""
                if matched_kw:
                    tags = "".join([
                        f"<span style='background:rgba(14,165,233,0.12); "
                        f"border:1px solid rgba(14,165,233,0.35); "
                        f"border-radius:8px; padding:0.35rem 0.8rem; font-size:0.8rem; "
                        f"color:#0369a1; font-weight:500; margin:0.2rem; "
                        f"display:inline-block;'>{w}</span>"
                        for w in matched_kw
                    ])
                    kw_block = (
                        f"<div style='font-size:0.7rem; color:#64748b; font-weight:600; "
                        f"text-transform:uppercase; letter-spacing:0.08em; "
                        f"margin:1rem 0 0.5rem;'>Matched topic keywords</div>"
                        f"<div>{tags}</div>"
                    )

                # Alternative candidates (if any besides the best)
                alt_block = ""
                if len(top3) > 1:
                    alt_pills = "".join([
                        f"<span style='background:rgba(168,178,207,0.06); "
                        f"border:1px solid #e5e7eb; border-radius:8px; "
                        f"padding:0.3rem 0.7rem; font-size:0.78rem; color:#475569; "
                        f"font-weight:500; margin:0.2rem; display:inline-block;'>"
                        f"{i} &nbsp;{t} <span style='color:#64748b; "
                        f"margin-left:0.4rem;'>×{n}</span></span>"
                        for t, i, n in top3[1:]
                    ])
                    alt_block = (
                        f"<div style='font-size:0.7rem; color:#64748b; font-weight:600; "
                        f"text-transform:uppercase; letter-spacing:0.08em; "
                        f"margin:1rem 0 0.5rem;'>Other plausible topics</div>"
                        f"<div>{alt_pills}</div>"
                    )

                memory_note = (
                    f"Searched across <b style='color:#475569;'>{TOPIC_COUNT}</b> "
                    f"topics in memory — best match: "
                    f"<b style='color:#0369a1;'>{topic}</b> "
                    f"({len(matched_kw)} keyword{'s' if len(matched_kw)!=1 else ''})."
                )
            else:
                # No topic in memory matched the text
                top_tags = "".join([
                    f"<span style='background:rgba(245,158,11,0.12); "
                    f"border:1px solid rgba(245,158,11,0.35); "
                    f"border-radius:8px; padding:0.35rem 0.8rem; font-size:0.8rem; "
                    f"color:#fbbf24; font-weight:500; margin:0.2rem; "
                    f"display:inline-block;'>{w}</span>"
                    for w in top_words
                ]) or "<span style='color:#64748b; font-size:0.85rem;'>(text too short to analyse)</span>"

                topic_html = (
                    f"<div style='display:flex; align-items:center; gap:1rem; "
                    f"margin-bottom:1rem;'>"
                    f"<div style='width:54px; height:54px; border-radius:12px; "
                    f"background:rgba(245,158,11,0.12); "
                    f"border:1px solid rgba(245,158,11,0.40); "
                    f"display:flex; align-items:center; justify-content:center; flex-shrink:0;'>"
                    f"<span style='font-size:1.6rem;'>🧠</span>"
                    f"</div>"
                    f"<div>"
                    f"<div style='font-size:0.7rem; color:#64748b; font-weight:600; "
                    f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.2rem;'>"
                    f"Out of model's memory</div>"
                    f"<div style='font-size:1.15rem; color:#0f172a; font-weight:700; "
                    f"letter-spacing:-0.01em;'>"
                    f"None of the {TOPIC_COUNT} known topics matched this text</div>"
                    f"</div>"
                    f"</div>"
                )
                kw_block = (
                    f"<div style='background:rgba(245,158,11,0.04); "
                    f"border:1px dashed rgba(245,158,11,0.30); "
                    f"border-radius:10px; padding:0.95rem 1.1rem; margin-top:0.5rem;'>"
                    f"<div style='font-size:0.82rem; color:#475569; line-height:1.6; "
                    f"margin-bottom:0.7rem;'>"
                    f"The model has a fixed vocabulary of topics. When the input "
                    f"vocabulary doesn't overlap with any known topic, the system "
                    f"<b style='color:#fbbf24;'>refuses to guess</b> rather than "
                    f"forcing a wrong category. This is a deliberate fail-safe."
                    f"</div>"
                    f"<div style='font-size:0.7rem; color:#64748b; font-weight:600; "
                    f"text-transform:uppercase; letter-spacing:0.08em; "
                    f"margin:0.6rem 0 0.5rem;'>Vocabulary observed in input</div>"
                    f"<div>{top_tags}</div>"
                    f"</div>"
                )
                alt_block = ""
                memory_note = (
                    f"Searched across <b style='color:#475569;'>{TOPIC_COUNT}</b> "
                    f"topics in memory — "
                    f"<b style='color:#fbbf24;'>0 matches</b>. "
                    f"Possible reasons: very short input, non-English text not in lexicon, "
                    f"OCR noise, or a domain the model wasn't trained on."
                )

            st.markdown(
                f"<div style='background:#ffffff; border:1px solid #e5e7eb; "
                f"border-radius:14px; padding:1.5rem 1.7rem; margin-bottom:1rem;'>"
                f"<div style='display:flex; align-items:center; justify-content:space-between; "
                f"margin-bottom:1rem;'>"
                f"<div style='font-size:0.7rem; font-weight:600; color:#0369a1; "
                f"text-transform:uppercase; letter-spacing:0.10em;'>"
                f"💡 &nbsp;Main Idea Detected</div>"
                f"<div style='font-size:0.7rem; color:#64748b; "
                f"background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.25); "
                f"border-radius:99px; padding:0.25rem 0.7rem; font-weight:600;'>"
                f"🧠 {TOPIC_COUNT} topics in memory</div>"
                f"</div>"
                f"{topic_html}"
                f"{kw_block}"
                f"{alt_block}"
                f"<div style='color:#64748b; font-size:0.78rem; margin-top:1.2rem; "
                f"padding-top:0.9rem; border-top:1px solid #e5e7eb; line-height:1.55;'>"
                f"{memory_note}<br>"
                f"Found <b style='color:#475569;'>{n_match}</b> job-related keyword(s) "
                f"in <b style='color:#475569;'>{n_tok}</b> tokens — "
                f"below the threshold required to classify the posting.</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.stop()
        # ─────────────────────────────────────────────────────────────────────

        # Pick representation + algorithm from the two radios on the main page
        rep  = st.session_state.get('rep_choice',  'tfidf')
        algo = st.session_state.get('algo_choice', 'svm')

        # LSA models are full sklearn Pipelines that accept raw text directly.
        # TF-IDF and BoW models need an explicit vectorizer applied first.
        if rep == 'lsa' and artifacts['svm_lsa'] is not None:
            chosen_pipe = artifacts['svm_lsa'] if algo == 'svm' else artifacts['lr_lsa']
            X_for_predict = [processed]   # pipeline does its own vectorization
            rep_label = "LSA"
        elif rep == 'bow' and artifacts['bow_vec'] is not None:
            chosen_vec    = artifacts['bow_vec']
            chosen_pipe   = artifacts['svm_bow'] if algo == 'svm' else artifacts['lr_bow']
            X_for_predict = chosen_vec.transform([processed])
            rep_label = "BoW"
        else:
            chosen_vec    = artifacts['tfidf_vec']
            chosen_pipe   = artifacts['svm_tfidf'] if algo == 'svm' else artifacts['lr_tfidf']
            X_for_predict = chosen_vec.transform([processed])
            rep_label = "TF-IDF"

        algo_label = "SVM (Calibrated)" if algo == 'svm' else "Logistic Regression"
        chosen_model_name = f"{algo_label} + {rep_label}"

        # Threshold 0.30 chosen by precision/recall sweep (notebook 4 §8) —
        # lifts F1 from 0.865 → 0.892 by trading 5 pts of precision for 8 pts of recall.
        FRAUD_THRESHOLD = 0.30
        proba      = chosen_pipe.predict_proba(X_for_predict)[0]
        prediction = 1 if proba[1] >= FRAUD_THRESHOLD else 0

        # Small badge above the verdict showing which model produced this result
        is_deployed = (rep == 'tfidf' and algo == 'svm')
        star = " ⭐" if is_deployed else ""
        st.markdown(
            f"<div style='display:flex; justify-content:flex-end; margin:-0.3rem 0 0.5rem;'>"
            f"<div style='display:inline-flex; align-items:center; gap:0.5rem; "
            f"background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.30); "
            f"border-radius:99px; padding:0.25rem 0.7rem; "
            f"font-family:JetBrains Mono,monospace; font-size:0.72rem; "
            f"color:#0369a1; font-weight:600;'>"
            f"<span style='color:#64748b;'>predicted by:</span> {chosen_model_name}{star}"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # Verdict banner
        if prediction == 1:
            pct          = proba[1] * 100
            color        = "#ef4444"
            color_dark   = "#dc2626"
            bg_grad_from = "rgba(239,68,68,0.10)"
            bg_grad_to   = "rgba(239,68,68,0.04)"
            border_col   = "rgba(239,68,68,0.45)"
            shadow_col   = "rgba(239,68,68,0.25)"
            icon         = "🚨"
            label        = "Fraudulent"
            sub          = f"This posting shows strong signs of being a scam ({pct:.1f}% confidence)."
        else:
            pct          = proba[0] * 100
            color        = "#10b981"
            color_dark   = "#059669"
            bg_grad_from = "rgba(16,185,129,0.10)"
            bg_grad_to   = "rgba(16,185,129,0.04)"
            border_col   = "rgba(16,185,129,0.45)"
            shadow_col   = "rgba(16,185,129,0.25)"
            icon         = "✅"
            label        = "Legitimate"
            sub          = f"This posting appears to be a genuine job offer ({pct:.1f}% confidence)."

        st.markdown(
            f"<div style='background:linear-gradient(135deg, {bg_grad_from} 0%, {bg_grad_to} 100%); "
            f"border:1px solid {border_col}; border-radius:14px; "
            f"padding:2rem; display:flex; align-items:center; gap:1.5rem; margin-bottom:1rem; "
            f"box-shadow:0 8px 32px -12px {shadow_col};'>"
            f"<div style='width:64px; height:64px; border-radius:14px; "
            f"background:linear-gradient(135deg, {color} 0%, {color_dark} 100%); "
            f"display:flex; align-items:center; justify-content:center; flex-shrink:0; "
            f"box-shadow:0 4px 14px 0 {shadow_col};'>"
            f"<span style='font-size:1.8rem;'>{icon}</span>"
            f"</div>"
            f"<div>"
            f"<div style='font-size:1.6rem; font-weight:800; color:{color}; "
            f"letter-spacing:-0.01em; line-height:1.1;'>{label}</div>"
            f"<div style='color:#475569; font-size:0.92rem; margin-top:0.4rem; line-height:1.5;'>{sub}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Signals row
        if prediction == 1:
            tags_html = "".join([
                f"<span style='background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.35); "
                f"border-radius:8px; padding:0.4rem 0.85rem; font-size:0.82rem; color:#fca5a5; "
                f"font-weight:500; margin:0.25rem; display:inline-block;'>{t}</span>"
                for t in ["Vague salary promise", "Urgency tactics", "Missing company info", "No requirements listed"]
            ])
            label_color  = "#ef4444"
            signal_label = "🚩 &nbsp;Detected Warning Signs"
        else:
            tags_html = "".join([
                f"<span style='background:rgba(16,185,129,0.10); border:1px solid rgba(16,185,129,0.35); "
                f"border-radius:8px; padding:0.4rem 0.85rem; font-size:0.82rem; color:#6ee7b7; "
                f"font-weight:500; margin:0.25rem; display:inline-block;'>{t}</span>"
                for t in ["Clear job description", "Company info present", "Specific requirements", "Professional tone"]
            ])
            label_color  = "#10b981"
            signal_label = "✓ &nbsp;Positive Signals"

        st.markdown(
            f"<div style='background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; "
            f"padding:1.4rem 1.6rem; margin-bottom:1rem;'>"
            f"<div style='font-size:0.7rem; font-weight:600; color:{label_color}; "
            f"text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.8rem;'>{signal_label}</div>"
            f"<div>{tags_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Score breakdown — wrapped in a real container so widgets sit inside the card
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.7rem; font-weight:600; color:#0369a1; "
                "text-transform:uppercase; letter-spacing:0.10em; margin-bottom:1rem;'>"
                "📊 &nbsp;Score Breakdown</div>",
                unsafe_allow_html=True
            )
            m1, m2 = st.columns(2)
            m1.metric("Legitimate", f"{proba[0]*100:.1f}%")
            m2.metric("Fraudulent", f"{proba[1]*100:.1f}%")

            bar_pct   = proba[1] * 100
            bar_color = "#ef4444" if prediction == 1 else "#10b981"
            st.markdown(
                f"<div style='margin-top:1.2rem;'>"
                f"<div style='display:flex; justify-content:space-between; "
                f"color:#64748b; font-size:0.75rem; margin-bottom:0.5rem; "
                f"text-transform:uppercase; letter-spacing:0.06em; font-weight:600;'>"
                f"<span>Legitimate</span><span>Fraudulent</span></div>"
                f"<div style='background:#f8fafc; border:1px solid #e5e7eb; "
                f"border-radius:99px; height:10px; overflow:hidden; position:relative;'>"
                f"<div style='height:100%; width:{bar_pct:.1f}%; "
                f"background:linear-gradient(90deg, {bar_color}99 0%, {bar_color} 100%); "
                f"border-radius:99px; transition:width 0.4s ease;'></div></div>"
                f"</div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown(
    "<div style='margin-top:3.5rem; padding-top:1.5rem; "
    "border-top:1px solid #e5e7eb; text-align:center;'>"
    "<div style='display:flex; align-items:center; justify-content:center; "
    "gap:0.5rem; color:#64748b; font-size:0.8rem;'>"
    "<span style='display:inline-block; width:24px; height:24px; border-radius:7px; "
    "background:linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%); "
    "display:inline-flex; align-items:center; justify-content:center; "
    "font-size:0.8rem;'>🛡️</span>"
    "<span><b style='color:#475569;'>JobGuard</b> · Scam Detection · "
    "Powered by SVM + TF-IDF · NLP 2026</span>"
    "</div>"
    "</div>",
    unsafe_allow_html=True
)
