"""Feature extraction for phishing email detection."""
import re
import urllib.parse
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data on first import
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Suspicious keywords commonly found in phishing emails
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "account", "suspend", "suspend",
    "click", "link", "password", "login", "credential",
    "security", "alert", "confirm", "update", "unauthorized",
    "limited", "expires", "24 hours", "immediately", "now",
    "free", "winner", "prize", "won", "gift", "claim",
    "bank", "paypal", "credit card", "ssn", "social security",
    "lottery", "inheritance", "million", "payment",
    "action required", "notice", "restricted", "locked",
]

# Suspicious TLDs
SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".top", ".xyz", ".click", ".link", ".work"}

# Urgency indicators
URGENCY_WORDS = ["urgent", "immediately", "now", "hurry", "limited", "expires", "act now", "24 hours", "today", "last chance"]


def extract_features(subject: str, body: str) -> dict:
    """Extract features from an email subject and body."""
    text = f"{subject} {body}".lower()
    words = word_tokenize(text)
    text_no_punct = re.sub(r'[^\w\s]', ' ', text)
    words_clean = [w for w in word_tokenize(text_no_punct) if w not in STOP_WORDS and len(w) > 2]

    urls = extract_urls(text)

    features = {
        # Text statistics
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_count": sum(1 for c in text if c.isdigit()),

        # Suspicious keywords
        "suspicious_keyword_count": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text),
        "suspicious_keyword_ratio": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text) / max(len(SUSPICIOUS_KEYWORDS), 1),

        # Urgency indicators
        "urgency_word_count": sum(1 for u in URGENCY_WORDS if u in text),
        "has_urgent": int("urgent" in text),
        "has_immediately": int("immediately" in text),
        "has_limited_time": int(any(x in text for x in ["24 hours", "limited time", "expires"])),

        # Greeting / personalization patterns
        "has_dear_user": int(any(x in text for x in ["dear user", "dear customer", "dear valued"])),
        "has_greeting": int(any(x in text for x in ["dear ", "hello ", "hi "])),

        # Spelling / grammar (proxy: ratio of non-dictionary short words)
        "non_stopword_count": len(words_clean),

        # URL features
        "url_count": len(urls),
        "has_url": int(len(urls) > 0),
        "has_ip_url": int(any(is_ip_url(u) for u in urls)),
        "suspicious_tld_count": sum(1 for u in urls if has_suspicious_tld(u)),
        "has_suspicious_tld": int(any(has_suspicious_tld(u) for u in urls)),
        "domain_count": len(set(extract_domains(urls))),
        "has_http": int(any(u.startswith("http://") for u in urls)),
        "https_ratio": np.mean([int(u.startswith("https://")) for u in urls]) if urls else 0,
        "url_length_avg": np.mean([len(u) for u in urls]) if urls else 0,
        "has_url_shortener": int(any(is_url_shortener(u) for u in urls)),

        # HTML / formatting hints
        "html_tag_count": len(re.findall(r'<[^>]+>', body)),
        "has_html": int(len(re.findall(r'<[^>]+>', body)) > 0),
        "link_count": body.lower().count("<a ") + body.lower().count("href="),

        # Subject-specific
        "subject_exclamation": int("!" in subject),
        "subject_uppercase_ratio": sum(1 for c in subject if c.isupper()) / max(len(subject), 1),
        "subject_suspicious_words": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in subject.lower()),
        "subject_word_count": len(word_tokenize(subject.lower())),
    }

    return features


def extract_urls(text: str) -> list:
    """Extract URLs from text."""
    pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
    return re.findall(pattern, text)


def extract_domains(urls: list) -> list:
    """Extract domains from URLs."""
    domains = []
    for url in urls:
        try:
            parsed = urllib.parse.urlparse(url)
            domains.append(parsed.netloc)
        except Exception:
            pass
    return domains


def is_ip_url(url: str) -> bool:
    """Check if URL uses an IP address instead of a domain."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.split(":")[0]
        # Simple IPv4 check
        return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', netloc))
    except Exception:
        return False


def has_suspicious_tld(url: str) -> bool:
    """Check if URL uses a suspicious top-level domain."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.split(":")[0]
        for tld in SUSPICIOUS_TLDS:
            if netloc.endswith(tld):
                return True
        return False
    except Exception:
        return False


def is_url_shortener(url: str) -> bool:
    """Check if URL is from a known shortener."""
    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "short.link", "tr.im"}
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc.lower() in shorteners
    except Exception:
        return False


if __name__ == "__main__":
    # Quick test
    sample = {
        "subject": "URGENT: Verify your account!!!",
        "body": "Dear user, click here to verify: http://192.168.1.1/login.html"
    }
    feats = extract_features(sample["subject"], sample["body"])
    for k, v in feats.items():
        print(f"{k}: {v}")
