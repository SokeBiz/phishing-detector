"""Tests for feature extraction module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector.features import (
    extract_features,
    extract_urls,
    is_ip_url,
    has_suspicious_tld,
    is_url_shortener,
)


class TestExtractUrls:
    def test_http_url(self):
        urls = extract_urls("Visit http://example.com")
        assert "http://example.com" in urls

    def test_https_url(self):
        urls = extract_urls("Secure link: https://secure.example.org/path")
        assert "https://secure.example.org/path" in urls

    def test_no_urls(self):
        urls = extract_urls("This is a normal email with no links.")
        assert urls == []

    def test_multiple_urls(self):
        urls = extract_urls("Links: http://a.com and https://b.org/page")
        assert len(urls) == 2


class TestIsIpUrl:
    def test_ip_address(self):
        assert is_ip_url("http://192.168.1.1/login") is True

    def test_domain_name(self):
        assert is_ip_url("https://google.com") is False

    def test_localhost(self):
        assert is_ip_url("http://127.0.0.1") is True


class TestHasSuspiciousTld:
    def test_suspicious_tld(self):
        assert has_suspicious_tld("http://malicious.tk") is True
        assert has_suspicious_tld("http://free-prize.ml") is True

    def test_safe_tld(self):
        assert has_suspicious_tld("https://google.com") is False
        assert has_suspicious_tld("https://github.org") is False


class TestIsUrlShortener:
    def test_shortener(self):
        assert is_url_shortener("http://bit.ly/abc123") is True
        assert is_url_shortener("http://tinyurl.com/xyz") is True

    def test_not_shortener(self):
        assert is_url_shortener("https://google.com") is False


class TestExtractFeatures:
    def test_phishing_email(self):
        feats = extract_features(
            "URGENT: Verify your account!!!",
            "Dear user, click here to verify: http://192.168.1.1/login.html "
            "Your account will be suspended! Act now!",
        )
        assert feats["has_url"] == 1
        assert feats["has_ip_url"] == 1
        assert feats["has_urgent"] == 1
        assert feats["exclamation_count"] >= 3
        assert feats["suspicious_keyword_count"] > 0

    def test_legitimate_email(self):
        feats = extract_features(
            "Meeting tomorrow",
            "Hi team, let's meet at 3pm to discuss the project. "
            "Please bring your laptops. Thanks!",
        )
        assert feats["has_url"] == 0
        assert feats["suspicious_keyword_count"] == 0
        assert feats["has_dear_user"] == 0

    def test_extracts_all_features(self):
        feats = extract_features("Test", "Hello world")
        expected_keys = [
            "text_length", "word_count", "url_count", "has_url",
            "has_ip_url", "suspicious_keyword_count", "has_urgent",
            "exclamation_count", "has_html", "has_dear_user",
        ]
        for key in expected_keys:
            assert key in feats, f"Missing feature: {key}"
        assert len(feats) >= 30
