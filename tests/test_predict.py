"""Tests for prediction module."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector.predict import predict, predict_from_file


class TestPredict:
    def test_phishing_email(self):
        result = predict(
            "URGENT: Verify your account now!!!",
            "Dear user, click http://192.168.1.1 to verify. "
            "Your account will be suspended!",
        )
        assert result["verdict"] == "PHISHING"
        assert result["phishing_probability"] > 0.5

    def test_legitimate_email(self):
        result = predict(
            "Weekly team sync",
            "Hi team, our weekly sync is at 2pm. "
            "Please review the agenda before the meeting.",
        )
        assert result["verdict"] == "LEGITIMATE"
        assert result["phishing_probability"] < 0.5

    def test_result_structure(self):
        result = predict("Hello", "Test message")
        assert "verdict" in result
        assert "confidence" in result
        assert "phishing_probability" in result
        assert "top_risk_factors" in result
        assert "features" in result

    def test_confidence_range(self):
        result = predict("Test", "Normal message")
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["phishing_probability"] <= 1


class TestPredictFromFile:
    def test_with_subject_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Subject: URGENT: Your account is locked!\n")
            f.write("Click here to unlock: http://evil.ml/login")
            fname = f.name
        try:
            result = predict_from_file(fname)
            assert result["verdict"] == "PHISHING"
        finally:
            os.unlink(fname)

    def test_without_subject_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Meeting reminder\n")
            f.write("Don't forget our standup at 9am tomorrow.")
            fname = f.name
        try:
            result = predict_from_file(fname)
            assert result["verdict"] in ("PHISHING", "LEGITIMATE")
        finally:
            os.unlink(fname)
