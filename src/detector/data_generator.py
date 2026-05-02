"""Generate synthetic phishing and legitimate email dataset."""
import csv
import random
import os


def generate_legit_email():
    """Generate a legitimate email sample."""
    subjects = [
        "Weekly team meeting tomorrow at 10am",
        "Invoice #12345 for your recent purchase",
        "Your Amazon order has shipped",
        "Meeting notes from yesterday",
        "Project update - Q3 roadmap",
        "Happy birthday!",
        "Package delivery confirmation",
        "Please review the attached document",
        "Coffee chat this Friday?",
        "Newsletter: March 2024 updates",
    ]
    bodies = [
        "Hi team, just a reminder about our weekly standup tomorrow. Please come prepared with your updates.",
        "Thank you for your purchase. Your invoice is attached. Please let us know if you have any questions.",
        "Your order #ORD-98765 has been shipped via FedEx. Track it at fedex.com/tracking.",
        "Attached are the meeting notes from yesterday's session. Action items are highlighted.",
        "Hi, I've updated the project timeline. Please review and provide feedback by Friday.",
        "Wishing you a wonderful birthday! Hope you have a great celebration.",
        "Your package has been delivered to your front porch. Let us know if you need anything else.",
        "Please find the document attached for your review. Looking forward to your feedback.",
        "Would you be free for a quick coffee chat this Friday afternoon? No pressure!",
        "Here are our March updates: new features, upcoming events, and community highlights.",
    ]
    idx = random.randint(0, len(subjects) - 1)
    return subjects[idx], bodies[idx]


def generate_phishing_email():
    """Generate a phishing email sample."""
    subjects = [
        "URGENT: Verify your account immediately!!!",
        "Your account has been suspended - click here now",
        "Free gift card waiting - claim in 24 hours!!!",
        "Security alert: unauthorized login detected",
        "You won $1000 - claim your prize NOW",
        "Confirm your password or lose access forever",
        "Action required: update payment information",
        "Limited offer: free iPhone for first 100 users!!!",
        "Your PayPal account needs verification",
        "Click here to avoid account deletion",
    ]
    bodies = [
        "Dear user, your account will be suspended in 24 hours. Click here to verify: http://192.168.1.1/login.html. URGENT!!!",
        "We detected suspicious activity. Click this link immediately to secure your account: http://suspicious-bank.xyz/login",
        "Congratulations! You've won a $1000 gift card. Claim now at http://free-gift-site.com/claim?user=you. Limited time!!!",
        "WARNING: Someone tried to access your account from Russia. Click here to verify your identity: http://verify-paypal.tk/secure",
        "You are a lucky winner! Get your free iPhone by clicking: http://192.168.5.5/free-iphone.html. Act fast!!!",
        "Dear valued customer, please confirm your password to avoid service disruption. Submit here: http://phishing-site.com/form",
        "Your payment failed. Update your credit card immediately: http://secure-bank-update.com/payment. Don't lose access!!!",
        "Exclusive offer! Free vacation package. Click now: http://192.168.99.99/vacation-free. Only 24 hours left!!!",
        "PayPal Alert: Your account is limited. Click to restore access: http://paypa1-security.com/restore",
        "FINAL NOTICE: Account deletion scheduled. Click here to stop it: http://urgent-verify.com/stop-delete?token=abc123",
    ]
    idx = random.randint(0, len(subjects) - 1)
    return subjects[idx], bodies[idx]


def generate_dataset(n_samples=1000, output_path="data/emails.csv"):
    """Generate balanced synthetic dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for _ in range(n_samples // 2):
        subj, body = generate_legit_email()
        rows.append([subj, body, 0])
    for _ in range(n_samples // 2):
        subj, body = generate_phishing_email()
        rows.append([subj, body, 1])

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "body", "label"])
        writer.writerows(rows)

    print(f"Generated {n_samples} samples at {output_path}")


if __name__ == "__main__":
    generate_dataset(2000, "data/emails.csv")
