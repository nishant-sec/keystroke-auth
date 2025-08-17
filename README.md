# ⌨️ Free-Text Keystroke Dynamics Authentication

This project introduces a passwordless authentication system that verifies your identity based on your unique typing rhythm. Using an ESN-SVM hybrid model, it achieves 98.7% accuracy in recognizing users via free-text, making your typing pattern your password.

## Features

  - **Passwordless Login**: Authenticates you based on *how* you type, not *what* you type.
  - **High Accuracy**: Achieves a 98.7% success rate in distinguishing users.
  - **AI-Powered**: An Echo State Network (ESN) learns your typing rhythm, and a Support Vector Machine (SVM) makes the final verification decision.
  - **Free-Text Analysis**: Works with any text, from emails to documents, without requiring specific passphrases.

## Getting Started

### Prerequisites

  - Python 3.8+

### Installation & Execution

1.  Clone the repository:
    ```bash
    git clone git@github.com:nishant-sec/keystroke-auth.git
    cd keystroke-auth
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python app.py
    ```

## How It Works?

The system captures subtle patterns in your typing, such as keystroke latency, hold times, and the rhythm of your pauses and bursts.

1.  **Enrollment**: During a one-time setup, you'll type a few sample texts. The ESN analyzes these samples to create a unique mathematical "fingerprint" of your typing style.
2.  **Verification**: To log in, simply type any text. The system compares your current typing pattern to your stored fingerprint. The SVM then makes a decision, granting access if the patterns match.

This method is secure against shoulder-surfing and eliminates the need to remember complex passwords.


## Use Cases

This technology is ideal for:

  - **Continuous Authentication**: Passively verify a user's identity throughout a session.
  - **Enhanced Security**: Add a seamless second layer of security for accessing sensitive documents or applications.
  - **Behavioral Biometrics Research**: A practical tool for exploring AI-driven security and user identification.


## Contributing

Contributions are welcome\! Please feel free to open an issue to report a bug, suggest an enhancement, or submit a pull request.
