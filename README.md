# 🧠 SimpleRNN Text Generator (Built from Scratch)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Numpy](https://img.shields.io/badge/Numpy-Core-013243?style=flat&logo=numpy)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-2CC985?style=flat)

A character-level **Recurrent Neural Network (RNN)** implementation built entirely from scratch using only `NumPy`. This project demonstrates the core mathematics behind sequence generation without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

It features a modern, responsive **Dark Mode GUI** built with `CustomTkinter`, allowing users to visualize the training process, tune hyperparameters, and generate text in real-time.

---

## 📸 Screenshots

![App Screenshot](screenshot_placeholder.png)

---

## ✨ Key Features

* **Pure Math Implementation:** The RNN logic (Forward Pass, Backpropagation Through Time - BPTT, Gradient Descent) is implemented manually using matrix operations.
* **Modular Design:** Clean separation between Logic (`rnn_backend.py`) and Interface (`rnn_gui.py`) following MVC principles.
* **Modern GUI:** Built with `customtkinter` for a sleek, dark-themed experience.
* **Multithreading:** Training runs on a separate thread, ensuring the UI remains responsive and fluid during heavy computations.
* **Hyperparameter Tuning:** Users can customize `Epochs`, `Learning Rate`, and `Hidden Layer Size` directly from the sidebar.
* **Real-time Logging:** View loss metrics and training progress directly in the application.

---

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/SimpleRNN-Text-Generator.git](https://github.com/your-username/SimpleRNN-Text-Generator.git)
    cd SimpleRNN-Text-Generator
    ```

2.  **Install dependencies**
    You only need `numpy` for math and `customtkinter` for the GUI.
    ```bash
    pip install numpy customtkinter
    ```

---

## 🛠 Usage

1.  Run the application:
    ```bash
    python rnn_gui.py
    ```

2.  **Configure:** Set your desired Epochs (e.g., 2000) and Hidden Size (e.g., 32) in the sidebar.
3.  **Input Data:** Type or paste text into the main text box (e.g., *"hello world engineering"*).
4.  **Train:** Click the **"Start Training"** button. Watch the Loss decrease in the log box.
5.  **Test:** Enter a starting character (e.g., *"h"*) in the bottom section and click **"Generate"** to see the AI predict the rest of the sequence.

---

## 📂 Project Structure

```text
SimpleRNN-Text-Generator/
│
├── rnn_backend.py    # The "Brain" - Contains the RNN class, BPTT logic, and math.
├── rnn_gui.py        # The "Face" - Main application file handling UI and Threading.
├── README.md         # Project documentation.
└── requirements.txt  # List of dependencies.
# RNN-
