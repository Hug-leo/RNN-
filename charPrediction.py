import numpy as np
import sys

class SimpleRNN:
    def __init__(self, text_input, hidden_size=16):
        # --- 1. DATA PREPARATION ---
        self.vocab = sorted(list(set(text_input)))
        self.vocab_size = len(self.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        # --- 2. WEIGHT INITIALIZATION ---
        np.random.seed(42)
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(self.vocab_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(self.vocab_size, hidden_size) * 0.01
        self.h_prev = np.zeros((hidden_size, 1))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) 
        return e_x / e_x.sum(axis=0)

    def forward_step(self, char_index, h_prev):
        # Input vector (One-hot)
        x = np.zeros((self.vocab_size, 1))
        x[char_index] = 1
        
        # Hidden State Update
        h_t = np.tanh(np.dot(self.W_hh, h_prev) + np.dot(self.W_xh.T, x))
        
        # Output Calculation
        y = np.dot(self.W_hy, h_t)
        prob = self.softmax(y)
        
        return x, h_t, prob

    def train(self, input_str, target_str, epochs=2000, learning_rate=0.1):
        print(f"\n[INFO] Training model mapping: '{input_str}' -> '{target_str}'")
        print(f"[INFO] Vocabulary size: {self.vocab_size} | Hidden size: {self.hidden_size}")
        print("Training Progress:")
        
        for epoch in range(epochs):
            loss = 0
            h_prev = np.zeros_like(self.h_prev)
            
            inputs, hs, ps, targets = {}, {}, {}, {}
            hs[-1] = np.copy(h_prev)
            
            # --- FORWARD PASS ---
            for t in range(len(input_str)):
                char_in_idx = self.char_to_ix[input_str[t]]
                char_out_idx = self.char_to_ix[target_str[t]]
                
                x, h_t, prob = self.forward_step(char_in_idx, hs[t-1])
                
                inputs[t] = x
                hs[t] = h_t
                ps[t] = prob
                targets[t] = char_out_idx
                loss += -np.log(prob[char_out_idx, 0])
            
            # --- BACKWARD PASS (BPTT) ---
            dWxh, dWhh, dWhy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
            dh_next = np.zeros_like(h_prev)
            
            for t in reversed(range(len(input_str))):
                dy = np.copy(ps[t])
                dy[targets[t]] -= 1 
                
                dWhy += np.dot(dy, hs[t].T)
                
                dh = np.dot(self.W_hy.T, dy) + dh_next
                dh_raw = (1 - hs[t] * hs[t]) * dh 
                
                dWhh += np.dot(dh_raw, hs[t-1].T)
                dWxh += np.dot(inputs[t], dh_raw.T)
                dh_next = np.dot(self.W_hh.T, dh_raw)

            for param, dparam in zip([self.W_xh, self.W_hh, self.W_hy], [dWxh, dWhh, dWhy]):
                np.clip(dparam, -5, 5, out=dparam)
                param -= learning_rate * dparam

            if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                sys.stdout.write(f"\rEpoch {epoch}/{epochs} | Loss: {loss:.4f} | " + "#" * (epoch // (epochs // 10)))
                sys.stdout.flush()
        
        print("\n[INFO] Training Complete!")

    def predict(self, start_char, length):
        if start_char not in self.char_to_ix:
            return f"[ERROR] Character '{start_char}' not in vocabulary: {self.vocab}"
            
        h = np.zeros_like(self.h_prev)
        curr_ix = self.char_to_ix[start_char]
        output_str = start_char
        
        for _ in range(length):
            x, h, prob = self.forward_step(curr_ix, h)

            next_ix = np.argmax(prob)
            output_str += self.ix_to_char[next_ix]
            
            curr_ix = next_ix
            
        return output_str

def main():
    print("==========================================")
    print("   SIMPLE RNN TEXT GENERATOR (MINI-MODEL)")
    print("==========================================")

    while True:
        print("\n--- NEW TRAINING SESSION ---")
        raw_data = input("Enter text to train (or type 'quit' to exit program): ").strip()
        
        if raw_data.lower() == 'quit':
            print("Goodbye!")
            break
            
        if len(raw_data) < 2:
            print("[WARNING] Text must have at least 2 characters.")
            continue

        x_train = raw_data[:-1] 
        y_train = raw_data[1:]

        try:
            epochs_input = input("Enter epochs (default 2000): ").strip()
            num_epochs = int(epochs_input) if epochs_input else 2000
        except ValueError:
            num_epochs = 2000

        rnn = SimpleRNN(raw_data, hidden_size=24)
        rnn.train(x_train, y_train, epochs=num_epochs, learning_rate=0.1)
        
        print("-" * 40)
        print(f"Known Vocabulary: {rnn.vocab}")
        print("-" * 40)

        while True:
            user_input = input(f"\n[TEST] Enter start char (or 'new' for new text, 'quit' to exit): ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                return 
            
            if user_input.lower() == 'new':
                break 
            
            if len(user_input) > 0:
                start_char = user_input[0] 
                prediction = rnn.predict(start_char, length=len(raw_data)-1)
                print(f"-> AI Prediction: {prediction}")
            else:
                print("[WARNING] Please enter a character.")

if __name__ == "__main__":
    main()