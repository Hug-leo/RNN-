import numpy as np

class SimpleRNN:
    def __init__(self, hidden_size=32):
        self.hidden_size = hidden_size
        self.vocab = []
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.W_xh = None
        self.W_hh = None
        self.W_hy = None
        self.h_prev = None
        self.is_trained = False

    def init_weights(self, text_input):
        self.vocab = sorted(list(set(text_input)))
        self.vocab_size = len(self.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        
        np.random.seed(42)

        self.W_xh = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_hy = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.h_prev = np.zeros((self.hidden_size, 1))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward_step(self, char_index, h_prev):
        x = np.zeros((self.vocab_size, 1))
        x[char_index] = 1
        h_t = np.tanh(np.dot(self.W_hh, h_prev) + np.dot(self.W_xh.T, x))
        y = np.dot(self.W_hy, h_t)
        prob = self.softmax(y)
        return x, h_t, prob

    def train(self, input_text, epochs, lr, progress_callback=None, log_callback=None):

        x_train = input_text[:-1]
        y_train = input_text[1:]
        self.init_weights(input_text)
        
        for epoch in range(epochs):
            loss = 0
            h_prev = np.zeros_like(self.h_prev)
            hs = {-1: np.copy(h_prev)}
            inputs, ps, targets = {}, {}, {}

            for t in range(len(x_train)):
                char_in_idx = self.char_to_ix[x_train[t]]
                char_out_idx = self.char_to_ix[y_train[t]]
                
                x, h_t, prob = self.forward_step(char_in_idx, hs[t-1])
                inputs[t] = x
                hs[t] = h_t
                ps[t] = prob
                targets[t] = char_out_idx
                loss += -np.log(prob[char_out_idx, 0])

            dWxh = np.zeros_like(self.W_xh)
            dWhh = np.zeros_like(self.W_hh)
            dWhy = np.zeros_like(self.W_hy)
            dh_next = np.zeros_like(h_prev)

            for t in reversed(range(len(x_train))):
                dy = np.copy(ps[t])
                dy[targets[t]] -= 1
                dWhy += np.dot(dy, hs[t].T)
                dh = np.dot(self.W_hy.T, dy) + dh_next
                dh_raw = (1 - hs[t]**2) * dh
                dWhh += np.dot(dh_raw, hs[t-1].T)
                dWxh += np.dot(inputs[t], dh_raw.T)
                dh_next = np.dot(self.W_hh.T, dh_raw)

            for param, dparam in zip([self.W_xh, self.W_hh, self.W_hy], 
                                     [dWxh, dWhh, dWhy]):
                np.clip(dparam, -5, 5, out=dparam)
                param -= lr * dparam

            if progress_callback and (epoch % (epochs // 100 + 1) == 0 or epoch == epochs - 1):
                progress_callback((epoch + 1) / epochs)
            
            if log_callback and (epoch % (epochs // 10 + 1) == 0):
                log_callback(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

        self.is_trained = True
        if progress_callback: progress_callback(1.0)
        if log_callback: log_callback("--> Training hoàn tất! AI đã sẵn sàng.")

    def predict(self, start_char, length):
        if not self.is_trained:
            return "Lỗi: Bạn chưa bấm Train!"
        if start_char not in self.char_to_ix:
            return f"Lỗi: Ký tự '{start_char}' chưa được học!"
            
        h = np.zeros_like(self.h_prev)
        curr_ix = self.char_to_ix[start_char]
        output_str = start_char
        
        for _ in range(length):
            x, h, prob = self.forward_step(curr_ix, h)
            next_ix = np.argmax(prob)
            output_str += self.ix_to_char[next_ix]
            curr_ix = next_ix
            
        return output_str