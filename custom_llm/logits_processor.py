from transformers import LogitsProcessor

class HyperfocusLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        hyper_tokens,
        avoid_tokens,
        hyper_bias=1.5,
        avoid_bias=1.5,
        min_step_to_focus=5,
        ramp_steps=20,
        topic_boost_steps=10,
        decay_window=20,
        min_gap_between_hyperfocus=4,
    ):
        super().__init__()
        self.hyper_tokens = set(hyper_tokens)
        self.avoid_tokens = set(avoid_tokens)
        self.hyper_bias = hyper_bias
        self.avoid_bias = avoid_bias
        self.min_step_to_focus = min_step_to_focus
        self.ramp_steps = ramp_steps
        self.topic_boost_steps = topic_boost_steps
        self.decay_window = decay_window
        self.min_gap_between_hyperfocus = min_gap_between_hyperfocus

        self.step = 0
        self.recent_tokens = []
        self.last_hyperfocus_step = -min_gap_between_hyperfocus

    def __call__(self, input_ids, scores):
        if self.step < self.min_step_to_focus:
            self.step += 1
            return scores

        ramp_factor = min((self.step - self.min_step_to_focus) / self.ramp_steps, 1.0)
        bias = scores.new_zeros(scores.shape[-1])

        num_recent_hyper = sum(1 for t in self.recent_tokens if t in self.hyper_tokens)
        hyper_decay = max(0.9 ** num_recent_hyper, 0.5)

        if self.step < self.min_step_to_focus + self.topic_boost_steps:
            for token_id in self.hyper_tokens:
                if self.step - self.last_hyperfocus_step >= self.min_gap_between_hyperfocus:
                    bias[token_id] += self.hyper_bias * 1.5 * hyper_decay
        else:
            hyper_bias_factor = self.hyper_bias * ramp_factor * hyper_decay
            for token_id in self.hyper_tokens:
                if self.step - self.last_hyperfocus_step >= self.min_gap_between_hyperfocus:
                    bias[token_id] += hyper_bias_factor * 0.2

        for token_id in self.avoid_tokens:
            bias[token_id] -= self.avoid_bias

        scores = scores + bias

        last_token_id = input_ids[0, -1].item()
        self.recent_tokens.append(last_token_id)
        if len(self.recent_tokens) > self.decay_window:
            self.recent_tokens.pop(0)

        if last_token_id in self.hyper_tokens:
            self.last_hyperfocus_step = self.step

        self.step += 1
        return scores
