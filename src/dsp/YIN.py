import numpy as np

YIN_THRESHOLD = 0.2


class YIN:
    def __init__(self, frame_len, sample_rate):
        self.yin_frame = np.zeros(frame_len, dtype=float)
        self.sample_rate = sample_rate
        self.input_frame = None

    def absolute_threshold(self):
        buffer_len = len(self.yin_frame)
        buffer = self.yin_frame

        tau = 2

        for tau in range(buffer_len):
            if buffer[tau] < YIN_THRESHOLD:
                while tau + 1 < buffer_len and buffer[tau + 1] < buffer[tau]:
                    tau += 1
                break

        return -1 if (tau == buffer_len or buffer[tau] >= YIN_THRESHOLD) else tau

    def difference_function(self):
        yin_buffer = self.yin_frame
        yin_len = len(yin_buffer)
        input_buffer = self.input_frame

        for tau in range(yin_len):
            buf = input_buffer[:(yin_len - tau)].astype(np.float32)
            buf2 = input_buffer[tau:].astype(np.float32)
            dif_buf = (buf2 - buf) ** 2

            yin_buffer[tau] = np.sum(dif_buf)

    @staticmethod
    def parabolic_interpolation(x_vector, x_):
        if x_ < 0:
            return x_

        x = int(x_)

        x_adjusted = 0

        if x < 1:
            if x_vector[x] <= x_vector[x + 1]:
                x_adjusted = x
            else:
                x_adjusted = x + 1

        elif x > len(x_vector) - 1:
            if x_vector[x] <= x_vector[x - 1]:
                x_adjusted = x
            else:
                x_adjusted = x - 1

        else:
            den = x_vector[(x + 1) % len(x_vector)] + x_vector[x - 1] - 2 * x_vector[x]
            delta = x_vector[x - 1] - x_vector[(x + 1) % len(x_vector)]

            if den == 0.0:
                return x_
            else:
                return x_ + delta / (2 * den)

        return float(x_adjusted)

    def cumulative_normalized_difference(self):
        buffer = self.yin_frame
        buffer_len = len(self.yin_frame)

        buffer[0] = 1
        cumsum = np.cumsum(buffer[1:])
        indexes = cumsum <= 0.0
        cumsum[indexes] = 1.0
        tau = np.arange(1, buffer_len)
        buffer[1:] *= tau / cumsum

    def detect_pitch(self, input_frame):
        self.input_frame = input_frame

        if self.input_frame is None:
            assert "set input frame"
            return -1

        self.difference_function()
        self.cumulative_normalized_difference()
        tau = self.absolute_threshold()
        tau = self.parabolic_interpolation(self.yin_frame, tau)

        return tau
