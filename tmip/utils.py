import time

class IdentityTransform():
  def fit(self, x):
    return self

  def transform(self, x):
    return x

  def inverse_transform(self, x):
    return x


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600.0)
        if x >= 60:
            return '{:.1f}m'.format(x / 60.0)
        return '{}s'.format(round(x))

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0
