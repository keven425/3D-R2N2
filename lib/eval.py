import numpy as np

def cross_entropy_batch(logits, labels):
  ce = cross_entropy(logits, labels)
  ce = np.mean(ce)
  return ce


def cross_entropy(logits, labels):
  _shape = list(logits.shape[:-1]) + [1]
  _max = np.max(logits, axis=-1).reshape(_shape)
  e_x = np.exp(logits - _max)
  p = e_x.take(labels) / e_x.sum(axis=-1)
  return -np.log(p)

def softmax(logits):
  # input contains log probabiliy
  _shape = list(logits.shape[:-1]) + [1]
  _max = np.max(logits, axis=-1).reshape(_shape)
  e_x = np.exp(logits - _max)
  return e_x / e_x.sum(axis=-1).reshape(_shape)
