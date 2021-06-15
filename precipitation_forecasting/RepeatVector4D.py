from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.keras.backend import repeat_elements

class RepeatVector4D(Layer):
  """Repeats the input n times.
  Example:
  ```python
  inp = tf.keras.Input(shape=(4,4,1))
  # now: model.output_shape == (None, 4,4,1)
  # note: `None` is the batch dimension
  output = RepeatVector4D(3)(inp)
  # now: model.output_shape == (None, 3, 4, 4, 1)
  model = tf.keras.Model(inputs=inp, outputs=output)
  ```
  Args:
    n: Integer, repetition factor.
  Input shape:
    4D tensor of shape `(None, height, width, channels)`.
  Output shape:
    5D tensor of shape `(None, n, height, width, channels)`.
  """

  def __init__(self, n, **kwargs):
    super(RepeatVector4D, self).__init__(**kwargs)
    self.n = n
    self.input_spec = InputSpec(ndim=4)
  
  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

  def call(self, inputs):
    inputs = array_ops.expand_dims(inputs, 1)
    repeat = repeat_elements(inputs, self.n, axis=1)
    return repeat

  def get_config(self):
    config = {'n': self.n}
    base_config = super(RepeatVector4D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
