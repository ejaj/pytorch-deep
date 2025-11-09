import numpy
from onnx.helper import make_node

from onnx.reference import ReferenceEvaluator

node = make_node('EyeLike', ['X'], ['Y'])

sess = ReferenceEvaluator(node)

x = numpy.random.randn(4, 2).astype(numpy.float32)
feeds = {'X': x}

print(sess.run(None, feeds))
