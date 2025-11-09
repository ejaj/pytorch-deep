import numpy
from onnx import numpy_helper, TensorProto, AttributeProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]
# Let's define a function for a linear regression
# The first step consists in creating a constant
# equal to the input parameter of the function.
cst = make_node('Constant', [], ['B'])

att = AttributeProto()
att.name = "value"
# This line indicates the value comes from the argument
# named 'bias' the function is given.
att.ref_attr_name = "bias"
att.type = AttributeProto.TENSOR
cst.attribute.append(att)

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,  # domain name
    'LinearRegression',  # function name
    ['X', 'A'],  # input names
    ['Y'],  # output names
    [cst, node1, node2],  # nodes
    opset_imports,  # opsets
    ["bias"])  # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain,
               # bias is now an argument of the function and is defined as a tensor
               bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])),
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
