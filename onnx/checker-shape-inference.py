import onnx.parser
import onnx.checker

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
try:
    onnx_model = onnx.parser.parse_model(input)
    onnx.checker.check_model(onnx_model)
except Exception as e:
    print(e)
onnx_model = onnx.parser.parse_model(input)
inferred_model = shape_inference.infer_shapes(onnx_model)

print(inferred_model)
