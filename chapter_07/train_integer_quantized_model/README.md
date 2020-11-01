## Integer quantization

In this module, a Jupyter notebook is provided to cover the folloowing topics:

1. Build a full model with ResNet feature vector.
2. Convert a full model into integer quantization TFLite model.
3. Measure accuracy in scoring test data by TFLite.

`tflite_int8_model` folder is created as the result of successful conversion of original model to TFLite model. 

In addition, this module demonstrates the use of `from_keras_model` API. After a full model is trained, it can be converted to TFLite model directly without having to save it as a saved model. `from_keras_model` API takes a model as soon as it was trained to start the conversion process. 