model.py :- Creates the dataset, adds noise, creates and trains the model.
Additionally, it converts the tensorflow model to a flat buffer (quantized and non-quantized).
IMPORTANT: Update tensorflow version to 2.x (Mine is:2.0.0-dev20190922)


To convert the flat buffer to c++, run command :
xxd -i model_name.tflite > model_cpp_name.cc
