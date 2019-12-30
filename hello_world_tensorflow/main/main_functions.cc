/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"
#include "constants.h"
#include "output_handler.h"
#include "obj_model.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>



namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;


// Model size- will require trial and error.
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() {
  //Error Reporting
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

	error_reporter->Report("============== RUNNNG SETUP ============\n.");
  model = tflite::GetModel(g_obj_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // To access all our required operations
  static tflite::ops::micro::AllOpsResolver resolver;


	error_reporter->Report("Building Interpreter.\n");


  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
	error_reporter->Report("Interpreter Built\n.");
	
	error_reporter->Report("Allocating Tensors\n.");

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors failed, Memory Issue mostly.");
    return;
  }

	error_reporter->Report("Tensors Allocated\n.");
  // Pointers to the model's input and output tensors.
  	error_reporter->Report("Obtaining Input Tensor ptr\n.");
  input = interpreter->input(0);
    error_reporter->Report("Input Tensor Obtained\n.");
    error_reporter->Report("Obtaining Output Tensor ptr\n.");
  output = interpreter->output(0);
      error_reporter->Report("Output Tensor Obtained\n.");


	error_reporter->Report("============== SETUP SUCCESSFUL ============\n.");
}


void loop() {

error_reporter->Report("============== LOOP() STARTS ============\n.");

	error_reporter->Report("Placing Image in tensor here\n.");
  // Here, we place our image in the form of array in the model's input tensor
  unsigned int x_val=1; 	// IMAGE DATA GOES IN X_VAL
  input->data.f[0] = x_val;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on Image: %f\n",
                           static_cast<double>(x_val));
    return;
  }

  // Read the predicted y value from the model's output tensor
  float y_val = output->data.f[0];

  // Output the object name.
  HandleOutput(error_reporter, x_val, y_val);

}
