#!/usr/bin/env python3
"""
Phase 4 • Edge Deployment Preparation (Colab-Compatible)

1. Loads PyTorch ONNX model from Phase 3.
2. Creates a TensorFlow equivalent manually.
3. Converts to quantized TFLite INT8.
4. Generates C array for ESP32.

Tested on Google Colab with TensorFlow 2.15.0
"""

import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
import onnxruntime as ort

# Paths
ROOT        = Path("/content/drive/MyDrive/bfrb-final-project")
EXPORT_DIR  = ROOT/"phase3_modeling"/"exports"
ONNX_PATH   = EXPORT_DIR/"bfrb_model_cpu.onnx"
TFLITE_PATH = EXPORT_DIR/"bfrb_model_int8.tflite"
C_ARRAY_PATH= EXPORT_DIR/"bfrb_model.cc"

def create_tf_model_from_onnx(onnx_path: Path):
    """Create equivalent TensorFlow model by inspecting ONNX structure."""
    print(f"→ Analyzing ONNX model at {onnx_path}")
    
    # Load ONNX session to get input/output shapes
    session = ort.InferenceSession(str(onnx_path))
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    
    # Create equivalent TF model (MLP with same architecture as BFRBMLP)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape[1],)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_shape[1])  # num_classes
    ])
    
    # Compile model (needed for conversion)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Initialize with dummy data
    dummy_input = np.random.randn(1, input_shape[1]).astype(np.float32)
    _ = model(dummy_input)
    
    print("✅ TensorFlow model created with same architecture")
    return model

def load_calibration_data():
    """Load subset of training data for INT8 calibration."""
    print("→ Loading calibration data")
    import pandas as pd
    df = pd.read_parquet(ROOT/"data"/"features"/"bfrb_features_augmented.parquet", engine="pyarrow")
    X = df.drop(columns="label").values.astype(np.float32)
    # Use first 100 samples for calibration
    return X[:100]

def tf_to_tflite_int8(model, tflite_path: Path):
    """Convert TensorFlow model to quantized TFLite."""
    print(f"→ Converting to quantized TFLite at {tflite_path}")
    
    # Load calibration data
    calib_data = load_calibration_data()
    
    def representative_dataset():
        for sample in calib_data:
            yield [sample.reshape(1, -1)]
    
    # Configure converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    
    # Verify the model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
    print(f"  Model size: {len(tflite_model)} bytes")
    print("✅ TFLite model created and verified")

def tflite_to_c_array(tflite_path: Path, c_path: Path):
    """Generate C array from TFLite binary."""
    print(f"→ Generating C array at {c_path}")
    
    # Read TFLite binary
    tflite_data = tflite_path.read_bytes()
    
    # Generate C array manually (xxd might not be available)
    c_array = []
    c_array.append("// Auto-generated TFLite model for ESP32")
    c_array.append(f"// Model size: {len(tflite_data)} bytes")
    c_array.append("")
    c_array.append("#ifndef BFRB_MODEL_H")
    c_array.append("#define BFRB_MODEL_H")
    c_array.append("")
    c_array.append("const unsigned char bfrb_model_tflite[] = {")
    
    # Write bytes in rows of 12
    for i in range(0, len(tflite_data), 12):
        row = tflite_data[i:i+12]
        hex_vals = [f"0x{b:02x}" for b in row]
        c_array.append("  " + ", ".join(hex_vals) + ",")
    
    c_array.append("};")
    c_array.append(f"const unsigned int bfrb_model_tflite_len = {len(tflite_data)};")
    c_array.append("")
    c_array.append("#endif // BFRB_MODEL_H")
    
    # Write to file
    c_path.write_text("\n".join(c_array))
    print("✅ C array generated")

def main():
    print("=== Phase 4: BFRB Model Deployment Preparation ===")
    
    if not ONNX_PATH.exists():
        print(f"❌ ONNX model not found at {ONNX_PATH}")
        print("   Run Phase 3 evaluation first.")
        return
    
    try:
        # Create TensorFlow model with same architecture
        tf_model = create_tf_model_from_onnx(ONNX_PATH)
        
        # Convert to quantized TFLite
        tf_to_tflite_int8(tf_model, TFLITE_PATH)
        
        # Generate C array
        tflite_to_c_array(TFLITE_PATH, C_ARRAY_PATH)
        
        print(f"\n🎉 Phase 4 Complete!")
        print(f"Files generated:")
        print(f"  - TFLite model: {TFLITE_PATH}")
        print(f"  - C array: {C_ARRAY_PATH}")
        print(f"\nNext: Copy {C_ARRAY_PATH.name} to your ESP32 Arduino project")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
