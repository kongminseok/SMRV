import tensorflow as tf
from tensorflow.keras.models import load_model

@tf.keras.utils.register_keras_serializable()
def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    
    # Define the loss
    loss = -(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5  ##small epsilon value to prevent division by zero.
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss 

    return loss

# MobileNet 모델 로드 (사전 학습된 모델 사용)
model = load_model('/home/kongminseok/문서/my_pkg/my_pkg/model/mobilenetv2_weights.h5', 
                   custom_objects={"orientation_loss": orientation_loss})

#model.summary()
## SavedModel 형식으로 저장
#tf.saved_model.save(model, 'saved_model')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# INT8 양자화 적용: TensorFlow Lite 변환기는 Optimize.DEFAULT 설정 시 가중치와 활성값을 8비트 정수로 양자화하여 메모리 사용을 줄이고 추론 속도를 높인다.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)
    
# CLI command
# mo --input_model quantized_model.tflite --framework tflite --output_dir ./quantized_model_openvino --input_shape [1,224,224,3]
