import yaml
import subprocess
import copy
import random
import os
import shutil
import argparse
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from random import shuffle
from imgaug.augmentables import KeypointsOnImage
from IPython.display import SVG

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Lambda, LeakyReLU, Reshape
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from ultralytics import YOLO

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # GENERAL 설정 불러오기
    BIN = config['GENERAL']['BIN']
    BACKBONE = config['GENERAL']['BACKBONE']
    OVERLAP = config['GENERAL']['OVERLAP']
    W = config['GENERAL']['W']
    ALPHA = config['GENERAL']['ALPHA']
    MAX_JIT = config['GENERAL']['MAX_JIT']
    NORM_H = config['GENERAL']['NORM_H']
    NORM_W = config['GENERAL']['NORM_W']
    BATCH_SIZE = config['GENERAL']['BATCH_SIZE']
    EPOCH = config['GENERAL']['EPOCH']
    AUGMENTATION = config['GENERAL']['AUGMENTATION']

    # VEHICLES 설정 불러오기
    VEHICLES = config['VEHICLES']['VEHICLES_LIST']

    return BIN, BACKBONE, OVERLAP, W, ALPHA, MAX_JIT, NORM_H, NORM_W, BATCH_SIZE, EPOCH, AUGMENTATION, VEHICLES

BIN, BACKBONE, OVERLAP, W, ALPHA, MAX_JIT, NORM_H, NORM_W, BATCH_SIZE, EPOCH, AUGMENTATION, VEHICLES = load_config()


####################################################### Preprocessing #######################################################
def parse_annotation(label_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(os.path.join(label_dir, label_file)).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = (float(line[3]) + np.pi / 2) % (2 * np.pi)

                obj = {
                    'name': line[0],
                    'image': image_file,
                    'xmin': int(float(line[4])),
                    'ymin': int(float(line[5])),
                    'xmax': int(float(line[6])),
                    'ymax': int(float(line[7])),
                    'dims': np.array([float(number) for number in line[8:11]]),
                    'new_alpha': new_alpha
                }

                dims_avg[obj['name']] = (dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']) / (dims_cnt[obj['name']] + 1)
                dims_cnt[obj['name']] += 1
                all_objs.append(obj)
    
    return all_objs, dims_avg

def adjust_dimensions(obj, dims_avg):
    obj['dims'] -= dims_avg[obj['name']]

def compute_anchors(angle):
    anchors = []
    wedge = 2. * np.pi / BIN

    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])
    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])
    return anchors

def calculate_orientation_and_confidence(obj, flip=False):
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    angle = 2 * np.pi - obj['new_alpha'] if flip else obj['new_alpha']
    anchors = compute_anchors(angle)

    for anchor in anchors:
        orientation[anchor[0]] = [np.cos(anchor[1]), np.sin(anchor[1])]
        confidence[anchor[0]] = 1.0

    confidence /= np.sum(confidence)
    return orientation, confidence

def process_object_data(all_objs, dims_avg):
    for obj in all_objs:
        adjust_dimensions(obj, dims_avg)
        
        obj['orient'], obj['conf'] = calculate_orientation_and_confidence(obj, flip=False)
        obj['orient_flipped'], obj['conf_flipped'] = calculate_orientation_and_confidence(obj, flip=True)

def prepare_input_and_output(train_inst, image_dir):
    xmin, ymin, xmax, ymax = train_inst['xmin'], train_inst['ymin'], train_inst['xmax'], train_inst['ymax']
    img = copy.deepcopy(cv2.imread(os.path.join(image_dir, train_inst['image']))[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)

    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
    
    img = cv2.resize(img, (NORM_H, NORM_W))
    img -= np.array([[[103.939, 116.779, 123.68]]])
    
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']
   
 
####################################################### Training #######################################################
def augment_image(image):
    seq = iaa.Sequential([
    iaa.Crop(px=(0, 7)),  
    iaa.Crop(px=(7, 0)),  
    iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    
    return seq.augment_images([image])[0]

def data_gen(all_objs, image_dir, batch_size):
    num_obj = len(all_objs)
    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        if not AUGMENTATION:
            x_batch = np.zeros((batch_size, 224, 224, 3))
            d_batch = np.zeros((batch_size, 3))
            o_batch = np.zeros((batch_size, BIN, 2))
            c_batch = np.zeros((batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                # input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(train_inst=all_objs[key],image_dir=image_dir)

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

            yield x_batch, {
                'dimension': d_batch,
                'orientation': o_batch,
                'confidence': c_batch
            }

        if AUGMENTATION:
            x_batch = np.zeros((2 * batch_size, 224, 224, 3))
            d_batch = np.zeros((2 * batch_size, 3))
            o_batch = np.zeros((2 * batch_size, BIN, 2))
            c_batch = np.zeros((2 * batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                # input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(train_inst=all_objs[key],image_dir=image_dir)

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

                # Augmented images
                x_batch[idx + batch_size, :] = augment_image(image)
                d_batch[idx + batch_size, :] = dimension.copy()
                o_batch[idx + batch_size, :] = orientation.copy()
                c_batch[idx + batch_size, :] = confidence.copy()

            yield x_batch, {
                'dimension': d_batch,
                'orientation': o_batch,
                'confidence': c_batch
            }

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj:
            r_bound = num_obj

####################################################### Training #######################################################
def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

@tf.keras.saving.register_keras_serializable()
def orientation_loss(y_true, y_pred):    
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    
    loss = -(y_true[:,:,0] * y_pred[:,:,0] + y_true[:,:,1] * y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    epsilon = 1e-5
    anchors = anchors + epsilon
    loss = loss / anchors
    loss = tf.reduce_mean(loss)
    loss = 2 - 2 * loss 

    return loss

def build_model(input_shape=(224, 224, 3), bin_size=6):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    # Dimension branch
    dimension = Dense(512)(x)
    dimension = LeakyReLU(alpha=0.1)(dimension)
    dimension = Dropout(0.2)(dimension)
    dimension_output = Dense(3, name='dimension')(dimension)

    # Orientation branch
    orientation = Dense(256)(x)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.2)(orientation)
    orientation = Dense(bin_size * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((bin_size, -1))(orientation)
    orientation_output = Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = Dense(256)(x)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.2)(confidence)
    confidence_output = Dense(bin_size, activation='softmax', name='confidence')(confidence)

    model = tf.keras.Model(inputs=inputs, outputs={
        'dimension': dimension_output,
        'orientation': orientation_output,
        'confidence': confidence_output
    })
    return model

####################################################### Utility #######################################################
def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the given file path exists. If not, create it.

    Args:
        file_path (str): Full path to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_or_initialize_model(model_path, input_shape=(224, 224, 3), bin_size=6):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f'Loaded model from {model_path}')
        print(model.summary())
    else:
        print(f'{model_path} does not exist, starting training from scratch.')
        model = build_model(input_shape=input_shape, bin_size=bin_size)
    return model

def get_last_epoch(history_path):
    try:
        history_df = pd.read_csv(history_path)
        last_epoch = history_df.index[-1] + 1
    except FileNotFoundError:
        last_epoch = 0
    print(f'Last epoch number: {last_epoch}')
    return last_epoch

def compile_model(model):
    optimizer = Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss={
            'dimension': 'mean_squared_error',          # 객체의 실제 크기(높이, 너비, 길이)를 정확히 예측하는 것을 목표
            'orientation': orientation_loss,            # 각 구간의 방향 벡터 간 내적을 기반으로 실제 방향에 가장 가까운 구간으로 예측되도록 함
            'confidence': 'binary_crossentropy',        # 신뢰도가 가장 높은 구간이 실제 방향에 가까운 구간이 되도록 학습
        },
        loss_weights={
            'dimension': 5.0, 
            'orientation': 1.5, 
            'confidence': 0.5,
        },
        metrics={
            'dimension': 'mse', 
            'orientation': 'mse', 
            'confidence': 'accuracy',
        }
    )

def prepare_generators(all_objs, image_dir, batch_size=BATCH_SIZE, train_split=0.85):
    np.random.shuffle(all_objs)
    total_samples = len(all_objs)
    train_samples = int(train_split * total_samples)
    
    train_gen = data_gen(all_objs[:train_samples], image_dir, batch_size)
    valid_gen = data_gen(all_objs[train_samples:], image_dir, batch_size)
    
    steps_per_epoch = int(np.ceil(train_samples / batch_size))
    validation_steps = int(np.ceil((total_samples - train_samples) / batch_size))
    
    print(f'Train/Validation split: {train_samples}/{total_samples - train_samples}\n')
    
    return train_gen, valid_gen, steps_per_epoch, validation_steps

def update_training_history(history, history_path):
    """
    Update the training history file by appending new history data.
    
    Args:
        history: Training history object from model.fit.
        history_path: Path to the history CSV file.
    """
    if os.path.exists(history_path):
        # Load existing history
        previous_history = pd.read_csv(history_path)
        # Append new training history
        updated_history = pd.concat([previous_history, pd.DataFrame(history.history)], ignore_index=True)
    else:
        # Initialize new history DataFrame
        updated_history = pd.DataFrame(history.history)
    
    # Save updated history
    updated_history.to_csv(history_path, index=False)
    print(f"Training history updated at: {history_path}\n")

def generate_performance_plots(history_path, output_dir=None, select_model=BACKBONE):
    """
    Plot training metrics from the history file and save the plots.
    
    Args:
        history_path: Path to the history CSV file.
        output_path: Path to save the result plot.
        select_model: Model name for the plot title.
    """
    # Set default output path to the current working directory
    if output_dir is None:
        output_path = os.path.join(os.getcwd(), f"{select_model}_results_plot.png")
    else:
        output_path = os.path.join(output_dir, f"{select_model}_results_plot.png")
    
    # Load training history
    history_df = pd.read_csv(history_path)

    # Create subplots for training metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Define metrics and labels for each subplot
    plots = [
        {
            'columns': ['loss', 'val_loss'],
            'title': 'Model Loss',
            'ylabel': 'Loss',
            'xlabel': 'Epoch',
            'grid': True,
        },
        {
            'columns': ['dimension_mse', 'val_dimension_mse'],
            'title': 'Dimension MSE',
            'ylabel': 'MSE',
            'xlabel': 'Epoch',
            'grid': True,
        },
    ]

    # Plot each metric in its corresponding subplot
    for ax, plot in zip(axes, plots):
        for col in plot['columns']:
            ax.plot(history_df[col], label=col.replace('_', ' ').title())
        ax.set_title(plot['title'])
        ax.set_ylabel(plot['ylabel'])
        ax.set_xlabel(plot['xlabel'])
        ax.grid(plot['grid'])
        ax.legend(loc='lower left')

    # Set the main title before adjusting layout
    plt.suptitle(f"Training and Validation Metrics for {select_model}", fontsize=16)

    # Adjust layout considering the title space
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust the bottom, left, right, and top margins
    plt.savefig(output_path)
    print(f"Results plot saved to {output_path}\n")
    
def process_model_to_openvino(
    model_path, 
    tflite_output_path, 
    custom_objects=None, 
    quantize=True, 
    openvino_output_dir=None, 
    input_shape="[1,224,224,3]"
):
    """
    Load a pre-trained Keras model, convert it to TensorFlow Lite format with optional quantization, 
    and optionally generate an OpenVINO CLI command for further conversion.

    Ensure you have installed the required libraries:
        pip install openvino-dev[tensorflow,tflite]

    Args:
        model_path (str): Path to the pre-trained Keras model file.
        tflite_output_path (str): Path to save the converted TensorFlow Lite (TFLite) model.
        custom_objects (dict): Dictionary of custom objects required by the Keras model 
        (e.g., custom layers or loss functions).
        quantize (bool): Whether to apply INT8 quantization to reduce model size and improve inference speed 
        (default: True).
        openvino_output_dir (str, optional): Directory to save the converted OpenVINO model. 
        If None, skips OpenVINO conversion.
        input_shape (str): Input shape for the OpenVINO model (default: "[1,224,224,3]").

    Returns:
        str or None: CLI command for converting the model to OpenVINO format if `openvino_output_dir` is specified, otherwise None.
    """
    # Load the pre-trained Keras model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)

    # Convert to TensorFlow Lite format
    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_output_path}")

    # Generate OpenVINO CLI command if output directory is provided
    if openvino_output_dir:
        openvino_command = [
            "mo",  # Model Optimizer command
            "--input_model", tflite_output_path,
            "--framework", "tflite",
            "--output_dir", openvino_output_dir,
            "--input_shape", input_shape,
        ]

        print("Executing OpenVINO CLI command...")
        try:
            # Run the OpenVINO Model Optimizer command
            result = subprocess.run(openvino_command, check=True, capture_output=True, text=True)
            print("OpenVINO model optimization completed successfully.")
            print(result.stdout)  # Optional: Print command output
        except subprocess.CalledProcessError as error:
            print("Error during OpenVINO optimization:")
            print(error.stderr)  # Print the error message from the subprocess
            
# Function to handle copying and deletion
def process_yolo_to_openvino(source='yolov8n_openvino_model', destinations='../'):
    """
    Load a YOLOv8n PyTorch model, export it to OpenVINO format, 
    copy the exported model folder to multiple destinations, 
    and clean up the source folder after the process.

    Args:
        source (str): The source folder containing the exported OpenVINO model.
        destinations (list or str): List of paths or a single path where the source folder will be copied.

    Raises:
        Exception: If errors occur during copying or deleting folders.
    """
    
    # Load a YOLOv8n PyTorch model
    model = YOLO("yolov8n.pt")

    # Export the model
    model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
    
    print('\n')
    # Check if the source folder exists
    if not os.path.exists(source):
        print(f"Source folder does not exist: {source}")
        return
    
    for destination in destinations:
        try:
            # Check if the destination folder exists
            if not os.path.exists(destination):
                print(f"Destination folder does not exist: {destination}")
                continue  # Skip to the next destination

            # Copy the source folder to the destination
            shutil.copytree(source, destination, dirs_exist_ok=True)
            print(f"Folder successfully copied to: {destination}")
        
        except Exception as e:
            # Print any error that occurs during the copy process
            print(f"An error occurred while copying to {destination}: {str(e)}")
    
    try:
        # Delete the source folder after copying
        shutil.rmtree(source)
        print(f"Source folder deleted: {source}")
    except Exception as e:
        # Print any error that occurs during deletion
        print(f"An error occurred while deleting the source folder: {str(e)}")
            
def set_seed(seed=42):
    """
    Set the seed for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed set to: {seed}\n")
                
def parse_arguments(backbone=BACKBONE):
    """
    Parse command-line arguments for model paths, directories, and configurations.

    Args:
        backbone (str): Default backbone model name (default: "mobilenetv2").

    Returns:
        argparse.Namespace: Parsed arguments with model paths and directories.
    """
    parser = argparse.ArgumentParser(description="Script for model conversion and training management.")
    
    # Model paths
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=f"../{backbone}_weights.h5", 
        help="Path to the pre-trained model (default: '../{backbone}_weights.h5')."
    )
    parser.add_argument(
        "--history_path", 
        type=str, 
        default=f"history/{backbone}_training_history.csv", 
        help="Path to the training history file (default: 'history/{backbone}_training_history.csv')."
    )
    parser.add_argument(
        "--tflite_output_path", 
        type=str, 
        default="../quantized_model.tflite", 
        help="Path to save the converted TensorFlow Lite model (default: '../quantized_model.tflite')."
    )
    parser.add_argument(
        "--openvino_output_dir", 
        type=str, 
        default="../quantized_model_openvino", 
        help="Directory to save the OpenVINO-optimized model (default: '../quantized_model_openvino')."
    )
    parser.add_argument(
        "--history_dir", 
        type=str, 
        default="history", 
        help="Directory where training history files are stored (default: 'history')."
    )
    
    # Data paths
    parser.add_argument(
        "--label_dir", 
        type=str, 
        default="./dataset/kitti/label_2", 
        help="Directory for label files (default: './dataset/kitti/label_2')."
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./dataset/kitti/image_2", 
        help="Directory for image files (default: './dataset/kitti/image_2')."
    )

    return parser.parse_args()

####################################################### Main #######################################################
if __name__ == '__main__':
    # Set the seed at the beginning of the script
    set_seed(42)
    
     # Parse arguments
    args = parse_arguments(backbone="mobilenetv2")
    
    all_objs, dims_avg = parse_annotation(args.label_dir)
    process_object_data(all_objs, dims_avg)
    
    # Ensure the directory for history_path exists
    ensure_directory_exists(args.history_path)

    early_stop = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.001, 
        patience=10, 
        mode='min', 
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        args.model_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True
    )
    
    tensorboard = TensorBoard(
        log_dir='logs/', 
        histogram_freq=0, 
        write_graph=True, 
        write_images=False
    )
    
    # Load or initialize model
    model = load_or_initialize_model(args.model_path)
    compile_model(model)
    
    # Get last epoch number
    last_epoch = get_last_epoch(args.history_path)
    
    # Prepare data generators
    train_gen, valid_gen, steps_per_epoch, validation_steps = prepare_generators(all_objs=all_objs, image_dir=args.image_dir, batch_size=BATCH_SIZE)
    
    # Train the model
    history = model.fit(
        train_gen, 
        initial_epoch=last_epoch,
        steps_per_epoch=steps_per_epoch, 
        epochs=EPOCH, 
        verbose=1, 
        validation_data=valid_gen, 
        validation_steps=validation_steps, 
        callbacks=[early_stop, checkpoint, tensorboard], 
        shuffle=True,
        )
    
    # Update training history with the latest results.
    update_training_history(history=history, history_path=args.history_path)
    
    # Generate performance plots for the trained model.
    generate_performance_plots(history_path=args.history_path, output_dir='history')

    # Load a pre-trained Keras model, convert it to TensorFlow Lite format with optional quantization, 
    # and optionally generate an OpenVINO CLI command for further conversion.
    process_model_to_openvino(
        model_path=args.model_path,
        tflite_output_path=args.tflite_output_path,
        custom_objects={"orientation_loss": orientation_loss},
        quantize=True,
        openvino_output_dir=args.openvino_output_dir
    )

    # Load a YOLOv8n PyTorch model, export it to OpenVINO format
    process_yolo_to_openvino(source='yolov8n_openvino_model', destinations='../')
    
