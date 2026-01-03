"""
Pediatric Tuberculosis Detection using Transfer Learning
========================================================
This script implements a 4-scenario comparison study for detecting TB in 
pediatric chest X-rays using MobileNetV2 transfer learning.

Scenarios:
- Scenario 0: Adult baseline (train on adults, test on children)
- Scenario A: Transfer learning only
- Scenario B: Transfer learning + data augmentation
- Scenario C: Two-stage transfer (adults → children)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from datetime import datetime
from pathlib import Path

# Configure matplotlib styling for consistent, professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION ⚙️
# ============================================================================

class Config:
    """Central configuration class for all paths and hyperparameters"""
    
    # --- VinDr Dataset Paths ---
    # VinDr-PCXR is a large-scale pediatric chest X-ray dataset
    VINDR_TRAIN_LABELS = "Data/VINDR_PCXR/image_labels_train.csv"
    VINDR_TRAIN_DIR = "Data/VINDR_PCXR/train"
    VINDR_TEST_LABELS = "Data/VINDR_PCXR/image_labels_test.csv"
    VINDR_TEST_DIR = "Data/VINDR_PCXR/test"
    
    # --- External TB Source Paths ---
    # Shenzhen: Adult TB dataset for transfer learning
    SHENZHEN_META = r"Data\Shenzhen\shenzhen_metadata.csv"
    SHENZHEN_IMG_DIR = r"Data\Shenzhen\images\images"
    
    # Pediatric TB datasets from multiple sources
    SHENZHEN_PEDIATRIC_META = "shenzhen_pediatric_tb_records.csv"
    MONTGOMERY_PEDIATRIC_META = "montgomery_pediatric_tb_records.csv"
    MONTGOMERY_IMG_DIR = "Data/Montgomery/images/images"
    ATLAS_DIR = "Data/Atlas_CXR_TB_Children"
    
    # --- Image & Training Parameters ---
    IMG_SIZE = 224  # Input size for MobileNetV2 (224x224)
    BATCH_SIZE = 8  # Small batch size due to limited data
    EPOCHS_ADULT = 5  # Fewer epochs for adult pre-training
    EPOCHS_PEDIATRIC = 15  # More epochs for pediatric fine-tuning
    LEARNING_RATE = 1e-4  # Conservative learning rate
    
    # --- Data Balance Configuration ---
    # Aim for balanced dataset: equal TB and Normal cases
    TOTAL_TB_COUNT = 81  # Target number of TB cases
    N_TB_CASES = TOTAL_TB_COUNT
    N_NORMAL_CASES = TOTAL_TB_COUNT  # Match TB count for balance
    
    # --- Reproducibility ---
    SEED = 42  # Fixed random seed for reproducible results
    
    # --- Output Directory ---
    RESULTS_DIR = "results/Final Results"

# Instantiate configuration
config = Config()

# Create output directories if they don't exist
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(config.RESULTS_DIR, 'plots'), exist_ok=True)

# Set random seeds for reproducibility across numpy and TensorFlow
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# Optimize CPU usage for better performance
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ============================================================================
# UTILITIES
# ============================================================================

def log_print(message):
    """
    Print message with timestamp for better tracking of execution progress.
    
    Args:
        message (str): Message to print
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_image_auto(path):
    """
    Load and preprocess medical images (DICOM or standard formats).
    
    This function handles multiple image formats commonly used in medical imaging:
    - DICOM (.dcm, .dicom): Medical standard format
    - Standard images (.png, .jpg, .jpeg, .tif, .bmp)
    
    Preprocessing steps:
    1. Load image in grayscale
    2. Normalize pixel values to [0, 1]
    3. Resize to model input size (224x224)
    4. Convert to 3-channel (RGB) for MobileNetV2
    
    Args:
        path (str): Path to image file
        
    Returns:
        numpy.ndarray: Preprocessed image of shape (224, 224, 3), or None if failed
    """
    path = str(path)
    img = None
    
    # Handle DICOM format (medical imaging standard)
    if path.lower().endswith(('.dcm', '.dicom')):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            # Normalize DICOM pixel values to [0, 1]
            img -= np.min(img)
            img /= (np.max(img) + 1e-8)  # Add epsilon to avoid division by zero
        except Exception as e:
            log_print(f"  Error loading DICOM {path}: {e}")
            return None
    
    # Handle standard image formats
    elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log_print(f"  Error loading image {path}")
            return None
        # Normalize to [0, 1]
        img = img.astype("float32") / 255.0
    
    if img is None:
        return None
    
    # Resize to model input size
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    
    # Convert grayscale to 3-channel RGB (required by MobileNetV2)
    img = np.stack([img, img, img], axis=-1)
    
    return img

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, scenario_name, save_path):
    """
    Create and save a confusion matrix heatmap.
    
    A confusion matrix shows the counts of true positives, true negatives,
    false positives, and false negatives, helping visualize model performance.
    
    Args:
        cm (array): 2x2 confusion matrix
        scenario_name (str): Name of the scenario for title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'TB'], 
                yticklabels=['Normal', 'TB'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {scenario_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved confusion matrix: {save_path}")

def plot_roc_curve(y_true, y_pred_proba, scenario_name, save_path):
    """
    Create and save ROC (Receiver Operating Characteristic) curve.
    
    ROC curve plots True Positive Rate vs False Positive Rate at various
    classification thresholds. AUC (Area Under Curve) summarizes performance:
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random classifier
    - AUC < 0.5: Worse than random
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        scenario_name (str): Name of the scenario
        save_path (str): Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {scenario_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved ROC curve: {save_path}")

def plot_precision_recall_curve(y_true, y_pred_proba, scenario_name, save_path):
    """
    Create and save Precision-Recall curve.
    
    Precision-Recall curves are useful for imbalanced datasets:
    - Precision: Of predicted positives, how many are correct?
    - Recall: Of actual positives, how many did we catch?
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        scenario_name (str): Name of the scenario
        save_path (str): Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {scenario_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved PR curve: {save_path}")

def plot_training_history(history, scenario_name, save_path):
    """
    Create comprehensive training history visualization.
    
    Shows 4 subplots:
    1. Loss: Training vs validation loss over epochs
    2. Accuracy: Training vs validation accuracy over epochs
    3. AUC: Training vs validation AUC over epochs
    4. Precision & Recall: All metrics over epochs
    
    Args:
        history: Keras History object from model.fit()
        scenario_name (str): Name of the scenario
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: AUC curves (if available)
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        axes[1, 0].set_title('Model AUC', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Precision & Recall (if available)
    if 'precision' in history.history and 'recall' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'Training History - {scenario_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved training history: {save_path}")

def plot_all_scenarios_comparison(results_dict, save_path):
    """
    Create bar chart comparison of all scenarios across all metrics.
    
    Displays 5 metrics (Accuracy, Precision, Recall, F1, AUC) as separate
    bar charts for easy comparison between scenarios.
    
    Args:
        results_dict (dict): Dictionary mapping scenario names to results
        save_path (str): Path to save the plot
    """
    scenarios = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Create a bar chart for each metric
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[s][metric] for s in scenarios]
        colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
        bars = axes[idx].bar(range(len(scenarios)), values, color=colors, alpha=0.8)
        
        axes[idx].set_xticks(range(len(scenarios)))
        axes[idx].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[idx].set_ylabel(label, fontsize=11)
        axes[idx].set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 1.0])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height, 
                          f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplots if fewer than 6 scenarios
    if len(scenarios) <= 5:
        if len(scenarios) <= 3:
            fig.delaxes(axes[3])
            fig.delaxes(axes[4])
        if len(scenarios) <= 4:
            fig.delaxes(axes[4])
        fig.delaxes(axes[5])
    
    plt.suptitle('Scenario Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved comparison plot: {save_path}")

def plot_all_roc_curves(results_dict, save_path):
    """
    Plot all ROC curves on the same graph for direct comparison.
    
    Overlaying ROC curves makes it easy to see which scenario
    performs best overall.
    
    Args:
        results_dict (dict): Dictionary mapping scenario names to results
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # Plot each scenario's ROC curve
    for idx, (scenario_name, results) in enumerate(results_dict.items()):
        y_true = results['y_true']
        y_pred_proba = results['predictions']
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2.5, 
                label=f'{scenario_name} (AUC = {roc_auc:.3f})')
    
    # Add random classifier baseline
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', 
             label='Random Classifier', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves - All Scenarios', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"  Saved combined ROC curves: {save_path}")

# ============================================================================
# DATA LOADING
# ============================================================================

def create_balanced_pediatric_dataset():
    """
    Create a balanced pediatric TB dataset from multiple sources.
    
    Strategy:
    1. Load VinDr pediatric X-rays (normal cases)
    2. Load TB cases from multiple sources:
       - VinDr TB cases
       - Shenzhen pediatric TB
       - Montgomery pediatric TB
       - Atlas TB children
    3. Balance TB and Normal cases (~81 each)
    4. Split into train (60%), validation (20%), test (20%)
    
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames with image paths and labels
    """
    log_print("="*70)
    log_print("CREATING BALANCED PEDIATRIC DATASET")
    log_print("="*70)
    
    # --- Load VinDr Dataset ---
    df_train = pd.read_csv(config.VINDR_TRAIN_LABELS)
    df_train['source_dir'] = config.VINDR_TRAIN_DIR
    
    df_test = pd.read_csv(config.VINDR_TEST_LABELS)
    df_test['source_dir'] = config.VINDR_TEST_DIR
    
    df_vindr = pd.concat([df_train, df_test], ignore_index=True)
    
    # --- Extract Normal Cases from VinDr ---
    # Select only cases with "No finding" and no TB
    vindr_normal_candidates = df_vindr[
        (df_vindr['No finding'] == 1.0) & 
        (df_vindr['Tuberculosis'] == 0.0)
    ].copy()
    
    # Sample normal cases to match TB count
    normal_sample_size = min(config.N_NORMAL_CASES, len(vindr_normal_candidates))
    normal_balanced = vindr_normal_candidates.sample(
        n=normal_sample_size, 
        random_state=config.SEED
    ).reset_index(drop=True)
    
    normal_balanced['data_source'] = 'vindr_normal'
    normal_balanced['full_path'] = normal_balanced.apply(
        lambda row: os.path.join(row['source_dir'], row['image_id']), 
        axis=1
    )
    log_print(f"  VinDr Normal: {len(normal_balanced)}")
    
    # --- Collect TB Cases from Multiple Sources ---
    tb_data = []
    
    # 1. VinDr TB cases
    vindr_tb = df_vindr[df_vindr['Tuberculosis'] == 1.0].copy()
    vindr_tb = vindr_tb.sample(n=min(15, len(vindr_tb)), random_state=config.SEED)
    vindr_tb['data_source'] = 'vindr_tb'
    vindr_tb['full_path'] = vindr_tb.apply(
        lambda row: os.path.join(row['source_dir'], row['image_id']), 
        axis=1
    )
    tb_data.append(vindr_tb)
    log_print(f"  VinDr TB: {len(vindr_tb)}")
    
    # 2. Shenzhen Pediatric TB
    try:
        df_shenzhen_tb = pd.read_csv(config.SHENZHEN_PEDIATRIC_META)
        df_shenzhen_tb['full_path'] = df_shenzhen_tb['study_id'].apply(
            lambda x: os.path.join(config.SHENZHEN_IMG_DIR, x)
        )
        df_shenzhen_tb['data_source'] = 'shenzhen_pediatric_tb'
        tb_data.append(df_shenzhen_tb)
        log_print(f"  Shenzhen Pediatric TB: {len(df_shenzhen_tb)}")
    except FileNotFoundError:
        log_print(f"  WARNING: Shenzhen Pediatric metadata not found")
    
    # 3. Montgomery Pediatric TB
    try:
        df_montgomery_tb = pd.read_csv(config.MONTGOMERY_PEDIATRIC_META)
        df_montgomery_tb['full_path'] = df_montgomery_tb['study_id'].apply(
            lambda x: os.path.join(config.MONTGOMERY_IMG_DIR, x)
        )
        df_montgomery_tb['data_source'] = 'montgomery_pediatric_tb'
        tb_data.append(df_montgomery_tb)
        log_print(f"  Montgomery Pediatric TB: {len(df_montgomery_tb)}")
    except FileNotFoundError:
        log_print(f"  WARNING: Montgomery Pediatric metadata not found")
    
    # 4. Atlas TB Children
    atlas_path = Path(config.ATLAS_DIR)
    valid_extensions = {'.jpg', '.png', '.jpeg'}
    atlas_files = [f for f in atlas_path.rglob('*') if f.suffix.lower() in valid_extensions]
    
    df_atlas_tb = pd.DataFrame({
        'image_id': [f.name for f in atlas_files],
        'full_path': [str(f) for f in atlas_files],
        'data_source': 'atlas_tb'
    })
    tb_data.append(df_atlas_tb)
    log_print(f"  Atlas TB: {len(df_atlas_tb)}")
    
    # --- Combine and Balance TB Cases ---
    df_tb = pd.concat(tb_data, ignore_index=True)
    df_tb['label'] = 1  # TB = 1
    
    # Downsample if we have too many TB cases
    if len(df_tb) > config.N_TB_CASES:
        df_tb = df_tb.sample(n=config.N_TB_CASES, random_state=config.SEED)
        log_print(f"  Sampled down to {config.N_TB_CASES} TB cases")
    elif len(df_tb) < config.N_TB_CASES:
        log_print(f"  WARNING: Only {len(df_tb)} TB cases (target: {config.N_TB_CASES})")
    
    # --- Combine TB and Normal Cases ---
    normal_balanced['label'] = 0  # Normal = 0
    df_final = pd.concat([df_tb, normal_balanced], ignore_index=True)
    
    # Shuffle the combined dataset
    df_final = df_final.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    
    log_print(f"\nFinal Dataset: {len(df_final)} (TB:{(df_final['label']==1).sum()}, Normal:{(df_final['label']==0).sum()})")
    
    # --- Split into Train/Val/Test ---
    # 60% train, 20% validation, 20% test
    train_df, temp_df = train_test_split(
        df_final, 
        test_size=0.4,  # 40% for val+test
        stratify=df_final['label'],  # Maintain class balance
        random_state=config.SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5,  # Split 40% into 20%+20%
        stratify=temp_df['label'],
        random_state=config.SEED
    )
    
    log_print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def load_pediatric_images(df, split_name):
    """
    Load actual image data from DataFrame of paths.
    
    Handles both DICOM and standard image formats, with automatic
    path resolution for VinDr dataset extensions.
    
    Args:
        df (DataFrame): DataFrame with 'full_path' and 'label' columns
        split_name (str): Name of split (for logging)
        
    Returns:
        tuple: (images_array, labels_array)
    """
    log_print(f"\nLoading {split_name} images...")
    images = []
    labels = []
    failed_count = 0
    
    for _, row in df.iterrows():
        img_path_str = row['full_path']
        
        # Handle VinDr DICOM files (may have .dicom or .dcm extension)
        if row['data_source'].startswith('vindr') and not Path(img_path_str).exists():
            # Try different extensions
            found_path = next((
                Path(row['source_dir']) / f"{row['image_id']}{ext}"
                for ext in ['.dicom', '.dcm', '']
                if (Path(row['source_dir']) / f"{row['image_id']}{ext}").exists()
            ), None)
            img_path_str = str(found_path) if found_path else None
        
        # Load image if path exists
        if img_path_str and Path(img_path_str).exists():
            img = load_image_auto(img_path_str)
            if img is not None:
                images.append(img)
                labels.append(row['label'])
            else:
                failed_count += 1
        else:
            failed_count += 1
    
    log_print(f"  Loaded: {len(images)}/{len(df)} ({failed_count} failed)")
    
    return np.array(images), np.array(labels)

def load_adult_dataset():
    """
    Load Shenzhen adult TB dataset for transfer learning.
    
    Filters for adult patients (age > 15) and extracts TB labels
    from the 'findings' column.
    
    Returns:
        tuple: (images_array, labels_array)
    """
    log_print("\nLoading adult dataset...")
    
    df = pd.read_csv(config.SHENZHEN_META)
    
    # Filter for adults only (age > 15)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[df['age'] > 15].fillna(0)
    
    # Extract TB label from findings column
    def label_tb(finding):
        """Check if 'tb' or 'tuberculosis' appears in findings"""
        return 1 if 'tb' in str(finding).lower() or 'tuberculosis' in str(finding).lower() else 0
    
    df['label'] = df['findings'].apply(label_tb)
    
    # Load images
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(config.SHENZHEN_IMG_DIR, row["study_id"])
        if os.path.exists(img_path):
            img = load_image_auto(img_path)
            if img is not None:
                images.append(img)
                labels.append(row["label"])
    
    log_print(f"  Loaded: {len(images)} (TB: {sum(labels)})")
    
    return np.array(images), np.array(labels)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def create_augmentation_layer():
    """
    Create moderate data augmentation layer for training.
    
    Augmentations applied:
    - Random rotation: ±5% (18 degrees)
    - Random horizontal flip
    - Random zoom: ±10%
    - Random translation: ±5% horizontal and vertical
    - Random contrast: ±15%
    - Random brightness: ±15%
    
    These augmentations help the model generalize better by training
    on slightly modified versions of the training images.
    
    Returns:
        tf.keras.Sequential: Sequential model with augmentation layers
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05, seed=config.SEED),  # ±5% rotation
        tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),  # Horizontal flip
        tf.keras.layers.RandomZoom(0.1, seed=config.SEED),  # ±10% zoom
        tf.keras.layers.RandomTranslation(0.05, 0.05, seed=config.SEED),  # ±5% translation
        tf.keras.layers.RandomContrast(0.15, seed=config.SEED),  # ±15% contrast
        tf.keras.layers.RandomBrightness(0.15, seed=config.SEED)  # ±15% brightness
    ], name='augmentation')

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(use_augmentation=False, base_trainable_layers=20):
    """
    Build MobileNetV2-based transfer learning model.
    
    Architecture:
    1. Optional augmentation layer (if use_augmentation=True)
    2. MobileNetV2 base (pretrained on ImageNet)
       - Most layers frozen
       - Last N layers trainable for fine-tuning
    3. Global Average Pooling (reduces spatial dimensions)
    4. Dropout (0.5) for regularization
    5. Dense layer (128 units, ReLU)
    6. Dropout (0.3)
    7. Output layer (1 unit, sigmoid for binary classification)
    
    Args:
        use_augmentation (bool): Whether to include augmentation
        base_trainable_layers (int): Number of top MobileNetV2 layers to train
        
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Input layer for 224x224 RGB images
    input_layer = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    
    # Apply augmentation if requested
    if use_augmentation:
        x = create_augmentation_layer()(input_layer)
    else:
        x = input_layer
    
    # Load pretrained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        weights='imagenet',  # Use ImageNet pretrained weights
        include_top=False,   # Exclude the original classification head
        input_tensor=x
    )
    
    # Freeze all base model layers initially
    base_model.trainable = False
    
    # Unfreeze top layers for fine-tuning (if specified)
    if base_trainable_layers > 0:
        for layer in base_model.layers[-base_trainable_layers:]:
            layer.trainable = True
    
    # Build classification head
    x = GlobalAveragePooling2D()(base_model.output)  # Reduce to 1D feature vector
    x = Dropout(0.5, seed=config.SEED)(x)  # Strong dropout for regularization
    x = Dense(128, activation='relu')(x)  # Hidden layer
    x = Dropout(0.3, seed=config.SEED)(x)  # Moderate dropout
    output = Dense(1, activation='sigmoid')(x)  # Binary classification output
    
    # Create final model
    model = Model(input_layer, output)
    
    # Compile with Adam optimizer and comprehensive metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='binary_crossentropy',  # Standard for binary classification
        metrics=[
            'accuracy',  # Overall correctness
            tf.keras.metrics.AUC(name='auc'),  # Area under ROC curve
            tf.keras.metrics.Precision(name='precision'),  # Positive predictive value
            tf.keras.metrics.Recall(name='recall')  # Sensitivity/True positive rate
        ]
    )
    
    return model

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_images, test_labels, scenario_name):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Computes:
    - Accuracy: Overall correctness
    - Precision: Of predicted TB cases, how many are correct?
    - Recall: Of actual TB cases, how many did we detect?
    - F1-Score: Harmonic mean of precision and recall
    - AUC: Area under ROC curve
    - Confusion Matrix: Breakdown of predictions
    
    Also generates:
    - Confusion matrix plot
    - ROC curve
    - Precision-Recall curve
    
    Args:
        model: Trained Keras model
        test_images (array): Test images
        test_labels (array): True labels
        scenario_name (str): Name of scenario for logging
        
    Returns:
        dict: Dictionary containing all metrics and predictions
    """
    log_print(f"\nEvaluating {scenario_name}...")
    
    # Get model predictions
    pred_probs = model.predict(test_images, verbose=0).flatten()  # Probabilities
    pred_labels = (pred_probs >= 0.5).astype(int)  # Binary predictions (threshold=0.5)
    
    # Compute confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    
    # Extract TP, TN, FP, FN from confusion matrix
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Calculate metrics manually for robustness
    accuracy = np.mean(pred_labels == test_labels)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-Score: Harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC: Area under ROC curve
    try:
        auc_score = roc_auc_score(test_labels, pred_probs) if len(np.unique(test_labels)) > 1 else 0.0
    except Exception:
        auc_score = 0.0
    
    # Print results
    log_print(f"\n{'='*70}")
    log_print(f"{scenario_name} RESULTS")
    log_print(f"{'='*70}")
    log_print(f"Accuracy: {accuracy:.4f}")
    log_print(f"Precision: {precision:.4f}")
    log_print(f"Recall: {recall:.4f}")
    log_print(f"F1-Score: {f1:.4f}")
    log_print(f"AUC: {auc_score:.4f}")
    log_print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Print detailed classification report
    print("\n" + classification_report(test_labels, pred_labels, 
                                       target_names=['Normal', 'TB'], 
                                       zero_division=0))
    
    # Generate and save visualizations
    plots_dir = os.path.join(config.RESULTS_DIR, 'plots')
    scenario_clean = scenario_name.replace(' ', '_').lower()
    
    # Confusion matrix plot
    plot_confusion_matrix(cm, scenario_name, 
                         os.path.join(plots_dir, f'{scenario_clean}_confusion_matrix.png'))
    
    # ROC and PR curves (only if we have both classes in test set)
    if len(np.unique(test_labels)) > 1:
        plot_roc_curve(test_labels, pred_probs, scenario_name, 
                      os.path.join(plots_dir, f'{scenario_clean}_roc_curve.png'))
        plot_precision_recall_curve(test_labels, pred_probs, scenario_name, 
                                   os.path.join(plots_dir, f'{scenario_clean}_pr_curve.png'))
    
    # Return all results for comparison
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'confusion_matrix': cm,
        'predictions': pred_probs,
        'y_true': test_labels
    }

# ============================================================================
# SCENARIOS
# ============================================================================

def scenario_0_baseline(adult_images, adult_labels, test_images, test_labels):
    """
    Scenario 0: Adult Baseline (Negative Control)
    
    Purpose: Test if a model trained ONLY on adults can detect TB in children.
    This serves as a baseline to demonstrate the domain shift problem.
    
    Steps:
    1. Train on adult chest X-rays only
    2. Test on pediatric chest X-rays
    3. Expected: Poor performance due to anatomical differences
    
    Args:
        adult_images: Adult training images
        adult_labels: Adult training labels
        test_images: Pediatric test images
        test_labels: Pediatric test labels
        
    Returns:
        tuple: (model, results_dict, training_history)
    """
    log_print("\n" + "="*70)
    log_print("SCENARIO 0: ADULT BASELINE (Train on Adults, Test on Pediatric)")
    log_print("="*70)
    
    # Build model without augmentation
    model = build_model(use_augmentation=False, base_trainable_layers=20)
    
    # Split adult data into train/validation
    adult_train_x, adult_val_x, adult_train_y, adult_val_y = train_test_split(
        adult_images, adult_labels, 
        test_size=0.2, 
        random_state=config.SEED
    )
    
    # Train on adults only
    history = model.fit(
        adult_train_x, adult_train_y,
        validation_data=(adult_val_x, adult_val_y),
        epochs=config.EPOCHS_ADULT,  # Fewer epochs for adults
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # Test on PEDIATRIC data (cross-domain evaluation)
    results = evaluate_model(model, test_images, test_labels, 
                           "Scenario 0 (Adult Baseline)")
    
    # Save model and plots
    model.save(os.path.join(config.RESULTS_DIR, 'scenario_0_baseline.keras'))
    plot_training_history(history, "Scenario 0", 
                         os.path.join(config.RESULTS_DIR, 'plots', 'scenario_0_training.png'))
    
    return model, results, history

def scenario_a(train_images, train_labels, val_images, val_labels, 
               test_images, test_labels):
    """
    Scenario A: Transfer Learning Only
    
    Purpose: Use pretrained ImageNet features with minimal pediatric data,
    no augmentation. Tests if ImageNet features transfer to pediatric TB detection.
    
    Steps:
    1. Load MobileNetV2 pretrained on ImageNet
    2. Fine-tune top 20 layers on pediatric data
    3. No data augmentation
    
    Args:
        train_images: Pediatric training images
        train_labels: Pediatric training labels
        val_images: Validation images
        val_labels: Validation labels
        test_images: Test images
        test_labels: Test labels
        
    Returns:
        tuple: (model, results_dict, training_history)
    """
    log_print("\n" + "="*70)
    log_print("SCENARIO A: TRANSFER LEARNING ONLY")
    log_print("="*70)
    
    # Build model without augmentation
    model = build_model(use_augmentation=False, base_trainable_layers=20)
    
    # Train on pediatric data
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config.EPOCHS_PEDIATRIC,  # More epochs for limited data
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate
    results = evaluate_model(model, test_images, test_labels, "Scenario A")
    
    # Save model and plots
    model.save(os.path.join(config.RESULTS_DIR, 'scenario_a.keras'))
    plot_training_history(history, "Scenario A", 
                         os.path.join(config.RESULTS_DIR, 'plots', 'scenario_a_training.png'))
    
    return model, results, history

def scenario_b(train_images, train_labels, val_images, val_labels, 
               test_images, test_labels):
    """
    Scenario B: Transfer Learning + Data Augmentation
    
    Purpose: Enhance Scenario A with data augmentation to increase
    effective training data size and improve generalization.
    
    Steps:
    1. Load MobileNetV2 pretrained on ImageNet
    2. Apply moderate augmentation during training
    3. Fine-tune on augmented pediatric data
    
    Expected: Better generalization than Scenario A due to augmentation
    
    Args:
        train_images: Pediatric training images
        train_labels: Pediatric training labels
        val_images: Validation images
        val_labels: Validation labels
        test_images: Test images
        test_labels: Test labels
        
    Returns:
        tuple: (model, results_dict, training_history)
    """
    log_print("\n" + "="*70)
    log_print("SCENARIO B: TRANSFER LEARNING + AUGMENTATION")
    log_print("="*70)
    
    # Build model WITH augmentation
    model = build_model(use_augmentation=True, base_trainable_layers=20)
    
    # Train on augmented pediatric data
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config.EPOCHS_PEDIATRIC,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate
    results = evaluate_model(model, test_images, test_labels, "Scenario B")
    
    # Save model and plots
    model.save(os.path.join(config.RESULTS_DIR, 'scenario_b.keras'))
    plot_training_history(history, "Scenario B", 
                         os.path.join(config.RESULTS_DIR, 'plots', 'scenario_b_training.png'))
    
    return model, results, history

def scenario_c(adult_images, adult_labels, 
               train_images, train_labels, 
               val_images, val_labels, 
               test_images, test_labels):
    """
    Scenario C: Two-Stage Transfer Learning (Domain Adaptation)
    
    Purpose: Progressively adapt from ImageNet → Adults → Children
    to bridge the domain gap through staged learning.
    
    Steps:
    1. Stage 1: Pre-train on adult TB data (domain-specific features)
    2. Stage 2: Fine-tune on pediatric data (age-specific adaptation)
    
    Expected: Best performance by leveraging both adult TB knowledge
    and pediatric-specific adaptation.
    
    Args:
        adult_images: Adult training images
        adult_labels: Adult labels
        train_images: Pediatric training images
        train_labels: Pediatric labels
        val_images: Validation images
        val_labels: Validation labels
        test_images: Test images
        test_labels: Test labels
        
    Returns:
        tuple: (model, results_dict, training_history)
    """
    log_print("\n" + "="*70)
    log_print("SCENARIO C: TWO-STAGE TRANSFER LEARNING")
    log_print("="*70)
    
    # --- Stage 1: Pre-train on Adult TB Data ---
    log_print("\nStage 1: Pre-training on adults...")
    model = build_model(use_augmentation=False, base_trainable_layers=20)
    
    # Split adult data
    adult_train_x, adult_val_x, adult_train_y, adult_val_y = train_test_split(
        adult_images, adult_labels, 
        test_size=0.2, 
        random_state=config.SEED
    )
    
    # Train on adults (learn TB-specific features)
    model.fit(
        adult_train_x, adult_train_y,
        validation_data=(adult_val_x, adult_val_y),
        epochs=config.EPOCHS_ADULT,  # Fewer epochs for pre-training
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # --- Stage 2: Fine-tune on Pediatric Data ---
    log_print("\nStage 2: Fine-tuning on pediatric...")
    
    # Continue training on pediatric data
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config.EPOCHS_PEDIATRIC,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate final model
    results = evaluate_model(model, test_images, test_labels, "Scenario C")
    
    # Save model and plots
    model.save(os.path.join(config.RESULTS_DIR, 'scenario_c.keras'))
    plot_training_history(history, "Scenario C", 
                         os.path.join(config.RESULTS_DIR, 'plots', 'scenario_c_training.png'))
    
    return model, results, history

# ============================================================================
# COMPARISON
# ============================================================================

def compare_scenarios(results_dict):
    """
    Compare all scenarios side-by-side.
    
    Creates:
    1. CSV file with all metrics
    2. Bar chart comparison across metrics
    3. Combined ROC curve plot
    
    Args:
        results_dict (dict): Dictionary mapping scenario names to results
    """
    log_print("\n" + "="*70)
    log_print("SCENARIO COMPARISON")
    log_print("="*70)
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame({
        'Scenario': list(results_dict.keys()),
        'Accuracy': [r['accuracy'] for r in results_dict.values()],
        'Precision': [r['precision'] for r in results_dict.values()],
        'Recall': [r['recall'] for r in results_dict.values()],
        'F1-Score': [r['f1_score'] for r in results_dict.values()],
        'AUC': [r['auc'] for r in results_dict.values()]
    })
    
    # Print comparison table
    print("\n" + metrics_df.to_string(index=False))
    
    # Save to CSV
    metrics_df.to_csv(os.path.join(config.RESULTS_DIR, 'comparison.csv'), index=False)
    
    # Generate comparison visualizations
    plots_dir = os.path.join(config.RESULTS_DIR, 'plots')
    plot_all_scenarios_comparison(results_dict, 
                                 os.path.join(plots_dir, 'all_scenarios_comparison.png'))
    plot_all_roc_curves(results_dict, 
                       os.path.join(plots_dir, 'all_scenarios_roc_curves.png'))
    
    log_print(f"\nResults saved to: {config.RESULTS_DIR}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Execute the complete 4-scenario comparative study.
    
    Workflow:
    1. Load and balance pediatric dataset from multiple sources
    2. Load adult dataset for transfer learning
    3. Run 4 scenarios:
       - Scenario 0: Adult baseline (negative control)
       - Scenario A: Transfer learning only
       - Scenario B: Transfer learning + augmentation
       - Scenario C: Two-stage transfer (adults → children)
    4. Compare all scenarios with comprehensive visualizations
    
    All results, models, and plots are saved to the results directory.
    """
    log_print("="*70)
    log_print("PEDIATRIC TB DETECTION: 4-SCENARIO STUDY WITH VISUALIZATIONS")
    log_print("="*70)
    log_print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- Step 1: Load Data ---
    # Create balanced pediatric dataset and split into train/val/test
    train_df, val_df, test_df = create_balanced_pediatric_dataset()
    
    # Load actual image data
    train_images, train_labels = load_pediatric_images(train_df, 'train')
    val_images, val_labels = load_pediatric_images(val_df, 'validation')
    test_images, test_labels = load_pediatric_images(test_df, 'test')
    
    # Load adult dataset for transfer learning scenarios
    adult_images, adult_labels = load_adult_dataset()
    
    # --- Step 2: Run All Scenarios ---
    results = {}
    
    # Scenario 0: Adult Baseline (train on adults, test on children)
    _, results['Scenario 0 (Adult Baseline)'], _ = scenario_0_baseline(
        adult_images, adult_labels, 
        test_images, test_labels
    )
    
    # Scenario A: Transfer Learning Only
    _, results['Scenario A'], _ = scenario_a(
        train_images, train_labels, 
        val_images, val_labels, 
        test_images, test_labels
    )
    
    # Scenario B: Transfer Learning + Augmentation
    _, results['Scenario B'], _ = scenario_b(
        train_images, train_labels, 
        val_images, val_labels, 
        test_images, test_labels
    )
    
    # Scenario C: Two-Stage Transfer Learning
    _, results['Scenario C'], _ = scenario_c(
        adult_images, adult_labels, 
        train_images, train_labels, 
        val_images, val_labels, 
        test_images, test_labels
    )
    
    # --- Step 3: Compare All Scenarios ---
    compare_scenarios(results)
    
    # --- Completion ---
    log_print("\n" + "="*70)
    log_print("STUDY COMPLETE")
    log_print("="*70)
    log_print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"\nAll visualizations saved in: {os.path.join(config.RESULTS_DIR, 'plots')}")

# Entry point
if __name__ == "__main__":
    main()