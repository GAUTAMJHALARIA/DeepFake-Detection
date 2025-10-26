"""
Explainable AI (XAI) module for deepfake detection.
Provides Grad-CAM visualization for model interpretability.
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from settings import settings


def generate_gradcam_heatmaps(
    images: np.ndarray, predictions: np.ndarray, model
) -> List[np.ndarray]:
    """
    Generate Grad-CAM heatmaps for a batch of images.

    Args:
        images: Batch of preprocessed images (batch_size, height, width, channels)
        predictions: Model predictions (batch_size, num_classes)
        model: Trained Keras model or SavedModel

    Returns:
        List of heatmap images
    """
    try:
        print(f"Generating Grad-CAM heatmaps for {len(images)} images")

        # Initialize Grad-CAM
        gradcam = GradCAM(model, layer_name=settings.GRADCAM_LAYER)

        heatmaps = []
        for i, image in enumerate(images):
            try:
                # Get the predicted class index
                class_idx = 0  # For binary classification, use class 0

                # Generate heatmap
                heatmap = gradcam.generate_heatmap(
                    image, class_idx=class_idx, alpha=0.4
                )
                heatmaps.append(heatmap)

                if i % 10 == 0:
                    print(f"Generated heatmap {i + 1}/{len(images)}")

            except Exception as e:
                print(f"Error generating heatmap for image {i}: {e}")
                # Create a simple fallback heatmap
                fallback = np.zeros((64, 64, 3), dtype=np.uint8)
                heatmaps.append(fallback)

        print(f"Successfully generated {len(heatmaps)} heatmaps")

        # Convert numpy arrays to lists for JSON serialization
        serializable_heatmaps = []
        for heatmap in heatmaps:
            if isinstance(heatmap, np.ndarray):
                serializable_heatmaps.append(heatmap.tolist())
            else:
                serializable_heatmaps.append(heatmap)

        return serializable_heatmaps

    except Exception as e:
        print(f"Error in generate_gradcam_heatmaps: {e}")
        import traceback

        traceback.print_exc()
        # Return empty heatmaps as fallback
        return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(len(images))]


def generate_explanations(
    predictions: np.ndarray, annotations: List[Dict[str, Any]], threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Generate textual explanations for predictions.

    Args:
        predictions: Model predictions
        annotations: Frame annotations
        threshold: Classification threshold

    Returns:
        Dictionary containing explanations
    """
    try:
        fake_count = np.sum(predictions > threshold)
        total_frames = len(predictions)
        fake_percentage = (fake_count / total_frames) * 100

        explanations: Dict[str, Any] = {
            "overall_explanation": f"The model analyzed {total_frames} frames and detected potential deepfake content in {fake_count} frames ({fake_percentage:.1f}%).",
            "confidence_scores": {
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "mean": float(np.mean(predictions)),
            },
            "frame_analysis": [],  # Will be List[Dict[str, Any]]
        }

        # Add frame-level explanations
        for i, (pred, ann) in enumerate(zip(predictions, annotations)):
            frame_explanation = {
                "frame_number": i,
                "timestamp": ann.get("timestamp", 0.0),
                "prediction": float(pred),
                "label": "fake" if pred > threshold else "real",
                "confidence": "high" if abs(pred - threshold) > 0.3 else "medium",
            }
            explanations["frame_analysis"].append(frame_explanation)

        return explanations

    except Exception as e:
        print(f"Error generating explanations: {e}")
        return {
            "overall_explanation": "Unable to generate explanations due to an error."
        }


class GradCAM:
    """Traditional Grad-CAM implementation for model interpretability."""

    def __init__(self, model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained Keras model or SavedModel
            layer_name: Name of the target layer for Grad-CAM
        """
        self.model = model
        self.layer_name = layer_name or settings.GRADCAM_LAYER
        self.grad_model = self._build_grad_model()

    def _build_grad_model(self):
        """Build gradient model for Grad-CAM."""
        try:
            print(f"Building Grad-CAM model for layer: {self.layer_name}")
            print(f"Model type: {type(self.model)}")
            print(f"Model has layers: {hasattr(self.model, 'layers')}")
            print(f"Model has signatures: {hasattr(self.model, 'signatures')}")

            if hasattr(self.model, "layers"):
                print(f"Number of layers: {len(self.model.layers)}")
                layer_names = [layer.name for layer in self.model.layers]
                print(f"Available layers: {layer_names[:10]}...")
                print("Using traditional Grad-CAM approach with Keras model")
                return "use_original_model"
            elif hasattr(self.model, "signatures"):
                print(f"Model has signatures: {list(self.model.signatures.keys())}")
                print("Using SavedModel approach for Grad-CAM")
                return "use_savedmodel"
            else:
                print("Model does not have layers or signatures attribute!")
                return None
        except Exception as e:
            print(f"Error building Grad-CAM model: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_heatmap(
        self, image: np.ndarray, class_idx: int = 0, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image with comprehensive debugging.

        Args:
            image: Input image (height, width, channels)
            class_idx: Index of the target class
            alpha: Transparency for overlay

        Returns:
            Heatmap image
        """
        print("🔍 GRAD-CAM ENTRY POINT DEBUG:")
        print(f"   📊 Input image shape: {image.shape}")
        print(f"   📊 Input image dtype: {image.dtype}")
        print(f"   📊 Input image min/max: {np.min(image):.6f} / {np.max(image):.6f}")
        print(f"   📊 class_idx: {class_idx}")
        print(f"   📊 alpha: {alpha}")

        if self.grad_model is None:
            print("   ❌ Grad-CAM model not available!")
            raise ValueError("Grad-CAM model not available")

        try:
            print("   🔄 Converting input to tensor...")
            if len(image.shape) == 3:
                image_batch = tf.expand_dims(
                    tf.convert_to_tensor(image, dtype=tf.float32), axis=0
                )
                print(f"   📊 Expanded to batch: {image_batch.shape}")
            else:
                image_batch = tf.convert_to_tensor(image, dtype=tf.float32)
                print(f"   📊 Direct tensor conversion: {image_batch.shape}")

            print(f"   📊 Final image_batch shape: {image_batch.shape}")
            print(f"   📊 Final image_batch dtype: {image_batch.dtype}")
            print(f"   📊 grad_model type: {self.grad_model}")

            if self.grad_model == "use_original_model":
                print("   ✅ Calling traditional Grad-CAM...")
                return self._generate_heatmap_traditional(image_batch, class_idx, alpha)
            elif self.grad_model == "use_savedmodel":
                print("   ✅ Calling SavedModel Grad-CAM...")
                return self._generate_heatmap_savedmodel(image_batch, class_idx, alpha)
            else:
                print(f"   ❌ Unsupported grad_model type: {self.grad_model}")
                raise ValueError("Unsupported grad_model type")

        except Exception as e:
            print(f"   ❌ Error in generate_heatmap: {e}")
            print(f"   📍 Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            raise e

    def _generate_heatmap_traditional(
        self, image_batch: tf.Tensor, class_idx: int, alpha: float
    ) -> np.ndarray:
        """Generate traditional Grad-CAM heatmap with comprehensive debugging."""
        try:
            print("=" * 60)
            print("🎯 STARTING TRADITIONAL GRAD-CAM")
            print("=" * 60)

            # Debug: Input validation
            print("📊 INPUT DEBUG:")
            print(f"   - image_batch shape: {image_batch.shape}")
            print(f"   - image_batch dtype: {image_batch.dtype}")
            print(f"   - class_idx: {class_idx}")
            print(f"   - alpha: {alpha}")

            # Debug: Model structure
            print("🏗️ MODEL STRUCTURE DEBUG:")
            print(f"   - Total layers: {len(self.model.layers)}")
            print(f"   - Layer names: {[layer.name for layer in self.model.layers]}")

            # Get EfficientNetB0 base model (layer 1)
            if len(self.model.layers) < 2:
                raise ValueError(
                    "❌ EfficientNetB0 base model not found! Need at least 2 layers."
                )

            efficientnet_base = self.model.layers[1]
            print(f"✅ Using EfficientNetB0 base: {efficientnet_base.name}")
            print(f"   - EfficientNetB0 type: {type(efficientnet_base)}")
            print(
                f"   - EfficientNetB0 has layers: {hasattr(efficientnet_base, 'layers')}"
            )

            if hasattr(efficientnet_base, "layers"):
                print(
                    f"   - EfficientNetB0 internal layers: {len(efficientnet_base.layers)}"
                )

            print("\n🔄 FORWARD PASS DEBUG:")

            # Traditional Grad-CAM: Compute gradients with respect to final conv features
            with tf.GradientTape(persistent=True) as tape:
                print("   📹 GradientTape created and watching image_batch")
                tape.watch(image_batch)

                print("   🚀 Starting forward pass through EfficientNetB0...")
                # Forward pass through EfficientNetB0 base
                conv_outputs = efficientnet_base(image_batch)
                print("   ✅ EfficientNetB0 forward pass successful!")
                print(f"   📊 conv_outputs shape: {conv_outputs.shape}")
                print(f"   📊 conv_outputs dtype: {conv_outputs.dtype}")
                print(
                    f"   📊 conv_outputs min/max: {tf.reduce_min(conv_outputs):.6f} / {tf.reduce_max(conv_outputs):.6f}"
                )

                print("   🚀 Starting forward pass through full model...")
                # Forward pass through the rest of the model
                predictions = self.model(image_batch)
                print("   ✅ Full model forward pass successful!")
                print(f"   📊 predictions shape: {predictions.shape}")
                print(f"   📊 predictions dtype: {predictions.dtype}")
                print(f"   📊 predictions values: {predictions.numpy()}")

                class_output = predictions[:, class_idx]
                print(f"   📊 class_output shape: {class_output.shape}")
                print(f"   📊 class_output value: {class_output.numpy()}")

            print("\n🔍 GRADIENT COMPUTATION DEBUG:")
            print("   🧮 Computing gradients of class_output w.r.t. conv_outputs...")

            # Get gradients with respect to conv features
            grads = tape.gradient(class_output, conv_outputs)

            print(f"   📊 grads type: {type(grads)}")
            print(f"   📊 grads is None: {grads is None}")

            if grads is None:
                print("   ❌ GRADIENT COMPUTATION FAILED!")
                print("   🔍 Debugging gradient computation...")

                # Try to understand why gradients are None
                print(f"   📊 conv_outputs requires_grad: {conv_outputs.dtype}")
                print(f"   📊 class_output requires_grad: {class_output.dtype}")

                raise ValueError(
                    "❌ Failed to compute gradients - model may not be compatible with Grad-CAM"
                )

            print("   ✅ Gradient computation successful!")
            print(f"   📊 grads shape: {grads.shape}")
            print(f"   📊 grads dtype: {grads.dtype}")
            print(
                f"   📊 grads min/max: {tf.reduce_min(grads):.6f} / {tf.reduce_max(grads):.6f}"
            )

            print("\n🧮 GRAD-CAM COMPUTATION DEBUG:")

            # Traditional Grad-CAM computation
            # Global average pooling of gradients
            print("   📊 Computing global average pooling of gradients...")
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            print("   ✅ Pooled gradients computed!")
            print(f"   📊 pooled_grads shape: {pooled_grads.shape}")
            print(f"   📊 pooled_grads dtype: {pooled_grads.dtype}")
            print(
                f"   📊 pooled_grads min/max: {tf.reduce_min(pooled_grads):.6f} / {tf.reduce_max(pooled_grads):.6f}"
            )

            # Weight the feature maps by gradients
            print("   📊 Weighting feature maps by gradients...")
            conv_outputs = conv_outputs[0]  # Remove batch dimension
            print(f"   📊 conv_outputs (no batch) shape: {conv_outputs.shape}")

            print("   📊 Computing heatmap = conv_outputs @ pooled_grads...")
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            print(f"   📊 heatmap (before squeeze) shape: {heatmap.shape}")

            heatmap = tf.squeeze(heatmap)
            print(f"   📊 heatmap (after squeeze) shape: {heatmap.shape}")

            # Apply ReLU and normalize
            print("   📊 Applying ReLU and normalization...")
            heatmap = tf.maximum(heatmap, 0)
            print(
                f"   📊 heatmap (after ReLU) min/max: {tf.reduce_min(heatmap):.6f} / {tf.reduce_max(heatmap):.6f}"
            )

            heatmap = heatmap / tf.math.reduce_max(heatmap)
            print(
                f"   📊 heatmap (after normalization) min/max: {tf.reduce_min(heatmap):.6f} / {tf.reduce_max(heatmap):.6f}"
            )

            heatmap = heatmap.numpy()
            print(f"   📊 heatmap (numpy) shape: {heatmap.shape}")
            print(f"   📊 heatmap (numpy) dtype: {heatmap.dtype}")

            print("\n🎨 VISUALIZATION DEBUG:")

            # Resize heatmap to match input image
            print(f"   📊 Resizing heatmap from {heatmap.shape} to (64, 64)...")
            heatmap_resized = cv2.resize(heatmap, (64, 64))
            print(f"   📊 heatmap_resized shape: {heatmap_resized.shape}")

            # Create colored heatmap
            print("   📊 Creating colored heatmap with 'jet' colormap...")
            heatmap_colored = cm.get_cmap("jet")(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            print(f"   📊 heatmap_colored shape: {heatmap_colored.shape}")
            print(f"   📊 heatmap_colored dtype: {heatmap_colored.dtype}")

            # Overlay on original image
            print("   📊 Creating overlay with original image...")
            original_image = (image_batch[0].numpy() * 255).astype(np.uint8)
            print(f"   📊 original_image shape: {original_image.shape}")
            print(f"   📊 original_image dtype: {original_image.dtype}")

            overlay = cv2.addWeighted(
                original_image, 1 - alpha, heatmap_colored, alpha, 0
            )
            print(f"   📊 overlay shape: {overlay.shape}")
            print(f"   📊 overlay dtype: {overlay.dtype}")

            print("\n🎉 TRADITIONAL GRAD-CAM SUCCESSFUL!")
            print(f"   ✅ Final overlay shape: {overlay.shape}")
            print("=" * 60)
            return overlay

        except Exception as e:
            print("\n❌ TRADITIONAL GRAD-CAM FAILED!")
            print(f"   🚨 Error: {e}")
            print(f"   📍 Error type: {type(e).__name__}")
            print("=" * 60)
            import traceback

            traceback.print_exc()
            raise e

    def _generate_heatmap_savedmodel(
        self, image_batch: tf.Tensor, class_idx: int, alpha: float
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap using SavedModel with comprehensive debugging."""
        try:
            print("=" * 60)
            print("🎯 STARTING SAVEDMODEL GRAD-CAM")
            print("=" * 60)

            # Debug: Input validation
            print("📊 INPUT DEBUG:")
            print(f"   - image_batch shape: {image_batch.shape}")
            print(f"   - image_batch dtype: {image_batch.dtype}")
            print(f"   - class_idx: {class_idx}")
            print(f"   - alpha: {alpha}")

            # Debug: SavedModel structure
            print("🏗️ SAVEDMODEL STRUCTURE DEBUG:")
            print(f"   - Available signatures: {list(self.model.signatures.keys())}")

            # Get the default signature
            if "serving_default" in self.model.signatures:
                signature = self.model.signatures["serving_default"]
                print("   - Using 'serving_default' signature")
            else:
                signature_name = list(self.model.signatures.keys())[0]
                signature = self.model.signatures[signature_name]
                print(f"   - Using '{signature_name}' signature")

            print(f"   - Signature input spec: {signature.inputs}")
            print(f"   - Signature output spec: {signature.outputs}")

            print("\n🔄 FORWARD PASS DEBUG:")

            # For SavedModel, we need to use a different approach
            # Since we can't easily access intermediate layers, we'll use a simplified approach
            print("   🚀 Starting forward pass through SavedModel...")

            # Forward pass through the model
            predictions = signature(image_batch)
            print("   ✅ SavedModel forward pass successful!")
            print(f"   📊 predictions type: {type(predictions)}")

            # Extract the prediction value
            if isinstance(predictions, dict):
                # Get the first output
                output_key = list(predictions.keys())[0]
                pred_values = predictions[output_key]
                print(f"   📊 Using output key: {output_key}")
            else:
                pred_values = predictions

            print(f"   📊 pred_values shape: {pred_values.shape}")
            print(f"   📊 pred_values dtype: {pred_values.dtype}")
            print(f"   📊 pred_values values: {pred_values.numpy()}")

            class_output = pred_values[:, class_idx]
            print(f"   📊 class_output shape: {class_output.shape}")
            print(f"   📊 class_output value: {class_output.numpy()}")

            print("\n🔍 GRADIENT COMPUTATION DEBUG:")
            print("   🧮 Computing gradients of class_output w.r.t. image_batch...")

            # Compute gradients with respect to input
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(image_batch)
                predictions = signature(image_batch)

                if isinstance(predictions, dict):
                    pred_values = predictions[list(predictions.keys())[0]]
                else:
                    pred_values = predictions

                class_output = pred_values[:, class_idx]

            # Get gradients with respect to input image
            grads = tape.gradient(class_output, image_batch)

            print(f"   📊 grads type: {type(grads)}")
            print(f"   📊 grads is None: {grads is None}")

            if grads is None:
                print("   ❌ GRADIENT COMPUTATION FAILED!")
                raise ValueError("❌ Failed to compute gradients with SavedModel")

            print("   ✅ Gradient computation successful!")
            print(f"   📊 grads shape: {grads.shape}")
            print(f"   📊 grads dtype: {grads.dtype}")
            print(
                f"   📊 grads min/max: {tf.reduce_min(grads):.6f} / {tf.reduce_max(grads):.6f}"
            )

            print("\n🧮 GRAD-CAM COMPUTATION DEBUG:")

            # Simplified Grad-CAM using input gradients
            # Global average pooling of gradients
            print("   📊 Computing global average pooling of gradients...")
            pooled_grads = tf.reduce_mean(tf.abs(grads), axis=(1, 2, 3))
            print("   ✅ Pooled gradients computed!")
            print(f"   📊 pooled_grads shape: {pooled_grads.shape}")
            print(f"   📊 pooled_grads value: {pooled_grads.numpy()}")

            # Create heatmap from gradients
            print("   📊 Creating heatmap from gradients...")
            grads_squeezed = tf.squeeze(grads, axis=0)  # Remove batch dimension
            print(f"   📊 grads_squeezed shape: {grads_squeezed.shape}")

            # Convert to grayscale and normalize
            grads_gray = tf.reduce_mean(tf.abs(grads_squeezed), axis=-1)
            print(f"   📊 grads_gray shape: {grads_gray.shape}")

            # Apply ReLU and normalize
            print("   📊 Applying ReLU and normalization...")
            heatmap = tf.maximum(grads_gray, 0)
            print(
                f"   📊 heatmap (after ReLU) min/max: {tf.reduce_min(heatmap):.6f} / {tf.reduce_max(heatmap):.6f}"
            )

            heatmap = heatmap / tf.math.reduce_max(heatmap)
            print(
                f"   📊 heatmap (after normalization) min/max: {tf.reduce_min(heatmap):.6f} / {tf.reduce_max(heatmap):.6f}"
            )

            heatmap = heatmap.numpy()
            print(f"   📊 heatmap (numpy) shape: {heatmap.shape}")
            print(f"   📊 heatmap (numpy) dtype: {heatmap.dtype}")

            print("\n🎨 VISUALIZATION DEBUG:")

            # Resize heatmap to match input image
            print(f"   📊 Resizing heatmap from {heatmap.shape} to (64, 64)...")
            heatmap_resized = cv2.resize(heatmap, (64, 64))
            print(f"   📊 heatmap_resized shape: {heatmap_resized.shape}")

            # Create colored heatmap
            print("   📊 Creating colored heatmap with 'jet' colormap...")
            heatmap_colored = cm.get_cmap("jet")(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            print(f"   📊 heatmap_colored shape: {heatmap_colored.shape}")
            print(f"   📊 heatmap_colored dtype: {heatmap_colored.dtype}")

            # Overlay on original image
            print("   📊 Creating overlay with original image...")
            original_image = (image_batch[0].numpy() * 255).astype(np.uint8)
            print(f"   📊 original_image shape: {original_image.shape}")
            print(f"   📊 original_image dtype: {original_image.dtype}")

            overlay = cv2.addWeighted(
                original_image, 1 - alpha, heatmap_colored, alpha, 0
            )
            print(f"   📊 overlay shape: {overlay.shape}")
            print(f"   📊 overlay dtype: {overlay.dtype}")

            print("\n🎉 SAVEDMODEL GRAD-CAM SUCCESSFUL!")
            print(f"   ✅ Final overlay shape: {overlay.shape}")
            print("=" * 60)
            return overlay

        except Exception as e:
            print("\n❌ SAVEDMODEL GRAD-CAM FAILED!")
            print(f"   🚨 Error: {e}")
            print(f"   📍 Error type: {type(e).__name__}")
            print("=" * 60)
            import traceback

            traceback.print_exc()
            raise e
