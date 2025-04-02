import numpy as np
from typing import Dict, Any

def confidence_ensemble(cnn_preds: np.ndarray, 
                       vit_preds: np.ndarray, 
                       threshold: float = 0.15) -> np.ndarray:
    """
    Combine predictions from CNN and ViT models using confidence-based ensemble
    
    Args:
        cnn_preds: CNN model predictions (n_samples, n_classes)
        vit_preds: ViT model predictions (n_samples, n_classes)
        threshold: Confidence difference threshold for decision
    
    Returns:
        Ensemble predictions (n_samples, n_classes)
    """
    final_preds = []
    
    for cnn_pred, vit_pred in zip(cnn_preds, vit_preds):
        cnn_label = np.argmax(cnn_pred)
        cnn_conf = np.max(cnn_pred)
        vit_label = np.argmax(vit_pred)
        vit_conf = np.max(vit_pred)
        
        # Case 1: Models agree
        if cnn_label == vit_label:
            ensemble_pred = cnn_pred if cnn_conf > vit_conf else vit_pred
        
        # Case 2: Models disagree but one is much more confident
        elif abs(cnn_conf - vit_conf) > threshold:
            ensemble_pred = cnn_pred if cnn_conf > vit_conf else vit_pred
        
        # Case 3: Models disagree with similar confidence
        else:
            ensemble_pred = (cnn_pred + vit_pred) / 2  # Average predictions
            
        final_preds.append(ensemble_pred)
    
    return np.array(final_preds)

def explain_decision(cnn_pred: np.ndarray, 
                     vit_pred: np.ndarray, 
                     threshold: float = 0.15) -> Dict[str, Any]:
    """
    Generate explanation for ensemble decision
    
    Args:
        cnn_pred: Single CNN prediction (n_classes,)
        vit_pred: Single ViT prediction (n_classes,)
        threshold: Confidence difference threshold
    
    Returns:
        Dictionary with decision explanation
    """
    cnn_label = np.argmax(cnn_pred)
    cnn_conf = np.max(cnn_pred)
    vit_label = np.argmax(vit_pred)
    vit_conf = np.max(vit_pred)
    
    if cnn_label == vit_label:
        decision = "Models agreed - selected higher confidence prediction"
        final_pred = cnn_pred if cnn_conf > vit_conf else vit_pred
    else:
        if abs(cnn_conf - vit_conf) > threshold:
            decision = "Models disagreed - selected higher confidence prediction"
            final_pred = cnn_pred if cnn_conf > vit_conf else vit_pred
        else:
            decision = "Models disagreed with similar confidence - averaged predictions"
            final_pred = (cnn_pred + vit_pred) / 2
    
    return {
        'final_pred': final_pred,
        'final_label': "Real" if np.argmax(final_pred) == 0 else "Fake",
        'final_confidence': float(np.max(final_pred)),
        'decision': decision,
        'cnn_pred': {
            'label': "Real" if cnn_label == 0 else "Fake",
            'confidence': float(cnn_conf)
        },
        'vit_pred': {
            'label': "Real" if vit_label == 0 else "Fake",
            'confidence': float(vit_conf)
        }
    }