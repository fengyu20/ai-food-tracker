import json
from pathlib import Path
from typing import Dict, Any

from models.core.taxonomy import Taxonomy

def validate_against_ground_truth(
    predicted_result: Dict[str, Any],
    ground_truth_file: str,
    taxonomy: Taxonomy
) -> Dict[str, Any]:

    if not ground_truth_file or not Path(ground_truth_file).exists():
        return {'error': 'Ground truth file not found or not provided'}

    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    gt_set = set(item['classTitle'] for item in ground_truth_data.get('objects', []) if 'classTitle' in item)
    
    if not gt_set:
        return {'error': 'No ground truth classes found in annotation file.'}

    detected_items = predicted_result.get('detected_items', [])
    detected_set = set(item['subcategory'] for item in detected_items)
    
    tp_set = gt_set.intersection(detected_set)
    fp_set = detected_set.difference(gt_set)
    fn_set = gt_set.difference(detected_set)
    
    # Calculate precision, recall, and F1 score
    precision = len(tp_set) / len(detected_set) if detected_set else 0
    recall = len(tp_set) / len(gt_set) if gt_set else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    validation_metrics = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3)
    }

    parent_matches = 0
    for item in detected_items:
        if item['subcategory'] in tp_set:
            pred_parent = item.get('parent_category')
            expected_parent = taxonomy.get_parent_for_subcategory(item['subcategory'])
            if pred_parent == expected_parent:
                parent_matches += 1
    parent_accuracy = parent_matches / len(tp_set) if tp_set else 0

    return {
        'validation_results': {
            "true_positives": sorted(list(tp_set)),
            "false_positives": sorted(list(fp_set)),
            "false_negatives": sorted(list(fn_set)),
            "unlisted_detections": [item['ai_subcategory'] for item in predicted_result.get('unlisted_foods', [])],
            'accuracy_metrics': validation_metrics
        },
        'ground_truth_classes': sorted(list(gt_set)),
        'total_ground_truth': len(gt_set),
        'parent_accuracy': round(parent_accuracy, 3),
        'f1_score': validation_metrics['f1_score'] # for backward compatibility in reports
    }