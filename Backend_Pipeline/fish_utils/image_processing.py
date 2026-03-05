import numpy as np

def yolo_to_cv2(yolo_box, img_shape):
    """Converts YOLO (cx, cy, w, h) normalized to CV2 (x1, y1, x2, y2) pixels."""
    h, w = img_shape[:2]
    cx_n, cy_n, w_n, h_n = yolo_box
    x1 = int((cx_n - w_n/2) * w)
    y1 = int((cy_n - h_n/2) * h)
    x2 = int((cx_n + w_n/2) * w)
    y2 = int((cy_n + h_n/2) * h)
    return (x1, y1, x2, y2)

def cv2_to_yolo(cv2_box, img_shape, class_id):
    """Converts CV2 pixels to YOLO normalized format with a specific class ID."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = cv2_box
    bw, bh = x2 - x1, y2 - y1
    # Ensure values stay within 0-1 range
    return (
        int(class_id), 
        max(0, min(1, (x1 + bw/2)/w)), 
        max(0, min(1, (y1 + bh/2)/h)), 
        max(0, min(1, bw/w)), 
        max(0, min(1, bh/h))
    )

def translate_to_global(local_cv2_box, cell_global_cv2_box):
    """
    Translates coordinates from a local cell patch back to the global image.
    
    Args:
        local_cv2_box: (x1, y1, x2, y2) inside the patch.
        cell_global_cv2_box: (x1, y1, x2, y2) of the cell in the global image.
    """
    gx1, gy1, _, _ = cell_global_cv2_box
    lx1, ly1, lx2, ly2 = local_cv2_box
    
    return (lx1 + gx1, ly1 + gy1, lx2 + gx1, ly2 + gy1)