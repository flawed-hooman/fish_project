import math
from .image_processing import cv2_to_yolo

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def get_distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def apply_fusion_logic(detections, img_shape):
    """
    Processes local detections to find fusions and remaps IDs.
    IDs: 0: Green, 1: Red, 2: Aqua, 3: Fusion.
    """
    greens = [d for d in detections if d['cid'] == 0]
    reds = [d for d in detections if d['cid'] == 1]
    aquas = [d for d in detections if d['cid'] == 2]

    used_g, used_r = set(), set()
    final_results = []
    
    # 1. Match Green and Red for Fusion (ID 3)
    pairs = []
    for i, g in enumerate(greens):
        for j, r in enumerate(reds):
            dist = get_distance(get_center(g['box']), get_center(r['box']))
            thresh = (get_diag(g['box']) + get_diag(r['box'])) / 2.0
            if dist <= thresh:
                pairs.append((dist, i, j))
    
    pairs.sort() # Prioritize closest pairs
    for _, gi, rj in pairs:
        if gi in used_g or rj in used_r: continue
        used_g.add(gi); used_r.add(rj)
        
        gc, rc = get_center(greens[gi]['box']), get_center(reds[rj]['box'])
        fc = ((gc[0]+rc[0])/2, (gc[1]+rc[1])/2)
        # Scale for fusion box size
        side = (get_diag(greens[gi]['box']) + get_diag(reds[rj]['box'])) / 2.828
        f_box = (int(fc[0]-side), int(fc[1]-side), int(fc[0]+side), int(fc[1]+side))
        
        final_results.append({'cid': 3, 'box': f_box})

    # 2. Add remaining individual signals
    for i, g in enumerate(greens):
        if i not in used_g: final_results.append(g)
    for i, r in enumerate(reds):
        if i not in used_r: final_results.append(r)
    for a in aquas:
        final_results.append(a)

    return final_results