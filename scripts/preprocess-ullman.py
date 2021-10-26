import os

import numpy as np

from scipy import stats

import cv2, imutils

from matplotlib import pyplot as plt
from matplotlib import animation, rc
plt.style.use("bmh")
rc("animation", html="html5")

from tqdm import tqdm
from icecream import ic

import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
import argparse

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_BG = (250, 253, 251)
COLOR_MAT_GREEN = (48, 203, 154)
COLOR_MAT_PURPLE = (135, 0, 138)
COLOR_MAT_BROWN = (40, 44, 165)

PROJ_DIR = os.path.expanduser("~/projects/bayesian_symbolic_physics")
DATA_DIR = os.path.join(PROJ_DIR, "data/ullman")
SCRIPT_DIR = os.path.join(PROJ_DIR, "script")
DATA_FNS = sorted(list(filter(lambda f: "mp4" in f, os.listdir(DATA_DIR)))) 
DATA_FNS = DATA_FNS[6:] + DATA_FNS[:6] # move "world10" to the end

def cv2_imshow(ax, img, **kwargs):
    return ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **kwargs)

def remove_wall(img, edge=20):
    img[:edge,:] = np.array([COLOR_BG])
    img[:,:edge] = np.array([COLOR_BG])
    img[-edge:,:] = np.array([COLOR_BG])
    img[:,-edge:] = np.array([COLOR_BG])
    return img

def read_frames(fn):
    vidcap = cv2.VideoCapture(os.path.join(DATA_DIR, fn))
    success, image = vidcap.read()
    count, frames = 0, []
    while success:
        frames.append(remove_wall(image))
        success, image = vidcap.read()
        count += 1
    return frames

def plot_imgs(imgs, fig_w=5, fig_h=4):
    fig, axes = plt.subplots(1, len(imgs), figsize=(fig_w * len(imgs), fig_h))
    for (ax, img) in zip(axes, imgs):
        cv2_imshow(ax, img, cmap="gray")
        ax.axis("off")
    
def find_contours(image, vis=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 3)
    _, th = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(th, 50, 100)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if vis:
        plot_imgs([image, gray, blurred, th, edges])
    return imutils.grab_contours(cnts)

from scipy.spatial import distance

def resolve_overlap(cnt0):
    peri = cv2.arcLength(cnt0, True)
    approx = cv2.approxPolyDP(cnt0, 0.02 * peri, True)
    if len(approx) <= 4 or len(approx) >= 10:
        return [approx]
    else:
        peri = cv2.arcLength(cnt0, True)
        cnt = cv2.approxPolyDP(cnt0, 0.02 * peri, True)
        cnt = cnt.squeeze()
        if cnt.shape[0] == 8:
            v = distance.pdist(cnt)     # this returns condensed distance matrix
            m = distance.squareform(v)  # convert to pairwise distance matrix
            idx1, idx2 = np.unravel_index(m.argmax(), m.shape)
            tri1 = [cnt[idx1-1], cnt[idx1], cnt[idx1+1]]
            tri2 = [cnt[idx2-1], cnt[idx2], cnt[idx2+1]]
            tri1.append((tri1[0] - tri1[1]) + (tri1[2] - tri1[1]) + tri1[1])
            tri2.append((tri2[0] - tri2[1]) + (tri2[2] - tri2[1]) + tri2[1])
            return [np.array(tri1), np.array(tri2)]
        else:
            return [approx]

def detect_shape(cnt):
    shape = "unidentified"
    if len(cnt) == 3:
        shape = "triangle"
    elif len(cnt) == 4:
        shape = "rectangle"
    elif len(cnt) == 5:
        shape = "pentagon"
    elif len(cnt) >= 6:
        shape = "circle"
    return shape
    
def annotate_contours(image, cnts):
    for c0 in cnts:
        for c in resolve_overlap(c0):
            shape = detect_shape(c)
            if shape == "rectangle":
                cv2.drawContours(image, [c], -1, COLOR_GREEN, 4)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX, cY = int((M["m10"] / M["m00"])), int((M["m01"] / M["m00"]))
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK, 4)
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    return image

def process(frame):
    cnts = find_contours(frame)
    frame = annotate_contours(frame, cnts)
    return frame

OBJECTS = ["red_circle", "blue_circle", "yellow_circle"]
TEMPLATES = {obj: cv2.imread(os.path.join(DATA_DIR, "templates", f"{obj}.png")) for obj in OBJECTS}
CIRCLE_MASK = 255 - cv2.imread(os.path.join(DATA_DIR, "templates", "circle_mask.png"), 0)

def template_match(image, template, mask=None):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)

def l2sq(pt1, pt2):
    return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2

def remove_duplicates(pts, min_distance=40):
    pts_clean = []
    for pt in pts:
        duplicated = False
        for pt_other in pts_clean:
            if l2sq(pt, pt_other) < min_distance**2:
                duplicated = True
                break
        if not duplicated:
            pts_clean.append(pt)
    return pts_clean
    
def annotate_match(image, match, threshold=0.8, obj=None, image_remove=None):
    h, w, _ = TEMPLATES[obj].shape # the last dimension is the channel
    y, x = np.where((match >= threshold) * (match <= 1.0))
    pts = remove_duplicates(zip(x, y))
    for pt in pts:
        cX, cY = pt[0] + w, pt[1] + h
        cv2.rectangle(image, pt, (cX, cY), COLOR_GREEN, 2)
        if image_remove is not None:
            image_remove[pt[1]:cY+1,pt[0]:cX+1] = COLOR_BG
        if obj is not None:
            cv2.putText(image, obj, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK, 4)
            cv2.putText(image, obj, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    return pts

def find_mats(cnt0, minimum_area=0):
    peri = cv2.arcLength(cnt0, True)
    approx = cv2.approxPolyDP(cnt0, 0.02 * peri, True)
    if len(approx) == 4 and cv2.contourArea(approx) >= minimum_area:
        return [cv2.boundingRect(approx)] # (x, y, w, h)
    else:
        peri = cv2.arcLength(cnt0, True)
        cnt = cv2.approxPolyDP(cnt0, 0.02 * peri, True)
        cnt = cnt.squeeze()
        if cnt.shape[0] == 8:
            v = distance.pdist(cnt)     # this returns condensed distance matrix
            m = distance.squareform(v)  # convert to pairwise distance matrix
            idx1, idx2 = np.unravel_index(m.argmax(), m.shape)
            tri1 = [cnt[idx1-1], cnt[idx1], cnt[idx1+1]]
            tri2 = [cnt[idx2-1], cnt[idx2], cnt[idx2+1]]
            tri1.append((tri1[0] - tri1[1]) + (tri1[2] - tri1[1]) + tri1[1])
            tri2.append((tri2[0] - tri2[1]) + (tri2[2] - tri2[1]) + tri2[1])
            return [cv2.boundingRect(np.array(tri1)), cv2.boundingRect(np.array(tri2))]
        else:
            return None

def find_mats_all(cnts):
    mats = []
    for cnt in cnts:
        mat = find_mats(cnt)
        if mat is not None:
            mats += mat
    return mats

def annotate_mat(image, mat):
    pt, c = (mat[0], mat[1]), (mat[0] + mat[2], mat[1] + mat[3])
    cv2.rectangle(image, pt, c, COLOR_GREEN, 2)
    cv2.putText(image, "mat", c, cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK, 4)
    cv2.putText(image, "mat", c, cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)

def process_template_matching(frame, objects):
    frame_annotated = frame.copy()
    frame_removed = frame.copy()
    pts_all = []
    for obj in objects:
        match = template_match(frame, TEMPLATES[obj], mask=(CIRCLE_MASK if "circle" in obj else None))
        pts_all.append(annotate_match(frame_annotated, match, obj=obj, image_remove=frame_removed))
    return frame_annotated, frame_removed, pts_all

def find_color(frame, mat):
    patch = frame[mat[1]:mat[1]+mat[3],mat[0]:mat[0]+mat[2]]
    pixels = np.array(patch).reshape(-1, 3)
    color = stats.mode(pixels, axis=0).mode[0]
    if np.all(color - np.array(COLOR_MAT_GREEN) < 5):
        return "green"
    if np.all(color - np.array(COLOR_MAT_PURPLE) < 5):
        return "purple"
    if np.all(color - np.array(COLOR_MAT_BROWN) < 5):
        return "brown"

def parse_frames(frames):
    symbolic = {"static": {}, "dynamic": []}
    
    frames_annotated, frames_removed = [], []
    for frame in tqdm(frames, desc="template matching for circles"):
        frame_annotated, frame_removed, pts_all = process_template_matching(frame, OBJECTS)
        frames_annotated.append(frame_annotated)
        frames_removed.append(frame_removed)
        
        symbolic_dynamic = {}
        for (obj, pts) in zip(OBJECTS, pts_all):
            if len(pts) > 0:
                symbolic_dynamic[obj] = pts
        symbolic["dynamic"].append(symbolic_dynamic)
    
    frame_bg = np.min(frames_removed[1:-1:10], axis=0).astype(dtype=np.uint8)
    cnts = find_contours(frame_bg.copy())
    for frame_annotated in tqdm(frames_annotated, desc="annotating mats"):
        annotate_contours(frame_annotated, cnts)
    
    for mat in tqdm(find_mats_all(cnts), desc="detect mats and their colors"):
        # Majority vote for mat colors
        colors = [find_color(frame, mat) for frame in frames]
        # Remove Nones
        colors = list(filter(lambda c: c is not None, colors))
        if len(colors) > 0:
            color = stats.mode(colors).mode[0]
            name = f"{color}_rectangle"
            if name not in symbolic["static"]:
                symbolic["static"][name] = [mat]
            else:
                symbolic["static"][name].append(mat)
    
    return symbolic, frames_annotated


if __name__ == "__main__":
    print("Data files from Ullman (2018)", f"{len(DATA_FNS)=}")
    ic(DATA_FNS)

    parser = argparse.ArgumentParser()
    parser.add_argument("fn_id", type=int)
    args = parser.parse_args()

    fn = DATA_FNS[args.fn_id-1]
    print(f"{fn=}")

    frames = read_frames(fn)

    symbolic, frames_annotated = parse_frames(frames)

    writer = cv2.VideoWriter(os.path.join(f"{DATA_DIR}-annotated", fn), cv2.VideoWriter_fourcc(*"MP4V"), 20.0, (640, 480))
    for frame in frames_annotated:
        writer.write(frame)
    writer.release()

    with open(os.path.join(DATA_DIR, "processed", fn.replace("mp4", "json")), "w") as json_file:
        json.dump(symbolic, json_file, cls=NpEncoder)
