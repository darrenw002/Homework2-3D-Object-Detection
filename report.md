# **Homework2 3D Object Detection**

**PointPillars (KITTI & nuScenes) \+ CenterPoint (nuScenes)**  
 **Windows \+ WSL2 Hybrid Workflow**

**This project demonstrates an end-to-end 3D object detection workflow using OpenMMLab’s MMDetection3D.**  
 **It includes model inference, point cloud visualization, dataset preparation, troubleshooting notes, and score-based metrics for PointPillars and CenterPoint.**

---

## **Table of Contents**

* **Overview**

* **Project Structure**

* **Environment Setup**

  * **Windows (CPU)**

  * **Windows (CUDA attempt)**

  * **WSL2 \+ Ubuntu (CUDA – CenterPoint)**

* **Dataset Preparation**

* **Running Inference**

  * **PointPillars — KITTI**

  * **PointPillars — nuScenes**

  * **CenterPoint — nuScenes (CUDA)**

* **Visualization**

* **Metrics**

* **Troubleshooting**

* **Results Summary**

* **Takeaways and Limitations**

---

## **Overview**

**This project compares three inference configurations:**

| Model | Dataset | Device | Notes |
| ----- | ----- | ----- | ----- |
| **PointPillars (Car-only)** | **KITTI** | **CPU** | **Strong, high-confidence detections** |
| **PointPillars (General)** | **nuScenes** | **CPU** | **Many low-confidence detections** |
| **CenterPoint** | **nuScenes** | **CUDA (WSL2)** | **Best confidence & calibration** |

**Because Windows cannot run sparse voxel ops, all CenterPoint GPU inference is performed inside WSL2, while visualization and reporting are done back in Windows.**

---

## **Project Structure**

**`3d_detection_project/`**  
**`│`**  
**`├── mmdet3d_inference2.py`**  
**`├── scripts/`**  
**`│   ├── open3d_view_saved_ply.py`**  
**`│   └── export_kitti_calib.py`**  
**`│`**  
**`├── checkpoints/`**  
**`├── data/`**  
**`│   ├── kitti/`**  
**`│   └── nuscenes_demo/`**  
**`│`**  
**`├── outputs/`**  
**`└── results/`**

---

## **Environment Setup**

### **Windows (CPU Environment)**

**`py -3.10 -m venv .venv`**  
**`& .\.venv\Scripts\Activate.ps1`**

**`python -m pip install -U pip`**  
**`pip install openmim open3d opencv-python-headless==4.8.1.78 \`**  
  **`opencv-python==4.8.1.78 matplotlib tqdm moviepy pandas seaborn`**

**`` pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu ` ``**  
  **`--index-url https://download.pytorch.org/whl/cpu`**

**`pip install numpy==1.26.4`**  
**`mim install mmengine`**  
**`pip install mmcv==2.1.0 mmdet==3.2.0`**  
**`mim install mmdet3d`**

#### **Important:**

**Use `mim install` for mmcv/mmdet/mmdet3d — it ensures compiled ops (`mmcv._ext`) exist.**

---

### **Windows (CUDA Attempt – Not Supported)**

**Windows cannot run CenterPoint due to missing sparse voxel ops.**  
 **Use WSL2 instead.**

---

### **WSL2 \+ Ubuntu (CUDA — Required for CenterPoint)**

**`wsl --install`**  
**`wsl --set-default-version 2`**  
**`wsl --install -d Ubuntu`**

**Inside Ubuntu:**

**`conda create -n mmdet3d python=3.10 "numpy<2" -y`**  
**`conda activate mmdet3d`**

**`pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \`**  
  **`--index-url https://download.pytorch.org/whl/cu118`**

**`mim install "mmengine==0.10.7"`**  
**`mim install "mmcv==2.1.0"`**  
**`pip install "mmdet==3.2.0"`**  
**`mim install "mmdet3d==1.4.0"`**  
**`pip install numpy==1.26.4`**

---

## **Dataset Preparation**

### **KITTI**

**`mkdir data\kitti\training\{velodyne,image_2,label_2,calib}`**

**`Copy-Item external\mmdetection3d\demo\data\kitti\000008.bin data\kitti\training\velodyne\`**  
**`Copy-Item external\mmdetection3d\demo\data\kitti\000008.png data\kitti\training\image_2\`**  
**`Copy-Item external\mmdetection3d\demo\data\kitti\000008.txt data\kitti\training\label_2\`**

**Convert calibration:**

**`` python scripts/export_kitti_calib.py external/mmdetection3d/demo/data/kitti/000008.pkl ` ``**  
  **`data/kitti/training/calib/000008.txt`**

### **nuScenes Demo**

**`mkdir data\nuscenes_demo\{images,lidar}`**

**`Copy-Item external\mmdetection3d\demo\data\nuscenes\*CAM*jpg data\nuscenes_demo\images\`**  
**`Copy-Item external\mmdetection3d\demo\data\nuscenes\*.pcd.bin data\nuscenes_demo\lidar\`**

---

## **Running Inference**

### **PointPillars — KITTI**

**`` mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car ` ``**  
  **`--dest checkpoints\pointpillars_kitti`**

**`` python mmdet3d_inference2.py ` ``**  
  **`` --dataset kitti ` ``**  
  **`` --model checkpoints\pointpillars_kitti\pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ` ``**  
  **`` --checkpoint checkpoints\pointpillars_kitti\hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331.pth ` ``**  
  **`` --frame-number 000008 ` ``**  
  **`` --input-path data\kitti\training ` ``**  
  **`` --out-dir outputs\kitti_run1 ` ``**  
  **`` --device cpu ` ``**  
  **`--headless`**

---

### **PointPillars — nuScenes**

**nuScenes boxes have \>7 fields → add this helper to avoid unpacking errors:**

**`def ensure_bbox7(bbox):`**  
    **`arr = np.array(bbox, dtype=float).ravel()`**  
    **`if arr.size < 7:`**  
        **`raise ValueError("BBox must have ≥ 7 values.")`**  
    **`return arr[:7]`**

**Inference:**

**`` python mmdet3d_inference2.py ` ``**  
  **`` --dataset any ` ``**  
  **`` --input-path data\nuscenes_demo\lidar\sample.pcd.bin ` ``**  
  **`` --model checkpoints\nuscenes_pointpillars\pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py ` ``**  
  **`` --checkpoint checkpoints\nuscenes_pointpillars\hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.pth ` ``**  
  **`` --out-dir outputs\nuscenes_pointpillars ` ``**  
  **`` --device cpu ` ``**  
  **`` --score-thr 0.2 ` ``**  
  **`--headless`**

---

### **CenterPoint — nuScenes (CUDA, WSL2 Only)**

**`python mmdet3d_inference2.py \`**  
  **`--dataset any \`**  
  **`--input-path data/nuscenes_demo/lidar/sample.pcd.bin \`**  
  **`--model checkpoints/nuscenes_centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms.py \`**  
  **`--checkpoint checkpoints/nuscenes_centerpoint/centerpoint_01voxel_second_secfpn.pth \`**  
  **`--out-dir outputs/nuscenes_centerpoint \`**  
  **`--device cuda:0 \`**  
  **`--score-thr 0.2 \`**  
  **`--headless`**

**WSL2 cannot run Open3D GUI → copy `.ply` results back to Windows.**

---

## **Visualization**

**`` python scripts/open3d_view_saved_ply.py ` ``**  
  **`` --dir outputs\kitti_run1 ` ``**  
  **`` --basename 000008 ` ``**  
  **`` --width 1600 ` ``**  
  **`--height 1200`**

**Outputs include:**

* **`*_open3d.png`**

* **KITTI 2D projections: `*_2d_vis.png`**

* **`.ply` point clouds**

* **Prediction JSON files**

---

## **Metrics**

**A small Python script computes score statistics (mean, std, max, class counts):**

**`{`**  
  **`"kitti": {`**  
    **`"detections": 10,`**  
    **`"mean_score": 0.792,`**  
    **`"max_score": 0.975,`**  
    **`"class_counts": {"Car": 10}`**  
  **`},`**  
  **`"nuscenes_pointpillars": {`**  
    **`"detections": 365,`**  
    **`"mean_score": 0.127`**  
  **`},`**  
  **`"nuscenes_centerpoint": {`**  
    **`"detections": 264,`**  
    **`"mean_score": 0.244`**  
  **`}`**  
**`}`**

**Stored in:**

**`outputs/inference_stats.json`**

---

## **Troubleshooting**

### **1\. Missing `mmcv._ext`**

**Symptom:**  
 **Inferencers fail to import.**

**Fix:**

**`pip uninstall -y mmdet3d mmdet mmcv`**  
**`mim install "mmcv==2.1.0"`**  
**`mim install "mmdet==3.2.0"`**  
**`mim install "mmdet3d==1.4.0"`**

---

### **2\. nuScenes: `ValueError: too many values to unpack`**

**nuScenes boxes have \>7 fields → use:**

**`ensure_bbox7(bbox)`**

---

### **3\. CenterPoint GPU cannot run on Windows**

**Sparse ops unsupported → use WSL2 CUDA environment.**

---

### **4\. Open3D GUI does not run in WSL2**

**Copy `.ply` back to Windows and visualize there.**

---

## **Results Summary**

| Model | Dataset | Mean Score | High-Conf (≥0.7) | Notes |
| ----- | ----- | ----- | ----- | ----- |
| **PointPillars (KITTI)** | **KITTI** | **0.792** | **8** | **Best calibrated** |
| **PointPillars (nuScenes)** | **nuScenes** | **0.127** | **1** | **Many low-confidence detections** |
| **CenterPoint (nuScenes)** | **nuScenes** | **0.244** | **15** | **Best nuScenes performance** |

---

## **7\. Takeaways and Limitations**

### **Takeaways**

* A full multi-model 3D detection pipeline was successfully built: inference, `.ply` export, visualization, and score-based metrics.

* KITTI PointPillars produced high-confidence detections on the sample frame.

* CenterPoint outperformed PointPillars on nuScenes in terms of confidence and calibration.

* The environment setup (Windows \+ WSL2 hybrid workflow) produced a replicable and functional pipeline.

### **Limitations**

* CenterPoint cannot run natively on Windows due to missing sparse ops.

* No quantitative evaluation (mAP) was performed due to lack of annotation splits.

* Calibration conversion script can be refactored for future reuse.

* Runtime/FPS statistics were planned but not integrated.

