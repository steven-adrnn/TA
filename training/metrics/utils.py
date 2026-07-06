# metrics/utils.py
from sklearn import metrics
import numpy as np
import json, os

SUKU_LIST = ['BAL', 'BJR', 'BTK', 'BUG', 'CIN', 'DYK', 'JAW', 'MKS', 'MLK', 'MIN', 'PAP', 'SSK', 'SUN']

def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def _build_indo_demographics_mapping(root_dir):
    """
    Membangun mapping: actor_id (str) -> nama suku (str)
    dengan membaca real video dari Indo-MM.json, Indo-FS.json, Indo-VC.json.
 
    Format real video yang bisa dibaca:
      - Indo-MM real : 001_BAL_M_144_REAL   -> parts[0]='001', parts[1]='BAL'
      - Indo-FS real : 001_BAL_M_144_REAL   -> sama
      - Indo-VC real : 001_BAL_M_144_REAL   -> sama
 
    Return: dict, misal {'001': 'BAL', '031': 'BJR', ...}
    """
    mapping = {}
    json_folder = os.path.join(root_dir, 'preprocessing', 'dataset_json')
 
    # Baca dari semua dataset Indo agar mapping selengkap mungkin
    for ds_name in ['Indo-MM', 'Indo-FS', 'Indo-VC']:
        json_path = os.path.join(json_folder, f'{ds_name}.json')
        if not os.path.exists(json_path):
            print(f"[WARNING] _build_indo_demographics_mapping: {json_path} tidak ditemukan, dilewati.")
            continue
 
        with open(json_path, 'r') as f:
            data = json.load(f)
 
        try:
            # Struktur JSON: data[ds_name]['Indo-real'][mode][comp][vid_name]
            real_data = data[ds_name]['Indo-real']
            for mode in real_data:           # train, val, test
                for comp in real_data[mode]: # 720p, 144p
                    for vid_name in real_data[mode][comp]:
                        # Format: 001_BAL_M_144_REAL
                        parts = vid_name.split('_')
                        if len(parts) >= 2 and parts[1] in SUKU_LIST:
                            actor_id = parts[0]   # '001'
                            suku     = parts[1]   # 'BAL'
                            mapping[actor_id] = suku
        except (KeyError, TypeError):
            pass
 
    return mapping
 
 
def _get_suku_from_vidname(vid_name, indo_demographics):
    """
    Mengekstrak nama suku dari nama folder video.
 
    Format yang didukung:
      Real  : 001_BAL_M_144_REAL   -> langsung baca parts[1]
      FS/MM : 001_009_144_FS       -> lookup parts[0] (target) ke mapping
              001_009_144_FS+VC+LS -> sama
      VC    : 001_BAL_M_144_VC     -> langsung baca parts[1]
 
    Menurut konvensi dataset:
      - Untuk FS/MM, suku yang dipakai adalah suku TARGET (parts[0]).
      - Target ID selalu ada di mapping karena berasal dari real video.
    """
    parts = vid_name.split('_')
    if len(parts) < 1:
        return "UNKNOWN"
 
    # Kasus 1: parts[1] langsung berupa kode suku
    # Berlaku untuk: real video ("001_BAL_M_144_REAL") dan VC ("001_BAL_M_144_VC")
    if len(parts) >= 2 and parts[1] in SUKU_LIST:
        return parts[1]
 
    # Kasus 2: parts[1] adalah angka (ID source) -> ini video FS atau MM
    # Suku diambil dari TARGET (parts[0]) via lookup ke mapping
    target_id = parts[0]
    if target_id in indo_demographics:
        return indo_demographics[target_id]
 
    # Fallback: coba parts[1] sebagai ID (kadang ada format yang tidak terduga)
    if len(parts) >= 2 and parts[1] in indo_demographics:
        return indo_demographics[parts[1]]
 
    return "UNKNOWN"


def get_test_metrics(y_pred, y_true, img_names, dataset_name):
    def compute_per_group_threshold(fairness_preds, fairness_labels, demos, global_threshold):
        """
        Kalibrasi threshold per suku menggunakan Youden's J per-grup.
        Tujuan: menyeimbangkan FPR antar suku.
        """
        from sklearn import metrics as sk_metrics
        
        group_thresholds = {}
        unique_demos = [d for d in np.unique(demos) if d != "UNKNOWN"]
        
        for demo in unique_demos:
            mask = (demos == demo)
            y_t = fairness_labels[mask]
            y_p = fairness_preds[mask]
            
            if len(np.unique(y_t)) < 2 or len(y_t) < 5:
                group_thresholds[demo] = global_threshold
                continue
            
            fpr, tpr, thresholds = sk_metrics.roc_curve(y_t, y_p, pos_label=1)
            youden = tpr - fpr
            opt_idx = np.argmax(youden)
            group_thresholds[demo] = thresholds[opt_idx]
        
        return group_thresholds

    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        video_names = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())
        video_names = list(result_dict.keys())

        for video in image_arr:
            pred_sum = []
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum.append(float(frame[1]))
                label_sum += int(frame[2])
                leng += 1
            vid_pred = max(pred_sum)
            new_pred.append(vid_pred)
            vid_label = 1 if (label_sum / leng) >= 0.5 else 0
            new_label.append(vid_label)
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        v_ap = metrics.average_precision_score(new_label, new_pred)
        v_youden_idx = np.argmax(tpr - fpr)
        v_optimal_threshold = thresholds[v_youden_idx]
        v_pred_class = (new_pred >= v_optimal_threshold).astype(int)
        v_acc = (v_pred_class == new_label).sum() / len(v_pred_class)
        v_f1 = metrics.f1_score(new_label, v_pred_class, zero_division=0)

        return (v_auc, v_eer, v_ap, v_acc, v_f1, v_optimal_threshold, np.array(video_names), np.array(new_pred), np.array(new_label))

    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc

    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]


    prediction_class = (y_pred >= optimal_threshold).astype(int)
    prediction_class_05 = (y_pred >= 0.5).astype(int)
    y_true_clip = np.clip(y_true, a_min=0, a_max=1)
    correct = (prediction_class == y_true_clip).sum()
    acc = correct / len(prediction_class)
    f1 = metrics.f1_score(y_true_clip, prediction_class, zero_division=0)
    # if type(img_names[0]) is not list:
    #     # calculate video-level auc for the frame-level methods.
    #     v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    # else:
    #     # video-level methods
    #     v_auc=auc

    is_frame_level_model = type(img_names[0]) is not list

    if is_frame_level_model:
        # INI JIKA MODEL XCEPTION (Frame-level)
        v_auc, v_eer, v_ap, v_acc, v_f1, v_optimal_threshold, vid_names, vid_preds, vid_labels = get_video_metrics(img_names, y_pred, y_true)
        
        # Gunakan data yang sudah diagregasi ke level video untuk Fairness
        fairness_names = vid_names
        fairness_preds = vid_preds
        fairness_labels = vid_labels
        optimal_threshold = v_optimal_threshold
    else:
        # INI JIKA MODEL FAIRCROSSDOMAIN (Sudah Video-level)
        v_auc, v_eer, v_ap, v_acc, v_f1 = auc, eer, ap, acc, f1
        
        vid_names = []
        for s in img_names:
            if isinstance(s, list):
                s = s[0]
            vid_name = s.split('\\')[-2] if '\\' in s else s.split('/')[-2]
            vid_names.append(vid_name)
            
        fairness_names = np.array(vid_names)
        fairness_preds = y_pred
        fairness_labels = y_true

    fairness_pred_class_opt = (fairness_preds >= optimal_threshold).astype(int)
    fairness_pred_class_05 = (fairness_preds >= 0.5).astype(int)
    fairness_labels_clip = np.clip(fairness_labels, a_min=0, a_max=1)

    is_indo = any(x in dataset_name for x in ['Indo-MM', 'Indo-FS', 'Indo-VC', 'IndoDeepfake'])
    
    gfpr, gtpr, efpr, etpr, eo = 0.0, 0.0, 0.0, 0.0, 0.0
    if is_indo: # <--- HANYA MASUK SINI JIKA DATASET ADALAH INDO
        # Sesuaikan path ini dengan letak folder preprocessing Anda
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        training_dir = os.path.dirname(current_dir) 
        root_dir = os.path.dirname(training_dir) 
        
        indo_demographics = _build_indo_demographics_mapping(root_dir)
        if not indo_demographics:
            print(f"[WARNING] get_test_metrics: indo_demographics kosong! "
                  f"Fairness metrics tidak akan dihitung.")
            gfpr, gtpr, efpr, eo = 0.0, 0.0, 0.0, 0.0
        else:
            demos = []
            for vid_name in fairness_names:
                suku = _get_suku_from_vidname(vid_name, indo_demographics)
                demos.append(suku)

            demos = np.array(demos)
            unique_demos = [d for d in np.unique(demos) if d != "UNKNOWN"]

            if len(unique_demos) > 1:
                group_thresholds = compute_per_group_threshold(
                    fairness_preds, fairness_labels_clip, demos, optimal_threshold
                )
                fairness_pred_class_pergroup = np.zeros_like(fairness_pred_class_opt)
                for i, (name, pred) in enumerate(zip(fairness_names, fairness_preds)):
                    suku = _get_suku_from_vidname(name, indo_demographics)
                    thresh = group_thresholds.get(suku, optimal_threshold)
                    fairness_pred_class_pergroup[i] = int(pred >= thresh)
                    
                group_fprs_05, group_tprs_05 = [], []

                group_fprs_opt, group_tprs_opt = [], []
                
                print(f"\n--- DEBUG FAIRNESS VIDEO-LEVEL ({dataset_name}) | Opt Thresh: {optimal_threshold:.4f} ---")
                
                for demo in unique_demos:
                    mask = (demos == demo)
                    y_t_demo = fairness_labels_clip[mask]
                    
                    y_p_demo_opt = fairness_pred_class_pergroup[mask]
                    y_p_demo_05 = fairness_pred_class_05[mask]

                    if len(np.unique(y_t_demo)) < 2:
                        continue

                    cm_05 = metrics.confusion_matrix(y_t_demo, y_p_demo_05, labels=[0, 1]).ravel()
                    cm_opt = metrics.confusion_matrix(y_t_demo, y_p_demo_opt, labels=[0, 1]).ravel()
                    
                    if len(cm_05) == 4 and len(cm_opt) == 4:
                        tn_05, fp_05, fn_05, tp_05 = cm_05
                        tn_opt, fp_opt, fn_opt, tp_opt = cm_opt
                        
                        fpr_val_05 = fp_05 / (fp_05 + tn_05) if (fp_05 + tn_05) > 0 else 0.0
                        tpr_val_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0.0
                        
                        fpr_val_opt = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0.0
                        tpr_val_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0

                        group_fprs_opt.append(fpr_val_opt)
                        group_tprs_opt.append(tpr_val_opt)
                        
                        print(f"Suku: {demo:<4} | [Opt: {optimal_threshold:.2f}] Real (TN:{tn_opt:<2}, FP:{fp_opt:<2}) -> FPR: {fpr_val_opt:.2f} | Fake (FN:{fn_opt:<2}, TP:{tp_opt:<2})")
                        print(f"            | [Netral: 0.50] Real (TN:{tn_05:<2}, FP:{fp_05:<2}) -> FPR: {fpr_val_05:.2f} | Fake (FN:{fn_05:<2}, TP:{tp_05:<2})")

                print("--------------------------------------------------")

                if len(group_fprs_opt) >= 2:
                    gfpr = max(group_fprs_opt) - min(group_fprs_opt)
                    gtpr = max(group_tprs_opt) - min(group_tprs_opt)
                    efpr = float(np.std(group_fprs_opt))
                    etpr = float(np.std(group_tprs_opt))
                    # eo   = (max(group_fprs_opt) - min(group_fprs_opt)) + \
                    #        (max(group_tprs_opt) - min(group_tprs_opt))
                    # eo_std = efpr + etpr                                    # ← metrik EO alternatif
                    eo = gfpr + gtpr
            else:
                print(f"[WARNING] get_test_metrics ({dataset_name}): "
                      f"Hanya {len(unique_demos)} group suku valid ditemukan. "
                      f"Fairness metrics tidak dapat dihitung.")
 
    return {
        'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'f1': f1,
        'video_acc': v_acc, 'video_auc': v_auc, 'video_eer': v_eer,
        'video_ap': v_ap, 'video_f1': v_f1,
        'gfpr': gfpr, 'gtpr': gtpr, 'efpr': efpr, 'etpr': etpr, 'eo': eo,
        'pred': y_pred, 'video_auc': v_auc, 'label': y_true,
        # 'pred': y_pred, 'label': y_true,
        'opt_thresh': optimal_threshold
    }