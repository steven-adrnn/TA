# dataset/balanced_sampler.py
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class DemographicBalancedSampler(Sampler):
    """
    Memastikan setiap batch mengandung sampel dari
    sebanyak mungkin kelompok demografis.

    Cara kerja:
    - Kelompokkan semua index dataset berdasarkan demographic_label
    - Saat iterasi, ambil secara round-robin dari setiap grup
    - Shuffle dalam setiap grup di awal setiap epoch
    """

    def __init__(self, dataset, batch_size, num_demographics=14):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_demographics = num_demographics

        # Bangun mapping: demographic_label -> [list of indices]
        self.group_indices = defaultdict(list)
        self._build_group_indices()

        print(f"[BalancedSampler] Distribusi demografis dalam dataset:")
        for (suku, kelas), idxs in sorted(self.group_indices.items()):
            label_kelas = "Real" if kelas == 0 else "Fake"
            print(f"  Suku {suku:2d} | {label_kelas:4s}: {len(idxs):5d} sampel")

    def _build_group_indices(self):
        """
        Mengelompokkan berdasarkan tuple: (demographic_label, class_label)
        class_label -> 0 (Real), 1 (Fake)
        """
        dataset = self.dataset
        print("[BalancedSampler] Membangun demographic & class label cache...")

        for idx in range(len(dataset)):
            image_paths = dataset.data_dict['image'][idx]
            if not isinstance(image_paths, list):
                image_paths = [image_paths]

            first_path = image_paths[0].replace('\\', '/')
            video_name = first_path.split('/')[-2]

            demo_label = int(dataset._get_demographic_label(video_name, first_path))
            
            # Ambil label kelas (pastikan formatnya hanya 0 dan 1)
            raw_label = int(dataset.data_dict['label'][idx])
            class_label = 1 if raw_label > 0 else 0

            # Simpan menggunakan tuple (Suku, Real/Fake)
            self.group_indices[(demo_label, class_label)].append(idx)

        print(f"[BalancedSampler] Total kombinasi grup (Suku, Kelas) yang terdeteksi: {len(self.group_indices)}")

    def __iter__(self):
        # Shuffle dalam setiap grup
        shuffled_groups = {}
        for k, idxs in self.group_indices.items():
            arr = np.array(idxs, dtype=np.int64)
            np.random.shuffle(arr)
            shuffled_groups[k] = arr.tolist()

        pointers = {k: 0 for k in shuffled_groups}
        all_indices = []

        # Ambil daftar suku yang unik (0-13) dari keys
        unique_demos = sorted(list(set([k[0] for k in shuffled_groups.keys()])))

        exhausted = set()
        
        # Round-robin: Ambil 1 Real dan 1 Fake dari setiap Suku
        while len(exhausted) < len(shuffled_groups):
            for g in unique_demos:
                # 1. Ambil 1 sampel Real (label 0)
                if (g, 0) not in exhausted and (g, 0) in shuffled_groups:
                    if pointers[(g, 0)] < len(shuffled_groups[(g, 0)]):
                        all_indices.append(shuffled_groups[(g, 0)][pointers[(g, 0)]])
                        pointers[(g, 0)] += 1
                    else:
                        exhausted.add((g, 0))

                # 2. Ambil 1 sampel Fake (label 1)
                if (g, 1) not in exhausted and (g, 1) in shuffled_groups:
                    if pointers[(g, 1)] < len(shuffled_groups[(g, 1)]):
                        all_indices.append(shuffled_groups[(g, 1)][pointers[(g, 1)]])
                        pointers[(g, 1)] += 1
                    else:
                        exhausted.add((g, 1))

        return iter(all_indices)

    def __len__(self):
        return sum(len(idxs) for idxs in self.group_indices.values())