import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data_manager import setup_seed

from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import (
    partition_data,
    partition_test_by_train_distribution,
    DatasetSplit,
    average_weights,
)


class Anchor(BaseLearner):
    """Representative Feature Anchor with Task-balanced Memory and Batch Anchor Contrastive.

    핵심 설계
    ---------
    1) Raw data replay 없이 feature anchor만 저장한다.
    2) anchor 생성은 current-task 대표성에 old-anchor overlap penalty를 더한 overlap-aware k-means representative anchor로 수행한다.
    3) 각 client는 고정된 anchor_budget B개만 유지한다.
    4) FR/LS는 실제 forgetting 원인과 불일치할 수 있으므로 pruning/loss weight에는 쓰지 않고 진단 로그로만 유지한다.
    5) Memory는 task-balanced하게 유지한다.

    학습 loss
    --------
    L = CE_new + anchor_lambda * AnchorKL + anchor_contrast_lambda * BatchAnchorContrastive

    - AnchorKL: 저장된 old representative feature u에 대해 classifier head의 예전 분포 p_saved를 보존한다.
    - BatchAnchorContrastive: current-task feature가 old anchors보다 current batch task-center에
      상대적으로 더 가깝도록 만드는 anchor-based contrastive loss다.
      current task anchor가 아직 생성되기 전이므로, 현재 mini-batch mean feature를 positive proxy로 사용한다.
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        # client_id -> list[anchor]
        # anchor = {
        #   "task": int,
        #   "u": Tensor[feature_dim],           # stored feature anchor on CPU
        #   "p": Tensor[num_seen_at_save],      # saved local prediction on CPU
        #   "confidence": float,
        #   "selection_risk": float,            # uncertainty score used at creation
        #   "radius": float,                    # cluster mean distance from u
        #   "sigma": float,                     # cluster scalar feature std
        #   "p_var": float,                     # cluster prediction variance
        #   "support": int,                     # number of samples represented by this anchor
        #   "boundary_u": Tensor[feature_dim],    # lowest-margin feature inside the same cluster
        #   "boundary_p": Tensor[num_seen],       # saved prediction at boundary_u
        #   "boundary_risk": float,               # 1 - margin at boundary_u
        #   "boundary_margin": float,             # saved top1-top2 probability margin at boundary_u
        #   "boundary_top1": int,
        #   "boundary_top2": int,
        #   "ls": float,
        #   "fr": float,
        #   "importance": float,
        # }
        self.client_anchors = {}          # risk anchors: KL / FR / LS / importance

        # client_id -> {task_id: acc_at_task_end}
        # FR이 실제 forgetting을 반영하는지 보려고, 각 client가 해당 task를 막 배운 직후의
        # task별 accuracy를 저장해둔다. 이후 task에서 현재 accuracy와 비교해 drop을 찍는다.
        self.client_task_baseline_acc = {}

        # ===== Anchor memory =====
        # CIFAR10 진단용 기본값: 전체 10개 중 current task는 최대 1개, old anchor는 최대한 유지
        self.anchor_budget = args.get("anchor_budget",5)
        self.anchor_temp = args.get("anchor_temp", 1.0)

        # representative anchor 수. 기존 anchor_topk/ratio는 호환성 때문에 남겨두지만,
        # 현재 representative k-means 생성에서는 사용하지 않는다.
        self.anchor_topk = args.get("anchor_topk", 32)
        self.anchor_topk_ratio = args.get("anchor_topk_ratio", 0.05)

        self.anchor_per_task = args.get("anchor_per_task", 3)
        self.old_anchor_min = args.get("old_anchor_min", 3)
        self.current_anchor_max = args.get("current_anchor_max", 2)

        # ===== Overlap-aware anchor selection =====
        # 기존 k-means는 current task 내부 대표성만 보고 anchor를 뽑았다.
        # 로그상 old/current anchor 영역이 심하게 겹치는 것이 관찰되어,
        # 이제는 후보 anchor를 넉넉히 만든 뒤 old anchors와 너무 가까운 후보를 덜 선택한다.
        # 클래스별 anchor는 만들지 않고, task-level representative 후보 안에서만 재선택한다.
        self.anchor_candidate_multiplier = args.get("anchor_candidate_multiplier", 4)
        self.anchor_overlap_lambda = args.get("anchor_overlap_lambda", 0.5)
        self.anchor_diversity_lambda = args.get("anchor_diversity_lambda", 0.3)
        self.anchor_overlap_use_boundary = args.get("anchor_overlap_use_boundary", False)
        self.debug_anchor_selection = args.get("debug_anchor_selection", False)

        # ===== Distribution-aware anchor region =====
        # 기존 anchor는 center feature u 한 점만 저장했다.
        # 이제는 k-means cluster의 compact summary(radius/sigma/p_var/support)를 함께 저장하고,
        # 학습 시 u 주변 feature region에도 KL을 걸어 point-wise overfitting을 줄인다.
        self.anchor_region_samples = args.get("anchor_region_samples", 4)
        self.anchor_region_noise = args.get("anchor_region_noise", 0.5)
        self.anchor_region_lambda = args.get("anchor_region_lambda", 1.0)

        # ===== Loss weights =====
        self.anchor_lambda = args.get("anchor_lambda", 0.01)

        # Batch anchor contrastive.
        # current task anchor는 local training이 끝난 뒤 생성되므로, 학습 중에는
        # 현재 mini-batch mean feature를 positive proxy로 쓰고 old anchors를 negative로 둔다.
        # 목적은 absolute하게 old anchor와 멀어지는 것이 아니라, current features가
        # old anchors보다 current batch center에 상대적으로 더 붙도록 만드는 것이다.
        self.anchor_contrast_lambda = args.get("anchor_contrast_lambda", 0.01)
        self.anchor_contrast_temp = args.get("anchor_contrast_temp", 0.2)
        self.anchor_contrast_use_boundary = args.get("anchor_contrast_use_boundary", False)
        self.anchor_contrast_detach_center = args.get("anchor_contrast_detach_center", True)

        # Diagnostics only: threshold is used only in FEATURE-OVERLAP-CHECK logging.
        self.anchor_sep_threshold = args.get("anchor_sep_threshold", 0.55)

        # ===== Anchor retention / weighting =====
        # FR은 실제 old accuracy drop과 불일치하는 케이스가 있어 pruning/loss weight에서 제외한다.
        # 대신 task별 coverage를 보장하고, 필요하면 support 기반으로 loss weight를 줄 수 있게 한다.
        self.task_anchor_min = args.get("task_anchor_min", 2)
        self.anchor_weight_mode = args.get("anchor_weight_mode", "task_uniform")
        # choices: "task_uniform", "uniform", "support", "sqrt_support", "task_sqrt_support"

        # ===== Importance hyperparameters =====
        # FR/LS rank는 진단용으로만 유지한다. pruning과 loss weighting에는 사용하지 않는다.
        self.importance_beta = args.get("importance_beta", 0.5)   # diagnostic compatibility only
        # self.importance_gamma = args.get("importance_gamma", 0.3) # age bonus

        # Debug print frequency. 0이면 anchor debug 출력 안 함.
        self.debug_anchor_prob = args.get("debug_anchor_prob", 0.001)

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )

        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)

    # ------------------------------------------------------------------
    # Basic probability / anchor utilities
    # ------------------------------------------------------------------
    def _normalize_prob(self, p):
        eps = 1e-8
        p = torch.clamp(p, eps, 1.0)
        return p / torch.clamp(p.sum(), min=eps)

    def _kl_prob(self, p, q):
        eps = 1e-8
        p = self._normalize_prob(p)
        q = self._normalize_prob(q)
        return torch.sum(p * torch.log(torch.clamp(p / torch.clamp(q, eps, 1.0), eps, 1e8)))

    def _fc_logits_from_feature(self, model, feature):
        out = model.fc(feature)
        return out["logits"] if isinstance(out, dict) else out

    def _predict_anchor_prob(self, model, anchor_u):
        """Stored feature anchor를 현재 model의 FC에 통과시켜 prediction을 얻는다."""
        model.eval()
        u = anchor_u.cuda().unsqueeze(0)

        with torch.no_grad():
            logits = self._fc_logits_from_feature(model, u)
            logits = logits[:, : self._total_classes]
            prob = F.softmax(logits / self.anchor_temp, dim=1).squeeze(0)
            prob = self._normalize_prob(prob)

        return prob.detach().cpu()

    def _percentile_ranks(self, values):
        """Return percentile ranks in [0, 1]. Ties are handled stably enough for pruning."""
        n = len(values)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        arr = np.asarray(values, dtype=np.float64)
        order = np.argsort(arr)
        ranks = np.zeros(n, dtype=np.float64)
        for r, idx in enumerate(order):
            ranks[idx] = r / (n - 1)
        return ranks.tolist()

    # ------------------------------------------------------------------
    # Risk-aware anchor construction
    # ------------------------------------------------------------------
    def _select_topk_count(self, num_samples):
        if num_samples <= 0:
            return 0
        if self.anchor_topk is not None and self.anchor_topk > 0:
            return max(1, min(int(self.anchor_topk), num_samples))
        k = int(np.ceil(num_samples * float(self.anchor_topk_ratio)))
        return max(1, min(k, num_samples))

    def _kmeans_representative_indices(self, features, num_clusters, num_iters=20):
        """Select representative sample indices by k-means in normalized feature space.

        반환값은 cluster center 자체가 아니라, 각 center에 가장 가까운 실제 sample index다.
        이렇게 하면 anchor u가 실제 model feature 중 하나라서 p_saved 계산이 안정적이다.
        """
        n = int(features.size(0))
        k = max(1, min(int(num_clusters), n))
        if n == 0:
            return []
        if n <= k:
            return list(range(n))

        x = F.normalize(features.float(), dim=1)

        # Deterministic farthest-point initialization.
        # 첫 center는 전체 평균에 가장 가까운 sample로 두고, 이후는 기존 center들과 가장 먼 sample을 고른다.
        mean = x.mean(dim=0, keepdim=True)
        first = torch.argmin(torch.cdist(x, mean).squeeze(1)).item()
        selected = [first]

        for _ in range(1, k):
            centers = x[selected]
            dist_to_centers = torch.cdist(x, centers)
            min_dist = dist_to_centers.min(dim=1).values
            for idx in selected:
                min_dist[idx] = -1.0
            selected.append(int(torch.argmax(min_dist).item()))

        centers = x[selected].clone()

        for _ in range(int(num_iters)):
            dist = torch.cdist(x, centers)
            labels = dist.argmin(dim=1)
            new_centers = []
            for c in range(k):
                mask = labels == c
                if mask.sum() == 0:
                    new_centers.append(centers[c])
                else:
                    new_centers.append(F.normalize(x[mask].mean(dim=0), dim=0))
            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        # 각 center에 가장 가까운 실제 sample을 선택한다. 중복은 피한다.
        dist = torch.cdist(x, centers)
        selected_indices = []
        used = set()
        for c in range(k):
            order = torch.argsort(dist[:, c])
            chosen = None
            for idx in order.tolist():
                if idx not in used:
                    chosen = idx
                    break
            if chosen is None:
                chosen = int(order[0].item())
            selected_indices.append(chosen)
            used.add(chosen)

        return selected_indices

    def _build_anchor(self, model, loader, client_id=None):
        """Build distribution-aware representative feature anchors.

        기존 버전은 k-means cluster에서 center에 가까운 sample feature u 한 점만 저장했다.
        이 버전은 같은 anchor 개수를 유지하되, 각 anchor가 대표하는 cluster의 compact
        distribution summary를 함께 저장한다.

        저장 정보
        ---------
        u       : cluster representative feature
        p       : representative feature의 saved prediction
        radius  : cluster sample들이 u에서 평균적으로 얼마나 떨어져 있는지
        sigma   : cluster feature의 scalar std. training 때 u 주변 perturbation scale로 사용
        p_var   : cluster 내부 prediction variance. anchor가 대표하는 region의 불안정성 진단용
        support : cluster sample 수
        """
        model.eval()

        all_features = []
        all_risks = []
        all_confidences = []
        all_probs = []
        all_logits = []

        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.cuda()
                features = model.extract_vector(inputs)
                logits = self._fc_logits_from_feature(model, features)
                logits = logits[:, : self._total_classes]
                probs = F.softmax(logits / self.anchor_temp, dim=1)

                confidence = probs.max(dim=1).values

                if probs.size(1) >= 2:
                    top2 = torch.topk(probs, k=2, dim=1).values
                    margin = top2[:, 0] - top2[:, 1]
                    risk = 1.0 - margin
                else:
                    risk = 1.0 - confidence

                all_features.append(features.detach().cpu())
                all_risks.append(risk.detach().cpu())
                all_confidences.append(confidence.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_logits.append(logits.detach().cpu())

        if len(all_features) == 0:
            raise RuntimeError("Cannot build anchor from an empty loader.")

        features = torch.cat(all_features, dim=0)
        risks = torch.cat(all_risks, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        probs_all = torch.cat(all_probs, dim=0)
        logits_all = torch.cat(all_logits, dim=0)

        m = min(int(self.anchor_per_task), features.size(0))

        # --------------------------------------------------------------
        # Overlap-aware candidate selection
        # --------------------------------------------------------------
        # 1) k-means representative 후보를 anchor_per_task보다 넉넉히 만든다.
        # 2) 각 후보의 대표성/support, risk, p_var를 계산한다.
        # 3) old anchors와의 cosine overlap이 큰 후보는 감점한다.
        # 4) 같은 current task 후보끼리도 너무 비슷한 것은 diversity penalty로 피한다.
        #
        # 이렇게 해도 클래스별 anchor는 아니며, current task feature 분포에서
        # task-level representative를 고르는 방식은 유지한다.
        candidate_k = min(
            features.size(0),
            max(m, int(np.ceil(float(m) * float(self.anchor_candidate_multiplier))))
        )
        candidate_idx = self._kmeans_representative_indices(features, num_clusters=candidate_k)

        x_norm = F.normalize(features.float(), dim=1)
        cand_norm = F.normalize(features[candidate_idx].float(), dim=1)
        cand_labels = torch.cdist(x_norm, cand_norm).argmin(dim=1)

        # old anchor matrix for overlap penalty. CPU tensor is enough because anchor building runs no grad.
        old_anchor_mat = None
        if client_id is not None and self._cur_task > 0:
            old_feats = []
            for a in self.client_anchors.get(client_id, []):
                if int(a.get("task", -1)) >= self._cur_task:
                    continue
                if a.get("u", None) is not None:
                    old_feats.append(a["u"].float())
                if self.anchor_overlap_use_boundary and a.get("boundary_u", None) is not None:
                    old_feats.append(a["boundary_u"].float())
            if len(old_feats) > 0:
                old_anchor_mat = F.normalize(torch.stack(old_feats, dim=0), dim=1)

        candidate_infos = []
        for cand_pos, idx in enumerate(candidate_idx):
            cluster_mask = cand_labels == cand_pos
            if cluster_mask.sum() == 0:
                cluster_mask[idx] = True

            cluster_probs = probs_all[cluster_mask]
            support = int(cluster_mask.sum().item())
            p_var = float(cluster_probs.var(dim=0).mean().item()) if support > 1 else 0.0
            risk_val = float(risks[idx].item())

            # representative score: support를 기본으로, 불확실/분산이 있는 region을 약간 선호한다.
            represent_score = float(np.log1p(max(support, 1)))
            quality_score = represent_score * (1.0 + 0.20 * max(risk_val, 0.0) + 0.20 * max(p_var, 0.0))

            overlap = 0.0
            if old_anchor_mat is not None:
                a_norm = F.normalize(features[idx].float().unsqueeze(0), dim=1)
                overlap = float(torch.matmul(a_norm, old_anchor_mat.t()).max().item())

            candidate_infos.append({
                "idx": int(idx),
                "cand_pos": int(cand_pos),
                "support": support,
                "risk": risk_val,
                "p_var": p_var,
                "quality": quality_score,
                "overlap": overlap,
            })

        selected_infos = []
        selected_idx = []
        selected_norms = []

        for _ in range(m):
            best = None
            best_score = None
            for info in candidate_infos:
                if info["idx"] in selected_idx:
                    continue

                diversity_penalty = 0.0
                if len(selected_norms) > 0:
                    a_norm = F.normalize(features[info["idx"]].float().unsqueeze(0), dim=1)
                    selected_mat = torch.cat(selected_norms, dim=0)
                    diversity_penalty = float(torch.matmul(a_norm, selected_mat.t()).max().item())

                score = (
                    float(info["quality"])
                    - float(self.anchor_overlap_lambda) * float(info["overlap"])
                    - float(self.anchor_diversity_lambda) * float(diversity_penalty)
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best = dict(info)
                    best["score"] = float(score)
                    best["diversity_penalty"] = float(diversity_penalty)

            if best is None:
                break
            selected_infos.append(best)
            selected_idx.append(int(best["idx"]))
            selected_norms.append(F.normalize(features[int(best["idx"])].float().unsqueeze(0), dim=1))

        if len(selected_idx) == 0:
            selected_idx = candidate_idx[:m]
            selected_infos = [info for info in candidate_infos if info["idx"] in selected_idx]

        if self.debug_anchor_selection:
            dbg = []
            for info in selected_infos:
                dbg.append({
                    "idx": int(info.get("idx", -1)),
                    "support": int(info.get("support", 0)),
                    "risk": round(float(info.get("risk", 0.0)), 4),
                    "pvar": round(float(info.get("p_var", 0.0)), 6),
                    "quality": round(float(info.get("quality", 0.0)), 4),
                    "old_overlap": round(float(info.get("overlap", 0.0)), 4),
                    "dup": round(float(info.get("diversity_penalty", 0.0)), 4),
                    "score": round(float(info.get("score", 0.0)), 4),
                })
            print("[ANCHOR-SELECTION] Task {}, Client {} | candidates={} selected={}".format(
                self._cur_task, client_id, len(candidate_infos), dbg
            ))

        # Re-assign every sample to the final selected representative anchors for compact region summary.
        rep_norm = F.normalize(features[selected_idx].float(), dim=1)
        labels = torch.cdist(x_norm, rep_norm).argmin(dim=1)

        anchor_items = []
        for cluster_id, idx in enumerate(selected_idx):
            u = features[idx]
            selection_risk = float(risks[idx].item())
            confidence = float(confidences[idx].item())

            cluster_mask = labels == cluster_id
            if cluster_mask.sum() == 0:
                cluster_mask[idx] = True

            cluster_features = features[cluster_mask]
            cluster_probs = probs_all[cluster_mask]
            support = int(cluster_features.size(0))

            # Feature-region summary. Use scalar values to keep memory compact.
            diff = cluster_features.float() - u.float().unsqueeze(0)
            dist = torch.norm(diff, dim=1)
            radius = float(dist.mean().item()) if dist.numel() > 0 else 0.0
            sigma = float(diff.std(dim=0).mean().item()) if support > 1 else max(radius, 1e-6)
            p_var = float(cluster_probs.var(dim=0).mean().item()) if support > 1 else 0.0

            # Boundary proxy inside this cluster.
            # 클래스별 anchor를 만들지 않고, cluster 내부에서 현재 모델이 가장 불확실한 점
            # = margin이 가장 작은 점을 하나 저장한다.
            cluster_indices = torch.where(cluster_mask)[0]
            cluster_risks = risks[cluster_indices]
            boundary_local_pos = int(torch.argmax(cluster_risks).item())
            boundary_idx = int(cluster_indices[boundary_local_pos].item())

            boundary_u = features[boundary_idx]
            boundary_prob = probs_all[boundary_idx]
            boundary_logits = logits_all[boundary_idx]

            if boundary_prob.numel() >= 2:
                b_top2_vals, b_top2_idx = torch.topk(boundary_prob, k=2)
                boundary_margin = float((b_top2_vals[0] - b_top2_vals[1]).item())
                boundary_top1 = int(b_top2_idx[0].item())
                boundary_top2 = int(b_top2_idx[1].item())
                boundary_risk = float(1.0 - boundary_margin)
            else:
                boundary_top1 = int(torch.argmax(boundary_prob).item())
                boundary_top2 = boundary_top1
                boundary_margin = float(boundary_prob.max().item())
                boundary_risk = float(1.0 - boundary_margin)

            if boundary_top1 != boundary_top2:
                boundary_logit_margin = float((boundary_logits[boundary_top1] - boundary_logits[boundary_top2]).item())
            else:
                boundary_logit_margin = 0.0

            with torch.no_grad():
                logits_u = self._fc_logits_from_feature(
                    model,
                    u.cuda().unsqueeze(0)
                ).squeeze(0)

                logits_u = logits_u[: self._total_classes]
                p = F.softmax(logits_u / self.anchor_temp, dim=0)
                p = self._normalize_prob(p).detach().cpu()

                logits_b = self._fc_logits_from_feature(
                    model,
                    boundary_u.cuda().unsqueeze(0)
                ).squeeze(0)
                logits_b = logits_b[: self._total_classes]
                p_boundary = F.softmax(logits_b / self.anchor_temp, dim=0)
                p_boundary = self._normalize_prob(p_boundary).detach().cpu()

            anchor_items.append(
                {
                    "u": u.detach().cpu(),
                    "p": p,
                    "confidence": confidence,
                    "selection_risk": selection_risk,
                    "radius": radius,
                    "sigma": sigma,
                    "p_var": p_var,
                    "support": support,
                    "boundary_u": boundary_u.detach().cpu(),
                    "boundary_p": p_boundary,
                    "boundary_risk": boundary_risk,
                    "boundary_margin": boundary_margin,
                    "boundary_logit_margin": boundary_logit_margin,
                    "boundary_top1": boundary_top1,
                    "boundary_top2": boundary_top2,
                    "topk": m,
                    "selection_type": "overlap_aware_distribution_kmeans",
                }
            )

        return anchor_items

    # ------------------------------------------------------------------
    # Anchor scoring / memory update
    # ------------------------------------------------------------------
    def _compute_anchor_scores(self, client_id, local_model):
        """Compute FR, LS, and rank-based importance for one client's anchors."""
        anchors = self.client_anchors.get(client_id, [])
        if len(anchors) == 0:
            return []

        # 1) Raw FR/LS/Age 계산
        for anchor in anchors:
            u = anchor["u"]
            p_saved = anchor["p"]

            p_local_now = self._predict_anchor_prob(local_model, u)
            p_global_now = self._predict_anchor_prob(self._network, u)

            seen_dim = min(p_saved.shape[0], p_local_now.shape[0], p_global_now.shape[0])

            p_saved_ = self._normalize_prob(p_saved[:seen_dim])
            p_local_ = self._normalize_prob(p_local_now[:seen_dim])
            p_global_ = self._normalize_prob(p_global_now[:seen_dim])

            ls = self._kl_prob(p_global_, p_local_)
            fr = torch.tensor(0.0)
            if self._cur_task > anchor["task"]:
                fr = self._kl_prob(p_saved_, p_local_)

            anchor["ls"] = max(float(ls), 0.0)
            anchor["fr"] = max(float(fr), 0.0)
            anchor["age"] = max(int(self._cur_task - anchor["task"]), 0)

        # 2) Rank 기반 importance 계산
        fr_rank = self._percentile_ranks([a.get("fr", 0.0) for a in anchors])
        ls_rank = self._percentile_ranks([a.get("ls", 0.0) for a in anchors])
        age_rank = self._percentile_ranks([a.get("age", 0.0) for a in anchors])
        risk_rank = self._percentile_ranks([a.get("selection_risk", 0.0) for a in anchors])

        for i, anchor in enumerate(anchors):
            # 중요: FR은 실제 forgetting drop과 불일치하는 반례가 있어 더 이상
            # pruning/loss weight의 기준으로 사용하지 않는다.
            # importance 필드는 기존 로그 호환을 위해 FR 값을 그대로 보관하는 diagnostic 값이다.
            importance = max(float(anchor.get("fr", 0.0)), 0.0)

            anchor["fr_rank"] = float(fr_rank[i])
            anchor["ls_rank"] = float(ls_rank[i])
            anchor["age_rank"] = float(age_rank[i])
            anchor["risk_rank"] = float(risk_rank[i])
            anchor["importance"] = float(importance)

        return self._anchor_score_summary(client_id)

    def _anchor_score_summary(self, client_id):
        scores = []
        for a in self.client_anchors.get(client_id, []):
            scores.append(
                {
                    "task": int(a["task"]),
                    "conf": round(float(a.get("confidence", 0.0)), 4),
                    "risk": round(float(a.get("selection_risk", 0.0)), 4),
                    "FR": round(float(a.get("fr", 0.0)), 6),
                    "LS": round(float(a.get("ls", 0.0)), 6),
                    "age": int(a.get("age", 0)),
                    "I": round(float(a.get("importance", 0.0)), 6),
                    "rad": round(float(a.get("radius", 0.0)), 4),
                    "sig": round(float(a.get("sigma", 0.0)), 4),
                    "pvar": round(float(a.get("p_var", 0.0)), 6),
                    "sup": int(a.get("support", 1)),
                    "br": round(float(a.get("boundary_risk", 0.0)), 4),
                    "bm": round(float(a.get("boundary_margin", 0.0)), 4),
                }
            )
        return scores


    def _anchor_pool_task_summary(self, client_id):
        """Compact task-wise summary for checking which task anchors survive pruning."""
        anchors = self.client_anchors.get(client_id, [])
        out = {}
        for a in anchors:
            t = int(a.get("task", -1))
            if t not in out:
                out[t] = {"count": 0, "FR": [], "LS": [], "I": [], "radius": [], "support": []}
            out[t]["count"] += 1
            out[t]["FR"].append(float(a.get("fr", 0.0)))
            out[t]["LS"].append(float(a.get("ls", 0.0)))
            out[t]["I"].append(float(a.get("importance", 0.0)))
            out[t]["radius"].append(float(a.get("radius", 0.0)))
            out[t]["support"].append(float(a.get("support", 1)))

        summary = {}
        for t, v in sorted(out.items()):
            summary[t] = {
                "count": v["count"],
                "FR_mean": round(float(np.mean(v["FR"])), 6) if v["FR"] else 0.0,
                "LS_mean": round(float(np.mean(v["LS"])), 6) if v["LS"] else 0.0,
                "I_mean": round(float(np.mean(v["I"])), 6) if v["I"] else 0.0,
                "radius_mean": round(float(np.mean(v["radius"])), 4) if v.get("radius") else 0.0,
                "support_sum": int(np.sum(v["support"])) if v.get("support") else 0,
            }
        return summary

    def _anchor_prune_score(self, anchor):
        """FR-free score used only to break ties inside task-balanced memory.

        FR is unreliable as a forgetting proxy in current logs, so memory retention
        should primarily preserve task coverage. This score only decides which
        anchors to keep when a task has more anchors than its quota.
        """
        support = max(float(anchor.get("support", 1)), 1.0)
        risk = max(float(anchor.get("selection_risk", 0.0)), 0.0)
        p_var = max(float(anchor.get("p_var", 0.0)), 0.0)
        boundary_risk = max(float(anchor.get("boundary_risk", 0.0)), 0.0)

        # Coverage first, with small bonuses for boundary/unstable representative regions.
        return float(np.log1p(support) * (1.0 + 0.20 * risk + 0.30 * boundary_risk + 0.20 * p_var))

    def _prune_anchor_memory_task_balanced(self, client_id):
        """Keep anchors with task-balanced coverage instead of FR-based pruning.

        Motivation: logs showed cases such as drop≈99 but FR≈0.01 and same_pred=100%.
        Therefore FR can not decide which anchors are important. We keep at least a
        small quota per task whenever the budget allows, and fill remaining slots by
        a FR-free coverage score.
        """
        anchors = self.client_anchors.get(client_id, [])
        if len(anchors) <= self.anchor_budget:
            return

        groups = {}
        for a in anchors:
            groups.setdefault(int(a.get("task", -1)), []).append(a)

        task_ids = sorted(groups.keys())
        if len(task_ids) == 0:
            self.client_anchors[client_id] = []
            return

        # If there are many tasks, lower the per-task minimum so the total budget is respected.
        per_task_min = int(self.task_anchor_min)
        per_task_min = max(1, min(per_task_min, max(1, self.anchor_budget // len(task_ids))))

        keep = []
        kept_ids = set()

        # 1) Guarantee coverage for each task.
        for t in task_ids:
            candidates = sorted(groups[t], key=self._anchor_prune_score, reverse=True)
            for a in candidates[:per_task_min]:
                if len(keep) >= self.anchor_budget:
                    break
                keep.append(a)
                kept_ids.add(id(a))

        # 2) Fill remaining budget with FR-free score.
        if len(keep) < self.anchor_budget:
            rest = [a for a in anchors if id(a) not in kept_ids]
            rest = sorted(rest, key=self._anchor_prune_score, reverse=True)
            keep.extend(rest[: self.anchor_budget - len(keep)])

        self.client_anchors[client_id] = keep[: self.anchor_budget]

    def _update_anchor_memory(self, client_id, task_id, u, p, local_model, confidence=0.0, selection_risk=0.0, radius=0.0, sigma=0.0, p_var=0.0, support=1, boundary_u=None, boundary_p=None, boundary_risk=0.0, boundary_margin=0.0, boundary_logit_margin=0.0, boundary_top1=-1, boundary_top2=-1):
        if client_id not in self.client_anchors:
            self.client_anchors[client_id] = []

        new_anchor = {
            "task": int(task_id),
            "u": u.detach().cpu(),
            "p": p.detach().cpu(),
            "confidence": float(confidence),
            "selection_risk": float(selection_risk),
            "radius": float(radius),
            "sigma": float(sigma),
            "p_var": float(p_var),
            "support": int(support),
            "boundary_u": None if boundary_u is None else boundary_u.detach().cpu(),
            "boundary_p": None if boundary_p is None else boundary_p.detach().cpu(),
            "boundary_risk": float(boundary_risk),
            "boundary_margin": float(boundary_margin),
            "boundary_logit_margin": float(boundary_logit_margin),
            "boundary_top1": int(boundary_top1),
            "boundary_top2": int(boundary_top2),
            "ls": 0.0,
            "fr": 0.0,
            "age": 0,
            "importance": 0.0,
        }

        self.client_anchors[client_id].append(new_anchor)
        self._compute_anchor_scores(client_id, local_model)

        # Budget pruning
        # FR은 실제 forgetting과 불일치할 수 있으므로 memory retention 기준에서 제외한다.
        # 대신 task-balanced coverage를 보장하고, 같은 task 안에서는 support/risk/p_var 기반
        # FR-free score로 대표성이 큰 anchor를 우선 유지한다.
        self._prune_anchor_memory_task_balanced(client_id)

        # pruning 이후 rank를 다시 맞춰둔다.
        self._compute_anchor_scores(client_id, local_model)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _old_anchor_weights(self, client_id, device="cuda"):
        """Return old anchors and normalized FR-free weights for loss terms.

        FR is kept only as a diagnostic value. For training, use task-balanced
        weights so that one task with high anchor KL/FR does not dominate and
        anchors with falsely low FR are still preserved.
        """
        anchors = [a for a in self.client_anchors.get(client_id, []) if a.get("task", -1) < self._cur_task]
        if len(anchors) == 0:
            return anchors, None

        mode = str(self.anchor_weight_mode).lower()

        if mode in ["support", "sqrt_support", "uniform"]:
            if mode == "support":
                vals = [max(float(a.get("support", 1)), 1.0) for a in anchors]
            elif mode == "sqrt_support":
                vals = [np.sqrt(max(float(a.get("support", 1)), 1.0)) for a in anchors]
            else:
                vals = [1.0 for _ in anchors]

            weights = torch.tensor(vals, dtype=torch.float32, device=device)
            weights = weights / torch.clamp(weights.sum(), min=1e-12)
            return anchors, weights

        # Default: each old task receives the same total weight, regardless of anchor count.
        # Within each task, either uniform or sqrt-support weights are used.
        task_to_indices = {}
        for i, a in enumerate(anchors):
            task_to_indices.setdefault(int(a.get("task", -1)), []).append(i)

        weights = torch.zeros(len(anchors), dtype=torch.float32, device=device)
        num_tasks = max(len(task_to_indices), 1)

        for _, indices in task_to_indices.items():
            if mode == "task_sqrt_support":
                vals = torch.tensor(
                    [np.sqrt(max(float(anchors[i].get("support", 1)), 1.0)) for i in indices],
                    dtype=torch.float32,
                    device=device,
                )
                vals = vals / torch.clamp(vals.sum(), min=1e-12)
            else:
                vals = torch.ones(len(indices), dtype=torch.float32, device=device) / max(len(indices), 1)

            for local_j, anchor_idx in enumerate(indices):
                weights[anchor_idx] = vals[local_j] / num_tasks

        weights = weights / torch.clamp(weights.sum(), min=1e-12)
        return anchors, weights

    def _anchor_loss(self, model, client_id):
        anchors = [a for a in self.client_anchors.get(client_id, []) if a.get("task", -1) < self._cur_task]
        if len(anchors) == 0 or self.anchor_lambda <= 0:
            return torch.tensor(0.0, device="cuda")

        # FR이 아니라 task-balanced/support 기반 weight를 사용한다.
        anchors, weights = self._old_anchor_weights(client_id, device="cuda")

        loss = torch.tensor(0.0, device="cuda")
        debug_items = []

        for w, anchor in zip(weights, anchors):
            u = anchor["u"].cuda().unsqueeze(0)
            p_saved = anchor["p"].cuda()

            logits = self._fc_logits_from_feature(model, u).squeeze(0)
            min_dim = min(p_saved.shape[0], logits.shape[0], self._total_classes)

            p_saved_ = self._normalize_prob(p_saved[:min_dim])
            log_p_now = F.log_softmax(logits[:min_dim] / self.anchor_temp, dim=0)
            center_kl = F.kl_div(log_p_now, p_saved_, reduction="sum")

            # Distribution-aware region preservation.
            # u 한 점만 맞추는 것을 막기 위해, 저장된 cluster radius/sigma를 이용해
            # u 주변의 작은 feature region에서도 같은 saved distribution을 유지시킨다.
            region_kl = torch.tensor(0.0, device="cuda")
            n_region = int(self.anchor_region_samples)
            sigma = float(anchor.get("sigma", 0.0))
            radius = float(anchor.get("radius", 0.0))
            scale = max(sigma, radius * float(self.anchor_region_noise), 0.0)

            if n_region > 0 and scale > 1e-8 and self.anchor_region_lambda > 0:
                # scale이 너무 큰 경우 feature norm 기준으로 clamp해서 perturbation 폭주 방지.
                u_norm = torch.norm(u.detach()).item()
                max_scale = max(1e-6, 0.05 * u_norm)
                scale = min(scale, max_scale)

                noise = torch.randn(n_region, u.size(1), device="cuda") * scale
                u_region = u.repeat(n_region, 1) + noise
                logits_region = self._fc_logits_from_feature(model, u_region)
                logits_region = logits_region[:, :min_dim]
                log_p_region = F.log_softmax(logits_region / self.anchor_temp, dim=1)
                p_target = p_saved_.unsqueeze(0).expand(n_region, -1)
                region_kl = F.kl_div(log_p_region, p_target, reduction="batchmean")

            kl = center_kl + float(self.anchor_region_lambda) * region_kl
            loss += w * kl

            if self.debug_anchor_prob > 0:
                with torch.no_grad():
                    saved_pred = int(p_saved_[:min_dim].argmax().item())
                    now_pred = int(logits[:min_dim].argmax().item())
                    debug_items.append(
                        {
                            "task": int(anchor["task"]),
                            "w": round(float(w.item()), 4),
                            "I": round(float(anchor.get("importance", 0.0)), 4),
                            "kl": round(float(kl.detach().item()), 4),
                            "center": round(float(center_kl.detach().item()), 4),
                            "region": round(float(region_kl.detach().item()), 4),
                            "rad": round(float(anchor.get("radius", 0.0)), 4),
                            "br": round(float(anchor.get("boundary_risk", 0.0)), 4),
                            "sup": int(anchor.get("support", 1)),
                            "saved": saved_pred,
                            "now": now_pred,
                        }
                    )

        if self.debug_anchor_prob > 0 and self._cur_task > 0 and torch.rand(1).item() < self.debug_anchor_prob:
            print("[ANCHOR] Task {}, Client {} | {}".format(self._cur_task, client_id, debug_items))

        return loss

    def _anchor_batch_contrastive_loss(self, model, inputs, client_id):
        """Anchor-based contrastive loss using current batch center as positive.

        Why this loss?
        --------------
        Diagnostics showed that old/new feature regions are highly overlapped:
        old images can be close to old anchors but even closer to current/new anchors.
        Absolute separation is too harsh because cosine similarities are already high.

        During local training, current-task anchors are not created yet. Therefore:
            positive = normalized current mini-batch mean feature
            negatives = stored old anchors

        The loss is InfoNCE-style:
            -log exp(sim(z, pos)/tau) / [exp(sim(z, pos)/tau) + sum exp(sim(z, old_anchor)/tau)]

        This does not use class-wise anchors and does not store raw data.
        """
        if self._cur_task == 0 or self.anchor_contrast_lambda <= 0:
            return torch.tensor(0.0, device="cuda")

        anchors = [a for a in self.client_anchors.get(client_id, []) if a.get("task", -1) < self._cur_task]
        if len(anchors) == 0:
            return torch.tensor(0.0, device="cuda")

        negatives = []
        for a in anchors:
            if a.get("u", None) is not None:
                negatives.append(a["u"])
            if self.anchor_contrast_use_boundary and a.get("boundary_u", None) is not None:
                negatives.append(a["boundary_u"])

        if len(negatives) == 0:
            return torch.tensor(0.0, device="cuda")

        neg_feats = torch.stack([x.float() for x in negatives], dim=0).cuda()
        neg_feats = F.normalize(neg_feats, dim=1)

        cur_feats = model.extract_vector(inputs)
        cur_feats = F.normalize(cur_feats, dim=1)

        # task-level positive proxy from the current mini-batch
        center_source = cur_feats.detach() if self.anchor_contrast_detach_center else cur_feats
        pos = F.normalize(center_source.mean(dim=0, keepdim=True), dim=1)

        tau = max(float(self.anchor_contrast_temp), 1e-6)
        pos_logit = torch.matmul(cur_feats, pos.t()) / tau          # [B, 1]
        neg_logits = torch.matmul(cur_feats, neg_feats.t()) / tau   # [B, K]

        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _compute_loss(self, model, inputs, targets, client_id, feature_teacher=None):
        logits = model(inputs)["logits"]

        if self._cur_task > 0:
            fake_targets = targets - self._known_classes
            ce_loss = F.cross_entropy(
                logits[:, self._known_classes:self._total_classes],
                fake_targets,
            )
        else:
            ce_loss = F.cross_entropy(logits[:, :self._total_classes], targets)

        anchor_loss = self._anchor_loss(model, client_id)
        contrast_loss = self._anchor_batch_contrastive_loss(model, inputs, client_id)

        loss = (
            ce_loss
            + self.anchor_lambda * anchor_loss
            + self.anchor_contrast_lambda * contrast_loss
        )

        return loss, ce_loss.detach(), anchor_loss.detach(), contrast_loss.detach()

    # ------------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------------
    def _collect_old_error_stats(self, model, loader):
        stats = {"old_acc": 0.0, "old_to_new": 0.0, "old_to_wrong_old": 0.0, "total_old": 0}
        if self._cur_task == 0:
            return stats

        model.eval()
        total_old = correct_old = old_to_new = old_to_wrong_old = 0

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                mask = targets < self._known_classes
                if mask.sum() == 0:
                    continue

                inputs = inputs[mask]
                targets = targets[mask]
                logits = model(inputs)["logits"][:, :self._total_classes]
                preds = logits.argmax(dim=1)

                total_old += targets.size(0)
                correct_old += (preds == targets).sum().item()
                old_to_new += (preds >= self._known_classes).sum().item()
                old_to_wrong_old += ((preds < self._known_classes) & (preds != targets)).sum().item()

        if total_old > 0:
            stats["old_acc"] = 100.0 * correct_old / total_old
            stats["old_to_new"] = 100.0 * old_to_new / total_old
            stats["old_to_wrong_old"] = 100.0 * old_to_wrong_old / total_old
            stats["total_old"] = total_old
        return stats
    
    def _anchor_retention_stats(self, model, client_id):
        anchors = [a for a in self.client_anchors.get(client_id, []) if a["task"] < self._cur_task]
        if len(anchors) == 0:
            return {}

        same_pred = 0
        total = 0
        mean_kl = 0.0

        model.eval()
        with torch.no_grad():
            for a in anchors:
                u = a["u"].cuda().unsqueeze(0)
                p_saved = a["p"].cuda()

                logits = self._fc_logits_from_feature(model, u).squeeze(0)
                min_dim = min(p_saved.shape[0], logits.shape[0], self._total_classes)

                p_saved_ = self._normalize_prob(p_saved[:min_dim])
                p_now = F.softmax(logits[:min_dim] / self.anchor_temp, dim=0)

                saved_pred = int(p_saved_.argmax().item())
                now_pred = int(p_now.argmax().item())

                kl = F.kl_div(
                    torch.log(torch.clamp(p_now, min=1e-8)),
                    p_saved_,
                    reduction="sum",
                )

                same_pred += int(saved_pred == now_pred)
                mean_kl += float(kl.item())
                total += 1

        return {
            "anchor_same_pred": 100.0 * same_pred / total,
            "anchor_mean_kl": mean_kl / total,
            "num_old_anchors": total,
        }


    def _task_class_range(self, task_id):
        """Return [start, end) class range for a task.

        CIFAR10 5-task 설정처럼 init_cls == increment인 경우도 되고,
        첫 task class 수와 이후 increment가 다른 일반 설정도 최대한 처리한다.
        """
        init_cls = int(self.args.get("init_cls", getattr(self, "each_task", 0)))
        increment = int(self.args.get("increment", getattr(self, "each_task", init_cls)))

        if task_id == 0:
            start, end = 0, init_cls
        else:
            start = init_cls + (int(task_id) - 1) * increment
            end = start + increment

        end = min(end, self._total_classes)
        return start, end

    def _eval_model_task_accs(self, model, loader):
        """Evaluate accuracy for each seen task on a client's local test split."""
        model.eval()
        correct = {t: 0 for t in range(self._cur_task + 1)}
        total = {t: 0 for t in range(self._cur_task + 1)}

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                logits = model(inputs)["logits"][:, : self._total_classes]
                preds = logits.argmax(dim=1)

                for t in range(self._cur_task + 1):
                    start, end = self._task_class_range(t)
                    if end <= start:
                        continue
                    mask = (targets >= start) & (targets < end)
                    if mask.sum() == 0:
                        continue
                    total[t] += int(mask.sum().item())
                    correct[t] += int((preds[mask] == targets[mask]).sum().item())

        accs = {}
        for t in range(self._cur_task + 1):
            accs[t] = 100.0 * correct[t] / total[t] if total[t] > 0 else None
        return accs

    def _store_current_task_baseline_acc(self, client_id, local_model, local_loader):
        """Save the accuracy right after learning the current task.

        Later tasks compare against this value:
            forgetting_drop = baseline_acc_at_task_end - current_acc
        """
        if client_id not in self.client_task_baseline_acc:
            self.client_task_baseline_acc[client_id] = {}

        task_accs = self._eval_model_task_accs(local_model, local_loader)
        cur_acc = task_accs.get(self._cur_task, None)
        if cur_acc is not None:
            self.client_task_baseline_acc[client_id][int(self._cur_task)] = float(cur_acc)

    def _anchor_signal_by_task(self, model, client_id):
        """Aggregate old-anchor FR/LS/I and anchor prediction retention by task."""
        anchors = [
            a for a in self.client_anchors.get(client_id, [])
            if int(a.get("task", -1)) < self._cur_task
        ]
        if len(anchors) == 0:
            return {}

        out = {}
        model.eval()
        with torch.no_grad():
            for a in anchors:
                t = int(a.get("task", -1))
                out.setdefault(t, {"FR": [], "LS": [], "I": [], "same": 0, "kl": [], "n": 0})

                u = a["u"].cuda().unsqueeze(0)
                p_saved = a["p"].cuda()
                logits = self._fc_logits_from_feature(model, u).squeeze(0)
                min_dim = min(p_saved.shape[0], logits.shape[0], self._total_classes)

                p_saved_ = self._normalize_prob(p_saved[:min_dim])
                p_now = F.softmax(logits[:min_dim] / self.anchor_temp, dim=0)
                p_now = self._normalize_prob(p_now)

                saved_pred = int(p_saved_.argmax().item())
                now_pred = int(p_now.argmax().item())
                kl = F.kl_div(torch.log(torch.clamp(p_now, min=1e-8)), p_saved_, reduction="sum")

                out[t]["FR"].append(float(a.get("fr", 0.0)))
                out[t]["LS"].append(float(a.get("ls", 0.0)))
                out[t]["I"].append(float(a.get("importance", 0.0)))
                out[t]["same"] += int(saved_pred == now_pred)
                out[t]["kl"].append(float(kl.item()))
                out[t]["n"] += 1

        summary = {}
        for t, v in sorted(out.items()):
            n = max(int(v["n"]), 1)
            summary[t] = {
                "n": int(v["n"]),
                "FR_mean": float(np.mean(v["FR"])) if v["FR"] else 0.0,
                "LS_mean": float(np.mean(v["LS"])) if v["LS"] else 0.0,
                "I_mean": float(np.mean(v["I"])) if v["I"] else 0.0,
                "same_pred": 100.0 * float(v["same"]) / n,
                "anchor_KL": float(np.mean(v["kl"])) if v["kl"] else 0.0,
            }
        return summary

    def _debug_fr_signal_vs_forgetting(self, client_id, local_model, local_loader):
        """Print the direct check: does FR increase when real task accuracy drops?

        For each old task, this prints:
        - baseline: client accuracy right after that task was learned
        - now: current local accuracy on that task
        - drop: baseline - now, i.e. actual forgetting proxy
        - FR/LS/I: current anchor signal for that same task
        If FR_mean and drop move together across tasks/clients, FR is meaningful.
        """
        if self._cur_task == 0:
            return

        self._compute_anchor_scores(client_id, local_model)
        local_task_accs = self._eval_model_task_accs(local_model, local_loader)
        global_task_accs = self._eval_model_task_accs(self._network, local_loader)
        anchor_by_task = self._anchor_signal_by_task(local_model, client_id)
        baselines = self.client_task_baseline_acc.get(client_id, {})

        print("\n[FR-FORGET-CHECK] Task {}, Client {}".format(self._cur_task, client_id))
        print("  task | base_acc | local_now | drop | global_now | FR | LS | I | same_pred | anchor_KL | n")

        for t in range(self._cur_task):
            base = baselines.get(t, None)
            local_now = local_task_accs.get(t, None)
            global_now = global_task_accs.get(t, None)
            sig = anchor_by_task.get(t, {})

            if base is None or local_now is None:
                drop = None
            else:
                drop = float(base) - float(local_now)

            def fmt(x):
                return "NA" if x is None else "{:.2f}".format(float(x))

            print(
                "  {:>4} | {:>8} | {:>9} | {:>5} | {:>10} | {:>6.4f} | {:>6.4f} | {:>6.4f} | {:>9.2f} | {:>9.4f} | {:>1}".format(
                    t,
                    fmt(base),
                    fmt(local_now),
                    fmt(drop),
                    fmt(global_now),
                    float(sig.get("FR_mean", 0.0)),
                    float(sig.get("LS_mean", 0.0)),
                    float(sig.get("I_mean", 0.0)),
                    float(sig.get("same_pred", 0.0)),
                    float(sig.get("anchor_KL", 0.0)),
                    int(sig.get("n", 0)),
                )
            )



    def _debug_anchor_signal_vs_perf(self, client_id, local_model, local_loader):
        if self._cur_task == 0:
            return

        anchor_scores = self._compute_anchor_scores(client_id, local_model)
        anchor_stats = self._anchor_retention_stats(local_model, client_id)
        local_grouped = self._eval_model_grouped(local_model, local_loader)["grouped"]
        global_grouped = self._eval_model_grouped(self._network, local_loader)["grouped"]
        local_old = self._collect_old_error_stats(local_model, local_loader)
        global_old = self._collect_old_error_stats(self._network, local_loader)

        print("\n[SIGNAL-CHECK] Task {}, Client {}".format(self._cur_task, client_id))
        print(
            "  Local  total/old/new: {:.2f}/{:.2f}/{:.2f} | old_to_new: {:.2f} | old_wrong_old: {:.2f}".format(
                float(local_grouped.get("total", 0.0)),
                float(local_grouped.get("old", 0.0)),
                float(local_grouped.get("new", 0.0)),
                local_old["old_to_new"],
                local_old["old_to_wrong_old"],
            )
        )
        print(
            "  Global total/old/new: {:.2f}/{:.2f}/{:.2f} | old_to_new: {:.2f} | old_wrong_old: {:.2f}".format(
                float(global_grouped.get("total", 0.0)),
                float(global_grouped.get("old", 0.0)),
                float(global_grouped.get("new", 0.0)),
                global_old["old_to_new"],
                global_old["old_to_wrong_old"],
            )
        )
        print(
            "  Anchor-Retention | same_pred: {:.2f} | mean_KL: {:.4f} | num_old: {}".format(
                anchor_stats.get("anchor_same_pred", -1),
                anchor_stats.get("anchor_mean_kl", -1),
                anchor_stats.get("num_old_anchors", 0),
            )
        )
        print("  Anchors: {}".format(anchor_scores))


    def _feature_anchor_matrix(self, anchors, use_boundary=True):
        """Return normalized anchor feature matrix and task ids.

        This is diagnostic-only. It lets us compare real image features with
        stored old anchors and temporary current-task anchors.
        """
        feats = []
        tasks = []
        kinds = []
        for a in anchors:
            t = int(a.get("task", -1))
            if a.get("u", None) is not None:
                feats.append(a["u"].float())
                tasks.append(t)
                kinds.append("center")
            if use_boundary and a.get("boundary_u", None) is not None:
                feats.append(a["boundary_u"].float())
                tasks.append(t)
                kinds.append("boundary")

        if len(feats) == 0:
            return None, [], []

        mat = torch.stack(feats, dim=0).cuda()
        mat = F.normalize(mat, dim=1)
        return mat, tasks, kinds

    def _safe_sim_stats(self, values):
        if values is None or len(values) == 0:
            return {"mean": None, "p25": None, "p50": None, "p75": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }

    def _debug_feature_overlap_diagnosis(self, client_id, local_model, local_loader, current_anchor_items=None):
        """Deep diagnostic for the real failure mode.

        This answers three concrete questions:
        1) Do real old images still map near old anchors?
        2) Do real old images map closer to current-task anchors than old anchors?
        3) Do current-task images invade old anchor regions?

        Important: call this AFTER local update and BEFORE adding current anchors to memory.
        current_anchor_items may be built from the current task train loader, but must not
        yet be inserted into self.client_anchors.
        """
        if self._cur_task == 0:
            return

        old_anchors = [a for a in self.client_anchors.get(client_id, []) if int(a.get("task", -1)) < self._cur_task]
        if len(old_anchors) == 0:
            print("\n[FEATURE-OVERLAP-CHECK] Task {}, Client {} | no old anchors".format(self._cur_task, client_id))
            return

        tmp_new_anchors = []
        if current_anchor_items is not None:
            for item in current_anchor_items:
                item_copy = dict(item)
                item_copy["task"] = int(self._cur_task)
                tmp_new_anchors.append(item_copy)

        old_mat_all, old_tasks, old_kinds = self._feature_anchor_matrix(old_anchors, use_boundary=True)
        new_mat_all, _, _ = self._feature_anchor_matrix(tmp_new_anchors, use_boundary=True)

        if old_mat_all is None:
            print("\n[FEATURE-OVERLAP-CHECK] Task {}, Client {} | empty old anchor matrix".format(self._cur_task, client_id))
            return

        old_tasks_arr = np.asarray(old_tasks, dtype=np.int64)

        local_model.eval()

        # Per-task old sample diagnostics.
        per_task = {
            t: {
                "n": 0,
                "correct": 0,
                "old_to_new": 0,
                "old_to_wrong_old": 0,
                "old_anchor_sim": [],
                "same_task_anchor_sim": [],
                "new_anchor_sim": [],
                "new_closer_than_old": 0,
                "new_closer_than_same_task": 0,
            }
            for t in range(self._cur_task)
        }

        # Current-task new sample overlap with old anchors.
        new_stats = {
            "n": 0,
            "old_anchor_sim": [],
            "over_sep_th": 0,
            "over_050": 0,
            "over_070": 0,
        }

        with torch.no_grad():
            for _, inputs, targets in local_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()

                feats = local_model.extract_vector(inputs)
                feats = F.normalize(feats, dim=1)
                logits = local_model(inputs)["logits"][:, : self._total_classes]
                preds = logits.argmax(dim=1)

                for t in range(self._cur_task + 1):
                    start, end = self._task_class_range(t)
                    if end <= start:
                        continue

                    mask = (targets >= start) & (targets < end)
                    if mask.sum() == 0:
                        continue

                    f_t = feats[mask]
                    y_t = targets[mask]
                    p_t = preds[mask]

                    sim_old_all = torch.matmul(f_t, old_mat_all.t()).max(dim=1).values

                    if t < self._cur_task:
                        d = per_task[t]
                        d["n"] += int(mask.sum().item())
                        d["correct"] += int((p_t == y_t).sum().item())
                        d["old_to_new"] += int((p_t >= self._known_classes).sum().item())
                        d["old_to_wrong_old"] += int(((p_t < self._known_classes) & (p_t != y_t)).sum().item())
                        d["old_anchor_sim"].extend(sim_old_all.detach().cpu().tolist())

                        same_idx = np.where(old_tasks_arr == int(t))[0]
                        if len(same_idx) > 0:
                            same_mat = old_mat_all[torch.tensor(same_idx, dtype=torch.long, device="cuda")]
                            same_sim = torch.matmul(f_t, same_mat.t()).max(dim=1).values
                            d["same_task_anchor_sim"].extend(same_sim.detach().cpu().tolist())
                        else:
                            same_sim = sim_old_all
                            d["same_task_anchor_sim"].extend(sim_old_all.detach().cpu().tolist())

                        if new_mat_all is not None:
                            sim_new = torch.matmul(f_t, new_mat_all.t()).max(dim=1).values
                            d["new_anchor_sim"].extend(sim_new.detach().cpu().tolist())
                            d["new_closer_than_old"] += int((sim_new > sim_old_all).sum().item())
                            d["new_closer_than_same_task"] += int((sim_new > same_sim).sum().item())

                    else:
                        new_stats["n"] += int(mask.sum().item())
                        new_stats["old_anchor_sim"].extend(sim_old_all.detach().cpu().tolist())
                        new_stats["over_sep_th"] += int((sim_old_all > float(self.anchor_sep_threshold)).sum().item())
                        new_stats["over_050"] += int((sim_old_all > 0.50).sum().item())
                        new_stats["over_070"] += int((sim_old_all > 0.70).sum().item())

        # Anchor quality summary by old task.
        anchor_quality = {}
        for t in range(self._cur_task):
            a_t = [a for a in old_anchors if int(a.get("task", -1)) == t]
            if len(a_t) == 0:
                continue
            anchor_quality[t] = {
                "count": len(a_t),
                "conf": [round(float(a.get("confidence", 0.0)), 4) for a in a_t],
                "risk": [round(float(a.get("selection_risk", 0.0)), 4) for a in a_t],
                "pvar": [round(float(a.get("p_var", 0.0)), 6) for a in a_t],
                "rad": [round(float(a.get("radius", 0.0)), 4) for a in a_t],
                "sup": [int(a.get("support", 1)) for a in a_t],
            }

        print("\n[FEATURE-OVERLAP-CHECK] Task {}, Client {}".format(self._cur_task, client_id))
        print("  old_anchor_count={} | temp_current_anchor_count={} | sep_th={:.3f}".format(
            len(old_anchors),
            len(tmp_new_anchors),
            float(self.anchor_sep_threshold),
        ))

        print("  [old task samples]")
        print("  task | n | acc | old->new | old->wrong_old | old_sim_mean/p50 | same_task_sim_mean/p50 | new_sim_mean/p50 | new>old | new>same")
        for t in range(self._cur_task):
            d = per_task[t]
            n = max(int(d["n"]), 1)
            old_sim = self._safe_sim_stats(d["old_anchor_sim"])
            same_sim = self._safe_sim_stats(d["same_task_anchor_sim"])
            new_sim = self._safe_sim_stats(d["new_anchor_sim"])

            def fmt_pair(stat):
                if stat["mean"] is None:
                    return "NA/NA"
                return "{:.3f}/{:.3f}".format(stat["mean"], stat["p50"])

            print(
                "  {:>4} | {:>4} | {:>5.2f} | {:>8.2f} | {:>14.2f} | {:>16} | {:>22} | {:>16} | {:>7.2f} | {:>8.2f}".format(
                    t,
                    int(d["n"]),
                    100.0 * d["correct"] / n,
                    100.0 * d["old_to_new"] / n,
                    100.0 * d["old_to_wrong_old"] / n,
                    fmt_pair(old_sim),
                    fmt_pair(same_sim),
                    fmt_pair(new_sim),
                    100.0 * d["new_closer_than_old"] / n,
                    100.0 * d["new_closer_than_same_task"] / n,
                )
            )

        n_new = max(int(new_stats["n"]), 1)
        new_old_sim = self._safe_sim_stats(new_stats["old_anchor_sim"])
        print("  [current task samples]")
        print(
            "  n={} | new->old_anchor_sim mean/p50/p75/max={}/{}/{}/{} | sim>sep_th={:.2f}% | sim>0.50={:.2f}% | sim>0.70={:.2f}%".format(
                int(new_stats["n"]),
                "NA" if new_old_sim["mean"] is None else "{:.3f}".format(new_old_sim["mean"]),
                "NA" if new_old_sim["p50"] is None else "{:.3f}".format(new_old_sim["p50"]),
                "NA" if new_old_sim["p75"] is None else "{:.3f}".format(new_old_sim["p75"]),
                "NA" if new_old_sim["max"] is None else "{:.3f}".format(new_old_sim["max"]),
                100.0 * new_stats["over_sep_th"] / n_new,
                100.0 * new_stats["over_050"] / n_new,
                100.0 * new_stats["over_070"] / n_new,
            )
        )
        print("  [old anchor quality] {}".format(anchor_quality))

    def _debug_class_dist(self, data_loader, name=""):
        from collections import Counter
        cnt = Counter()
        for batch in data_loader:
            if len(batch) == 3:
                _, _, targets = batch
            else:
                _, targets = batch
            cnt.update(targets.cpu().numpy().tolist())
        print("\n[DATA DIST] {}".format(name))
        print(dict(sorted(cnt.items())))

    
    # ------------------------------------------------------------------
    # Federated training
    # ------------------------------------------------------------------
    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()

        if not hasattr(self, "local_task_curve"):
            self.local_task_curve = []
        if not hasattr(self, "local_client_curve"):
            self.local_client_curve = []
        if not hasattr(self, "local_client_grouped_curve"):
            self.local_client_grouped_curve = []

        local_mean_list = []
        local_client_acc_list = []
        local_grouped_acc_list = []

        user_groups = partition_data(
            train_dataset.labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"],
        )

        test_user_groups = partition_test_by_train_distribution(
            train_dataset.labels,
            test_loader.dataset.labels,
            user_groups,
            n_parties=self.args["num_users"],
        )

        prog_bar = tqdm(range(self.args["com_round"]))

        for _, com in enumerate(prog_bar):
            local_weights = []
            local_accs = []
            local_grouped_accs = []

            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)

            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=4,
                )

                local_test_loader = DataLoader(
                    DatasetSplit(test_loader.dataset, test_user_groups[idx]),
                    batch_size=256,
                    shuffle=False,
                    num_workers=4,
                )

                local_model = copy.deepcopy(self._network)
                if self._cur_task > 0 and idx in self.client_anchors:
                    self._compute_anchor_scores(idx, local_model)
                if self._cur_task == 0:
                    w = self._local_update(local_model, local_train_loader, idx)
                else:
                    w = self._local_finetune(local_model, local_train_loader, idx)
                local_model.load_state_dict(w)

                local_eval = self._eval_model_grouped(local_model, local_test_loader)
                local_grouped = local_eval["grouped"]
                local_accs.append(float(local_grouped["total"]))
                local_grouped_accs.append(local_grouped)

                # 마지막 communication round에서만 anchor를 생성/갱신한다.
                if com == self.args["com_round"] - 1:
                    if self.args.get("debug_anchor_signal", False):
                        self._debug_class_dist(local_train_loader, name="train-client-{}".format(idx))
                        self._debug_anchor_signal_vs_perf(idx, local_model, local_test_loader)

                    # FR이 실제 forgetting인지 확인하는 핵심 로그.
                    # 기본값 True라서 별도 옵션 없이 바로 찍힌다. 끄려면 --debug_fr_signal False 계열로 넘기면 된다.
                    if self.args.get("debug_fr_signal", True):
                        self._debug_fr_signal_vs_forgetting(idx, local_model, local_test_loader)

                    # Build current-task anchors as temporary probes first.
                    # IMPORTANT: these are not inserted into memory yet.
                    # The deep diagnosis below therefore sees:
                    #   trained local_model + old anchors + temporary current anchors
                    # which is the cleanest state for identifying the real failure mode.
                    anchor_items = self._build_anchor(
                        local_model,
                        local_train_loader,
                        client_id=idx,
                    )

                    if self.args.get("debug_feature_overlap", True):
                        self._debug_feature_overlap_diagnosis(
                            client_id=idx,
                            local_model=local_model,
                            local_loader=local_test_loader,
                            current_anchor_items=anchor_items,
                        )

                    for item in anchor_items:
                        self._update_anchor_memory(
                            client_id=idx,
                            task_id=self._cur_task,
                            u=item["u"],
                            p=item["p"],
                            local_model=local_model,
                            confidence=item["confidence"],
                            selection_risk=item["selection_risk"],
                            radius=item.get("radius", 0.0),
                            sigma=item.get("sigma", 0.0),
                            p_var=item.get("p_var", 0.0),
                            support=item.get("support", 1),
                            boundary_u=item.get("boundary_u", None),
                            boundary_p=item.get("boundary_p", None),
                            boundary_risk=item.get("boundary_risk", 0.0),
                            boundary_margin=item.get("boundary_margin", 0.0),
                            boundary_logit_margin=item.get("boundary_logit_margin", 0.0),
                            boundary_top1=item.get("boundary_top1", -1),
                            boundary_top2=item.get("boundary_top2", -1),
                        )

                    anchor_scores = self._compute_anchor_scores(idx, local_model)

                    print(
                        "Task {}, Client {}, RepAnchor added={}, anchor_pool={}, scores={}".format(
                            self._cur_task,
                            idx,
                            len(anchor_items),
                            self._anchor_pool_task_summary(idx),
                            anchor_scores,
                        )
                    )

                    # 현재 task를 막 학습한 직후의 task accuracy를 저장한다.
                    # 다음 task부터 이 값과 current task accuracy의 차이를 실제 forgetting drop으로 출력한다.
                    self._store_current_task_baseline_acc(idx, local_model, local_test_loader)
                    # FeatureKD disabled: do not store client teacher states.

                local_weights.append(copy.deepcopy(w))

                del local_train_loader, local_test_loader, local_model, w
                torch.cuda.empty_cache()

            local_stats = {
                "mean": float(np.mean(local_accs)),
                "std": float(np.std(local_accs)),
                "min": float(np.min(local_accs)),
                "max": float(np.max(local_accs)),
                "client_accs": local_accs,
                "client_grouped_accs": local_grouped_accs,
            }

            local_mean_list.append(local_stats["mean"])
            local_client_acc_list.append(local_stats["client_accs"])
            local_grouped_acc_list.append(local_stats["client_grouped_accs"])

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Global {:.2f}, Local-P {:.2f}".format(
                self._cur_task,
                com + 1,
                self.args["com_round"],
                test_acc,
                local_stats["mean"],
            )
            prog_bar.set_description(info)

            if self.wandb == 1:
                wandb.log(
                    {
                        "Task_{}, global_accuracy".format(self._cur_task): test_acc,
                        "Task_{}, local_personalized_accuracy".format(self._cur_task): local_stats["mean"],
                    }
                )

            del local_weights
            torch.cuda.empty_cache()

        self.local_task_curve.append(float(local_mean_list[-1]))
        self.local_client_curve.append(local_client_acc_list[-1])
        self.local_client_grouped_curve.append(local_grouped_acc_list[-1])

        print(
            "Task {}, Local personalized mean acc: {:.2f}, client accs: {}".format(
                self._cur_task,
                self.local_task_curve[-1],
                self.local_client_curve[-1],
            )
        )

    def _local_update(self, model, train_loader, client_id):
        # Task0용: 전체 모델 lr=0.01
        model.cuda()
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        for epoch in range(self.args["local_ep"]):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                loss, *_ = self._compute_loss(model, inputs, targets, client_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def _local_finetune(self, model, train_loader, client_id):
        # Task>0용: backbone lr 낮추고 fc는 기존 lr 유지
        model.cuda()
        model.train()

        optimizer = torch.optim.SGD(
            [
                {"params": model.convnet.parameters(), "lr": 0.001},
                {"params": model.fc.parameters(), "lr": 0.01},
            ],
            momentum=0.9,
            weight_decay=5e-4,
        )

        for epoch in range(self.args["local_ep"]):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                loss, *_ = self._compute_loss(model, inputs, targets, client_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.state_dict()
