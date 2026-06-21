import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader

from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import (
    partition_data,
    partition_test_by_train_distribution,
    DatasetSplit,
    average_weights,
)


class Anchor(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        # client_id -> list of anchors
        # anchor = {"task": int, "u": Tensor[feature_dim], "p": Tensor[num_classes],
        #           "ls": float, "fr": float, "importance": float}
        self.client_anchors = {}

        self.anchor_budget = args.get("anchor_budget", 5)
        self.anchor_temp = args.get("anchor_temp", 1.0) # 분포 부드러운 정도 조정
        self.anchor_lambda = args.get("anchor_lambda", 0.01)

        # Selective global KD. KD weight is exp(-LS):
        # low LS -> trust global teacher more, high LS -> trust global teacher less.
        self.kd_lambda = args.get("kd_lambda", 0.0)
        self.kd_temp = args.get("kd_temp", 1.0)

        self.anchor_logs = []

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

        self._fl_train(train_dataset, self.test_loader)

    def _kl_prob(self, p, q):
        eps = 1e-8
        p = torch.clamp(p, eps, 1.0) # 분포가 eps~1.0 사이의 값을 갖도록 자름
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(p * torch.log(p / q)) # KL 계산 공식, 두 분포가 얼마나 다른지, 값이 작을 수록 비슷한 분포

    def _normalize_prob(self, p): # 합이 1이 되는 확률 분포로 만듦
        eps = 1e-8
        p = torch.clamp(p, eps, 1.0)
        return p / torch.clamp(p.sum(), min=eps)

    def _fc_logits_from_feature(self, model, feature): # 피처 fc에 넣어서 로짓 반환
        out = model.fc(feature)
        return out["logits"] if isinstance(out, dict) else out

    def _build_anchor(self, model, loader): # Build one anchor (u, p) from a client's current task data, loader: 현재 클라이언트 데이터
        model.eval()

        feat_sum = None
        prob_sum = None
        total = 0

        for _, inputs, targets in loader:
            inputs = inputs.cuda()

            with torch.no_grad():
                features = model.extract_vector(inputs) # 피처 추출, shape(배치, 피처 차원)
                logits = model(inputs)["logits"] # 로짓, shape(배치, 클래스 수)
                probs = F.softmax(logits / self.anchor_temp, dim=1) # 로짓 > 확률로

            if feat_sum is None:
                feat_sum = features.sum(dim=0) # 모든 샘플의 피처 합 생성
                prob_sum = probs.sum(dim=0) # 모든 샘플의 확률 합 생성
            else:
                feat_sum += features.sum(dim=0)
                prob_sum += probs.sum(dim=0)

            total += inputs.size(0) # 샘플 총 개수 누적

        u = feat_sum / total # 평균 (대표 피처)
        p = prob_sum / total # 평균 분포
        p = p / p.sum() # 합이 1이 되도록 정규화

        return u.detach().cpu(), p.detach().cpu()

    def _predict_anchor_prob(self, model, anchor_u): # Predict p_now for a stored anchor feature u with a given model
        model.eval()
        u = anchor_u.cuda().unsqueeze(0) # 배치 차원 추가 (fc 구조 때문)

        with torch.no_grad():
            logits = self._fc_logits_from_feature(model, u)
            logits = logits[:, : self._total_classes] # 아직 안 배운 클래스 제거
            prob = F.softmax(logits / self.anchor_temp, dim=1).squeeze(0) # 확률, 배치 차원 제거

        return prob.detach().cpu()

    def _compute_anchor_scores(self, client_id, local_model): # Compute LS, FR, and importance for all anchors of a client
        if client_id not in self.client_anchors:
            return []

        scores = []

        for anchor in self.client_anchors[client_id]:
            u = anchor["u"]
            p_saved = anchor["p"]

            p_local_now = self._predict_anchor_prob(local_model, u) # u에 대한 현재 로컬 모델의 확률 분포
            p_global_now = self._predict_anchor_prob(self._network, u) # u에 대한 글로벌 모델의 확률 분포

            min_dim = min(p_saved.shape[0], p_local_now.shape[0], p_global_now.shape[0])

            p_saved_ = self._normalize_prob(p_saved[:min_dim])
            p_local_ = self._normalize_prob(p_local_now[:min_dim])
            p_global_ = self._normalize_prob(p_global_now[:min_dim])

            ls = self._kl_prob(p_global_, p_local_)
            fr = self._kl_prob(p_saved_, p_local_)

            # Numerical noise can make KL slightly negative; clamp for stable weighting.
            ls_value = max(float(ls), 0.0)
            fr_value = max(float(fr), 0.0)
            importance_value = ls_value * fr_value

            anchor["ls"] = ls_value
            anchor["fr"] = fr_value
            anchor["importance"] = importance_value

            scores.append(
                {
                    "task": anchor["task"],
                    "ls": ls_value,
                    "fr": fr_value,
                    "importance": importance_value,
                }
            )

        return scores

    def _update_anchor_memory(self, client_id, task_id, u, p, local_model):
        if client_id not in self.client_anchors:
            self.client_anchors[client_id] = []

        new_anchor = {
            "task": task_id,
            "u": u,
            "p": p,
            "ls": 0.0,
            "fr": 0.0,
            "importance": 0.0,
        }

        self.client_anchors[client_id].append(new_anchor)
        self._compute_anchor_scores(client_id, local_model)

        self.client_anchors[client_id] = sorted(
            self.client_anchors[client_id],
            key=lambda x: x["importance"],
            reverse=True,
        )[: self.anchor_budget]

    def _get_anchor_weights(self, client_id): # 각 anchor의 normalized importance weight
        """Return normalized importance weights for a client's anchors.

        Selection score remains I=LS*FR, but the loss weight is normalized so one
        very large score does not dominate training too aggressively.
        """
        anchors = self.client_anchors.get(client_id, [])
        if len(anchors) == 0:
            return []

        raw = torch.tensor(
            [max(float(a.get("importance", 0.0)), 0.0) for a in anchors],
            dtype=torch.float32,
            device="cuda",
        )

        if torch.sum(raw) <= 1e-12:
            raw = torch.ones_like(raw) # 모두 동일 가중치로 설정

        weights = raw / torch.sum(raw) # 합이 1이 되도록 중요도 가중치 생성
        return weights

    def _anchor_loss(self, model, client_id):
        """Importance-weighted anchor preservation loss.

        L_anchor = sum_i I_norm_i * KL(p_saved_i || p_now_i)
        """
        anchors = self.client_anchors.get(client_id, [])
        if len(anchors) == 0:
            return torch.tensor(0.0, device="cuda")

        weights = self._get_anchor_weights(client_id)
        loss = torch.tensor(0.0, device="cuda")

        for weight, anchor in zip(weights, anchors):
            u = anchor["u"].cuda().unsqueeze(0)
            p_saved = anchor["p"].cuda()

            logits = self._fc_logits_from_feature(model, u).squeeze(0)

            min_dim = min(p_saved.shape[0], logits.shape[0])
            p_saved_ = self._normalize_prob(p_saved[:min_dim])
            log_p_now = F.log_softmax(logits[:min_dim] / self.anchor_temp, dim=0)

            loss += weight * F.kl_div(log_p_now, p_saved_, reduction="sum")

        return loss

    def _anchor_kd_loss(self, model, client_id):
        """PPT-style selective KD on stored anchors.

        For each anchor i:
            LS_i = KL(g_i || l_i)
            w_i^kd = exp(-LS_i)
            L_kd = sum_i w_i^kd * LS_i

        g_i: global model prediction on anchor feature u_i
        l_i: local model prediction on anchor feature u_i
        """
        if self.kd_lambda <= 0: # kd_lambda=0이면 KD를 아예 끔
            return torch.tensor(0.0, device="cuda")

        anchors = self.client_anchors.get(client_id, [])
        if len(anchors) == 0:
            return torch.tensor(0.0, device="cuda")

        loss = torch.tensor(0.0, device="cuda")
        count = 0

        self._network.eval() # 글로벌이 teacher

        for anchor in anchors:
            u = anchor["u"].cuda().unsqueeze(0)

            with torch.no_grad():
                global_logits = self._fc_logits_from_feature(self._network, u).squeeze(0)

            local_logits = self._fc_logits_from_feature(model, u).squeeze(0)

            min_dim = min(global_logits.shape[0], local_logits.shape[0], self._total_classes)

            log_p_local = F.log_softmax(local_logits[:min_dim] / self.kd_temp, dim=0)
            p_global = F.softmax(global_logits[:min_dim] / self.kd_temp, dim=0)

            # KL(g_i || l_i). In torch, kl_div(input=log Q, target=P) computes KL(P || Q).
            ls = F.kl_div(log_p_local, p_global, reduction="sum")
            ls = torch.clamp(ls, min=0.0)

            # Detach the weight so gradients optimize LS, not the weighting rule itself.
            kd_weight = torch.exp(-ls.detach())
            loss += kd_weight * ls
            count += 1

        if count > 0:
            loss = loss / count

        return loss

    def _compute_loss(self, model, inputs, targets, client_id):
        logits = model(inputs)["logits"]
        ce_loss = F.cross_entropy(logits, targets)

        anchor_loss = self._anchor_loss(model, client_id)
        kd_loss = self._anchor_kd_loss(model, client_id)

        loss = ce_loss + self.anchor_lambda * anchor_loss + self.kd_lambda * kd_loss
        return loss, ce_loss.detach(), anchor_loss.detach(), kd_loss.detach()

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
            idxs_users = np.random.choice(
                range(self.args["num_users"]),
                m,
                replace=False,
            )

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

                if self._cur_task == 0:
                    w = self._local_update(local_model, local_train_loader, idx)
                else:
                    w = self._local_finetune(local_model, local_train_loader, idx)

                local_model.load_state_dict(w)

                local_eval = self._eval_model_grouped(local_model, local_test_loader)
                local_grouped = local_eval["grouped"]

                local_accs.append(float(local_grouped["total"]))
                local_grouped_accs.append(local_grouped)

                # Update anchors only after the last communication round of each task.
                if com == self.args["com_round"] - 1:
                    anchor_u, anchor_p = self._build_anchor(local_model, local_train_loader)
                    self._update_anchor_memory(
                        client_id=idx,
                        task_id=self._cur_task,
                        u=anchor_u,
                        p=anchor_p,
                        local_model=local_model,
                    )

                    anchor_scores = self._compute_anchor_scores(idx, local_model)
                    print(
                        "Task {}, Client {}, Anchor scores: {}".format(
                            self._cur_task,
                            idx,
                            anchor_scores,
                        )
                    )

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

            info = (
                "Task {}, Epoch {}/{} => Global {:.2f}, Local-P {:.2f}".format(
                    self._cur_task,
                    com + 1,
                    self.args["com_round"],
                    test_acc,
                    local_stats["mean"],
                )
            )
            prog_bar.set_description(info)

            if self.wandb == 1:
                wandb.log(
                    {
                        "Task_{}, global_accuracy".format(self._cur_task): test_acc,
                        "Task_{}, local_personalized_accuracy".format(
                            self._cur_task
                        ): local_stats["mean"],
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
        model.cuda()
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 170],
            gamma=0.1,
        )

        for epoch in range(self.args["local_ep"]):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                loss, _, _, _ = self._compute_loss(model, inputs, targets, client_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        return model.state_dict()

    def _local_finetune(self, model, train_loader, client_id):
        model.cuda()
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 170],
            gamma=0.1,
        )

        for epoch in range(self.args["local_ep"]):
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                loss, _, _, _ = self._compute_loss(model, inputs, targets, client_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        return model.state_dict()
