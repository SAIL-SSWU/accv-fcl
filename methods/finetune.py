import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_test_by_train_distribution
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from sklearn.metrics import confusion_matrix

# init_epoch = 200
# com_round = 100  
# num_users = 5 # 5, 
# frac = 1 # 




# local_bs = 128  # cifar100, 5w, 5 tasks, 1w for each task, 2k for each client
# local_ep = 5
# batch_size = 128
# num_workers = 4

tau=1

# 클라이언트의 데이터 클래스 분포 확인
def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        # | 并集
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    print(f"Client {client_id}:",
      sorted(temp.items(), key=lambda x:x[0]))




# 정답을 제외하고 나머지 클래스만 남김
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.acc = []
        self.local_task_curve = []
        self.local_client_curve = []

    def after_task(self):
        self._known_classes = self._total_classes # 학습한 클래스 기록
        self.pre_loader = self.test_loader # 테스트 데이터 저장 -> 이전 Task 성능 확인용
        self._old_network = self._network.copy().freeze() # t번째 태스크 종료 시점 모델


    def _ntd_loss(self, logits, dg_logits, targets): # 정답 외의 클래스 분포를 맞춤, logits: student 출력, dg_logits: teacher 출력, targets: 정답 클래스
        """Not-tue Distillation Loss"""
        KLDiv = nn.KLDivLoss(reduction="batchmean") # 두 확률분포 차이를 측정
        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self._total_classes) # 정답 클래스 로짓 제외
        pred_probs = F.log_softmax(logits / tau, dim=1) # student 분포 생성

        # Get smoothed global model prediction
        with torch.no_grad(): # teacher 학습 x
            dg_logits = refine_as_not_true(dg_logits, targets, self._total_classes) # teacher도 정답 클래스 로짓 제외
            dg_probs = torch.softmax(dg_logits / tau, dim=1) # teacher 분포 생성

        loss = (tau ** 2) * KLDiv(pred_probs, dg_probs) # CE에서 이미 정답을 맞출 것을 강하게 학습 > 비정답 클래스만 KD로 학습

        return loss


    def incremental_train(self, data_manager): # incremental_train을 위해 필요한 작업(데이터.로더 가져옴)
        self._cur_task += 1 # 현재 태스크 번호 증가
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        ) # 현재 태스크까지의 총 클래스 수
        self._network.update_fc(self._total_classes) # 추가된 클래스 수에 맞게 분류기 확장
        print("Learning on {}-{}".format(self._known_classes, self._total_classes)) # 현재 학습 중인 클래스

        train_dataset = data_manager.get_dataset(   #* get the data for one task # 현재 학습해야 하는 클래스의 데이터를 가져옴
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        ) 
        test_dataset = data_manager.get_dataset( # 현재까지 학습한 클래스의 테스트 데이터 가져옴
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader( # 테스트용 데이터로더 생성
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)
        

        # if self._cur_task == 0:
        #     # self._fl_train(train_dataset, self.test_loader)
        #     # torch.save(self._network.state_dict(), 'finetune.pkl')
        #     # print("save checkpoint >>>")

        #     self._network.cuda()
        #     state_dict = torch.load('finetune.pkl')
        #     self._network.load_state_dict(state_dict)
        #     test_acc = self._compute_accuracy(self._network, self.test_loader)
        #     print("For task 1, loading ckpt, acc:{}".format(test_acc))

        #     # return 
        # else:
        #     # return 
        #     acc = self._compute_accuracy(self._old_network, self.pre_loader)
        #     print("loading ckpt, acc:{}".format(acc))
            
        #     self._fl_train(train_dataset, self.test_loader)

        

    # def _local_update(self, model, train_data_loader):
    #     model.train()
    #     cp_model =  copy.deepcopy(model)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #     for iter in range(local_ep):
    #         for batch_idx, (_, images, labels) in enumerate(train_data_loader):
    #             images, labels = images.cuda(), labels.cuda()
    #             output = model(images)["logits"]
    #             loss_ce = F.cross_entropy(output, labels)
    #             with torch.no_grad():
    #                 dg_logits = cp_model(images.detach())["logits"]
    #             # only learn from out-distribution knowledge, overcome local forgetting
    #             loss_ntd = self._ntd_loss(output, dg_logits, labels)
    #             loss = loss_ce + loss_ntd 
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     return model.state_dict()

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader): # 데이터에서 이미지랑 라벨만
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"] # 모델에 이미지 넣고 로짓 출력
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict() # weight값 반환


    def per_cls_acc(self, val_loader, model): # 클래스별 정확도
        model.eval()
        all_preds = [] # 예측 리스트
        all_targets = [] # 정답 리스트
        with torch.no_grad():
            for i, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1) # 클래스별 샘플 수
        cls_hit = np.diag(cf) # 맞춘 개수 (대각선 추출)

        cls_acc = cls_hit / cls_cnt # 클래스 별 정확도
        return cls_acc
        # pdb.set_trace()
        # out_cls_acc = 'Per Class Accuracy: %s' % ((np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        # print(out_cls_acc)
        

        

    def _local_finetune(self, model, train_data_loader): # 현재 태스크에 새 클래스만 학습
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # print_data_stats(0, train_data_loader)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes # 현재 태스크에 추가된 클래스에 답지
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # self.per_cls_acc(self.test_loader, model)

        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        if not hasattr(self, "local_task_curve"):
            self.local_task_curve = []
        if not hasattr(self, "local_client_curve"):
            self.local_client_curve = []
        if not hasattr(self, "local_client_grouped_curve"):
            self.local_client_grouped_curve = []
            
        cls_acc_list = []
        local_mean_list = []
        local_client_acc_list = []
        local_grouped_acc_list = []

        user_groups = partition_data(
            train_dataset.labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"]
        )

        test_user_groups = partition_test_by_train_distribution(
            train_dataset.labels,
            test_loader.dataset.labels,
            user_groups,
            n_parties=self.args["num_users"]
        )

        prog_bar = tqdm(range(self.args["com_round"]))

        for _, com in enumerate(prog_bar):
            local_weights = []
            local_grouped_accs = []
            local_accs = []

            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)

            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=4
                )

                local_test_loader = DataLoader(
                    DatasetSplit(test_loader.dataset, test_user_groups[idx]),
                    batch_size=256,
                    shuffle=False,
                    num_workers=4
                )

                local_model = copy.deepcopy(self._network)

                if self._cur_task == 0:
                    w = self._local_update(local_model, local_train_loader)
                else:
                    w = self._local_finetune(local_model, local_train_loader)

                local_model.load_state_dict(w)

                # personalized local evaluation:
                # client idx local model -> client idx local test split
                local_eval = self._eval_model_grouped(local_model, local_test_loader)
                local_grouped = local_eval["grouped"]

                local_accs.append(float(local_grouped["total"]))
                local_grouped_accs.append(local_grouped)

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

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            if com % 1 == 0:
                cls_acc = self.per_cls_acc(self.test_loader, self._network)
                cls_acc_list.append(cls_acc)

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
                    wandb.log({
                        'Task_{}, global_accuracy'.format(self._cur_task): test_acc,
                        'Task_{}, local_personalized_accuracy'.format(self._cur_task): local_stats["mean"],
                    })

            del local_weights
            torch.cuda.empty_cache()

        acc_arr = np.array(cls_acc_list)
        acc_max = acc_arr.max(axis=0)

        if self._cur_task == 4:
            acc_max = self.per_cls_acc(self.test_loader, self._network)

        print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
        self.acc.append(acc_max)

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