import argparse
import wandb, os
import csv
import json
from datetime import datetime 
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from methods.anchor import Anchor
import warnings
warnings.filterwarnings('ignore')


def get_learner(model_name, args): # 모델 가져오기
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "target":
        return TARGET(args)
    elif name == "anchor":
        return Anchor(args)
    else:
        assert 0
        

def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    setup_seed(args["seed"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f'{args["dataset"]}_{args["method"]}_{args["exp_name"]}_{run_id}'

    task_csv_path = f"results/{result_name}_task.csv"
    summary_csv_path = f"results/{result_name}_summary.csv"

    data_manager = DataManager(
        args["dataset"],
        True,
        args["seed"],
        args["init_cls"],
        args["increment"],
    )

    learner = get_learner(args["method"], args)

    task_logs = []
    old_curve = []
    new_curve = []
    local_curve = []
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    for task in range(data_manager.nb_tasks):
        print(
            "All params: {}, Trainable params: {}".format(
                count_parameters(learner._network),
                count_parameters(learner._network, True),
            )
        )

        learner.incremental_train(data_manager)
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()

        grouped = cnn_accy["grouped"]

        local_p = None
        client_accs = None

        if hasattr(learner, "local_task_curve") and len(learner.local_task_curve) > 0:
            local_p = learner.local_task_curve[-1]

        if hasattr(learner, "local_client_curve") and len(learner.local_client_curve) > 0:
            client_accs = learner.local_client_curve[-1]

        old_acc = grouped.get("old", None)
        new_acc = grouped.get("new", None)
        total_acc = grouped.get("total", cnn_accy["top1"])

        old_curve.append(old_acc)
        new_curve.append(new_acc)
        local_curve.append(local_p)

        task_row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "exp_name": args["exp_name"],
            "method": args["method"],
            "dataset": args["dataset"],
            "seed": args["seed"],
            "task": task,
            "tasks": args["tasks"],
            "beta": args["beta"],
            "num_users": args["num_users"],
            "frac": args["frac"],
            "com_round": args["com_round"],
            "local_ep": args["local_ep"],
            "global_total": total_acc,
            "global_old": old_acc,
            "global_new": new_acc,
            "local_p": local_p,
            "client_accs": json.dumps(client_accs),
            "grouped_acc": json.dumps(grouped),
            "local_client_grouped_accs": json.dumps(
                getattr(learner, "local_client_grouped_curve", [[]])[-1]
            ),
        }

        append_csv(task_csv_path, task_row)
        task_logs.append(task_row)

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))

    global_curve = cnn_curve["top1"]

    summary_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "exp_name": args["exp_name"],
        "method": args["method"],
        "dataset": args["dataset"],
        "seed": args["seed"],
        "tasks": args["tasks"],
        "beta": args["beta"],
        "num_users": args["num_users"],
        "frac": args["frac"],
        "com_round": args["com_round"],
        "local_ep": args["local_ep"],
        "global_curve": json.dumps(global_curve),
        "local_p_curve": json.dumps(local_curve),
        "old_curve": json.dumps(old_curve),
        "new_curve": json.dumps(new_curve),
        "global_final": global_curve[-1],
        "global_avg": float(sum(global_curve) / len(global_curve)),
        "local_p_final": local_curve[-1],
        "local_p_avg": float(sum(local_curve) / len(local_curve)),
    }

    append_csv(summary_csv_path, summary_row)

    print(f"Saved task results to {task_csv_path}")
    print(f"Saved summary results to {summary_csv_path}")






def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb') # 딥러닝 실험 관리 툴
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="TARGET", help='wandb project')
    parser.add_argument('--group', type=str, default="exp1", help='wandb group')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0.3, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data') # 타겟
    parser.add_argument('--kd', type=int, default=25, help='for kd loss')
    parser.add_argument('--memory_size', type=int, default=300, help='the num of real data per task') # icarl
    parser.add_argument('--increment', type=int, default=None, help='classes per task')

    parser.add_argument('--anchor_budget', type=int, default=5)
    parser.add_argument('--anchor_lambda', type=float, default=0.01)
    parser.add_argument('--anchor_temp', type=float, default=1.0)
    parser.add_argument('--anchor_per_task', type=int, default=3)
    parser.add_argument('--kd_lambda', type=float, default=0.0)
    parser.add_argument('--kd_temp', type=float, default=1.0)
    parser.add_argument('--old_anchor_min', type=int, default=3)
    parser.add_argument('--current_anchor_max', type=int, default=2)
        

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset == "cifar10":
        args.num_class = 10
    elif args.dataset == "tiny_imagenet":
        args.num_class = 200
    else:
        args.num_class = 100

    if args.increment is None:
        args.init_cls = int(args.num_class / args.tasks)
        args.increment = args.init_cls
    else:
        args.init_cls = args.increment

    args.exp_name = f"{args.dataset}_{args.com_round}_{args.beta}_{args.method}_{args.exp_name}"
    if args.method == "target":
        dir = "run"
        if not os.path.exists(dir):
            os.makedirs(dir) 
        args.save_dir = os.path.join(dir, args.group+"_"+args.exp_name) # 합성 데이터 저장할 폴더 생성
    
    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
    args = vars(args) # 딕셔너리로 변환
    
    train(args)

