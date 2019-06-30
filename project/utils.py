import math
import pandas
import numpy as np
import torch

from collections import Counter
from tqdm import tqdm

def evaluation(step, args, model, test_loader):
    model.eval()
    pred_times, pred_events = [], []
    real_times, real_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        real_times.append(batch[0][:, -1].numpy())
        real_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch)
        pred_times.append(pred_time)
        pred_events.append(pred_event)
    pred_times = np.concatenate(pred_times).reshape(-1)
    real_times = np.concatenate(real_times).reshape(-1)
    pred_events = np.concatenate(pred_events).reshape(-1)
    real_events = np.concatenate(real_events).reshape(-1)
    time_error = abs_error(pred_times, real_times)
    acc, recall, f1 = clf_metric(pred_events, real_events, n_class=args.num_class)
    print("epoch: ", step)
    print("time_error: {}, Presicion: {}, Recall: {}, macro-F1: {}".format(time_error, acc, recall, f1))


class MyDataset:
    def __init__(self, args, subset):
        data = pandas.read_csv(f"data/{subset}.csv")
        self.subset = subset
        self.id = list(data['id'])
        self.time = list(data['time'])
        self.event = list(data['event'])
        self.seq_len = args.seq_len
        self.time_seqs, self.event_seqs = self.generate_sequence()
        self.statistic()

    def generate_sequence(self):
        pbar = tqdm(total=len(self.id) - self.seq_len + 1)
        time_seqs = []
        event_seqs = []
        cur_end = self.seq_len - 1
        while cur_end < len(self.id):
            pbar.update(1)
            cur_start = cur_end - self.seq_len + 1
            if self.id[cur_start] != self.id[cur_end]:
                cur_end += 1
                continue

            event_seqs.append(self.event[cur_start:cur_end + 1])
            time_seqs.append(self.time[cur_start:cur_end + 1])
            cur_end += 1
        return time_seqs, event_seqs

    def __getitem__(self, item):
        return self.time_seqs[item], self.event_seqs[item]

    def __len__(self):
        return len(self.time_seqs)

    @staticmethod
    def to_features(batch):
        times, events = [], []
        for time, event in batch:
            time = np.array([time[0]] + time)
            time = np.diff(time)
            times.append(time)
            events.append(event)
        return torch.FloatTensor(times), torch.LongTensor(events)

    def statistic(self):
        print("TOTAL SEQs:", len(self.time_seqs))
        # for i in range(10):
        #     print(self.time_seqs[i], "\n", self.event_seqs[i])
        intervals = np.diff(np.array(self.time))
        for thr in [0.001, 0.01, 0.1, 1, 10, 100]:
            print(f"<{thr} = {np.mean(intervals < thr)}")

    def importance_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / x.sum()


def abs_error(pred, real): return np.mean(np.abs(pred - real))

def clf_metric(pred, real, n_class):
    gold_count = Counter(real)
    pred_count = Counter(pred)
    prec = recall = 0
    pcnt = rcnt = 0
    for i in range(n_class):
        match_count = np.logical_and(pred == real, pred == i).sum()
        if gold_count[i] != 0:
            prec += match_count / gold_count[i]
            pcnt += 1
        if pred_count[i] != 0:
            recall += match_count / pred_count[i]
            rcnt += 1
    prec /= pcnt
    recall /= rcnt
    print(f"pcnt={pcnt}, rcnt={rcnt}")
    f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1

def eva_metric(pred, real, n_class):
    # Use confusion matrix
    real_count = Counter(real)
    pred_count = Counter(pred)
    prec, recall = [], []
    pNum = rNum = 0
    pp = rr = 0
    ppNum = rrNum = 0
    macroF1 = []
    for i in range(n_class):
        # TP
        TP = np.logical_and(pred == real, real == i).sum()
        # real_count[i] = TP + FN
        # pred_count[i] = TP + FP
        FN = real_count[i] - TP
        FP = pred_count[i] - TP
        if pred_count[i] != 0:
            pp = TP / pred_count[i]
            prec.append(pp)

        if real_count[i] != 0:
            rr = TP / real_count[i]
            recall.append(rr)

        # if rr is not None
        macroF1.append(2 * pp * rr / (pp + rr))

    prec /= pNum
    recall /= rNum
    print(f"pNum={pNum}, rNum={rNum}")
    macroF1 = np.mean(macroF1)
    return prec, recall, macroF1
