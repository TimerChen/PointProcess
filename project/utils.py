import math
import pandas
import numpy as np
import torch

from collections import Counter
from tqdm import tqdm


def to_diffTask(pred_events, real_events, n, type = 0):
    pes = []
    res = []
    for i in range(len(pred_events)):
        if type == 0:
            pes.append(1 if(pred_events[i]==6) else 0)
            res.append(1 if(real_events[i]==6) else 0)
        else:
            if real_events[i] != 6:
                pes.append(pred_events[i])
                res.append(real_events[i])

    return np.array(pes), np.array(res)


def evaluation(step, args, model, test_loader, result):
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


    time_error = np.mean(np.abs(pred_times - real_times))

    acc, recall, f1 = [], [], []
    eErr = [acc, recall, f1]

    # error and ticket
    pes, res = to_diffTask(pred_events, real_events, args.num_class, 0)
    tmp = eva_metric(pes, res, 2)
    for i in range(3):
        eErr[i].append(tmp[i])

    # for all error
    pes, res = to_diffTask(pred_events, real_events, args.num_class, 1)
    tmp = eva_metric(pes, res, args.num_class-1)
    for i in range(3):
        eErr[i].append(tmp[i])


    print("epoch: ", step)
    print("MAE: {}, Presicion: {}, Recall: {}, macro-F1: {}".format(time_error, acc, recall, f1))

    result['MAE'].append(time_error)
    result['presicion'].append(acc)
    result['recall'].append(recall)
    result['f1'].append(f1)


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

    def init_weight(self):
        count = Counter(self.event)
        percentage = [count[k] / len(self.event) for k in sorted(count.keys())]
        for i, p in enumerate(percentage):
            print(f"event{i} = {p * 100}%")
        weight = [len(self.event) / count[k] for k in sorted(count.keys())]
        return weight


def eva_metric(pred, real, n_class):
    # Use confusion matrix
    real_count = Counter(real)
    pred_count = Counter(pred)
    prec, recall = [], []
    pp = rr = 0
    macroF1 = []
    for i in range(n_class):
        # TP
        TP = np.logical_and(pred == real, real == i).sum()
        # real_count[i] = TP + FN
        # pred_count[i] = TP + FP
        rr = pp = None
        if pred_count[i] != 0:
            pp = TP / pred_count[i]
            prec.append(pp)

        if real_count[i] != 0:
            rr = TP / real_count[i]
            recall.append(rr)

        if TP != 0:
            macroF1.append(2 * pp * rr / (pp + rr))

    prec = np.mean(prec)
    recall = np.mean(recall)
    macroF1 = np.mean(macroF1)
    return prec, recall, macroF1
