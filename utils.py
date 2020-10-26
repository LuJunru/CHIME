import torch
import random
import numpy as np
import os
from config import configs
from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()
device = configs['device']
devices = configs['devices']
rev_num = configs['rev_num']
pad_index = configs['pad_index']
ans_num = configs['ans_num']


def random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    multi_gpu = False
    torch.cuda.set_device(device)
    if devices:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if len(devices) > 1:
            torch.cuda.manual_seed_all(seed)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])
            multi_gpu = True
    return multi_gpu


def get_batch(data_source, batch_size):
    batches = []
    for i in range(0, len(data_source) // batch_size + 1):
        batch = data_source[batch_size * i:batch_size * (i + 1)]
        if len(batch) == 0:
            continue
        que, rev, ans = [], [], []
        for batch_j in batch:
            que.append(batch_j.q_word)

            rev_org = batch_j.review_snippets[:rev_num]
            rev.append(rev_org)

            ans_h_org = []
            for n in range(len(batch_j.answerText)):
                try:
                    helpful_res = batch_j.helpful[n]
                    if helpful_res[1] != 0:
                        helpful_score = helpful_res[0] / helpful_res[1]
                    else:
                        helpful_score = 0.5
                    ans_h_org.append((batch_j.answerText[n], helpful_score))
                except:
                    ans_h_org.append((batch_j.answerText[n], 0.5))
            ans_h_org = sorted(ans_h_org, key=lambda x: (-x[1], -x[0].count(pad_index)))
            # Padding answers
            answer_number = ans_num
            ans_h_org = ans_h_org * (answer_number // len(ans_h_org)) + ans_h_org[:(answer_number % len(ans_h_org))]
            ans.append([ans_h_org_i[0] for ans_h_org_i in ans_h_org])
        if len(que) > 0:
            batches.append((torch.tensor(que).cuda(), torch.tensor(rev).cuda(), torch.tensor(ans).cuda()))
    return batches
