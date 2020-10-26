import time
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from apex import amp
from config import configs
from build_data import AmazonQA
from CHIME import CHIME_Model
from utils import get_batch, random_seed
from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


def evaluate(eval_model, data_source, eval_batch_size):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(get_batch(data_source, eval_batch_size)):
            lm_losses, _ = eval_model(batch)
            if multi_gpu:
                lm_losses = sum([lm_loss.mean() for lm_loss in lm_losses])
            total_loss += batch[0].size(0) * lm_losses.item()
    return total_loss / len(data_source)


def predict(load_model, batch_data, single_answer_length, sep_idx, beam_size):
    load_model.eval()
    with torch.no_grad():
        que, rev, ans = batch_data  # que: (N, SQ=TQ), rev: (N, R, SR), ans: (N, A, TA)
        outputs = []
        for k in range(que.size(0)):
            top_candidates = []
            seen_words = {}

            # Prepare data
            q = que[k:k + 1, :]
            r = rev[k:k + 1, :, :]
            output = torch.tensor([[[]]]).long().cuda()  # (1, 1, 0)

            # generate predictions
            _, output_ts = load_model((q, r, output))
            output_tbv, output_tbi = torch.topk(F.log_softmax(output_ts, dim=1), beam_size, dim=1)
            for p in range(beam_size):
                top_candidates.append(([output_tbi[:, p]], output_tbv[:, p]))  # save generation & beam scores
                seen_words[p] = {int(output_tbi[:, p].item()): 1}  # limit repeated words
            for t in range(single_answer_length - 1):
                candidates = {}
                for i in range(beam_size):
                    current_tokens = top_candidates[i][0]
                    if int(current_tokens[-1].item()) == sep_idx:
                        pass
                    else:
                        output_c = torch.tensor(current_tokens).unsqueeze(0).unsqueeze(0).float().cuda()
                        _, output_cs = load_model((q, r, output_c))
                        output_cbv, output_cbi = torch.topk(F.log_softmax(output_cs, dim=1), beam_size, dim=1)
                        for j in range(beam_size):
                            current_p = output_cbi[:, j]
                            current_p_i = int(current_p.item())
                            current_p_c = seen_words[i].get(current_p_i, 0)
                            if current_p != current_tokens[-1] and current_p_c <= 1:
                                seen_words[i][current_p_i] = current_p_c + 1
                                beam_score = (top_candidates[i][1] * len(current_tokens) + output_cbv[:, j]) / \
                                             ((len(current_tokens) + 1) ** 0.7)
                                candidates[(i, j)] = (current_tokens + [current_p], beam_score)
                            else:
                                pass
                if len(candidates) == 0:
                    break
                top_beams = sorted(candidates.items(), key=lambda x: -x[1][1])[:beam_size]  # beam_score排序候选答案
                for top_beam_i, top_beam in enumerate(top_beams):
                    top_candidates[top_beam_i] = top_beam[1]
            outputs.append(top_candidates[0][0])
        return outputs


if __name__ == "__main__":
    # ----Hyperparameters & PTEncoder----
    pretrained_config = AutoConfig.from_pretrained(configs['pretrained_weights'], output_hidden_states=True)
    pretrained_encoder = AutoModel.from_pretrained(configs['pretrained_weights'], config=pretrained_config).cuda()
    tokenizer = configs['tokenizer']
    multi_gpu = random_seed(configs['seed'])
    epochs = configs['epochs']
    task = configs['task']
    if 'fp16' in configs:
        fp16 = configs['fp16']
    else:
        fp16 = None

    # ----Loading data and initializing model----
    logger.info('Loading data...')
    running_data = AmazonQA()
    logger.info(
        "Size of train, valid and test data: %s, %s, %s" % (len(running_data.train.examples),
                                                            len(running_data.dev.examples),
                                                            len(running_data.test.examples)))
    batch_size = configs['batch_size']
    logger.info("Batch size of train, valid and test data per GPU : %s" % batch_size)
    batch_size *= len(configs['devices'])

    logger.info("Building model...")
    model = CHIME_Model(pretrained_encoder,
                        pretrained_config,
                        configs['cls_index'],
                        configs['sep_index'],
                        configs['pad_index'],
                        configs['msk_index'],
                        configs['seed']).cuda()

    if task == "Train":
        # ----Setting optimizers, scheduler and parameters----
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.95},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=configs['lr'])
        t_total = len(running_data.train.examples) // configs['gradient_accumulations'] * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * configs['warm_up_rate']),
                                                    num_training_steps=t_total)

        if fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16)
        if multi_gpu:
            model = nn.DataParallel(model, device_ids=configs['devices'])
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info("Number of model parameters: %s" % num_parameters)

        logger.info("Training...")
        model.train()  # Turn on the train mode

        best_val_loss = float("inf")
        best_model = None
        for epoch in range(1, epochs + 1):
            total_loss = 0.
            start_time = time.time()
            for i, batch in enumerate(get_batch(running_data.train.examples, batch_size)):
                optimizer.zero_grad()

                lm_losses, _ = model(batch)
                bsz = batch[0].size(0)
                if multi_gpu:
                    lm_losses = sum([lm_loss.mean() for lm_loss in lm_losses])
                with amp.scale_loss(lm_losses, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])

                if i % configs['gradient_accumulations'] == 0 and i > 0:
                    optimizer.step()
                    scheduler.step()

                total_loss += lm_losses.item()
                log_interval = len(running_data.train.examples) // (10 * bsz)
                if i % log_interval == 0:
                    cur_loss_lm = total_loss / log_interval  # average loss on each sample
                    elapsed = time.time() - start_time
                    logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.8f} | ms/batch {:5.2f} | '
                                'train lm loss {:5.4f} '.format(epoch, i, len(running_data.train) // bsz,
                                                                optimizer.param_groups[0]["lr"],
                                                                elapsed * 1000 / log_interval,
                                                                cur_loss_lm))
                    total_loss = 0.
                    start_time = time.time()

            val_loss_lm = evaluate(model, running_data.dev.examples, batch_size)
            logger.info('Epoch %s, Valid loss %.5s' % (epoch, val_loss_lm))
            if val_loss_lm < best_val_loss:
                best_val_loss = val_loss_lm
                best_model = model
                savepath = configs['model_output_path'] + "%s_%s.pth" % (configs['task_name'], str(epoch))
                torch.save(best_model.state_dict(), savepath)
                logger.info("model of epoch %s saved" % epoch)
    elif task == "Test":
        for epoch in range(1, epochs + 1):
            loadpath = configs['model_output_path'] + "%s_%s.pth" % (configs['task_name'], str(epoch))
            state_dict = torch.load(loadpath, map_location=torch.device("cpu"))
            if fp16:
                model = amp.initialize(model, opt_level=fp16)
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            test_loss_lm = evaluate(model, running_data.test.examples, batch_size)
            logger.info('Epoch %s, Test loss %.5s' % (epoch, test_loss_lm))
    elif task == "Predict":
        for epoch in range(1, epochs + 1):
            records = []
            loadpath = configs['model_output_path'] + "%s_%s.pth" % (configs['task_name'], str(epoch))
            state_dict = torch.load(loadpath, map_location=torch.device("cpu"))
            if fp16:
                model = amp.initialize(model, opt_level=fp16)
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            for i, batch in enumerate(get_batch(running_data.test.examples, batch_size)):
                ans = batch[2]  # (N, A, TA)
                prediction_beam = predict(model, batch, configs['single_answer_length'], configs['sep_index'],
                                          beam_size=configs['beam_size'])
                for k in range(len(prediction_beam)):
                    references = []
                    for j in range(ans.size(1)):
                        reference = tokenizer.decode(ans[k, j, :], skip_special_tokens=True)
                        references.append(reference)
                    record_p = {"batch": str(i + 1), "case": str(k + 1),
                                "question": tokenizer.decode(batch[0][k, :], skip_special_tokens=True),
                                "answers": references,
                                "prediction": tokenizer.decode(prediction_beam[k], skip_special_tokens=True)}
                    records.append(record_p)
            prediction_path = configs['prediction_output_path'] + "%s_%s.json" % (configs['task_name'], str(epoch))
            json.dump(records, open(prediction_path, "w"))
            logger.info("epoch %s prediction done" % epoch)
    elif task == "Analysis":
        loadpath = configs['model_output_path'] + "%s_%s.pth" % (configs['task_name'], str(epochs))
        state_dict = torch.load(loadpath, map_location=torch.device("cpu"))
        if fp16:
            model = amp.initialize(model, opt_level=fp16)
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        batch = get_batch(running_data.test.examples, batch_size)[0]
        ans = batch[2]
        rev = batch[1]
        que = batch[0]
        for i in range(10, 11):
            prediction_beam = predict(model, (que, rev[:, :i, :], ans), configs['single_answer_length'],
                                      configs['sep_index'], beam_size=configs['beam_size'])
            logger.info("After reading %s passages: %s" % (i, tokenizer.decode(prediction_beam[0], skip_special_tokens=True)))
    else:
        logger.info("Wrong task type")
