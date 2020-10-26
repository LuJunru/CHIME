import torch.nn as nn
import torch
from TransformerEncoder import Encoder


class CHIME_Model(nn.Module):
    def __init__(self, pretrained_encoder, config, cls_index, sep_index, pad_index, msk_index, seed):
        super(CHIME_Model, self).__init__()

        self.encoder = pretrained_encoder
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.activate = nn.Sigmoid()

        self.c_mem_encoder = Encoder(num_layers=1, model_dim=config.d_model, num_heads=8)
        self.a_mem_encoder = Encoder(num_layers=1, model_dim=config.d_model, num_heads=8)
        self.cgate_projector = nn.Linear(config.d_model * 2, 1, bias=True)
        self.agate_projector = nn.Linear(config.d_model * 2, 1, bias=True)

        self.pad_i = pad_index
        self.msk_i = msk_index
        self.cls_i = cls_index
        self.sep_i = sep_index
        self.seed = seed
        self.voc_size = config.vocab_size

        self.init_weights()

    def init_weights(self):
        self.decoder.weight = self.encoder.word_embedding.weight
        self.decoder.bias.data.zero_()

    def _get_padding_mask(self, seq, sen_1_leng):
        """
        cls q cls r sep a1 sep a2 sep pad
        cls q cls r a1 sep a2 sep pad pad
        :param seq: the target sequence
        :param sen_1_leng: the length of q + r + m parts
        :return: special attention mask following UniLM strategy
        """
        s_l = seq.size(0)
        ones = torch.ones((s_l, s_l)).to(seq.device)
        mask = ones.tril()
        mask[:, :sen_1_leng] = 1
        padding_mask = (seq != self.pad_i).float()
        mask *= padding_mask
        mask *= padding_mask.unsqueeze(1)
        return mask

    def _compute_loss(self, logits, labels, ans_mask=None, soft=False):
        """
        :param logits: [cls q cls r sep a1 a2 a3 sep pad pad pad]
        :param labels: [cls q cls r a1 a2 a3 sep pad pad pad pad]
        :return: the LM loss
        """
        if soft:
            logits = torch.log(logits)
            loss_fct = nn.NLLLoss(ignore_index=self.pad_i, reduction="none")
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_i, reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.size(2)), labels.reshape(-1).long())
        if torch.is_tensor(ans_mask):
            loss = (loss * ans_mask.reshape(-1)).sum()
        else:
            loss = loss.sum()
        not_ignore = labels.ne(self.pad_i).float()
        if torch.is_tensor(ans_mask):
            not_ignore *= ans_mask
        num_targets = not_ignore.long().sum().item()
        loss = loss / num_targets
        return loss

    def dynamic_padding(self, q, r, a):
        """
        :param q: (N, SQ)
        :param r: (N, SR)
        :param a: (N, TA)
        :return: model inputs [CLS Q CLS R SEP A SEP PAD*], targets [CLS Q CLS R A SEP PAD* PAD],
                 segment id mask [0 1 0 1], ans mask [0 0 1 0] and attention mask
        """
        bsz = q.size(0)
        qra = torch.ones((bsz, 1 + q.size(1) + 1 + r.size(1) + 1 + a.size(1) + 1)).to(q.device) * self.pad_i
        trg = torch.ones_like(qra) * self.pad_i
        seg_mask = torch.ones_like(qra)
        ans_mask = torch.zeros_like(qra)
        qAr_mask = torch.zeros_like(qra)
        attention_mask = torch.ones((bsz, qra.size(1), qra.size(1))).to(q.device)
        for bsz_i in range(bsz):
            q_c = q[bsz_i, :]
            r_c = r[bsz_i, :]
            a_c = a[bsz_i, :]
            n_q_c = q_c[q_c != self.pad_i].long()
            n_r_c = r_c[r_c != self.pad_i].long()
            n_a_c = a_c[a_c != self.pad_i].long()

            # masked q + r & a -> inputs
            # q + r & shifted a -> labels
            n_qra = torch.cat((self.cls_i, n_q_c, self.cls_i, n_r_c, self.sep_i, n_a_c, self.sep_i), dim=0)
            t_qra = torch.cat((self.cls_i, n_q_c, self.cls_i, n_r_c, n_a_c, self.sep_i), dim=0)
            qra[bsz_i, :len(n_qra)] = n_qra
            trg[bsz_i, :len(t_qra)] = t_qra

            # segment mask & answer mask & q + r mask & attention mask
            sen_1_len = len(n_qra) - 1 - len(n_a_c)  # length of cls + q + sep + r + sep
            seg_mask[bsz_i, :(len(n_q_c) + 2)] *= 0  # q parts
            seg_mask[bsz_i, sen_1_len:len(n_qra)] *= 0  # a parts
            qAr_mask[bsz_i, 1:(len(n_q_c) + 1)] = 1  # q parts
            qAr_mask[bsz_i, (len(n_q_c) + 2):(sen_1_len - 1)] = 1  # r parts
            # filling parts for q & r
            qAr_mask[bsz_i, len(n_qra):(len(n_qra) + r.size(1) + q.size(1) - len(n_r_c) - len(n_q_c))] = 1
            ans_mask[bsz_i, sen_1_len - 1:len(n_qra)] = 1  # a parts
            # filling parts for a
            ans_mask[bsz_i, (len(n_qra) + r.size(1) + q.size(1) - len(n_r_c) - len(n_q_c)):] = 1
            attention_mask[bsz_i, :, :] = self._get_padding_mask(qra[bsz_i, :], sen_1_len)
        return qra.long(), trg, ans_mask, qAr_mask, seg_mask.long(), attention_mask

    def memorygate(self, query, kv_memory, mem_encoder, gate_projector):
        zvalue, attn = mem_encoder(query, kv_memory, kv_memory)  # (B, LQ, H), (B, LQ, LK)
        gate = gate_projector(torch.cat((zvalue, query), dim=2))  # (B, LQ, 2H) -> (B, LQ, 1)
        gate = self.activate(gate)

        return gate, attn

    def splithiddenstates(self, all_hid, c_mask, a_mask):
        bsz, h_dim = all_hid.size(0), all_hid.size(2)
        c_condition = torch.cat([c_mask.unsqueeze(2)] * h_dim, dim=2)
        c_hid = all_hid[c_condition == 1].reshape((bsz, -1, h_dim))
        a_condition = torch.cat([a_mask.unsqueeze(2)] * h_dim, dim=2)
        a_hid = all_hid[a_condition == 1].reshape((bsz, -1, h_dim))
        return c_hid, a_hid

    def memoryupdater(self, c_mem, a_mem, qAr_mask, a_mask, all_hidden):
        c_hid, a_hid = self.splithiddenstates(all_hidden, qAr_mask, a_mask)
        if not torch.is_tensor(c_mem):
            c_mem, a_mem, c_gate = c_hid, a_hid, None
        else:
            # update context memory based on new inputs
            c_gate, _ = self.memorygate(c_mem, c_hid, self.c_mem_encoder, self.cgate_projector)
            c_mem = c_gate * c_mem + (1.0 - c_gate) * c_hid

            # update answer memory based on context memory
            a_gate, _ = self.memorygate(a_mem, c_mem, self.a_mem_encoder, self.agate_projector)
            a_mem = a_gate * a_mem + (1.0 - a_gate) * a_hid
        return c_mem, a_mem, c_gate

    def forward(self, qrah):
        # que: (N, SQ) == (N, TQ)
        # rev: (N, R, SR)
        # ans: (N, A, TA)
        que = qrah[0]
        rev = qrah[1]
        ans = qrah[2]

        self.cls_i = torch.tensor([self.cls_i]).to(que.device)
        self.sep_i = torch.tensor([self.sep_i]).to(que.device)

        lm_losses = []
        c_memory, a_memory = None, None
        chime_trgs = []

        for i in range(ans.size(1)):
            for j in range(rev.size(1)):
                qra, trg, ans_mask, qAr_mask, seg_mask, a_mask = self.dynamic_padding(que, rev[:, j, :], ans[:, i, :])
                hidden_states = self.encoder(input_ids=qra, token_type_ids=seg_mask, perm_mask=(1.0 - a_mask))
                last_hidden_states = hidden_states[0]
                c_memory, a_memory, cgate = self.memoryupdater(c_memory, a_memory, qAr_mask,
                                                               ans_mask, last_hidden_states)
            chime_trgs.append(trg[ans_mask == 1])
        output = self.decoder(a_memory)
        for chime_trg in chime_trgs:
            lm_loss = self._compute_loss(output, chime_trg, soft=False)
            lm_losses.append(lm_loss)
        logits = output[:, ans.size(2), :]

        return sum(lm_losses) / len(lm_losses), logits
