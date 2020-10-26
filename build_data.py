import os
import torch
import random
import json
from torchtext import data
from config import configs
from utils import random_seed
import re
pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.S)
_ = random_seed(configs['seed'])
tokenizer = configs['tokenizer']
pad_index = configs['pad_index']
root_path = configs['root_path']
pretrained_weights = configs['pretrained_weights']
data_size = configs['data_size']


def DataSelection(input_file, output_file):
    w = open(output_file, "w")
    w_c, r_c = 0, 0
    for line in open(input_file, "r"):
        r_c += 1
        line_dict = json.loads(line)
        if (line_dict["is_answerable"] == 1) and (line_dict["questionType"] == "descriptive") and \
                (len(line_dict["review_snippets"]) == 10):
            new_q = pattern.sub("", line_dict["questionText"])
            if len(new_q.split(" ")) <= 2:
                continue
            line_dict["questionText"] = new_q
            new_rs = []
            for review in line_dict["review_snippets"]:
                new_r = pattern.sub("", review)
                if len(new_r.split(" ")) <= 2:
                    break
                new_rs.append(new_r)
            if len(new_rs) != 10:
                continue
            line_dict["review_snippets"] = new_rs
            new_as = []
            for answer in line_dict["answers"]:
                new_a = answer
                new_at = pattern.sub("", new_a["answerText"])
                if new_at == "":
                    continue
                new_as.append(new_a)
            if len(new_as) == 0:
                continue
            line_dict["answers"] = new_as
            w.write(json.dumps(line_dict) + "\n")
            w_c += 1
    w.close()
    print("%s takes %s percent of %s" % (output_file, round(w_c / r_c, 2) * 100, input_file))


def preprocessor(input_data, refer_len, pre_tokenizer, pre_pad_index):
    input_data = " ".join(input_data).lower()
    try:
        tokens = pre_tokenizer.encode(input_data, add_special_tokens=False)
    except:
        print("input text contains non-utf8 words")
        tokens = []
    start_len = len(tokens)
    if start_len < refer_len:
        tokens += [pre_pad_index] * (refer_len - start_len)
    else:
        tokens = tokens[:refer_len]
    assert len(tokens) == refer_len
    return tokens


def q_preprocessor(input_data):
    return preprocessor(input_data, configs["question_length"], tokenizer, pad_index)


def r_preprocessor(input_data):
    return preprocessor(input_data, configs["single_review_length"], tokenizer, pad_index)


def a_preprocessor(input_data):
    return preprocessor(input_data, configs["single_answer_length"], tokenizer, pad_index)


class AmazonQA():
    def __init__(self):
        path = root_path + 'amazonqa'
        dataset_path = path + '/torchtext_' + pretrained_weights + '/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'val_examples.pt'
        test_examples_path = dataset_path + 'test_examples.pt'

        self.RAW = data.RawField()
        self.RAW.is_target = False
        self.Q_WORD = data.Field(batch_first=True, preprocessing=q_preprocessor)
        self.R_WORD = data.Field(batch_first=True, preprocessing=r_preprocessor)
        self.A_WORD = data.Field(batch_first=True, preprocessing=a_preprocessor)
        self.REVIEWS = data.NestedField(self.R_WORD)
        self.ANSWERS = data.NestedField(self.A_WORD)

        dict_fields = {'qid': ('qid', self.RAW),
                       'questionText': ('q_word', self.Q_WORD),
                       'asin': ('asin', self.RAW),
                       'category': ('category', self.RAW),
                       'review_snippets': ('reviews', self.REVIEWS),
                       'answers.answerText': ('answerText', self.ANSWERS),
                       'answers.helpful': ('helpful', self.RAW)}

        list_fields = [('qid', self.RAW),
                       ('q_word', self.Q_WORD),
                       ('asin', self.RAW),
                       ('category', self.RAW),
                       ('reviews', self.REVIEWS),
                       ('answerText', self.ANSWERS),
                       ('helpful', self.RAW)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)
            test_examples = torch.load(test_examples_path)
            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
            self.test = data.Dataset(examples=test_examples, fields=list_fields)
        else:
            train_path, valid_path, test_path = path + "/train-qar-filter.jsonl", \
                                                path + "/val-qar-filter.jsonl", path + "/test-qar-filter.jsonl"
            if not os.path.exists(train_path):
                print("building filtered subset...")
                DataSelection(path + "/train-qar.jsonl", train_path)
                DataSelection(path + "/val-qar.jsonl", valid_path)
                DataSelection(path + "/test-qar.jsonl", test_path)

            print("building splits...")
            self.train, self.dev, self.test = data.TabularDataset.splits(path=path, train=train_path,
                                                                         validation=valid_path, test=test_path,
                                                                         format='json', fields=dict_fields)
            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)
            torch.save(self.test.examples, test_examples_path)

        self.train.examples = random.sample(self.train.examples, int(data_size * len(self.train.examples)))
        self.dev.examples = random.sample(self.dev.examples, int(data_size * len(self.dev.examples)))
        self.test.examples = random.sample(self.test.examples, int(data_size * len(self.test.examples)))
