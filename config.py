from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='0,1', type=str, required=False, help='GPUs for using, in sequence')
    parser.add_argument('--model', default='xlnet-base-cased', type=str, required=False, help='Pre-trained Encoder')
    parser.add_argument('--root_path', default='', type=str, required=True, help='root path of your data and features')
    parser.add_argument('--data_size', default=1.0, type=float, required=False, help='percentage of data to be used')
    parser.add_argument('--epochs', default=5, type=int, required=False)
    parser.add_argument('--batch_size_perGPU', default=8, type=int, required=False, help='batch size of each GPU')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='learning rate')
    parser.add_argument('--warm_up_rate', default=0.2, type=float, required=False, help='percentage of warm up steps')
    parser.add_argument('--gradient_accumulations', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_output_path', default='', type=str, required=True, help='output path of saved models')
    parser.add_argument('--prediction_output_path', default='', type=str, required=True,
                        help='output path of predictions')
    parser.add_argument('--evaluation_output_path', default='', type=str, required=True,
                        help='output path of evaluations')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--task', type=str, default=None, required=True, help="task type",
                        choices=('Train', 'Test', 'Predict', 'Analyze', 'Evaluate'))
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='beam size for text generation')
    parser.add_argument('--question_length', default=40, type=int, required=False, help='fix length for questions')
    parser.add_argument('--single_review_length', default=124, type=int, required=False, help='fix length for reviews')
    parser.add_argument('--single_answer_length', default=82, type=int, required=False, help='fix length for answers')
    parser.add_argument('--fp16', default="O1", type=str, required=False, help='whether to use apex or not, and mode')
    parser.add_argument('--rev_num', default=10, type=int, required=False, help='the number of reviews to be used',
                        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    parser.add_argument('--ans_num', default=1, type=int, required=False, help='the number of answers to be used',
                        choices=(1, 2, 3, 4, 5))
    return parser.parse_args()


args = setup_args()
configs = {}

# settings for tokenizer
configs['pretrained_weights'] = args.model
configs['tokenizer'] = AutoTokenizer.from_pretrained(configs['pretrained_weights'])
configs['pad_index'] = configs['tokenizer'].pad_token_id
configs['cls_index'] = configs['tokenizer'].cls_token_id
configs['sep_index'] = configs['tokenizer'].sep_token_id
configs['msk_index'] = configs['tokenizer'].mask_token_id

# settings for building dataset
# 95% questions has tokenized question length <= 40
# 95% reviews has tokenized review length <= 124
# 95% answers has tokenized answer length <= 82
configs['question_length'] = args.question_length
configs['single_review_length'] = args.single_review_length
configs['single_answer_length'] = args.single_answer_length
configs['data_size'] = args.data_size

# settings for training
configs['ans_num'] = args.ans_num
configs['rev_num'] = args.rev_num
configs['batch_size'] = args.batch_size_perGPU
configs['epochs'] = args.epochs
configs['seed'] = args.seed
configs['lr'] = args.lr
configs['task'] = args.task
configs['task_name'] = configs['pretrained_weights'] + "_" + \
                       str(configs['rev_num']) + "R" + "_" + \
                       str(configs['ans_num']) + "A" + "_" + \
                       str(configs['data_size'])
configs['devices'] = [int(d) for d in args.devices.split(",")]
configs['device'] = configs['devices'][0]
configs['root_path'] = args.root_path
configs['fp16'] = args.fp16
configs['gradient_accumulations'] = args.gradient_accumulations
configs['warm_up_rate'] = args.warm_up_rate
configs['max_grad_norm'] = args.max_grad_norm
configs['model_output_path'] = args.model_output_path

# settings for prediction
configs['beam_size'] = args.beam_size
configs['prediction_output_path'] = args.prediction_output_path

# settings for evaluation
configs['evaluation_output_path'] = args.evaluation_output_path
