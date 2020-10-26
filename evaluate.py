import json
from config import configs
from rouge import Rouge
from bleurt.score import BleurtScorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()
rouge = Rouge()
scorer = BleurtScorer("bleurt-base-128")
weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1/3, 1/3, 1/3, 0), (0.25, 0.25, 0.25, 0.25)]

output_path = configs['evaluation_output_path'] + "%s.txt" % configs['task_name']
w = open(output_path, "w")
for epoch in range(1, configs['epochs'] + 1):
    w.write("Epoch " + str(epoch) + "\n")
    bleus = [0.0] * 4
    rouge_l = [0.0] * 1
    bert_score = [0.0] * 1
    bleurt_score = [0.0] * 1
    count = 0
    prediction_path = configs['prediction_output_path'] + "%s_%s.json" % (configs['task_name'], str(epoch))
    records = open(prediction_path, "r").readlines()[0]
    single_cands = []
    multi_refs = []
    for r_i, record in enumerate(json.loads(records)):
        w.write("Batch %s" % record["batch"] + " Case %s" % record["case"] + "\n")
        w.write("Question : " + record["question"] + "\n")
        answers = []
        for i, answer in enumerate(record["answers"]):
            answers.append(answer.lower())
            w.write("Reference Answer " + str(i + 1) + " : " + answer + "\n")
        w.write("Predicted Answer : " + record["prediction"] + "\n")
        w.write("=" * 20 + "\n")
        w.write("=" * 20 + "\n")

        # 1. Bleu 1-4
        try:
            rouges_score_i = rouge.get_scores([record["prediction"].lower()] * len(answers), answers)
            bleurt_score_i = scorer.score(answers, [record["prediction"].lower()] * len(answers))
            bleus_i = [sentence_bleu([answer.split(" ") for answer in answers],
                                     record["prediction"].lower().split(" "), weights=weight, auto_reweigh=True,
                                     smoothing_function=cc.method3) * 100 for weight in weights]
        except:
            print(record["prediction"].lower())
            continue
        rouges_score_i = max([rouges_score_ia["rouge-l"]["f"] for rouges_score_ia in rouges_score_i])
        rouge_l[0] += rouges_score_i * 100
        bleurt_score_i = max(bleurt_score_i)
        bleurt_score[0] += bleurt_score_i
        bleus = [bleu + bleus_i[j] for j, bleu in enumerate(bleus)]
        count += 1

        single_cands.append(record["prediction"].lower())
        multi_refs.append(answers)

    bert_score[0] += score(single_cands, multi_refs, lang="en", rescale_with_baseline=True)[2].mean().item()
    w.write("Beam Search Bleu-1 : " + str(round(bleus[0] / count, 3)) + "\n")
    w.write("Beam Search Bleu-2 : " + str(round(bleus[1] / count, 3)) + "\n")
    w.write("Beam Search Bleu-3 : " + str(round(bleus[2] / count, 3)) + "\n")
    w.write("Beam Search Bleu-4 : " + str(round(bleus[3] / count, 3)) + "\n")
    w.write("Beam Search Rouge-L : " + str(round(rouge_l[0] / count, 3)) + "\n")
    w.write("Beam Search BertScore : " + str(round(bert_score[0], 3)) + "\n")
    w.write("Beam Search BLEURT : " + str(round(bleurt_score[0] / count, 3)) + "\n")
    print("Epoch %s done" % str(epoch))
w.close()
