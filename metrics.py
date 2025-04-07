import glob
import pandas as pd
from tabulate import tabulate
import argparse

def classic_vle(pos, neg):
    cnt, tot = 0, len(pos)
    for i in range(len(pos)):
        if pos[i] > neg[i]:
            cnt += 1
    return (round(cnt/tot,3), cnt, tot)

def strict_vle(pos, neg):
    cnt, tot = 0, len(pos)
    for i in range(len(pos)):
        if pos[i] > 0.5 and neg[i] < 0.5:
            cnt += 1
    return (round(cnt/tot,3), cnt, tot)

def strict_vle_analysis(pos, neg):
    cnt, tot = 0, 0
    for i in range(len(pos)):
        if pos[i] > 0.5 and neg[i] < 0.5:
            cnt += 1
        if pos[i] > 0.5:
            tot += 1
    return (round(cnt/tot,3), round(tot/len(pos),3), tot)

def mcq(ans_a, ans_b):
    cnt_a, cnt_b, cnt_ab, tot = 0, 0, 0, 0
    for i in range(len(ans_a)):
        a_i, b_i = ans_a[i].split()[0], ans_b[i].split()[0]
        if (a_i == 'A' or a_i == 'B') and (b_i == 'A' or b_i == 'B'):
            tot += 1
            if a_i == 'A' and b_i == 'B':
                cnt_a += 1
                cnt_b += 1
                cnt_ab += 1
            elif a_i == 'A':
                cnt_a += 1
            elif b_i == 'B':
                cnt_b += 1

    return (round(cnt_a/tot,3), round(cnt_b/tot,3), round(cnt_ab/tot,3), tot)


def entail_evaluate(args):
    files = glob.glob(args.path+'/*.csv')
    print(args.path)

    for metric in [strict_vle, classic_vle, strict_vle_analysis]:
        print(metric)

        results = {}

        for file in sorted(files):
            name = file.split('/')[-1].split('.')[0]

            data = pd.read_csv(file)
            acc, _, _ = metric(data['pos_score'], data['neg_score'])

            results[name] = acc

        df = pd.DataFrame([results])
        print(tabulate(df, headers='keys', tablefmt='psql', maxheadercolwidths=10))




def mcq_evaluate(args):
    files = glob.glob(args.path+'/*.csv')
    print(args.path)

    results = {}

    for file in sorted(files):
        name = file.split('/')[-1].split('.')[0]

        data = pd.read_csv(file)
        res= mcq(data['gt_A'], data['gt_B'])

        results[name] = '\n'.join(list(map(str,res)))

    df = pd.DataFrame([results])
    print(tabulate(df, headers='keys', tablefmt='psql', maxheadercolwidths=10))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="./output/model_name",
        help="Directory to where model results are saved",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="entail",
        help="Evaluation type",
        choices=[
            "entail",
            "mcq",
        ],
    )

    args = parser.parse_args()

    if args.eval_type == 'entail':
        entail_evaluate(args)
    elif args.eval_type == 'mcq':
        mcq_evaluate(args)

