import glob
import pandas as pd
from tabulate import tabulate
import argparse


def strict_vle_closed(pos, neg):
    cnt, tot = 0,0
    for i in range(len(pos)):
        if pos[i] == 'Yes' and neg[i] == 'No':
            cnt += 1
            tot += 1
        elif (pos[i] == 'Yes' or pos[i] == 'No') and (neg[i] == 'Yes' or neg[i] == 'No'):
            tot += 1
    return (round(cnt/tot,3), cnt, tot)

def strict_vle_analysis_closed(pos, neg):
    cnt, pos_tot, tot = 0,0,0
    for i in range(len(pos)):
        if pos[i] == 'Yes' and neg[i] == 'No':
            cnt += 1
            tot += 1
            pos_tot += 1
        elif pos[i] == 'Yes' and (neg[i] == 'Yes' or neg[i] == 'No'):
            pos_tot += 1
            tot += 1
        elif (pos[i] == 'Yes' or pos[i] == 'No') and (neg[i] == 'Yes' or neg[i] == 'No'):
            tot += 1
    return (round(cnt/pos_tot,3), round(pos_tot/tot,3), tot)

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

    for metric in [strict_vle_closed, strict_vle_analysis_closed]:
        print(metric)

        results = {}

        for file in sorted(files):
            name = file.split('/')[-1].split('.')[0]

            data = pd.read_csv(file)
            acc, _, _ = metric(data['pos_pred'], data['neg_pred'])

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

