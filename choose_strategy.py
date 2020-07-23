#  ------------------------------------------------------------------
#  Author: Bowen Wu
#  Email: wubw6@mail2.sysu.edu.cn
#  Affiliation: Sun Yat-sen University, Guangzhou
#  Date: 14 JULY 2020
#  ------------------------------------------------------------------
import sys

data_path = sys.argv[1]

score = []

print("#" * 10, data_path, "#" * 10)
with open(data_path) as data:
    line = data.readlines()
    for l in line:
        d = l.split(" ")
        score.append(float(d[0]))

score_sorted_index = sorted(range(len(score)), key=lambda k: score[k], reverse=True)

for i in range(5):
    print(
        "strategy index:{}, score:{}".format(
            score_sorted_index[i], score[score_sorted_index[i]]
        )
    )
print("\n")
