import numpy as np

b = {}
Liste = [['macron', 'lepen', 'melanchon'], ['connard', 'macron', 'pecresse'], ['oui', 'non', 'pecresse', 'melanchon'], ['cacahuette']]
i = 0
retweet = [12, 4, 6, 7]
for liste1 in Liste:
    for mot in liste1:
        if mot not in b:
            b[mot] = [retweet[i]]
        else:
            b[mot].append(retweet[i])
    i += 1

a = {}
for mot in b:
    a[mot] = np.mean(b[mot])
print(b)
print(a)

