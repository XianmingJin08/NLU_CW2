import numpy as np

if __name__ == "__main__":

    # 2.1
    # with open('./train.de') as f:
    #     en_dict = {}
    #     for line in f:
    #         for w in line.split():
    #             if w in en_dict:
    #                 en_dict[w] +=1
    #             else:
    #                 en_dict[w] = 1
    #     print (len(en_dict.keys()))
    #     total_count = 0
    #     for k in en_dict.keys():
    #         total_count += en_dict[k]
    #     print (total_count)


    # 2.2
    # with open('./train.de') as f:
    #     en_dict = {}
    #     for line in f:
    #         for w in line.split():
    #             if w in en_dict:
    #                 en_dict[w] +=1
    #             else:
    #                 en_dict[w] = 1
    #     print (len(en_dict.keys()))
    #     total_unk = 0
    #     for k in en_dict.keys():
    #         if (en_dict[k] == 1):
    #             total_unk += en_dict[k]
    #     print (total_unk)

    # 2.3


    # with open('./train.de') as f:
    #     en_dict = {}
    #     for line in f:
    #         for w in line.split():
    #             if w in en_dict:
    #                 en_dict[w] +=1
    #             else:
    #                 en_dict[w] = 1
    #     print (len(en_dict.keys()))
    #     unknown = {}
    #     unk_words_bag = []
    #     for k in en_dict.keys():
    #         if (en_dict[k] == 1):
    #             print (k)

    # 2.4
    with open('./train.de') as f:
        de_dict = {}
        for line in f:
            for w in line.split():
                if w in de_dict:
                    de_dict[w] +=1
                else:
                    de_dict[w] = 1
        total_count = 0
        for k in de_dict.keys():
            total_count += de_dict[k]
    with open('./train.en') as f:
        en_dict = {}
        for line in f:
            for w in line.split():
                if w in en_dict:
                    en_dict[w] += 1
                else:
                    en_dict[w] = 1
        total_count = 0
        for k in en_dict.keys():
            total_count += en_dict[k]

    uniques_dict = {}
    uniques = 0
    for k in en_dict.keys():
        if (k in de_dict.keys()):
            if k in uniques_dict:
                uniques_dict [k] += 1
            else:
                uniques_dict [k] = 1
    print (len(uniques_dict.keys()))