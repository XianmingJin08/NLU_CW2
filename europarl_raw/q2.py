import numpy as np
import difflib

if __name__ == "__main__":

    # 2.1
    # with open('./train.en') as f:
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
    # with open('./train.de') as f:
    #     de_dict = {}
    #     for line in f:
    #         for w in line.split():
    #             if w in de_dict:
    #                 de_dict[w] +=1
    #             else:
    #                 de_dict[w] = 1
    #     total_count = 0
    #     for k in de_dict.keys():
    #         total_count += de_dict[k]
    # with open('./train.en') as f:
    #     en_dict = {}
    #     for line in f:
    #         for w in line.split():
    #             if w in en_dict:
    #                 en_dict[w] += 1
    #             else:
    #                 en_dict[w] = 1
    #     total_count = 0
    #     for k in en_dict.keys():
    #         total_count += en_dict[k]
    #
    # uniques_dict = {}
    # uniques = 0
    # for k in en_dict.keys():
    #     if (k in de_dict.keys()):
    #         if k in uniques_dict:
    #             uniques_dict [k] += 1
    #         else:
    #             uniques_dict [k] = 1
    # print (len(uniques_dict.keys()))


    # q5 find examples

    rare_words = {}

    with open('./train.en') as f:
        en_dict = {}
        for line in f:
            for w in line.split():
                if w in en_dict:
                    en_dict[w] +=1
                else:
                    en_dict[w] = 1
        for k in en_dict.keys():
            if (en_dict[k] <= 3):
                rare_words [k] = en_dict[k]

    print (len(rare_words.keys()))

    with open('model_translations_baseline.txt') as f:
        translated_unk_b = {}
        total_unk_b = 0
        for line in f:
            for w in line.split():
                if w in rare_words:
                    if (w not in translated_unk_b.keys()):
                        total_unk_b+=1
                    translated_unk_b [w] = 1
        # print (translated_unk_b)
        print (total_unk_b)

    with open('model_translations_q5.txt') as f:
        translated_unk_q5 = {}
        total_unk_q5 = 0
        for line in f:
            for w in line.split():
                if w in rare_words.keys():
                    if (w not in translated_unk_q5.keys()):
                        total_unk_q5+=1
                    translated_unk_q5 [w] = 1


        # print (translated_unk_q5)
        print (total_unk_q5)

    # diff = []
    # for w in translated_unk_q5.keys():
    #     if w not in translated_unk_b.keys():
    #         diff.append(w)
    # print (len(diff))

    # trans_q5 = open("model_translations_q5.txt").readlines()
    # trans_b = open("model_translations_baseline.txt").readlines()
    ref = open("test.en").readlines()

    print (len(ref))

    ref_dic = {}
    for r in ref:
        for w in r.split():
            if w in rare_words:
                ref_dic [w] = 1

    print (ref_dic)
    rare_b = 0
    rare_q5 = 0
    rare_bs = []
    rare_q5s = []
    for r in ref_dic.keys():
        if r in translated_unk_b:
            rare_b += 1
            rare_bs.append(r)
        if r in translated_unk_q5:
            rare_q5 +=1
            rare_q5s.append(r)
    print (rare_b)
    print (rare_q5)
    print (rare_bs)
    print (rare_q5s)
    # for line in difflib.unified_diff(trans_q5, trans_b):
    #     print ("newline")
    #     print (line)