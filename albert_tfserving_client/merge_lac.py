# 判断文件是否存在等各种状态都是需要调用特定的函数进行判断
# label:OOOOOOOOOOOOOOOOOOOOOOOOOOOO
# predict:[CLS]v-Bv-In-Bn-Id-Bn-Iv-Bv-Ia-Br-Br-In-Bn-Iv-Bv-Iv-Bv-Iv-Bv-Ia-Ba-Iu-Bn-Bn-Iv-Bv-Ivn-Bvn-Ian-B


def merge_line(line):
    sen, _, pred = line.strip().split('\t')
    senls = sen.split('\x02')
    predls = pred[len('predict:[CLS]\x02'):].split('\x02')
    return merge_line2(senls, predls)


def merge_line2(senls, predls):
    segment = []
    segi = []
    last_tag = ''
    for i, predi in enumerate(predls):
        if predi == '[CLS]':
            continue
        if i == len(senls) or predi == '[SEP]':
            segi.append(last_tag)
            segment.append(tuple(segi))
            break
        # print(predi)
        if '-' not in predi:
            continue
        tag, bi = predi.split('-')
        if bi == 'B' or not tag == last_tag:
            if not last_tag == '':
                segi.append(last_tag)
                segment.append(segi)
            # new seg
            segi = [senls[i]]
        else:
            segi.append(senls[i])
        last_tag = tag
    return (['{}_{}'.format(''.join(s[:-1]), s[-1]) for s in segment])


def merge(filep):
    with open(filep, 'r', encoding='utf8') as r:
        lines = r.readlines()
        for line in lines:
            print(merge_line(line))
