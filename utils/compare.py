from utils.attribute_combination import AttributeCombination as AC


def compare(label, pred, columns):
    label = AC.batch_from_string(label, attribute_names=columns)
    pred = AC.batch_from_string(pred, attribute_names=columns)

    # 算法报告的属性组合在根因中(TP)
    # 算法报告的属性组合不在根因中(FP)
    # 算法没有报告根因中的属性组合(FN)
    fn = len(label)
    tp, fp = 0, 0
    for rc_item in pred:
        if rc_item in label:
            fn -= 1
            tp += 1
        else:
            fp += 1
    return fn, tp, fp
