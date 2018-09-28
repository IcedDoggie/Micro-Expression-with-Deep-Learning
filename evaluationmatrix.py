import numpy as np


def fpr(matrix,n_exp):

    diag = []
    preMatrix = []
    reMatrix = []
    preList = []
    reList = []

    denominator_De_zero = 0.00001

    for index in range(n_exp):
        diag = matrix[index,index]
        col = sum(matrix[:,index])
        row = sum(matrix[index])
        # print(diag)
        # print(col)
        if index == 0:
            prec = diag/(col + denominator_De_zero)
            # print(prec) 
            rec = diag/(row + denominator_De_zero)

            # print(rec)
        else:
            prec = prec + diag/(col + denominator_De_zero) 
            # print(prec)
            rec = rec + diag/(row + denominator_De_zero)
            # print(rec)

    precision = prec / (n_exp)
    recall = rec /(n_exp)    # print(precision)
    # print(recall)
    f1 = 2 * precision * recall / (precision + recall)

        
    return [f1,precision,recall]

def weighted_average_recall(matrix, n_exp, total_N):
    # normal recognition accuracy
    # war = no. correct classified samples / total number of samples
    number_correct_classified = 0

    for index in range(n_exp):
        diag = matrix[index, index]
        number_correct_classified += diag

    war = number_correct_classified / total_N

    return war

def unweighted_average_recall(matrix, n_exp):
    # balanced recognition accuracy
    # uar = sum(accuracy of each class ) / number of classes
    sum_of_accuracy = 0
    for index in range(n_exp):
        diag = matrix[index, index]
        row = sum(matrix[index])
        accuracy_of_n_class = diag / row
        sum_of_accuracy += accuracy_of_n_class        

    uar = sum_of_accuracy / n_exp


    return uar