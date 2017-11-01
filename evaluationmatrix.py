import numpy as np

def fpr(matrix,n_exp):

    diag=[]
    preMatrix=[]
    reMatrix=[]
    preList=[]
    reList=[]

    denominator_De_zero = 0.00001

    for index in range(n_exp):
        diag=matrix[index,index]
        col=sum(matrix[:,index])
        row=sum(matrix[index])
        # print(diag)
        # print(col)
        if index == 0:
            prec=diag/(col)
            # print(prec)
            rec=diag/(row)

            # print(rec)
        else:
            prec = prec + diag/(col)
            # print(prec)
            rec = rec + diag/(row)
            # print(rec)

    precision=prec/(n_exp)
    recall=rec/(n_exp)
    f1=2*precision*recall/(precision+recall)

        
    return [f1,precision,recall]
