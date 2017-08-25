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
        if index ==0:
            prec=diag/(col + denominator_De_zero )
            # print(prec)
            rec=diag/(row + denominator_De_zero)

            # print(rec)
        else:
            prec = prec + diag/(col + denominator_De_zero)
            # print(prec)
            rec = rec + diag/(row + denominator_De_zero)
            # print(rec)

    precision=prec/(n_exp + denominator_De_zero)
    recall=rec/(n_exp + denominator_De_zero)
    f1=2*precision*recall/(precision+recall)

        
    return [f1,precision,recall]
