import numpy as np

def fpr(matrix,n_exp):

    diag=[]
    preMatrix=[]
    reMatrix=[]
    preList=[]
    reList=[]
    for index in range(n_exp):
        diag=matrix[index,index]
        col=sum(matrix[:,index])
        row=sum(matrix[index])

        if index ==0:
            prec=diag/col
            rec=diag/row
        else:
            prec = prec + diag/col
            rec = rec + diag/row


    precision=prec/n_exp
    recall=rec/n_exp
    f1=2*precision*recall/(precision+recall)

        
    return [f1,precision,recall]
