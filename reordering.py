import os


def readinput (path):
    inputList=sorted([f for f in os.listdir(path)])
    maxLength=max(len(infile) for infile in inputList)
    minLength=min(len(infile) for infile in inputList)
    # print(inputList) 
    # print("maxLength {: }".format(maxLength))
    # print("minLength {: }".format(minLength))
    if maxLength == minLength:      
        seqList=[path + infile for infile in inputList]

    else:
        # special designed condition for casme2 dB, SMIC should not be involved
        tempList=[]
        for index in inputList:
            if len(index) == 12:
                tempVidName=int(index[-5:-4])
            elif len(index) == 13:
                tempVidName=int(index[-6:-4])
            elif len(index) == 14:
                tempVidName=int(index[-7:-4])
            else:
                print ("Exceed the predefined range!")
            tempList.append(tempVidName)
        tempList=sorted(tempList)
        seqList=[path + 'reg_img' + str(infile) + '.jpg' for infile in tempList]

    return seqList
         
        
    
            
        
        
