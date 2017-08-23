def collectinglabel(Table, sub, videoName, workplace, db):

    if db == "SMIC":
        for var in range(len(Table)):
            result=-1
            
            if videoName == Table[var,0]:
                result=int(Table[var,1])
                break
    else:
        for var in range(len(Table)):
            result =-1
            if videoName == Table[var,1] and sub == Table[var,0]:
                if Table[var,2]=='happiness':
                    result=0
                    break
                if Table[var,2]=='disgust':
                    result=1
                    break
                if Table[var,2]=='repression':
                    result=2
                    break
                if Table[var,2]=='surprise':
                    result=3
                    break
                if Table[var,2]=='others':
                    result=4
                    break
        
         
    if result != -1 :
        with open(workplace + db+'_label.txt','a') as f:
            f.write(str(result) + '\n')
            f.close()
    else:
        
        if db == 'SMIC':
            print ('Cannot find the matching label for %s'%(videoName))
        else:
            print ('Cannot find the matching label for %s of %s'%(videoName,sub))

    
