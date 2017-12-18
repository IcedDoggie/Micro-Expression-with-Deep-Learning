def collectinglabel(Table, sub, videoName, workplace, db):
    if db == "SMIC_TIM10":
        counter = 0

        for var in ((Table[0, :, 0])):
            result = -1
            # print(var)
            if videoName == var:
                result = int(Table[0, counter, 1])
                if result == 1: # negative
                    result = 0
                elif result == 2: # positive
                    result = 1
                elif result == 3: # surprise
                    result = 2
                # print("found: %s" % (videoName) )
                break
            counter += 1
    elif db == "SAMM_TIM":
        counter = 0
        for var in ((Table[0, :, 0])):
            result = -1
            if videoName == var or videoName in var:
                result = (Table[0, counter, 1])
                if result == 'Anger': # negative
                    result = 0
                elif result == 'Contempt': # positive
                    result = 1
                elif result == 'Disgust': # surprise
                    result = 2
                elif result == 'Fear': # surprise
                    result = 3
                elif result == 'Happiness': # surprise
                    result = 4
                elif result == 'Other': # surprise
                    result = 5
                elif result == 'Sadness': # surprise
                    result = 6
                elif result == 'Surprise': # surprise
                    result = 7                    
                # print("found: %s" % (videoName) )
                # print(result)
                break
            counter += 1

    else:

        for var in range(len(Table)):

            result = -1
            if ( videoName == Table[var,1] or Table[var, 1] in videoName ) and sub == Table[var,0]:
                # print(Table[var])
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
                # if Table[var,2]=='sadness':
                #     result = 5
                #     break
                # if Table[var,2]=='fear':
                #     result = 6
                #     break
        

    if result != -1 :

        with open(workplace + db + '_label.txt','a') as f:
            f.write(str(result) + '\n')
            f.close()

    elif result == -1:
        
        if db == 'SMIC':
            print ('Cannot find the matching label for %s'%(videoName))
        else:
            print ('Cannot find the matching label for %s of %s'%(videoName,sub))
        # file_to_be_ignored = Table[videoName]

        # return file_to_be_ignored
    
