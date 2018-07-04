def collectinglabel(Table, sub, videoName, workplace, db, objective_flag):
    
    # note: cross cases must be put on top

    # SAMM-CASME case
    if "SAMM_CASME" in db:
        for var in range(len(Table[0, :, 0])):
            result = -1

            # for casme: sub[3:]
            # for samm: sub
            if videoName == Table[0, var, 1] and sub == Table[0, var, 0]:
                result = Table[0, var, 2]
                result -= 1
                break
            elif videoName == Table[0, var, 1] and sub[3:] == Table[0, var, 0]:
                result = Table[0, var, 2]
                result -= 1
                break
    
    # SMIC only case
    elif "SMIC" in db:
        counter = 0

        for var in ((Table[0, :, 0])):
            result = -1
            # print(var)
            if videoName == var:
                result = int(Table[0, counter, 1])
                if result == 1: # negative
                    result = 0
                    break
                elif result == 2: # positive
                    result = 1
                    break
                elif result == 3: # surprise
                    result = 2
                    break
                # print("found: %s" % (videoName) )
                # break
            counter += 1

    # SAMM only case
    elif "SAMM" in db:

        counter = 0
        if objective_flag == 0:
            for var in ((Table[0, :, 0])):
                result = -1
                # print(Table[0,counter, 1])
                if videoName == var or videoName in var:
                    result = (Table[0, counter, 1])
                    if result == 'Anger': # negative
                        result = 0
                        break
                    elif result == 'Contempt': # positive
                        result = 1
                        break
                    elif result == 'Disgust': # surprise
                        result = 2
                        break
                    elif result == 'Fear': # surprise
                        result = 3
                        break
                    elif result == 'Happiness': # surprise
                        result = 4
                        break
                    elif result == 'Other': # surprise
                        result = 5
                        break
                    elif result == 'Sadness': # surprise
                        result = 6
                        break
                    elif result == 'Surprise': # surprise
                        result = 7
                        break                    
                    # print("found: %s" % (videoName) )
                    # print(result)
                    # break
                counter += 1            

 
    # CASME2 usually.
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
    
