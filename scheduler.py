from subprocess import call
import time


start_time = time.time()
call(["python", "main.py", "--train=./train.py" , "--batch_size=10", "--spatial_epochs=100", "--temporal_epochs=100", "--train_id=default", "--dB=SAMM_CASME_Optical", "--spatial_size=224", "--flag=st"])
elapsed_time = time.time() - start_time
print(str(elapsed_time) + " seconds\n")

