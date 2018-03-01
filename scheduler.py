from subprocess import call


call(["python", "main.py", "--train=./train.py" , "--batch_size=10", "--spatial_epochs=100", "--temporal_epochs=100", "--train_id=default", "--dB=CASME2_TIM", "--spatial_size=224", "--flag=st"])
