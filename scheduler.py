from subprocess import call, Popen, run
import time
import threading
import queue
import os
from pynvml.pynvml import *

# start_time = time.time()
# call(["python", "main.py", "--train=./train.py" , "--batch_size=10", "--spatial_epochs=100", "--temporal_epochs=100", "--train_id=default", "--dB=SAMM_CASME_Optical", "--spatial_size=224", "--flag=st"])
# elapsed_time = time.time() - start_time
# print(str(elapsed_time) + " seconds\n")

def console(q, lock):
	while 1:
		input()
		free_mem = check_gpu_resources()
		print("Available VRAM: %0.2f %%" % (free_mem))
		print(q.qsize())
		input()
		with lock:
			cmd = input('> ')
		if free_mem > 1:
			q.put(cmd)

		if cmd == 'quit':
			break

def invalid_input(lock):
	with lock:
		print("Invalid Command")	

def action_help(lock):
	with lock:
		print("Run python scripts, for eg: python main.py --dB 'CASME2_Optical'")		

def run_process(lock):
	with lock:
		input()
		cmd = input('command> ')
		filename = input('filename> ')
		run_command(cmd, filename)


def check_gpu_resources():
	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
			nvmlDeviceGetName(handle),
			meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    
	free_memory = meminfo.free/1024.**2
	used_memory = meminfo.used/1024.**2
	total_memory = meminfo.total/1024.**2

	free_percentage =  ( free_memory / total_memory ) * 100 

	return free_percentage

def run_command(command, nohup_output):
	# log in text file
	# os.system("x-terminal-emulator -e /bin/bash")

	command = "nohup " + command + " > " + nohup_output + " &"
	print(command)
	# command = command.split(' ')
	
	# command.insert(0, '-e')
	# command.insert(0, 'x-terminal-emulator')
	# command.insert(0, 'nohup')
	# command.insert(-2, ' ')
	# nohup_output = "test"
	# print(command)
# "nohup python main.py --dB 'CASME2_Optical' 'CASME2_Strain_TIM10' --batch_size=1 --spatial_epochs=100 --temporal_epochs=100 --train_id='default_test' --spatial_size=224 --flag='st4se' > nohup_custom.log &"	
# ['nohup', 'python', 'main.py', '--dB', "'CASME2_Optical'", "'CASME2_Strain_TIM10'", '--batch_size=1', '--spatial_epochs=100', '--temporal_epochs=100', "--train_id='default_test'", '--spatial_size=224', "--flag='st4se'", '> nohup_custom.log &']
	run(command, shell=True, check=True)
	# call(['exec', 'bash'])
	# call(command

def main():
	cmd_actions = {'help': action_help, 'run_command': run_process}
	cmd_queue = queue.Queue()
	stdout_lock = threading.Lock()

	dj = threading.Thread(target=console, args=(cmd_queue, stdout_lock))
	dj.start()

	while 1:
		# getting the next command from queue
		cmd = cmd_queue.get()
		
		if cmd == 'quit':
			break

		# execute action
		action = cmd_actions.get(cmd, invalid_input)
		action(stdout_lock)

main()

