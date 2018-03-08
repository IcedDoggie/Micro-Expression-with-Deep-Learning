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
		print("%i process on queue." % (q.qsize()))
		input()
		with lock:
			cmd = input('> ')

			# q.put(cmd)


		if cmd == 'quit':
			break

def invalid_input(lock):
	with lock:
		print("Invalid Command")	

def action_help(lock):
	with lock:
		print("Run python scripts, for eg: python main.py --dB 'CASME2_Optical'")		

def run_process(q, lock):
	while 1:
		free_mem = check_gpu_resources()
		print("Available VRAM: %0.2f %%" % (free_mem))
		with lock:


			input()
			print("%i process on queue." % (q.qsize()))
			cmd = input('command> ')
			filename = input('filename> ')
			cmd = "nohup " + cmd + " > " + filename + "&"
			q.put(cmd)

		if free_mem >= 92:
			# print("born to run")
			cmd = q.get()
			run(cmd, shell=True, check=True)
		# else:
		# 	print("%i process new queue." % (q.qsize()))


# def run_command(command, nohup_output):
# 	command = "nohup " + command + " > " + nohup_output + " &"
# 	print(command)
# 	run(command, shell=True, check=True)


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



def main():
	cmd_queue = queue.Queue()
	stdout_lock = threading.Lock()	
	cmd_actions = {'help': action_help, 'run_process': run_process}


	dj = threading.Thread(target=run_process, args=(cmd_queue, stdout_lock))
	dj.start()

	while 1:
		# getting the next command from queue

		# print("queue")
		cmd = cmd_queue.qsize()

		# if cmd == 'quit':
		# 	break

		# 	# execute action
		# action = cmd_actions.get(cmd, invalid_input)
		# action(stdout_lock)

main()

