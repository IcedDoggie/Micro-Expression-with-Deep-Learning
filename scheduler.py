from subprocess import call, Popen, run
import time
import threading
import queue
import os
from pynvml.pynvml import *
import time

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

def run_process(q, lock, threshold):
	while 1:
		with lock:
			free_mem = check_gpu_resources()
			print("Available VRAM: %0.2f %%" % (free_mem))

			input()
			print("%i process on queue." % (q.qsize()))
			cmd = input('command> ')
			filename = input('filename> ')
			cmd = "nohup " + cmd + " > " + filename + "&"
			q.put(cmd)

			free_mem = check_gpu_resources()
			print("Available VRAM: %0.2f %%" % (free_mem))

		# if free_mem >= float(threshold):
		# 	cmd = q.get()
		# 	run(cmd, shell=True, check=True)


def check_gpu_resources():
	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		# print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
		# 	nvmlDeviceGetName(handle),
		# 	meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    
	free_memory = meminfo.free/1024.**2
	used_memory = meminfo.used/1024.**2
	total_memory = meminfo.total/1024.**2

	free_percentage =  ( free_memory / total_memory ) * 100 

	return free_percentage



def main():
	cmd_queue = queue.Queue()
	stdout_lock = threading.Lock()	
	cmd_actions = {'help': action_help, 'run_process': run_process}

	threshold = input('Threshold > ')


	dj = threading.Thread(target=run_process, args=(cmd_queue, stdout_lock, threshold))
	dj.start()

	flag = 0
	while 1:
		free_mem = check_gpu_resources()
		if free_mem > float(threshold) and flag == 0:
			cmd = cmd_queue.get()
			flag = 1
			run(cmd, shell=True, check=True)
			start = time.time()

		elif flag == 1 and (time.time()-start) > 120:
			flag = 0
		cmd = cmd_queue.qsize()


main()

