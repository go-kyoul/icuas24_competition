import subprocess
import time


thr = 0.12
for i in range(1):
    node1_process = subprocess.Popen(['python', 'sim_ws/src/icuas24_competition/scripts/gs_planner.py'])
    node2_process = subprocess.Popen(['python', 'sim_ws/src/icuas24_competition/scripts/fruit_detector.py'])

    node1_process.wait()
    node2_process.terminate()
    assos_process = subprocess.run(['python', 'sim_ws/src/icuas24_competition/scripts/assosiation.py', 
                                    str(thr), str(i)])

            
