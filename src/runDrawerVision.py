#! /usr/bin/env python3

import socket
import numpy as np
import time
import os
# import interfacesReach2GraspVision
import interfacesDrawerCV
import pandas as pd

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 5006)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)



# interface = interfacesReach2GraspVision.DiscreteActionsRobot()
interface = interfacesDrawerCV.DiscreteActionsRobot()
interface.reset()
# interface.userDisplay.updateFeedback((100,100,100))
# interface.userDisplay.updateText(2)
robot_vel = np.array([0.0,0.0,0.0])
robot_pos = np.array([0.0, 0.0, 0.0])
target_pos  = np.array([0.0, 0.0, 0.0])

interface.target_pos = target_pos + [-0.20, -0.40, 0]

sock.setblocking(False)
while True:

	newestData = None

	while newestData == None:
		keepReceiving = True
		while keepReceiving:
			try:
				interface.userDisplay.checkClose()
				interface.vision_inference_step()
				data, fromAddr = sock.recvfrom(2048)
				if data:
					newestData = data
					if data[0] != 5:
						keepReceiving = False
			except socket.error as why:
				keepReceiving = False

	data = newestData
		
	command = data[0]
	val1 	= data[1]
	val2 	= data[2]
	val3 	= data[3]
	val4 	= data[4]
	val5 	= data[5]
	val6 	= data[6]
	val7 	= data[7]
	val8 	= data[8]
	val9 	= data[9]
	val10 	= data[10]

	if command == 0:
		if val1 == 1:		# Reset 
			interface.reset()
			print("RESET")
		if val1 == 2:		# Set Mode
			interface.setMode(val2)
			print("Mode: ", val2)
		if val1 == 3:		# Set 
			interface.targetRadius = val2/1000.0
		if val1 == 4:		# Set 
			# interface.targetHoldTime = val2
			interface.holdTimeSteps  = val2/0.2
			# interface.holdTimeSteps = 1
			print(interface.holdTimeSteps)

		if val1 == 6:
			H = val2
			M = val3
			S = val4
			dataDir = "Data/Bravo1/"+ time.strftime("%Y%m%d") + "/" + str(H).zfill(2) + str(M).zfill(2)+ str(S).zfill(2)
			if not os.path.exists(dataDir):
				os.makedirs(dataDir)

		if val1 == 5:  #Start logger
			trialInd = len(os.listdir(dataDir))
			print(trialInd)
			fn = dataDir + "/data00" + str(trialInd) + ".csv"
			fnp = dataDir + "/param00" + str(trialInd) + ".csv"
			fnk = dataDir + "/robot00" + str(trialInd) + ".csv"
			interface.saveTrialParams(fnp)
			interface.startLogger(fn)
			interface.startLogger2(fnk)
			interface.inTrial = 1

		if val1 == 7:		# Set Alpha
			interface.assistAlpha = val2/100.0
			print("Alpha ", val2/100.0)

		if val1 == 8:		# Set AutoGrasp
			interface.AG = val2

		if val1 == 9:		# Set clicker hold time
			interface.runningSwitchBinNum  = val2

		if val1 == 10:		# Set autocenter flag
			interface.autoCenterOverTarget  = val2

		if val1 == 11:		# Set autocenter dist
			interface.autoCenterDist  = val2*.01

		if val1 == 12:		# Set clicker hold time
			interface.wristStartX  = val2*.1
			print("WRISTX2: ", interface.wristStartX)

		if val1 == 13:		# Set clicker hold time
			interface.wristStartZ  = val2*.1

		if val1 == 14:		# Set clicker hold time
			interface.operationalModeReset = val2

		if val1 == 15:		# Set clicker hold time
			interface.wl[2] = val2*.01

		if val1 == 16:		# Set clicker hold time
			interface.lowGainMode = val2
			
		if val1 == 17:		# Set clicker hold time
			interface.graspOrientation = val2
		if val1 == 18:		# 
			interface.runningSwitchBinNum = val2
		if val1 == 19:		# 
			interface.runningSwitchThresh = val2*0.1
		if val1 == 20:		# 
			interface.runningGraspBinNum = val2
		if val1 == 21:		# 
			interface.runningGraspThresh = val2*0.1

		if val1 == 22:
			interface.wl[0] = ((val2-1) *(val3 + val4/100.0))/ 100.0
			interface.wl[1] = ((val5-1) *(val6 + val7/100.0))/ 100.0
			interface.wl[2] = ((val8-1) *(val9 + val10/100.0))/ 100.0
			print("WL", interface.wl )
		if val1 == 23:			
			interface.wu[0] = ((val2-1) *(val3 + val4/100.0))/ 100.0
			interface.wu[1] = ((val5-1) *(val6 + val7/100.0))/ 100.0
			interface.wu[2] = ((val8-1) *(val9 + val10/100.0))/ 100.0
			print("WU", interface.wu )
		if val1 == 25:		# Stop for Beta Stop Task
			sock.sendto(str(interface.pose[1]),("127.0.0.1", 43210))
			interface.stopRobot()
		if val1 == 26:		# Set robot dynamics
			interface.kv_f 	= val2/10.0
			interface.ki_f 	= val3/1000.0
			interface.gkv 	= val4/10.0
			interface.gki 	= val5/1000.0
			print("DYN:", interface.kv_f , interface.ki_f , interface.gkv ,interface.gki )
		if val1 == 27:		
			if val2 == 1:
				print("CLOSE")
				interface.setGripper(interface.gripperClose)
			if val2 == 2:
				interface.setGripper(interface.gripperOpen)

		if val1 == 28: # set next path goal
			interface.goalVal[1] = -((val2-1) *(val3 + val4/100.0))/ 1000.0 - 0.4
			interface.goalVal[0] = -((val5-1) *(val6 + val7/100.0))/ 1000.0 - 0.2
			interface.goalVal[2] = ((val8-1) *(val9 + val10/100.0) + 256)/ 1000.0

		if val1 == 29: # set goal Decode
			interface.goalDim = val2
			interface.goalMet = 0
			# update goal text
			interface.userDisplay.updateText(val2)

			print("GOAL:", interface.goalVal, interface.goalDim)

		if val1 == 30:
			print("CHANGE, val2")
			if val2 == 1:
				interface.userDisplay.changeBGColor((255,0,0))
			elif val2 == 0:
				interface.userDisplay.changeBGColor((0,255,0))

		if val1 == 31:		# Set Wrist Orientation
			interface.wristStartX  = ((val2-1) *(val3 + val4/100.0))/ 10.0
			interface.wristStartY  = ((val5-1) *(val6 + val7/100.0))/ 10.0
			interface.wristStartZ  = ((val8-1) *(val9 + val10/100.0))/ 10.0

		if val1 == 32:  #flipping detected
			print('FLIP STOP')
			interface.flipStop()

		if val1 == 33:
			interface.UseAutoGraspTD 		= val2
			interface.WaitForGraspSignal 	= val3
			interface.UseHeightOffset 		= val4
			interface.AutoGraspHorzDist     = val5/100.0
			interface.AutoGraspVertDist 	= val6/100.0

			if interface.UseHeightOffset == 0:
				interface.AutoGraspVertDist = 0

		if val1 == 34:  # Read goal pos from file
			interface.readObjects()
			# interface.goToObjects()
			interface.initializeBI()
			interface.assistStart = 0

		if val1 == 35:
			interface.AssistMode = val2

		if val1 == 36:		# robot position
			interface.view = val2

		if val1 == 37:		# use mode switch
			interface.UseModeSwitch = val2

		if val1 == 38:		# use mode switch
			interface.PreviewTarget = val2
			print("preview, ", interface.PreviewTarget)

		if val1 == 39:
			interface.runningInputBinNum = val2
			interface.runningInputThresh = val3*0.1

		if val1 == 40: # set bg color
			if val2 == 0:
				interface.userDisplay.changeBGColor((0,0,0))
			if val2 == 1:
				interface.userDisplay.changeBGColor((255,0,0))
			if val2 == 2:
				interface.userDisplay.changeBGColor((0,255,0))
			if val2 == 3:
				interface.userDisplay.changeBGColor((0,0,255))

		if val1 == 41: # set belief 
			interface.beliefThresh = val2/10.0
			interface.distB = val3/10.0
			interface.distK = val4/10.0
			interface.velB = val5/10.0
			interface.velK = val6/10.0
			interface.pDiag = val7/10.0
			interface.slowThresh = val8/1000.0
			print("BELIEF PARAMS: ", interface.beliefThresh,interface.distB,interface.distK,interface.velB, interface.velK, interface.pDiag)

		if val1 == 42: # set gripper lock out
			interface.lockGripperClosed = val2

		if val1 == 43: # set slow mode params
			interface.UseSlowMode = val2
			interface.SlowDistanceThreshold = val3/100.0
			interface.kv_s 	= val4/100.0
			interface.ki_s 	= val5/1000.0

	if command == 2:	# Read target1Pos
		target_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
		target_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
		target_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
		interface.t1_pos = target_pos + [-0.20, -0.40, 0]
		# print('targetPos: ', interface.t1_pos )

	if command == 3:	# Read target2Pos
		target_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
		target_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
		target_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
		interface.t2_pos = target_pos + [-0.20, -0.40, 0]
		# print('targetPos: ', interface.t2_pos)

	# if command == 3:	# Read targetPos
	# 	target_ang = -((val1-1) *(val2 + val3/100.0))/ 1000.0
	# 	interface.target_ang = target_ang

	if command == 4:	# Read position goal 
		robot_pos[1] = -((val1-1) *(val2 + val3/100.0))/ 1000.0
		robot_pos[0] = -((val4-1) *(val5 + val6/100.0))/ 1000.0
		robot_pos[2] = ((val7-1) *(val8 + val9/100.0) + 256)/ 1000.0
		interface.matlabPos = robot_pos
		print(robot_pos)

	if command == 5:	# Read Decoder Output
		interface.inputToAction(val10)
		interface.setVelocity(interface.V, interface.R)
		sock.sendto(str(interface.modeswitch).encode(),("127.0.0.1", 43210))

	if command == 6:	# Read Decoder Output (Path Task)
		# if interface.inTrial:
		print(interface.pose)
		interface.inputToAction(val10)
		interface.setVelocity(interface.V, interface.R)

		interface.checkPathGoal()
		sock.sendto(str(interface.goalMet),("127.0.0.1", 43210))
		if interface.goalMet:
			interface.V = [0,0,0]
			interface.R = [0,0,0]

	if command == 7:	# Read Decoder Output (Beta Stop Task)
		# if interface.inTrial:
		sock.sendto(str(interface.pose[1]),("127.0.0.1", 43210))
		interface.inputToAction(val10)
		interface.setVelocity(interface.V, interface.R)


	


sock.shutdown()
sock.close() 
