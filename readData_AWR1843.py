import serial
from serial import Serial
import time
import numpy as np
import cv2
#import pyqtgraph as pg
#from pyqtgraph.Qt import QtGui

#import pyqtgraph as pg
#from pyqtgraph.Qt import QtGui
from PyQt5 import QtWidgets
import pyqtgraph as pg

# Change the configuration file name
configFileName = 'xwr16xx_profile_2024_static_clutter.cfg'
#configFileName = 'xwr16xx_profile_2024_.cfg'
#configFileName = 'car_detection.cfg'
#configFileName = 'calib_corridor.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0;

# camera variables

camera_num = 0

camera_x = 1920 # width of camera picture
camera_z = 1080 # height of camera picture
CamObj = cv2.VideoCapture(camera_num, cv2.CAP_DSHOW)
#CamObj.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#CamObj.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamObj.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_z)
CamObj.set(cv2.CAP_PROP_FRAME_WIDTH, camera_x)

cameraFOV = 76
cameraFOV_rad = np.deg2rad(cameraFOV)

camera_center_z = int(camera_z/2)
camera_center_x = int(camera_x/2)

P = camera_x/2 # half of the screen where detected object can 
FOV_half_rad =  cameraFOV_rad/2
tan_FOV_half = np.tan(FOV_half_rad)
false_distance = (camera_x/2)/tan_FOV_half

# print vaiables
print_i = 0
image_saved_num = 0

#


# ------------------------------------------------------------------

def detection_2_pixels(x_array, y_array) -> list[int]:
    pixels_movement = []
    for i in range(0, len(x_array)):
        x_ = x_array[i]
        y_ = y_array[i]
        obj_tan = x_/y_
        cam_x = false_distance * obj_tan
        if np.abs(cam_x) <= camera_center_x - 5:
            #cam_x = camera_center_x - 5
            pixels_movement.append(int(cam_x))
    return pixels_movement

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Windows
    CLIport = Serial('COM7', 115200)
    Dataport = Serial('COM6', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;
                
            digOutSampleRate = int(splitWords[11]);
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = float(splitWords[5]);

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData18xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2**15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
    maxBufferSize = 2**15;
    tlvHeaderLengthInBytes = 8;
    pointLengthInBytes = 16;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)
    
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]
        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # Initialize the arrays
                x = np.zeros(numDetectedObj,dtype=np.float32)
                y = np.zeros(numDetectedObj,dtype=np.float32)
                z = np.zeros(numDetectedObj,dtype=np.float32)
                velocity = np.zeros(numDetectedObj,dtype=np.float32)
                
                for objectNum in range(numDetectedObj):
                    
                    # Read the data for each object
                    
                    x_temp = byteBuffer[idX:idX + 4].view(dtype=np.float32)

                    x[objectNum] = x_temp[0]
                    idX += 4
                    y_temp = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    y[objectNum] = y_temp[0]
                    idX += 4
                    z_temp = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    z[objectNum] = z_temp[0]
                    idX += 4
                    vel_temp = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    velocity[objectNum] = vel_temp[0] # improve
                    idX += 4
                
                # Store the data in the detObj dictionary
                detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity":velocity}
                dataOK = 1
                
 
        # Remove already processed data
        if idX > 0 and byteBufferLength>idX:
            shiftSize = totalPacketLen
            
                
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0         

    return dataOK, frameNumber, detObj

# ------------------------------------------------------------------

def draw_detections(pixels, image):
    for i in pixels:
        first = 0 # blue
        second = 0 #green
        third = 255 # red

        # centrum
        image[camera_center_z][camera_center_x + i][0] = first
        image[camera_center_z][camera_center_x + i][1] = second
        image[camera_center_z][camera_center_x + i][2] = third
        
        image[camera_center_z][camera_center_x + i - 3][0] = first
        image[camera_center_z][camera_center_x + i - 3][1] = second
        image[camera_center_z][camera_center_x + i - 3][2] = third
        
        image[camera_center_z][camera_center_x + i + 3][0] = first
        image[camera_center_z][camera_center_x + i + 3][1] = second
        image[camera_center_z][camera_center_x + i + 3][2] = third
        
        image[camera_center_z+1][camera_center_x + i - 3][0] = first
        image[camera_center_z+1][camera_center_x + i - 3][1] = second
        image[camera_center_z+1][camera_center_x + i - 3][2] = third
        
        image[camera_center_z+1][camera_center_x + i + 3][0] = first
        image[camera_center_z+1][camera_center_x + i + 3][1] = second
        image[camera_center_z+1][camera_center_x + i + 3][2] = third
        
        image[camera_center_z-1][camera_center_x + i - 3][0] = first
        image[camera_center_z-1][camera_center_x + i - 3][1] = second
        image[camera_center_z-1][camera_center_x + i - 3][2] = third
        
        image[camera_center_z-1][camera_center_x + i + 3][0] = first
        image[camera_center_z-1][camera_center_x + i + 3][1] = second
        image[camera_center_z-1][camera_center_x + i + 3][2] = third
        
        image[camera_center_z+2][camera_center_x + i - 3][0] = first
        image[camera_center_z+2][camera_center_x + i - 3][1] = second
        image[camera_center_z+2][camera_center_x + i - 3][2] = third
        
        image[camera_center_z+2][camera_center_x + i + 3][0] = first
        image[camera_center_z+2][camera_center_x + i + 3][1] = second
        image[camera_center_z+2][camera_center_x + i + 3][2] = third
        
        image[camera_center_z-2][camera_center_x + i - 3][0] = first
        image[camera_center_z-2][camera_center_x + i - 3][1] = second
        image[camera_center_z-2][camera_center_x + i - 3][2] = third
        
        image[camera_center_z-2][camera_center_x + i + 3][0] = first
        image[camera_center_z-2][camera_center_x + i + 3][1] = second
        image[camera_center_z-2][camera_center_x + i + 3][2] = third
        
        image[camera_center_z+3][camera_center_x + i - 3][0] = first
        image[camera_center_z+3][camera_center_x + i - 3][1] = second
        image[camera_center_z+3][camera_center_x + i - 3][2] = third
        
        image[camera_center_z+3][camera_center_x + i + 3][0] = first
        image[camera_center_z+3][camera_center_x + i + 3][1] = second
        image[camera_center_z+3][camera_center_x + i + 3][2] = third
        
        image[camera_center_z-3][camera_center_x + i - 3][0] = first
        image[camera_center_z-3][camera_center_x + i - 3][1] = second
        image[camera_center_z-3][camera_center_x + i - 3][2] = third
        
        image[camera_center_z-3][camera_center_x + i + 3][0] = first
        image[camera_center_z-3][camera_center_x + i + 3][1] = second
        image[camera_center_z-3][camera_center_x + i + 3][2] = third
        
        image[camera_center_z-3][camera_center_x + i + 0][0] = first
        image[camera_center_z-3][camera_center_x + i + 0][1] = second
        image[camera_center_z-3][camera_center_x + i + 0][2] = third
        
        image[camera_center_z+3][camera_center_x + i + 0][0] = first
        image[camera_center_z+3][camera_center_x + i + 0][1] = second
        image[camera_center_z+3][camera_center_x + i + 0][2] = third
        
        image[camera_center_z-3][camera_center_x + i + 1][0] = first
        image[camera_center_z-3][camera_center_x + i + 1][1] = second
        image[camera_center_z-3][camera_center_x + i + 1][2] = third
        
        image[camera_center_z+3][camera_center_x + i + 1][0] = first
        image[camera_center_z+3][camera_center_x + i + 1][1] = second
        image[camera_center_z+3][camera_center_x + i + 1][2] = third
        
        image[camera_center_z-3][camera_center_x + i - 1][0] = first
        image[camera_center_z-3][camera_center_x + i - 1][1] = second
        image[camera_center_z-3][camera_center_x + i - 1][2] = third
        
        image[camera_center_z+3][camera_center_x + i - 1][0] = first
        image[camera_center_z+3][camera_center_x + i - 1][1] = second
        image[camera_center_z+3][camera_center_x + i - 1][2] = third
        
        image[camera_center_z-3][camera_center_x + i + 2][0] = first
        image[camera_center_z-3][camera_center_x + i + 2][1] = second
        image[camera_center_z-3][camera_center_x + i + 2][2] = third
        
        image[camera_center_z+3][camera_center_x + i + 2][0] = first
        image[camera_center_z+3][camera_center_x + i + 2][1] = second
        image[camera_center_z+3][camera_center_x + i + 2][2] = third
        
        image[camera_center_z-3][camera_center_x + i - 2][0] = first
        image[camera_center_z-3][camera_center_x + i - 2][1] = second
        image[camera_center_z-3][camera_center_x + i - 2][2] = third
        
        image[camera_center_z+3][camera_center_x + i - 2][0] = first
        image[camera_center_z+3][camera_center_x + i - 2][1] = second
        image[camera_center_z+3][camera_center_x + i - 2][2] = third
        
        
    return image

def draw_detections_bb(pixels, image, h, w):
    img_with_box = image
    for i in pixels:
        if (i + camera_center_x) > int(w/2) & (i + camera_center_x) < (camera_x - int(w/2)):
            
            y = camera_center_z - int(h/2)
            x = i - int(w/2) + camera_center_x
            img_with_box = cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_with_box


def draw_line_in_center(image):
    
    #res_img = deepcopy(image)
    res_img = image
    for i in range(0, 120):
        
        res_img[camera_z-1-i][camera_center_x][0] = 255
        res_img[camera_z-1-i][camera_center_x-1][0] = 255
        res_img[camera_z-1-i][camera_center_x][1] = 0
        res_img[camera_z-1-i][camera_center_x-1][1] = 0
        res_img[camera_z-1-i][camera_center_x][2] = 0
        res_img[camera_z-1-i][camera_center_x-1][2] = 0

    return res_img

        
def save_txt(str, x, y):
    combined_array = np.column_stack((x, y))
    # Save to a text file
    np.savetxt(str, combined_array, fmt='%f', delimiter=',', header='x, y', comments='')

    print("Arrays saved to {txt}".format(txt = str))

# Funtion to update the data and display in the plot
def update(p_i, image_saved_num):
     
    dataOk = 0
    global detObj
    x = []
    y = []
      
    # Read and parse the received data
    dataOk, frameNumber, detObj = readAndParseData18xx(Dataport, configParameters)
    
    success, currentFrame = CamObj.read()
    image = currentFrame
    
    if dataOk and len(detObj["x"])>0:
        #print(detObj)
        x = detObj["x"]
        y = detObj["y"]
        z_dim = detObj["z"]
        
        # alt print loc
        if p_i % 5 == 0:

            #txt_file_name = "car_detection_test_1_day_{num}.txt".format(num = image_saved_num)
            #txt_file_name2 = "car_detection_test_1_day_{num}.jpg".format(num = image_saved_num)
            
            txt_file_name = "human_detection_inside_night_{num}.txt".format(num = image_saved_num)
            txt_file_name2 = "human_detection_inside_night_{num}.jpg".format(num = image_saved_num)
            
            save_txt(txt_file_name, x, y)
            cv2.imwrite(txt_file_name2, image)
            image_saved_num += 1
        # -----
        
        pixels_detected = detection_2_pixels(x, y) #ODLEGLOSC OD SRODKA OBRAZU!
        image1 = draw_detections(pixels_detected, image)
        image2 = draw_detections_bb(pixels_detected, image1, 240, 320)
        image2 = draw_line_in_center(image2)
        
        # print loc
        #if p_i % 20 == 0:

            #txt_file_name = "car_detection_test_1_day_{num}.txt".format(num = image_saved_num)
            #txt_file_name2 = "car_detection_test_1_day_{num}.jpg".format(num = image_saved_num)
            
            #txt_file_name = "calibration_test_pre_car_test_2_{num}.txt".format(num = image_saved_num)
            #txt_file_name2 = "calibration_test_pre_car_test_2_{num}.jpg".format(num = image_saved_num)
            
            #save_txt(txt_file_name, x, y)
            #cv2.imwrite(txt_file_name2, image2)
            #image_saved_num += 1
        # -------
        
        # s is the plot object
        s.setData(x,y)
        QtWidgets.QApplication.processEvents()
        #app = QtGui.QApplication([])
    
        if camera_x == 1920:
            # Define the desired dimensions for the resized image
            width = 1280
            height = 720

            # Resize the image
            resized_image = cv2.resize(image2, (width, height))
            cv2.imshow("frame", resized_image)
            cv2.waitKey(1)


        else:
        
            cv2.imshow("frame2", image2)
            cv2.waitKey(1)


    
    return dataOk, image_saved_num


# -------------------------    MAIN   -----------------------------------------  

# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# START QtAPPfor the plot
#app = QtGui.QApplication([])
app = QtWidgets.QApplication([])

# Set the plot 
pg.setConfigOption('background','w')
win = pg.GraphicsLayoutWidget(title="2D scatter plot")
p = win.addPlot()
p.setXRange(-4.5,4.5)
p.setYRange(0,9)
p.setLabel('left',text = 'Y position (m)')
p.setLabel('bottom', text= 'X position (m)')
s = p.plot([],[],pen=None,symbol='o')
win.show()

# Main loop 
detObj = {}  
frameData = {}    
currentIndex = 0
while True:
    try:
        # Update the data and check if the data is okay
        dataOk, image_saved_num = update(print_i, image_saved_num)
        print_i += 1
        
        if dataOk:
            # Store the current frame into frameData
            frameData[currentIndex] = detObj
            currentIndex += 1
        
        time.sleep(0.05) # Sampling frequency of 30 Hz
        
    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        win.close()
        break
        
    





